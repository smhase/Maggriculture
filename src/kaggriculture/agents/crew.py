"""Crew of specialized unit minds behind one official action dict.

Identities are internal. The engine only sees farmer/hands/market ops.
Hands vanish at EOD; slot k reuses specialty k the next day after re-hire.
"""

from __future__ import annotations

from typing import Any, Optional

from kaggriculture.agents.base import Agent
from kaggriculture.agents.navigation import (
    can_harvest_plant,
    nearest,
    needs_water,
    plant_needs_urgent_water,
    scan_tiles,
    step_toward,
)
from kaggriculture.agents.planner_agent import PlannerAgent
from kaggriculture.agents.profiles import CREW, StrategyProfile, resolve_swing
from kaggriculture.env.legal import hire_cost, is_action_legal
from kaggriculture.env.observation import parse_observation
from kaggriculture.env.rules import DEFAULT_EPISODE_STEPS, DEFAULT_TURNS_PER_DAY
from kaggriculture.env.state import GameState
from kaggriculture.env.tiles import is_empty, is_weed
from kaggriculture.planning.beam_search import beam_search, choose_crop
from kaggriculture.planning.macros import BuySeeds, ExpandFarm, LiquidateInventory, SellCommodity
from kaggriculture.planning.market_model import MarketTracker
from kaggriculture.planning.opponent_model import profile_opponent
from kaggriculture.planning.scheduler import schedule
from kaggriculture.planning.trace import DecisionTrace

HAND_SPECIALTIES = ("stable", "risk", "short_term", "long_term")


class CrewAgent(Agent):
    """Farmer planner plus specialist hands; optional HIRE via market."""

    name = "crew"
    version = "0.1.0"

    def __init__(
        self,
        *,
        episode_steps: int = DEFAULT_EPISODE_STEPS,
        turns_per_day: int = DEFAULT_TURNS_PER_DAY,
        profile: Optional[StrategyProfile] = None,
    ) -> None:
        super().__init__()
        self.episode_steps = int(episode_steps)
        self.turns_per_day = int(turns_per_day)
        self.profile = profile or CREW
        self.name = self.profile.name
        self._opp_money: Optional[float] = None
        self._market = MarketTracker()
        self._farmer = PlannerAgent(
            episode_steps=episode_steps,
            turns_per_day=turns_per_day,
            profile=self.profile,
        )

    def begin_episode(self) -> None:
        super().begin_episode()
        self._opp_money = None
        self._market.reset()

    def act(self, observation: Any, configuration: Optional[Any] = None) -> dict[str, Any]:
        episode_steps = self.episode_steps
        turns_per_day = self.turns_per_day
        if configuration is not None:
            if isinstance(configuration, dict):
                episode_steps = int(configuration.get("episodeSteps", episode_steps))
                turns_per_day = int(configuration.get("turnsPerDay", turns_per_day))
            else:
                episode_steps = int(getattr(configuration, "episodeSteps", episode_steps))
                turns_per_day = int(getattr(configuration, "turnsPerDay", turns_per_day))

        state = parse_observation(observation, episode_steps=episode_steps)
        profile = resolve_swing(
            self.profile,
            self_money=float(state.self_player.farm.money),
            opponent_money=float(state.opponent.farm.money),
        )
        crop = self._farmer.preferred_crop or choose_crop(state)
        opp = profile_opponent(state, previous_money=self._opp_money)
        self._opp_money = opp.money
        market = self._market.observe(state)
        result = beam_search(
            state,
            preferred_crop=crop,
            turns_per_day=turns_per_day,
            profile=profile,
            opponent=opp,
            market=market,
        )
        action = schedule(state, result.best_macro, turns_per_day=turns_per_day)
        claimed: set[tuple[int, int]] = set()
        farmer_target = getattr(result.best_macro, "target", None)
        if farmer_target is not None:
            claimed.add(tuple(farmer_target))

        traces: list[DecisionTrace] = [result.reasoning]
        traces[0].unit = "farmer"
        hands: list[list[Any]] = []
        for index, pos in enumerate(state.self_player.farm.hands):
            identity = HAND_SPECIALTIES[index % len(HAND_SPECIALTIES)]
            op, cause, target = _specialty_op(
                state,
                pos=tuple(pos),
                identity=identity,
                claimed=claimed,
                crop=crop,
                allow_plant=state.hour < turns_per_day - 1,
            )
            hands.append(op)
            if target is not None:
                claimed.add(target)
            traces.append(
                DecisionTrace(
                    source=identity,
                    unit=f"hand:{index}",
                    headline=f"{identity} {op[0]} ({cause})",
                    causes=[cause, f"identity={identity}"],
                    macro=str(op[0]),
                    target=list(target) if target else None,
                )
            )

        market = list(action.get("market") or [])
        hire_cause = _maybe_hire(state, profile, market)
        if hire_cause:
            traces[0].causes.append(hire_cause)
        action = {
            "farmer": list(action.get("farmer") or ["PASS"]),
            "hands": hands,
            "market": market[:10],
        }
        if not is_action_legal(state, action):
            action["hands"] = [["PASS"] for _ in state.self_player.farm.hands]
            traces[0].causes.append("hands_reset_illegal")
        if not isinstance(
            result.best_macro, (SellCommodity, LiquidateInventory, BuySeeds, ExpandFarm)
        ) and any((o or [None])[0] == "SELL" for o in action["market"]):
            traces[0].causes.append("opportunistic_sell")
        self.last_trace = traces  # type: ignore[assignment]
        return action


def _maybe_hire(
    state: GameState,
    profile: StrategyProfile,
    market: list[list[Any]],
) -> Optional[str]:
    if profile.hire == "never":
        return None
    if profile.hire == "later" and state.day < 8:
        return None
    n_hands = len(state.self_player.farm.hands)
    if n_hands >= int(profile.max_hands):
        return None
    cost = hire_cost(state.self_player.farm.hires_today)
    if state.self_player.farm.money < cost + int(profile.cash_reserve):
        return None
    if any((order or [None])[0] == "HIRE" for order in market):
        return "hire_already_queued"
    market.append(["HIRE"])
    return f"hire_slot={n_hands}"


def _specialty_op(
    state: GameState,
    *,
    pos: tuple[int, int],
    identity: str,
    claimed: set[tuple[int, int]],
    crop: str,
    allow_plant: bool,
) -> tuple[list[Any], str, Optional[tuple[int, int]]]:
    farm = state.self_player.farm
    private = state.self_player.private
    fx, fy = pos
    tile = farm.tiles[fy][fx]
    day = state.day
    seeds = int(private.seeds.get(crop, 0)) if private else 0

    urgent = [p for p in scan_tiles(state, lambda t, x, y: plant_needs_urgent_water(t, day)) if p not in claimed]
    waters = [p for p in scan_tiles(state, lambda t, x, y: needs_water(t, day)) if p not in claimed]
    harvests = [p for p in scan_tiles(state, lambda t, x, y: can_harvest_plant(t, day)) if p not in claimed]
    empties = [
        p
        for p in scan_tiles(state, lambda t, x, y: is_empty(t))
        if p not in claimed and seeds > 0 and allow_plant
    ]
    weeds = [p for p in scan_tiles(state, lambda t, x, y: is_weed(t)) if p not in claimed]

    if plant_needs_urgent_water(tile, day):
        return ["WATER"], "urgent_water_here", pos

    if identity == "stable":
        if needs_water(tile, day):
            return ["WATER"], "maintain_here", pos
        target = nearest(pos, urgent) or nearest(pos, waters)
        return _go(pos, target, ["WATER"]), "water", target
    if identity == "risk":
        if is_empty(tile) and seeds > 0 and allow_plant and not urgent:
            return ["PLANT", crop], "plant_here", pos
        target = nearest(pos, empties)
        arrive = ["PLANT", crop] if seeds and allow_plant else ["PASS"]
        return _go(pos, target, arrive), "plant", target
    if identity == "short_term":
        if can_harvest_plant(tile, day):
            return ["HARVEST"], "harvest_here", pos
        target = nearest(pos, harvests)
        return _go(pos, target, ["HARVEST"]), "harvest", target
    if is_weed(tile):
        return ["DIG"], "clear_weed_here", pos
    target = nearest(pos, weeds) or nearest(pos, empties)
    if target is not None:
        arrive = ["DIG"] if target in weeds else (["PLANT", crop] if allow_plant and seeds else ["PASS"])
        return _go(pos, target, arrive), "setup", target
    target = nearest(pos, waters)
    return _go(pos, target, ["WATER"]), "fallback_water", target


def _go(
    pos: tuple[int, int],
    target: Optional[tuple[int, int]],
    arrive: list[Any],
) -> list[Any]:
    if target is None:
        return ["PASS"]
    if target == pos:
        return arrive
    return step_toward(pos, target)
