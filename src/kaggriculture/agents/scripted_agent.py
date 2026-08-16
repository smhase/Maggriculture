"""Deterministic scripted season plan (wheat maintenance loop).

Phases by day:
  0–6   Bootstrap wheat on NW; never plant at hour 23.
  7–14  Optional single land unlock if cash reserve allows; keep wheat cycle.
  15–24 Scale carefully; harvest ASAP; sell every turn.
  25–27 Wind-down: stop planting; finish harvests/waters.
  28–29 Liquidate: sell only; clear nothing critical.

Key rule: never PASS-wait on one plant while others need water.
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
from kaggriculture.economics import analyze_crop
from kaggriculture.env.actions import make_action
from kaggriculture.env.legal import next_land_cost
from kaggriculture.env.observation import parse_observation
from kaggriculture.env.rules import CROPS, DEFAULT_EPISODE_STEPS, DEFAULT_TURNS_PER_DAY
from kaggriculture.env.state import GameState
from kaggriculture.env.tiles import is_empty, is_weed
from kaggriculture.logging import get_logger
from kaggriculture.planning.trace import DecisionTrace

logger = get_logger("agents.scripted")

CROP = "WHEAT"


class ScriptedAgent(Agent):
    name = "scripted"
    version = "0.2.0"

    def __init__(
        self,
        *,
        episode_steps: int = DEFAULT_EPISODE_STEPS,
        turns_per_day: int = DEFAULT_TURNS_PER_DAY,
    ) -> None:
        super().__init__()
        self.episode_steps = int(episode_steps)
        self.turns_per_day = int(turns_per_day)
        self._last_phase: Optional[str] = None

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
        phase = self._phase(state.day)
        if phase != self._last_phase:
            logger.debug(
                "day=%s phase=%s money=%s", state.day, phase, state.self_player.farm.money
            )
            self._last_phase = phase

        allow_plant = (
            phase not in ("winddown", "liquidate")
            and state.hour < turns_per_day - 1
            and analyze_crop(state, CROP).matures_before_terminal
        )

        market = self._market(state, phase, allow_plant)
        farmer = self._unit(state, 0, allow_plant)
        hands = [
            self._unit(state, i + 1, allow_plant)
            for i in range(len(state.self_player.farm.hands))
        ]
        action = make_action(farmer=farmer, hands=hands, market=market)
        self.last_trace = DecisionTrace(
            source=self.name,
            headline=f"scripted phase={phase} farmer={farmer[0]}",
            causes=[f"phase={phase}", f"allow_plant={allow_plant}"],
            macro=str(farmer[0]),
        )
        return action

    def _phase(self, day: int) -> str:
        if day <= 6:
            return "bootstrap"
        if day <= 14:
            return "expand"
        if day <= 24:
            return "scale"
        if day <= 27:
            return "winddown"
        return "liquidate"

    def _market(self, state: GameState, phase: str, allow_plant: bool) -> list[list[Any]]:
        private = state.self_player.private
        farm = state.self_player.farm
        if private is None:
            return []
        orders: list[list[Any]] = []

        for item, n in list(private.shed.items()):
            if int(n) > 0:
                orders.append(["SELL", item, int(n)])

        if phase == "liquidate":
            return orders[:10]

        if allow_plant:
            have = int(private.seeds.get(CROP, 0))
            empty = sum(1 for row in farm.tiles for t in row if is_empty(t))
            want = 4 if phase == "bootstrap" else min(6, max(2, empty))
            cost = int(CROPS[CROP]["seed"])
            afford = int(farm.money // cost) if cost else 0
            buy = min(max(0, want - have), afford, 4)
            land = next_land_cost(farm.unlocked_quadrants)
            reserve = int(land) + 1000 if (phase == "expand" and land) else 200
            while buy > 0 and farm.money - buy * cost < reserve:
                buy -= 1
            if buy > 0:
                orders.append(["BUY_SEED", CROP, buy])

        if phase == "expand":
            land = next_land_cost(farm.unlocked_quadrants)
            empty = sum(1 for row in farm.tiles for t in row if is_empty(t))
            # Only unlock NE once, with a healthy cash buffer
            if (
                land is not None
                and land <= 1000
                and empty < 6
                and farm.money >= land + 1500
            ):
                orders.append(["BUY_LAND"])

        return orders[:10]

    def _unit(self, state: GameState, unit_index: int, allow_plant: bool) -> list[Any]:
        farm = state.self_player.farm
        private = state.self_player.private
        if unit_index == 0:
            pos = farm.farmer
        else:
            if unit_index - 1 >= len(farm.hands):
                return ["PASS"]
            pos = farm.hands[unit_index - 1]

        fx, fy = pos
        tile = farm.tiles[fy][fx]
        day = state.day
        seeds = int(private.seeds.get(CROP, 0)) if private else 0

        urgent = scan_tiles(state, lambda t, x, y: plant_needs_urgent_water(t, day))
        waters = scan_tiles(state, lambda t, x, y: needs_water(t, day))
        harvests = scan_tiles(state, lambda t, x, y: can_harvest_plant(t, day))
        empties = scan_tiles(state, lambda t, x, y: is_empty(t)) if seeds > 0 and allow_plant else []
        weeds = scan_tiles(state, lambda t, x, y: is_weed(t))

        if plant_needs_urgent_water(tile, day):
            return ["WATER"]
        if can_harvest_plant(tile, day):
            return ["HARVEST"]
        if needs_water(tile, day) and not [p for p in urgent if p != pos]:
            return ["WATER"]
        if is_weed(tile) and not urgent and not waters and not harvests:
            return ["DIG"]
        if is_empty(tile) and seeds > 0 and allow_plant and not urgent and not waters:
            return ["PLANT", CROP]

        target = (
            nearest(pos, urgent)
            or nearest(pos, harvests)
            or nearest(pos, waters)
            or nearest(pos, empties)
            or nearest(pos, weeds)
        )
        if target is None:
            return ["PASS"]
        if target == pos:
            if needs_water(tile, day):
                return ["WATER"]
            if is_empty(tile) and seeds > 0 and allow_plant:
                return ["PLANT", CROP]
            if is_weed(tile):
                return ["DIG"]
            return ["PASS"]
        return step_toward(pos, target)
