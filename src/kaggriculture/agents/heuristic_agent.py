"""Minimal economic agent — wheat-focused profitable baseline.

Priority each turn:
1. Act on current tile (urgent water → harvest → water → dig → plant).
2. Never idle on a watered immature crop if other tiles need care.
3. Walk to nearest useful tile.
4. Do not plant on the last hour of the day (cannot water before EOD).
5. Sell shed every turn; buy a small seed buffer; expand land only with reserve cash.
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

logger = get_logger("agents.heuristic")


class MinimalEconomicAgent(Agent):
    name = "heuristic"
    version = "0.2.0"

    def __init__(
        self,
        *,
        episode_steps: int = DEFAULT_EPISODE_STEPS,
        preferred_crop: Optional[str] = None,
        turns_per_day: int = DEFAULT_TURNS_PER_DAY,
    ) -> None:
        super().__init__()
        self.episode_steps = int(episode_steps)
        self.preferred_crop = preferred_crop
        self.turns_per_day = int(turns_per_day)

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
        crop = self._choose_crop(state)
        can_plant = self._can_plant(state, crop, turns_per_day)
        market = self._market_orders(state, crop, can_plant)
        farmer, cause = self._farmer_op(state, crop, can_plant)
        hands = [["PASS"] for _ in state.self_player.farm.hands]
        action = make_action(farmer=farmer, hands=hands, market=market)
        causes = [cause]
        if not can_plant:
            causes.append("no_plant_last_hour_or_horizon")
        self.last_trace = DecisionTrace(
            source=self.name,
            headline=f"heuristic {farmer[0]} because {cause}",
            causes=causes,
            macro=str(farmer[0]),
        )
        return action

    def _can_plant(self, state: GameState, crop: str, turns_per_day: int) -> bool:
        if state.hour >= turns_per_day - 1:
            return False  # cannot water before EOD
        analysis = analyze_crop(state, crop)
        return analysis.matures_before_terminal

    def _choose_crop(self, state: GameState) -> str:
        if self.preferred_crop and self.preferred_crop in CROPS:
            if analyze_crop(state, self.preferred_crop).matures_before_terminal:
                return self.preferred_crop
        # Prefer fast one-time crops by capital efficiency among those that mature
        candidates = [
            analyze_crop(state, c)
            for c in CROPS
            if (not CROPS[c]["ongoing"]) and analyze_crop(state, c).matures_before_terminal
        ]
        if not candidates:
            # Fall back to fastest crop even if marginal
            return "WHEAT"
        best = max(
            candidates,
            key=lambda a: (a.capital_efficiency, -a.turns_to_first_yield, a.profit_per_turn_watered),
        )
        return best.crop

    def _market_orders(self, state: GameState, crop: str, can_plant: bool) -> list[list[Any]]:
        private = state.self_player.private
        farm = state.self_player.farm
        if private is None:
            return []
        orders: list[list[Any]] = []

        for item, n in list(private.shed.items()):
            if int(n) > 0:
                orders.append(["SELL", item, int(n)])

        if can_plant:
            empty = sum(1 for row in farm.tiles for t in row if is_empty(t))
            have = int(private.seeds.get(crop, 0))
            want = min(4, max(1, empty))
            cost = int(CROPS[crop]["seed"])
            afford = int(farm.money // cost) if cost else 0
            buy = min(want - have, afford, 4)
            land = next_land_cost(farm.unlocked_quadrants)
            reserve = int(land) if land and empty < 4 else 0
            while buy > 0 and farm.money - buy * cost < reserve + 200:
                buy -= 1
            if buy > 0:
                orders.append(["BUY_SEED", crop, buy])

        # Expand once when flush and cramped
        land = next_land_cost(farm.unlocked_quadrants)
        empty = sum(1 for row in farm.tiles for t in row if is_empty(t))
        if (
            land is not None
            and empty < 5
            and farm.money >= land + 1200
            and state.day <= 15
        ):
            orders.append(["BUY_LAND"])

        return orders[:10]

    def _farmer_op(self, state: GameState, crop: str, can_plant: bool) -> list[Any]:
        farm = state.self_player.farm
        private = state.self_player.private
        fx, fy = farm.farmer
        tile = farm.tiles[fy][fx]
        day = state.day
        seeds = int(private.seeds.get(crop, 0)) if private else 0
        pos = (fx, fy)

        urgent = scan_tiles(state, lambda t, x, y: plant_needs_urgent_water(t, day))
        waters = scan_tiles(state, lambda t, x, y: needs_water(t, day))
        harvests = scan_tiles(state, lambda t, x, y: can_harvest_plant(t, day))
        empties = scan_tiles(state, lambda t, x, y: is_empty(t)) if seeds > 0 and can_plant else []
        weeds = scan_tiles(state, lambda t, x, y: is_weed(t))

        # Prefer leaving current tile if others are urgent and current is already OK
        if plant_needs_urgent_water(tile, day):
            return ["WATER"], "urgent_water"
        if can_harvest_plant(tile, day):
            return ["HARVEST"], "harvest_here"
        if needs_water(tile, day) and not urgent:
            return ["WATER"], "water_here"
        if needs_water(tile, day) and (fx, fy) in urgent:
            return ["WATER"], "urgent_water"
        if is_weed(tile):
            return ["DIG"], "clear_weed"
        if is_empty(tile) and seeds > 0 and can_plant and not urgent and not waters:
            return ["PLANT", crop], "plant_here"
        if is_empty(tile) and seeds > 0 and can_plant and not urgent:
            if not waters:
                return ["PLANT", crop], "plant_here"

        target = (
            nearest(pos, urgent)
            or nearest(pos, harvests)
            or nearest(pos, waters)
            or nearest(pos, empties)
            or nearest(pos, weeds)
        )
        if target is None:
            return ["PASS"], "idle"
        if target == pos:
            if needs_water(tile, day):
                return ["WATER"], "water_here"
            if is_empty(tile) and seeds > 0 and can_plant:
                return ["PLANT", crop], "plant_here"
            if is_weed(tile):
                return ["DIG"], "clear_weed"
            return ["PASS"], "idle"
        return step_toward(pos, target), f"walk_to_{target}"


HeuristicAgent = MinimalEconomicAgent
