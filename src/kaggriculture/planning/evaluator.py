"""Hand-crafted state evaluator with terminal-horizon awareness.

value ≈
  cash
  + shed liquidation at current prices
  + seed recovery (seed cost, not sell — seeds aren't sold)
  + standing crop expected harvest value (discounted by maturity / horizon)
  + carried inventory liquidation
  + infrastructure (extra land, structures, animals)
  - expected restock / remaining land pressure
  - mild penalties for weeds, unwatered plants, and insolvency risk
"""

from __future__ import annotations

from dataclasses import dataclass

from kaggriculture.economics import analyze_animal, analyze_crop
from kaggriculture.env.legal import next_land_cost
from kaggriculture.env.rules import ANIMALS, CROPS, DEFAULT_TURNS_PER_DAY
from kaggriculture.env.state import GameState
from kaggriculture.env.tiles import has_animal, is_empty, is_plant, is_structure, is_weed


@dataclass(frozen=True)
class EvalBreakdown:
    cash: float
    shed_value: float
    inventory_value: float
    seed_value: float
    crop_value: float
    infrastructure_value: float
    future_cost: float
    weed_penalty: float
    thirst_penalty: float
    risk_penalty: float
    total: float


def evaluate_state(
    state: GameState,
    *,
    turns_per_day: int = DEFAULT_TURNS_PER_DAY,
) -> float:
    return evaluate_breakdown(state, turns_per_day=turns_per_day).total


def evaluate_breakdown(
    state: GameState,
    *,
    turns_per_day: int = DEFAULT_TURNS_PER_DAY,
) -> EvalBreakdown:
    farm = state.self_player.farm
    private = state.self_player.private
    prices = state.market.prices

    cash = float(farm.money)
    shed_value = 0.0
    inventory_value = 0.0
    seed_value = 0.0
    if private is not None:
        for item, n in private.shed.items():
            shed_value += int(n) * float(prices.get(item, 0))
        for inv in private.inventories:
            for item, n in inv.items():
                inventory_value += int(n) * float(prices.get(item, 0)) * 0.95
        for crop, n in private.seeds.items():
            if crop in CROPS:
                seed_value += int(n) * float(CROPS[crop]["seed"])

    crop_value = 0.0
    infrastructure_value = 0.0
    weed_penalty = 0.0
    thirst_penalty = 0.0
    empty_tiles = 0
    plant_count = 0
    thirsty = 0
    day = state.day
    hours_left_today = max(0, turns_per_day - 1 - state.hour)

    extra_land = max(0, len(farm.unlocked_quadrants) - 1)
    infrastructure_value += extra_land * 60.0

    for y, row in enumerate(farm.tiles):
        for x, tile in enumerate(row):
            if is_empty(tile):
                empty_tiles += 1
                continue
            if is_weed(tile):
                weed_penalty += 5.0
                continue
            if is_structure(tile):
                infrastructure_value += 25.0
                if has_animal(tile) and isinstance(tile, dict):
                    animal = str(tile.get("animal", ""))
                    if animal in ANIMALS:
                        try:
                            infrastructure_value += max(
                                0.0, analyze_animal(state, animal).expected_profit * 0.15
                            )
                        except (KeyError, TypeError, ValueError):
                            infrastructure_value += 40.0
                continue
            if not is_plant(tile):
                continue
            assert isinstance(tile, dict)
            plant_count += 1
            crop = tile["crop"]
            if crop not in CROPS:
                continue
            analysis = analyze_crop(state, crop)
            age = day - int(tile.get("planted_day", day))
            first = int(CROPS[crop]["first_yield_day"])
            peak = int(CROPS[crop]["max_yield_day"])
            units = float(tile.get("yield_units", 0))
            price = float(prices.get(crop, 0))

            if age >= first and units > 0:
                crop_value += units * price
            elif analysis.matures_before_terminal:
                progress = min(1.0, max(0.0, age / max(1.0, float(peak))))
                expected = analysis.expected_yield_watered * price * analysis.terminal_value_factor
                crop_value += expected * (0.35 + 0.65 * progress)

            if not tile.get("watered_today", False):
                thirsty += 1
                consec = int(tile.get("consecutive_unwatered", 0))
                if consec >= 1:
                    thirst_penalty += 40.0 if hours_left_today <= 2 else 15.0
                else:
                    thirst_penalty += 3.0

    wheat_seed = float(CROPS.get("WHEAT", {}).get("seed", 2) or 2)
    future_cost = min(empty_tiles, 8) * wheat_seed * 0.25
    land = next_land_cost(farm.unlocked_quadrants)
    if land is not None and empty_tiles < 4 and plant_count >= 6:
        shortfall = max(0.0, float(land) - cash)
        future_cost += shortfall * 0.05

    risk_penalty = 0.0
    if plant_count:
        risk_penalty += 8.0 * (thirsty / plant_count)
    if cash < 50 and plant_count == 0 and seed_value == 0:
        risk_penalty += 100.0

    total = (
        cash
        + shed_value
        + inventory_value
        + seed_value
        + crop_value
        + infrastructure_value
        - future_cost
        - weed_penalty
        - thirst_penalty
        - risk_penalty
    )
    return EvalBreakdown(
        cash=cash,
        shed_value=shed_value,
        inventory_value=inventory_value,
        seed_value=seed_value,
        crop_value=crop_value,
        infrastructure_value=infrastructure_value,
        future_cost=future_cost,
        weed_penalty=weed_penalty,
        thirst_penalty=thirst_penalty,
        risk_penalty=risk_penalty,
        total=total,
    )
