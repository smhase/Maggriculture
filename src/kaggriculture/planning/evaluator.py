"""Hand-crafted state evaluator with terminal-horizon awareness.

value ≈
  cash
  + shed liquidation at current prices
  + seed recovery (seed cost, not sell — seeds aren't sold)
  + standing crop expected harvest value (discounted by maturity / horizon)
  + carried inventory liquidation
  - mild penalties for weeds and unwatered plants near EOD
"""

from __future__ import annotations

from dataclasses import dataclass

from kaggriculture.economics import analyze_crop
from kaggriculture.env.rules import CROPS, DEFAULT_TURNS_PER_DAY
from kaggriculture.env.state import GameState
from kaggriculture.env.tiles import is_plant, is_weed


@dataclass(frozen=True)
class EvalBreakdown:
    cash: float
    shed_value: float
    inventory_value: float
    crop_value: float
    weed_penalty: float
    thirst_penalty: float
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
    if private is not None:
        for item, n in private.shed.items():
            shed_value += int(n) * float(prices.get(item, 0))
        for inv in private.inventories:
            for item, n in inv.items():
                # Carried goods sell only after shed deposit; still count near full value
                inventory_value += int(n) * float(prices.get(item, 0)) * 0.95

    crop_value = 0.0
    weed_penalty = 0.0
    thirst_penalty = 0.0
    day = state.day
    hours_left_today = max(0, turns_per_day - 1 - state.hour)

    for y, row in enumerate(farm.tiles):
        for x, tile in enumerate(row):
            if is_weed(tile):
                weed_penalty += 5.0
                continue
            if not is_plant(tile):
                continue
            assert isinstance(tile, dict)
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
                # Harvestable now
                crop_value += units * price
            elif analysis.matures_before_terminal:
                # Progress toward harvest; scale by age/peak
                progress = min(1.0, max(0.0, age / max(1.0, float(peak))))
                expected = analysis.expected_yield_watered * price * analysis.terminal_value_factor
                crop_value += expected * (0.35 + 0.65 * progress)
            # else worthless near terminal

            if not tile.get("watered_today", False):
                # Higher penalty if will weed tonight and little time left
                consec = int(tile.get("consecutive_unwatered", 0))
                if consec >= 1:
                    thirst_penalty += 40.0 if hours_left_today <= 2 else 15.0
                else:
                    thirst_penalty += 3.0

    total = cash + shed_value + inventory_value + crop_value - weed_penalty - thirst_penalty
    return EvalBreakdown(
        cash=cash,
        shed_value=shed_value,
        inventory_value=inventory_value,
        crop_value=crop_value,
        weed_penalty=weed_penalty,
        thirst_penalty=thirst_penalty,
        total=total,
    )
