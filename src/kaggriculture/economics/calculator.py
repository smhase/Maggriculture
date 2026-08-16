"""Economic analysis for crops, animals, and capital ranking.

Uses current market sell prices and official ``CROPS`` / ``ANIMALS`` constants.
Movement overhead is estimated coarsely (Manhattan to nearest empty tile /
shed); agents should refine with actual pathing later.

Terminal-horizon note: investments that mature after ``turns_remaining`` are
marked with ``terminal_value_factor`` near 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from kaggriculture.env.rules import (
    ANIMALS,
    CROPS,
    DEFAULT_TURNS_PER_DAY,
    MARKET_PARAMS,
)
from kaggriculture.env.state import GameState
from kaggriculture.env.tiles import is_empty


@dataclass(frozen=True)
class CropAnalysis:
    crop: str
    seed_cost: float
    first_yield_day: int
    max_yield_day: int
    ongoing: bool
    max_yield: int
    turns_to_first_yield: int
    turns_to_peak: int
    water_actions_estimate: int
    expected_yield_watered: float
    expected_yield_fertilized: float
    market_unit_price: float
    expected_revenue_watered: float
    expected_revenue_fertilized: float
    expected_profit_watered: float
    expected_profit_fertilized: float
    profit_per_turn_watered: float
    profit_per_tile: float
    capital_efficiency: float  # profit / seed_cost
    turns_remaining: int
    matures_before_terminal: bool
    terminal_value_factor: float
    notes: str = ""


@dataclass(frozen=True)
class AnimalAnalysis:
    animal: str
    acquisition_cost: float
    structure: str
    product: str
    first_yield_day: int
    interval: int
    max_held: int
    market_unit_price: float
    feed_cost_per_day_estimate: float
    turns_to_first_yield: int
    expected_productions_in_horizon: int
    expected_gross_revenue: float
    expected_feed_cost: float
    expected_profit: float
    profit_per_turn: float
    capital_efficiency: float
    turns_remaining: int
    matures_before_terminal: bool
    terminal_value_factor: float
    notes: str = ""


@dataclass(frozen=True)
class InvestmentRank:
    kind: str  # "crop" | "animal" | "land"
    name: str
    score: float
    profit: float
    profit_per_turn: float
    capital_efficiency: float
    detail: Any


def _unit_price(state: GameState, product: str) -> float:
    return float(state.market.prices.get(product, MARKET_PARAMS.get(product, {}).get("base", 0)))


def _wheat_feed_price(state: GameState) -> float:
    # Feeding uses wheat from inventory; opportunity cost ≈ market sell price
    # (or buy price if purchasing feed). Use sell price as conservative estimate.
    return _unit_price(state, "WHEAT")


def _empty_unlocked_count(state: GameState) -> int:
    tiles = state.self_player.farm.tiles
    return sum(1 for row in tiles for t in row if is_empty(t))


def analyze_crop(
    state: GameState,
    crop_type: str,
    *,
    turns_per_day: int = DEFAULT_TURNS_PER_DAY,
    fertilize: bool = False,
) -> CropAnalysis:
    if crop_type not in CROPS:
        raise ValueError(f"Unknown crop {crop_type!r}")
    cd = CROPS[crop_type]
    seed = float(cd["seed"])
    first = int(cd["first_yield_day"])
    peak = int(cd["max_yield_day"])
    ongoing = bool(cd["ongoing"])
    max_y = int(cd["max_yield"])
    price = _unit_price(state, crop_type)
    remaining = int(state.turns_remaining)
    max_day = max(0, (state.episode_steps - 1) // turns_per_day)
    # Harvest unlocks on calendar day (planted_day + first_yield_day).
    matures = (state.day + first) <= max_day
    days_to_first = first
    days_to_peak = peak if not ongoing else peak
    turns_to_first = max(0, (state.day + first) * turns_per_day - state.turn)
    turns_to_peak = max(0, (state.day + days_to_peak) * turns_per_day - state.turn)
    # Daily water from plant day through peak (inclusive of critical days)
    water_actions = peak + 1  # plant day through max_yield_day inclusive

    if ongoing:
        interval = max(1, int(cd["interval"]))
        n_prod = max_y
        last_age = first + (n_prod - 1) * interval
        days_to_peak = last_age
        turns_to_peak = max(0, (state.day + last_age) * turns_per_day - state.turn)
        expected_watered = float(n_prod)
        expected_fert = float(n_prod * 2) if fertilize else expected_watered
        notes = (
            f"Ongoing: {n_prod} scheduled yields (interval={interval}); "
            "fertilizer doubles yield on watered production days."
        )
    else:
        window_start = (peak + 1) // 2
        bonus_days = max(0, peak - window_start + 1)
        expected_watered = float(min(max_y, 1 + bonus_days))
        expected_fert = float(min(max_y, 1 + 2 * bonus_days))
        notes = (
            f"One-time: bonus window ages {window_start}–{peak}; "
            f"watered yield≈{expected_watered}, fertilized≈{expected_fert}."
        )

    rev_w = expected_watered * price
    rev_f = expected_fert * price
    fert_cost = _unit_price(state, "FERTILIZER") if fertilize else 0.0
    profit_w = rev_w - seed
    profit_f = rev_f - seed - fert_cost

    if not matures:
        tv = 0.0
    elif turns_to_peak <= remaining:
        tv = 1.0
    elif turns_to_first > remaining:
        tv = 0.0
    else:
        tv = max(0.0, (remaining - turns_to_first) / max(1.0, turns_to_peak - turns_to_first))

    # Occupancy-normalized return: profit / days the tile is tied up
    occ_days = max(1.0, float(days_to_peak if not ongoing else days_to_first + max_y))
    ppt = (profit_w * tv) / (occ_days * turns_per_day)

    return CropAnalysis(
        crop=crop_type,
        seed_cost=seed,
        first_yield_day=first,
        max_yield_day=peak,
        ongoing=ongoing,
        max_yield=max_y,
        turns_to_first_yield=turns_to_first,
        turns_to_peak=turns_to_peak,
        water_actions_estimate=water_actions,
        expected_yield_watered=expected_watered,
        expected_yield_fertilized=expected_fert,
        market_unit_price=price,
        expected_revenue_watered=rev_w,
        expected_revenue_fertilized=rev_f,
        expected_profit_watered=profit_w * tv,
        expected_profit_fertilized=profit_f * tv,
        profit_per_turn_watered=ppt,
        profit_per_tile=profit_w * tv,
        capital_efficiency=(profit_w * tv) / seed if seed else 0.0,
        turns_remaining=remaining,
        matures_before_terminal=matures,
        terminal_value_factor=tv,
        notes=notes,
    )


def analyze_animal(
    state: GameState,
    animal_type: str,
    *,
    turns_per_day: int = DEFAULT_TURNS_PER_DAY,
    care: bool = True,
) -> AnimalAnalysis:
    if animal_type not in ANIMALS:
        raise ValueError(f"Unknown animal {animal_type!r}")
    a = ANIMALS[animal_type]
    cost = float(a["cost"])
    product = str(a["product"])
    first = int(a["first_yield_day"])
    interval = max(1, int(a["interval"]))
    max_held = int(a["max_held"])
    price = _unit_price(state, product)
    remaining = int(state.turns_remaining)
    max_day = max(0, (state.episode_steps - 1) // turns_per_day)
    matures = (state.day + first) <= max_day
    days_left = max(0.0, float(max_day - state.day))
    turns_to_first = max(0, (state.day + first) * turns_per_day - state.turn)

    if not matures:
        n_prod = 0
        tv = 0.0
    else:
        n_prod = 0
        age = float(first)
        while age <= days_left and n_prod < 10_000:
            n_prod += 1
            age += interval
        tv = 1.0

    units_per_prod = 1.0 + (0.5 if care else 0.0)
    units_per_prod = min(float(max_held), units_per_prod)

    feed_price = _wheat_feed_price(state)
    feed_days = max(0.0, days_left)
    feed_cost = feed_days * feed_price
    gross = n_prod * units_per_prod * price
    profit = (gross - feed_cost - cost) * tv
    ppt = profit / max(1.0, float(remaining)) if remaining else 0.0

    notes = (
        f"Steady-state ~{1.0 / interval:.2f} products/day after day {first}; "
        f"feed wheat daily (opp. cost ≈ ${feed_price:.0f}/day). "
        "Structure build action + PLACE not included in acquisition_cost."
    )

    return AnimalAnalysis(
        animal=animal_type,
        acquisition_cost=cost,
        structure=str(a["structure"]),
        product=product,
        first_yield_day=first,
        interval=interval,
        max_held=max_held,
        market_unit_price=price,
        feed_cost_per_day_estimate=feed_price,
        turns_to_first_yield=turns_to_first,
        expected_productions_in_horizon=n_prod,
        expected_gross_revenue=gross * tv,
        expected_feed_cost=feed_cost,
        expected_profit=profit,
        profit_per_turn=ppt,
        capital_efficiency=profit / cost if cost else 0.0,
        turns_remaining=remaining,
        matures_before_terminal=matures,
        terminal_value_factor=tv,
        notes=notes,
    )


def rank_investments(
    state: GameState,
    *,
    include_animals: bool = True,
    include_land: bool = True,
) -> list[InvestmentRank]:
    """Rank crop/animal options by profit_per_turn * terminal factor, then capital efficiency."""
    ranks: list[InvestmentRank] = []

    for crop in CROPS:
        a = analyze_crop(state, crop)
        score = a.profit_per_turn_watered
        ranks.append(
            InvestmentRank(
                kind="crop",
                name=crop,
                score=score,
                profit=a.expected_profit_watered,
                profit_per_turn=a.profit_per_turn_watered,
                capital_efficiency=a.capital_efficiency,
                detail=a,
            )
        )

    if include_animals:
        for animal in ANIMALS:
            a = analyze_animal(state, animal)
            ranks.append(
                InvestmentRank(
                    kind="animal",
                    name=animal,
                    score=a.profit_per_turn,
                    profit=a.expected_profit,
                    profit_per_turn=a.profit_per_turn,
                    capital_efficiency=a.capital_efficiency,
                    detail=a,
                )
            )

    if include_land:
        from kaggriculture.env.legal import next_land_cost

        cost = next_land_cost(state.self_player.farm.unlocked_quadrants)
        empty = _empty_unlocked_count(state)
        # Land is valuable when tiles are scarce and horizon is long
        if cost is not None:
            days_left = state.turns_remaining / float(DEFAULT_TURNS_PER_DAY)
            # Heuristic: unlock if < 8 empty tiles and > 10 days left
            score = 0.0
            if empty < 8 and days_left > 10:
                # Compare to best crop profit density * 25 new tiles
                best_crop = max(
                    (analyze_crop(state, c).profit_per_tile for c in CROPS),
                    default=0.0,
                )
                score = (best_crop * 12.0 - cost) / max(1.0, float(state.turns_remaining))
            ranks.append(
                InvestmentRank(
                    kind="land",
                    name="BUY_LAND",
                    score=score,
                    profit=score * state.turns_remaining,
                    profit_per_turn=score,
                    capital_efficiency=score * state.turns_remaining / cost if cost else 0.0,
                    detail={"cost": cost, "empty_tiles": empty},
                )
            )

    ranks.sort(key=lambda r: (r.score, r.capital_efficiency), reverse=True)
    return ranks


def best_crop(state: GameState) -> CropAnalysis:
    analyses = [analyze_crop(state, c) for c in CROPS]
    return max(analyses, key=lambda a: (a.profit_per_turn_watered, a.capital_efficiency))
