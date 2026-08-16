"""Conservative beam search over macro sequences.

We do **not** reimplement the official engine. Futures are scored with a
lightweight mental model + the hand-crafted evaluator on the *current*
GameState, adjusted by estimated macro deltas (cash, crop EV, risk).

This is model-predictive control lite: pick the first macro of the best
short sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from kaggriculture.agents.navigation import manhattan
from kaggriculture.economics import analyze_crop
from kaggriculture.env.rules import CROPS, DEFAULT_TURNS_PER_DAY
from kaggriculture.env.state import GameState
from kaggriculture.planning.evaluator import evaluate_state
from kaggriculture.planning.macros import (
    BuySeeds,
    ClearWeed,
    ExpandFarm,
    HarvestField,
    Idle,
    LiquidateInventory,
    Macro,
    MaintainField,
    PlantCrop,
    SellCommodity,
    WaterUrgent,
    macro_label,
)
from kaggriculture.planning.scheduler import propose_macros


@dataclass(frozen=True)
class PlanNode:
    macros: tuple[Macro, ...]
    score: float


@dataclass(frozen=True)
class BeamResult:
    best_macro: Macro
    score: float
    beam: tuple[PlanNode, ...]
    reasoning: str


def beam_search(
    state: GameState,
    *,
    beam_width: int = 6,
    depth: int = 3,
    preferred_crop: str = "WHEAT",
    turns_per_day: int = DEFAULT_TURNS_PER_DAY,
) -> BeamResult:
    """Search short macro sequences; return the first macro of the best plan."""
    base = evaluate_state(state, turns_per_day=turns_per_day)
    root_candidates = propose_macros(
        state, preferred_crop=preferred_crop, turns_per_day=turns_per_day
    )
    # Hard priority: if any plant will weed tonight, only consider water/harvest first.
    urgent_first = [m for m in root_candidates if isinstance(m, (WaterUrgent, HarvestField))]
    if any(isinstance(m, WaterUrgent) for m in root_candidates):
        root_for_first = urgent_first
    else:
        root_for_first = root_candidates

    beam: list[PlanNode] = [
        PlanNode(macros=(m,), score=base + _macro_delta(state, m, turns_per_day))
        for m in root_for_first
    ]
    beam.sort(key=lambda n: n.score, reverse=True)
    beam = beam[:beam_width]

    for _ in range(1, max(1, depth)):
        expanded: list[PlanNode] = []
        for node in beam:
            for m in root_candidates:
                if node.macros and macro_label(m) == macro_label(node.macros[-1]):
                    if not isinstance(m, Idle):
                        continue
                seq = node.macros + (m,)
                total_delta = sum(
                    _macro_delta(state, mm, turns_per_day) * (0.75**i)
                    for i, mm in enumerate(seq)
                )
                expanded.append(PlanNode(macros=seq, score=base + total_delta))
        expanded.sort(key=lambda n: n.score, reverse=True)
        beam = expanded[:beam_width]
        if not beam:
            break

    if not beam:
        return BeamResult(best_macro=Idle(), score=base, beam=(), reasoning="empty beam → idle")

    best = beam[0]
    first = best.macros[0]
    reasoning = (
        f"base={base:.1f} best={best.score:.1f} plan="
        + " → ".join(macro_label(m) for m in best.macros)
    )
    return BeamResult(best_macro=first, score=best.score, beam=tuple(beam), reasoning=reasoning)


def choose_crop(state: GameState) -> str:
    """Pick preferred crop for planning: mature-able, high capital efficiency, fast."""
    candidates = []
    for crop, meta in CROPS.items():
        if meta["ongoing"]:
            continue
        a = analyze_crop(state, crop)
        if not a.matures_before_terminal:
            continue
        candidates.append(a)
    if not candidates:
        return "WHEAT"
    best = max(
        candidates,
        key=lambda a: (a.capital_efficiency, -a.turns_to_first_yield, a.profit_per_turn_watered),
    )
    return best.crop


def _path_delta(state: GameState, macros: Sequence[Macro], turns_per_day: int) -> float:
    return sum(
        _macro_delta(state, m, turns_per_day) * (0.75**i) for i, m in enumerate(macros)
    )


def _macro_delta(state: GameState, macro: Macro, turns_per_day: int) -> float:
    """Estimated value change from committing to this macro now."""
    farm = state.self_player.farm
    private = state.self_player.private
    pos = farm.farmer
    prices = state.market.prices

    def travel(target: Optional[tuple[int, int]]) -> float:
        if target is None:
            return 0.0
        # Each step costs opportunity; rough 1.5 value units per tile
        return -1.5 * manhattan(pos, target)

    if isinstance(macro, WaterUrgent):
        # Saving a plant from weeding is high value
        return 120.0 + travel(macro.target)

    if isinstance(macro, MaintainField):
        return 25.0 + travel(macro.target)

    if isinstance(macro, HarvestField):
        # Approximate harvest value at target or nearest harvestable already in propose
        bonus = 20.0
        if macro.target is not None:
            x, y = macro.target
            tile = farm.tiles[y][x]
            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                crop = tile.get("crop")
                units = float(tile.get("yield_units", 0))
                bonus = units * float(prices.get(crop, 0))
        return bonus + travel(macro.target)

    if isinstance(macro, PlantCrop):
        a = analyze_crop(state, macro.crop)
        if not a.matures_before_terminal or state.hour >= turns_per_day - 1:
            return -30.0
        return a.expected_profit_watered * 0.5 + travel(macro.target)

    if isinstance(macro, ClearWeed):
        return 8.0 + travel(macro.target)

    if isinstance(macro, BuySeeds):
        if macro.crop not in CROPS:
            return -10.0
        a = analyze_crop(state, macro.crop)
        if not a.matures_before_terminal:
            return -20.0
        have = int(private.seeds.get(macro.crop, 0)) if private else 0
        if have >= 3:
            return -5.0  # already stocked
        # Modest option value — must not outrank watering
        return 8.0 + 2.0 * min(macro.quantity, 3 - have)

    if isinstance(macro, SellCommodity):
        if private is None:
            return 0.0
        have = int(private.shed.get(macro.item, 0))
        qty = have if macro.quantity is None else min(have, int(macro.quantity))
        return qty * float(prices.get(macro.item, 0)) * 0.05  # small nudge; cash already in eval

    if isinstance(macro, LiquidateInventory):
        if private is None:
            return 0.0
        return 5.0 * sum(1 for n in private.shed.values() if int(n) > 0)

    if isinstance(macro, ExpandFarm):
        a = analyze_crop(state, choose_crop(state))
        return a.profit_per_tile * 8.0 - 50.0

    if isinstance(macro, Idle):
        # Penalize idling when work exists
        return -5.0

    return 0.0
