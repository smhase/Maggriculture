"""Beam search over macros with a forward mental model.

Search copies GameState via ``apply_macro``, evaluates leaves, and returns
the first macro of the principal variation. This is not the official engine.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Optional, Sequence

from kaggriculture.agents.navigation import manhattan
from kaggriculture.agents.profiles import StrategyProfile
from kaggriculture.economics import analyze_crop
from kaggriculture.env.rules import CROPS, DEFAULT_TURNS_PER_DAY
from kaggriculture.env.state import GameState
from kaggriculture.planning.evaluator import evaluate_breakdown, evaluate_state
from kaggriculture.planning.forward import apply_macro
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
from kaggriculture.planning.opponent_model import OpponentProfile, opponent_score_adjust
from kaggriculture.planning.scheduler import propose_macros
from kaggriculture.planning.trace import Alternative, DecisionTrace


@dataclass(frozen=True)
class PlanNode:
    macros: tuple[Macro, ...]
    score: float
    state: Optional[GameState] = None


@dataclass(frozen=True)
class BeamResult:
    best_macro: Macro
    score: float
    beam: tuple[PlanNode, ...]
    reasoning: DecisionTrace
    plan: tuple[Macro, ...] = ()


def beam_search(
    state: GameState,
    *,
    beam_width: int = 6,
    depth: int = 3,
    preferred_crop: str = "WHEAT",
    turns_per_day: int = DEFAULT_TURNS_PER_DAY,
    profile: Optional[StrategyProfile] = None,
    time_budget_s: Optional[float] = None,
    opponent: Optional[OpponentProfile] = None,
) -> BeamResult:
    """Search short macro sequences on a forward model; return the PV head."""
    width = int(profile.beam_width) if profile is not None else int(beam_width)
    max_depth = int(profile.search_depth) if profile is not None else int(depth)
    crop = preferred_crop
    budget = (
        float(profile.time_budget_s)
        if profile is not None and time_budget_s is None
        else (0.08 if time_budget_s is None else float(time_budget_s))
    )
    deadline = time.perf_counter() + max(0.001, budget)

    breakdown = evaluate_breakdown(state, turns_per_day=turns_per_day)
    base = breakdown.total
    last = _search_depth(
        state,
        depth=1,
        beam_width=width,
        preferred_crop=crop,
        turns_per_day=turns_per_day,
        profile=profile,
        deadline=deadline,
        opponent=opponent,
    )
    for d in range(2, max(1, max_depth) + 1):
        if time.perf_counter() >= deadline:
            break
        last = _search_depth(
            state,
            depth=d,
            beam_width=width,
            preferred_crop=crop,
            turns_per_day=turns_per_day,
            profile=profile,
            deadline=deadline,
            opponent=opponent,
        )

    if last is None or not last.macros:
        idle = Idle()
        reasoning = _trace(
            source=profile.name if profile else "planner",
            state=state,
            base=base,
            breakdown=breakdown,
            best=idle,
            plan=(idle,),
            score=base,
            beam=(),
            causes=["empty beam → idle"],
        )
        return BeamResult(best_macro=idle, score=base, beam=(), reasoning=reasoning, plan=(idle,))

    first = last.macros[0]
    causes = _causes(state, first, profile)
    if opponent is not None:
        causes.append(
            f"opp_money={opponent.money:.0f} plants={opponent.n_plants} "
            f"expand={opponent.expansion_stage} seller={opponent.likely_seller}"
        )
    reasoning = _trace(
        source=profile.name if profile else "planner",
        state=state,
        base=base,
        breakdown=breakdown,
        best=first,
        plan=last.macros,
        score=last.score,
        beam=(),  # filled below
        causes=causes,
    )
    # Rebuild a small beam snapshot from the last search via a second shallow pass
    # is unnecessary; attach alternatives from a depth-1 ranking.
    root = _root_ranked(state, crop, turns_per_day, profile, width, opponent)
    alts = [
        Alternative(macro=macro_label(node.macros[0]), score=node.score)
        for node in root[1:4]
        if node.macros
    ]
    reasoning.alternatives = alts
    reasoning.headline = (
        f"base={base:.1f} best={last.score:.1f} plan="
        + " → ".join(macro_label(m) for m in last.macros)
    )
    return BeamResult(
        best_macro=first,
        score=last.score,
        beam=tuple(root[:width]),
        reasoning=reasoning,
        plan=last.macros,
    )


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


def _search_depth(
    state: GameState,
    *,
    depth: int,
    beam_width: int,
    preferred_crop: str,
    turns_per_day: int,
    profile: Optional[StrategyProfile],
    deadline: float,
    opponent: Optional[OpponentProfile] = None,
) -> Optional[PlanNode]:
    beam: list[PlanNode] = [
        PlanNode(macros=(), score=_leaf_score(state, profile, turns_per_day, opponent), state=state)
    ]
    for _ply in range(max(1, depth)):
        if time.perf_counter() >= deadline:
            break
        expanded: list[PlanNode] = []
        for node in beam:
            if node.state is None:
                continue
            candidates = _candidates(node.state, preferred_crop, turns_per_day, profile, ply=_ply)
            for macro in candidates:
                if time.perf_counter() >= deadline:
                    break
                nxt = apply_macro(node.state, macro, turns_per_day=turns_per_day)
                seq = node.macros + (macro,)
                expanded.append(
                    PlanNode(
                        macros=seq,
                        score=_leaf_score(nxt, profile, turns_per_day, opponent),
                        state=nxt,
                    )
                )
        if not expanded:
            break
        expanded.sort(key=lambda n: n.score, reverse=True)
        beam = expanded[:beam_width]
    return beam[0] if beam else None


def _root_ranked(
    state: GameState,
    preferred_crop: str,
    turns_per_day: int,
    profile: Optional[StrategyProfile],
    beam_width: int,
    opponent: Optional[OpponentProfile] = None,
) -> list[PlanNode]:
    ranked: list[PlanNode] = []
    for macro in _candidates(state, preferred_crop, turns_per_day, profile, ply=0):
        nxt = apply_macro(state, macro, turns_per_day=turns_per_day)
        ranked.append(
            PlanNode(
                macros=(macro,),
                score=_leaf_score(nxt, profile, turns_per_day, opponent),
                state=nxt,
            )
        )
    ranked.sort(key=lambda n: n.score, reverse=True)
    return ranked[:beam_width]


def _candidates(
    state: GameState,
    preferred_crop: str,
    turns_per_day: int,
    profile: Optional[StrategyProfile],
    *,
    ply: int,
) -> list[Macro]:
    macros = propose_macros(
        state,
        preferred_crop=preferred_crop,
        turns_per_day=turns_per_day,
        profile=profile,
    )
    urgent_first = [m for m in macros if isinstance(m, (WaterUrgent, HarvestField))]
    if ply == 0 and any(isinstance(m, WaterUrgent) for m in macros):
        return urgent_first
    return macros


def _leaf_score(
    state: GameState,
    profile: Optional[StrategyProfile],
    turns_per_day: int,
    opponent: Optional[OpponentProfile] = None,
) -> float:
    score = evaluate_state(state, turns_per_day=turns_per_day)
    self_money = float(state.self_player.farm.money)
    opp_money = float(state.opponent.farm.money)
    objective = profile.objective if profile is not None else "score"
    risk = profile.risk if profile is not None else "stable"
    if objective == "win":
        score += (self_money - opp_money) * 1.5
    elif objective == "never_lose":
        if self_money < opp_money:
            score -= (opp_money - self_money) * 3.0 + 150.0
        else:
            score += min(self_money - opp_money, 400.0) * 0.25
    if risk == "aggressive":
        score += 15.0 * len(
            [q for q in state.self_player.farm.unlocked_quadrants if q]
        )
    elif risk == "safe":
        if self_money < (profile.cash_reserve if profile else 800):
            score -= 80.0
    if opponent is not None:
        score += opponent_score_adjust(state, opponent)
    return score


def _causes(state: GameState, macro: Macro, profile: Optional[StrategyProfile]) -> list[str]:
    causes: list[str] = []
    if isinstance(macro, WaterUrgent):
        causes.append("urgent plant would weed without water")
    elif isinstance(macro, HarvestField):
        causes.append("harvest yield now")
    elif isinstance(macro, PlantCrop):
        causes.append(f"plant {macro.crop}")
    elif isinstance(macro, Idle):
        causes.append("no higher-value macro")
    if profile is not None:
        causes.append(f"profile={profile.name} horizon={profile.horizon} risk={profile.risk}")
        causes.append(f"objective={profile.objective}")
    if state.hour >= 22:
        causes.append("late hour")
    return causes


def _trace(
    *,
    source: str,
    state: GameState,
    base: float,
    breakdown: object,
    best: Macro,
    plan: Sequence[Macro],
    score: float,
    beam: Sequence[PlanNode],
    causes: list[str],
) -> DecisionTrace:
    target = getattr(best, "target", None)
    bd = asdict(breakdown) if hasattr(breakdown, "total") else None
    return DecisionTrace(
        source=source,
        unit="farmer",
        headline="",
        macro=macro_label(best),
        target=list(target) if target is not None else None,
        plan=[macro_label(m) for m in plan],
        base_value=base,
        score=score,
        breakdown=bd,
        causes=causes,
        alternatives=[
            Alternative(macro=macro_label(n.macros[0]), score=n.score)
            for n in beam[1:4]
            if n.macros
        ],
    )


def _path_delta(state: GameState, macros: Sequence[Macro], turns_per_day: int) -> float:
    return sum(
        _macro_delta(state, m, turns_per_day) * (0.75**i) for i, m in enumerate(macros)
    )


def _macro_delta(state: GameState, macro: Macro, turns_per_day: int) -> float:
    """Legacy delta estimates (tests / debugging). Search uses apply_macro."""
    farm = state.self_player.farm
    private = state.self_player.private
    pos = farm.farmer
    prices = state.market.prices

    def travel(target: Optional[tuple[int, int]]) -> float:
        if target is None:
            return 0.0
        return -1.5 * manhattan(pos, target)

    if isinstance(macro, WaterUrgent):
        return 120.0 + travel(macro.target)
    if isinstance(macro, MaintainField):
        return 25.0 + travel(macro.target)
    if isinstance(macro, HarvestField):
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
            return -5.0
        return 8.0 + 2.0 * min(macro.quantity, 3 - have)
    if isinstance(macro, SellCommodity):
        if private is None:
            return 0.0
        have = int(private.shed.get(macro.item, 0))
        qty = have if macro.quantity is None else min(have, int(macro.quantity))
        return qty * float(prices.get(macro.item, 0)) * 0.05
    if isinstance(macro, LiquidateInventory):
        if private is None:
            return 0.0
        return 5.0 * sum(1 for n in private.shed.values() if int(n) > 0)
    if isinstance(macro, ExpandFarm):
        a = analyze_crop(state, choose_crop(state))
        return a.profit_per_tile * 8.0 - 50.0
    if isinstance(macro, Idle):
        return -5.0
    return 0.0
