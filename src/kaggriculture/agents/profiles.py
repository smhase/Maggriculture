"""Named strategy knobs for planner, heuristic, and crew minds.

These are research profiles, not engine objects. `act()` still returns only
the official farmer/hands/market dict.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional


@dataclass(frozen=True)
class StrategyProfile:
    name: str
    horizon: str = "medium"  # short | medium | long
    risk: str = "stable"  # aggressive | stable | safe
    hire: str = "never"  # never | later | crew
    objective: str = "score"  # score | win | never_lose | swing
    beam_width: int = 6
    depth: int = 3
    discount: float = 0.75
    time_budget_s: float = 0.08
    cash_reserve: int = 200
    max_hands: int = 0
    preferred_crop: Optional[str] = None

    @property
    def search_depth(self) -> int:
        if self.horizon == "short":
            return 1
        if self.horizon == "long":
            return max(self.depth, 4)
        return self.depth


def resolve_swing(profile: StrategyProfile, *, self_money: float, opponent_money: float) -> StrategyProfile:
    """Risky when behind, safe when ahead or tied."""
    if profile.objective != "swing":
        return profile
    if self_money < opponent_money:
        return replace(
            profile,
            name=f"{profile.name}:behind",
            risk="aggressive",
            objective="win",
            cash_reserve=0,
            horizon="short",
        )
    return replace(
        profile,
        name=f"{profile.name}:ahead",
        risk="safe",
        objective="never_lose",
        cash_reserve=max(profile.cash_reserve, 800),
        horizon="long",
    )


DEFAULT = StrategyProfile(name="planner", horizon="medium", depth=3, beam_width=6)

LONG_TERM = StrategyProfile(
    name="long_term",
    horizon="long",
    risk="stable",
    hire="never",
    objective="score",
    beam_width=8,
    depth=4,
    discount=0.9,
    time_budget_s=0.12,
    cash_reserve=400,
)

SHORT_TERM = StrategyProfile(
    name="short_term",
    horizon="short",
    risk="stable",
    hire="never",
    objective="score",
    beam_width=4,
    depth=1,
    discount=0.35,
    time_budget_s=0.04,
    cash_reserve=50,
)

SOLO = StrategyProfile(
    name="solo",
    horizon="long",
    risk="stable",
    hire="never",
    objective="score",
    beam_width=8,
    depth=4,
    max_hands=0,
)

RISK_TAKER = StrategyProfile(
    name="risk_taker",
    horizon="medium",
    risk="aggressive",
    hire="never",
    objective="win",
    cash_reserve=0,
    depth=3,
)

SAFE = StrategyProfile(
    name="safe",
    horizon="medium",
    risk="safe",
    hire="never",
    objective="never_lose",
    cash_reserve=1200,
    depth=3,
)

SWING = StrategyProfile(
    name="swing",
    horizon="medium",
    risk="stable",
    hire="never",
    objective="swing",
    depth=3,
)

ALWAYS_WIN = StrategyProfile(
    name="always_win",
    horizon="medium",
    risk="aggressive",
    hire="never",
    objective="win",
    cash_reserve=0,
    depth=3,
)

NEVER_LOSE = StrategyProfile(
    name="never_lose",
    horizon="long",
    risk="safe",
    hire="never",
    objective="never_lose",
    cash_reserve=1000,
    depth=4,
)

CREW = StrategyProfile(
    name="crew",
    horizon="long",
    risk="stable",
    hire="crew",
    objective="score",
    beam_width=6,
    depth=3,
    max_hands=4,
    cash_reserve=300,
)

NAMED_PROFILES: dict[str, StrategyProfile] = {
    "planner": DEFAULT,
    "long_term": LONG_TERM,
    "short_term": SHORT_TERM,
    "solo": SOLO,
    "risk_taker": RISK_TAKER,
    "safe": SAFE,
    "swing": SWING,
    "always_win": ALWAYS_WIN,
    "never_lose": NEVER_LOSE,
    "crew": CREW,
}
