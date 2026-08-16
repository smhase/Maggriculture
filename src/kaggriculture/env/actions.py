"""Helpers for building official Kaggriculture action dicts."""

from __future__ import annotations

from typing import Any, Optional, Sequence


UnitAction = list[Any]
MarketOrder = list[Any]


def default_action() -> dict[str, Any]:
    """Official default: farmer PASS, no hands, no market orders."""
    return {"farmer": ["PASS"], "hands": [], "market": []}


def make_action(
    farmer: Optional[Sequence[Any]] = None,
    hands: Optional[Sequence[Sequence[Any]]] = None,
    market: Optional[Sequence[Sequence[Any]]] = None,
) -> dict[str, Any]:
    """Build a Kaggle action dict with sensible defaults."""
    return {
        "farmer": list(farmer) if farmer is not None else ["PASS"],
        "hands": [list(h) for h in hands] if hands is not None else [],
        "market": [list(m) for m in market] if market is not None else [],
    }


def pass_action(num_hands: int = 0) -> dict[str, Any]:
    """PASS for the farmer and each hired hand."""
    return make_action(
        farmer=["PASS"],
        hands=[["PASS"] for _ in range(num_hands)],
        market=[],
    )
