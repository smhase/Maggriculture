"""Compatibility wrappers — prefer ``kaggriculture.env.legal``."""

from __future__ import annotations

from typing import Any

from kaggriculture.env.legal import get_legal_market_orders, get_legal_unit_actions
from kaggriculture.env.state import GameState


def farmer_candidates(state: GameState, *, unit_index: int = 0) -> list[list[Any]]:
    return get_legal_unit_actions(state, unit_index)


def market_candidates(state: GameState) -> list[list[Any]]:
    """Subset used by RandomLegalAgent (cheap buys/sells)."""
    orders = get_legal_market_orders(state)
    # Prefer wheat seed buy and sells for random noise
    preferred = [
        o
        for o in orders
        if (o[0] == "BUY_SEED" and o[1] == "WHEAT") or o[0] == "SELL"
    ]
    return preferred or orders
