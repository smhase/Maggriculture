"""Market price helpers for economics."""

from __future__ import annotations

from kaggriculture.env.rules import MARKET_PARAMS, market_price
from kaggriculture.env.state import GameState


def current_prices(state: GameState) -> dict[str, int]:
    return {str(k): int(v) for k, v in state.market.prices.items()}


def base_prices() -> dict[str, int]:
    return {k: int(v["base"]) for k, v in MARKET_PARAMS.items()}


__all__ = ["current_prices", "base_prices", "market_price"]
