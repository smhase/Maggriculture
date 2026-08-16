"""Simple market predictors used by the planner (Phase 13).

Kept: SMA, momentum, and inventory/supply pressure. Harvest/sale events are
folded into supply pressure when prices drop while inventory rises. Predictors
that need private opponent sheds are omitted because they are not observable.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional

from kaggriculture.env.rules import MARKET_I0
from kaggriculture.env.state import GameState


@dataclass(frozen=True)
class MarketSnapshot:
    prices: dict[str, float]
    inventory: dict[str, float]
    sma: dict[str, float]
    momentum: dict[str, float]
    supply_pressure: dict[str, float]


class MarketTracker:
    """Rolling public market features for one episode."""

    def __init__(self, window: int = 5) -> None:
        self.window = max(2, int(window))
        self._history: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=self.window))

    def reset(self) -> None:
        self._history.clear()

    def observe(self, state: GameState) -> MarketSnapshot:
        prices: dict[str, float] = {}
        inventory: dict[str, float] = {}
        sma: dict[str, float] = {}
        momentum: dict[str, float] = {}
        pressure: dict[str, float] = {}
        for item, price in state.market.prices.items():
            key = str(item)
            px = float(price)
            self._history[key].append(px)
            series = self._history[key]
            mean = sum(series) / len(series)
            prices[key] = px
            sma[key] = mean
            momentum[key] = px - mean
            inv = float(state.market.inventory.get(item, 0) or 0)
            inventory[key] = inv
            base = float(MARKET_I0) if MARKET_I0 else 1.0
            pressure[key] = inv / max(1.0, base)
        return MarketSnapshot(
            prices=prices,
            inventory=inventory,
            sma=sma,
            momentum=momentum,
            supply_pressure=pressure,
        )


def market_score_adjust(state: GameState, snapshot: MarketSnapshot) -> float:
    """Nudge the leaf when we hold goods into a falling or oversupplied market."""
    private = state.self_player.private
    if private is None:
        return 0.0
    adjust = 0.0
    for item, held in private.shed.items():
        qty = int(held)
        if qty <= 0:
            continue
        mom = snapshot.momentum.get(str(item), 0.0)
        pressure = snapshot.supply_pressure.get(str(item), 1.0)
        if mom < 0:
            adjust -= min(50.0, qty * abs(mom) * 0.15)
        if pressure > 1.15:
            adjust -= min(40.0, qty * (pressure - 1.0) * 4.0)
        if mom > 0 and pressure < 0.95:
            adjust += min(20.0, qty * 0.5)
    return adjust
