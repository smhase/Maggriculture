"""Typed GameState structures normalized from official observations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Union


Tile = Union[None, str, dict[str, Any]]  # None | "LOCKED" | structure dict


@dataclass(frozen=True)
class PrivateState:
    """Per-player private state (not visible to the opponent)."""

    shed: Mapping[str, int]
    seeds: Mapping[str, int]
    inventories: Sequence[Mapping[str, int]]


@dataclass(frozen=True)
class FarmState:
    """Public farm state visible to both players."""

    money: float
    tiles: Sequence[Sequence[Tile]]
    farmer: tuple[int, int]
    hands: Sequence[tuple[int, int]]
    unlocked_quadrants: Sequence[str]
    hires_today: int


@dataclass(frozen=True)
class PlayerState:
    """Public farm plus optional private inventory (None for opponent)."""

    player_id: int
    farm: FarmState
    private: Optional[PrivateState] = None


@dataclass(frozen=True)
class MarketState:
    inventory: Mapping[str, int]
    prices: Mapping[str, int]
    params: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class TownState:
    unlocked_shops: Sequence[str]


@dataclass(frozen=True)
class GameState:
    """Normalized view of one agent's observation.

    ``raw`` keeps the original observation for debugging. Agents submitted to
    Kaggle still receive the official observation; this structure is for our
    research agents and tests.
    """

    turn: int
    day: int
    hour: int
    turns_remaining: int
    player_id: int
    self_player: PlayerState
    opponent: PlayerState
    market: MarketState
    town: TownState
    episode_steps: int = 720
    remaining_overage_time: float = 60.0
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)
