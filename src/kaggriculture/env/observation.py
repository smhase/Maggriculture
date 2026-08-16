"""Parse official Kaggriculture observations into typed GameState."""

from __future__ import annotations

from typing import Any, Mapping

from kaggriculture.env.rules import DEFAULT_EPISODE_STEPS
from kaggriculture.env.state import (
    FarmState,
    GameState,
    MarketState,
    PlayerState,
    PrivateState,
    TownState,
)


def _as_mapping(obj: Any) -> Mapping[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, Mapping):
        return obj
    # kaggle Structify objects support attribute access + dict-like get
    if hasattr(obj, "items"):
        return dict(obj.items())
    return {k: getattr(obj, k) for k in dir(obj) if not k.startswith("_")}


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _int_counts(d: Any) -> dict[str, int]:
    if not d:
        return {}
    src = _as_mapping(d) if not isinstance(d, Mapping) else d
    return {str(k): int(v) for k, v in src.items()}


def _parse_farm(farm: Any) -> FarmState:
    farmer = _get(farm, "farmer", [0, 0])
    hands_raw = _get(farm, "hands", []) or []
    tiles = _get(farm, "tiles", []) or []
    # Materialize tiles as nested lists (avoid sharing mutable env state)
    tiles_copy = [list(row) for row in tiles]
    return FarmState(
        money=float(_get(farm, "money", 0.0)),
        tiles=tiles_copy,
        farmer=(int(farmer[0]), int(farmer[1])),
        hands=[(int(p[0]), int(p[1])) for p in hands_raw],
        unlocked_quadrants=list(_get(farm, "unlocked_quadrants", []) or []),
        hires_today=int(_get(farm, "hires_today", 0) or 0),
    )


def _parse_private(private: Any) -> PrivateState:
    inventories_raw = _get(private, "inventories", []) or []
    inventories = [_int_counts(inv) for inv in inventories_raw]
    return PrivateState(
        shed=_int_counts(_get(private, "shed", {})),
        seeds=_int_counts(_get(private, "seeds", {})),
        inventories=inventories,
    )


def parse_observation(
    obs: Any,
    *,
    episode_steps: int = DEFAULT_EPISODE_STEPS,
) -> GameState:
    """Convert a Kaggle observation (dict or Struct) into GameState."""
    raw = _as_mapping(obs) if not isinstance(obs, Mapping) else dict(obs)

    player_id = int(_get(obs, "player", 0))
    turn = int(_get(obs, "step", 0) or 0)
    day = int(_get(obs, "day", 0) or 0)
    hour = int(_get(obs, "hour", 0) or 0)
    # Empirically, final recorded step index is episode_steps - 1 (e.g. 719).
    # turns_remaining counts remaining agent decisions including the current one
    # as already started: max(0, episode_steps - turn).
    turns_remaining = max(0, int(episode_steps) - turn)

    farms = _get(obs, "farms", []) or []
    if len(farms) < 2:
        raise ValueError(f"Expected 2 farms in observation, got {len(farms)}")

    self_farm = _parse_farm(farms[player_id])
    opp_id = 1 - player_id
    opp_farm = _parse_farm(farms[opp_id])

    private_raw = _get(obs, "private", {})
    self_player = PlayerState(
        player_id=player_id,
        farm=self_farm,
        private=_parse_private(private_raw),
    )
    opponent = PlayerState(player_id=opp_id, farm=opp_farm, private=None)

    market_raw = _get(obs, "market", {}) or {}
    market = MarketState(
        inventory=_int_counts(_get(market_raw, "inventory", {})),
        prices=_int_counts(_get(market_raw, "prices", {})),
        params=_get(market_raw, "params", None),
    )

    town_raw = _get(obs, "town", {}) or {}
    town = TownState(unlocked_shops=list(_get(town_raw, "unlocked_shops", []) or []))

    return GameState(
        turn=turn,
        day=day,
        hour=hour,
        turns_remaining=turns_remaining,
        player_id=player_id,
        self_player=self_player,
        opponent=opponent,
        market=market,
        town=town,
        episode_steps=int(episode_steps),
        remaining_overage_time=float(_get(obs, "remainingOverageTime", 60) or 60),
        raw=raw,
    )
