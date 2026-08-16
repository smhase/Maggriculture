"""Interpretable opponent profile from public farm observations (Phase 12)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional

from kaggriculture.env.rules import CROPS
from kaggriculture.env.state import GameState
from kaggriculture.env.tiles import has_animal, is_empty, is_plant, is_weed


@dataclass(frozen=True)
class OpponentProfile:
    money: float
    money_delta: Optional[float]
    unlocked_quadrants: tuple[str, ...]
    expansion_stage: int
    n_hands: int
    hires_today: int
    crop_counts: dict[str, int] = field(default_factory=dict)
    harvestable_counts: dict[str, int] = field(default_factory=dict)
    animal_counts: dict[str, int] = field(default_factory=dict)
    n_weeds: int = 0
    n_empty: int = 0
    n_plants: int = 0
    likely_seller: bool = False
    market_pressure: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["unlocked_quadrants"] = list(self.unlocked_quadrants)
        return payload


def profile_opponent(
    state: GameState,
    *,
    previous_money: Optional[float] = None,
) -> OpponentProfile:
    """Summarize the opponent using only public observation fields."""
    farm = state.opponent.farm
    money = float(farm.money)
    delta = None if previous_money is None else money - float(previous_money)
    crop_counts: dict[str, int] = {}
    harvestable: dict[str, int] = {}
    animals: dict[str, int] = {}
    weeds = 0
    empty = 0
    plants = 0
    day = state.day
    for row in farm.tiles:
        for tile in row:
            if is_empty(tile):
                empty += 1
                continue
            if is_weed(tile):
                weeds += 1
                continue
            if has_animal(tile) and isinstance(tile, dict):
                name = str(tile.get("animal", "unknown"))
                animals[name] = animals.get(name, 0) + 1
                continue
            if not is_plant(tile):
                continue
            assert isinstance(tile, dict)
            plants += 1
            crop = str(tile.get("crop", "?"))
            crop_counts[crop] = crop_counts.get(crop, 0) + 1
            if _harvestable(tile, day):
                harvestable[crop] = harvestable.get(crop, 0) + 1
    pressure = {
        crop: float(count) for crop, count in harvestable.items() if count > 0
    }
    likely_seller = bool(harvestable) and (delta is None or delta >= 0)
    return OpponentProfile(
        money=money,
        money_delta=delta,
        unlocked_quadrants=tuple(farm.unlocked_quadrants),
        expansion_stage=max(0, len(farm.unlocked_quadrants) - 1),
        n_hands=len(farm.hands),
        hires_today=int(farm.hires_today),
        crop_counts=crop_counts,
        harvestable_counts=harvestable,
        animal_counts=animals,
        n_weeds=weeds,
        n_empty=empty,
        n_plants=plants,
        likely_seller=likely_seller,
        market_pressure=pressure,
    )


def opponent_score_adjust(state: GameState, profile: OpponentProfile) -> float:
    """Small, interpretable leaf adjustment — not a neural opponent model."""
    self_money = float(state.self_player.farm.money)
    adjust = 0.0
    if self_money < profile.money:
        adjust -= min(120.0, (profile.money - self_money) * 0.02)
    if profile.expansion_stage > max(0, len(state.self_player.farm.unlocked_quadrants) - 1):
        adjust -= 15.0
    private = state.self_player.private
    if private is not None:
        for crop, pressure in profile.market_pressure.items():
            held = int(private.shed.get(crop, 0))
            if held > 0 and pressure > 0:
                adjust -= min(40.0, pressure * 4.0)
    return adjust


def _harvestable(tile: Mapping[str, Any], day: int) -> bool:
    crop = tile.get("crop")
    if crop not in CROPS:
        return False
    age = day - int(tile.get("planted_day", day))
    return int(tile.get("yield_units", 0) or 0) > 0 and age >= int(CROPS[crop]["first_yield_day"])
