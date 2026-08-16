"""Compact GameState features for teacher datasets (Phase 14)."""

from __future__ import annotations

from typing import Any

from kaggriculture.env.state import GameState
from kaggriculture.env.tiles import has_animal, is_empty, is_plant, is_weed


def compact_features(state: GameState) -> dict[str, Any]:
    """Small JSON-friendly snapshot — not raw farms or observations."""
    farm = state.self_player.farm
    private = state.self_player.private
    plants = weeds = empty = animals = 0
    for row in farm.tiles:
        for tile in row:
            if is_empty(tile):
                empty += 1
            elif is_weed(tile):
                weeds += 1
            elif is_plant(tile):
                plants += 1
            elif has_animal(tile):
                animals += 1
    shed = dict(private.shed) if private is not None else {}
    seeds = dict(private.seeds) if private is not None else {}
    return {
        "turn": int(state.turn),
        "day": int(state.day),
        "hour": int(state.hour),
        "turns_remaining": int(state.turns_remaining),
        "money": float(farm.money),
        "opp_money": float(state.opponent.farm.money),
        "n_plants": plants,
        "n_weeds": weeds,
        "n_empty": empty,
        "n_animals": animals,
        "n_hands": len(farm.hands),
        "unlocked": len(farm.unlocked_quadrants),
        "shed_units": int(sum(int(n) for n in shed.values())),
        "seed_units": int(sum(int(n) for n in seeds.values())),
        "wheat_price": float(state.market.prices.get("WHEAT", 0) or 0),
    }
