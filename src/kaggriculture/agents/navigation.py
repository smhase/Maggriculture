"""Shared navigation / farm-scan helpers for scripted baselines."""

from __future__ import annotations

from typing import Callable, Optional, Sequence

from kaggriculture.env.rules import CROPS
from kaggriculture.env.state import GameState, Tile
from kaggriculture.env.tiles import has_animal, is_empty, is_plant, is_weed, tile_at


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def step_toward(src: tuple[int, int], dst: tuple[int, int]) -> list[str]:
    """One cardinal step reducing Manhattan distance (prefer x then y)."""
    sx, sy = src
    dx, dy = dst
    if sx < dx:
        return ["EAST"]
    if sx > dx:
        return ["WEST"]
    if sy < dy:
        return ["SOUTH"]
    if sy > dy:
        return ["NORTH"]
    return ["PASS"]


def scan_tiles(
    state: GameState,
    predicate: Callable[[Tile, int, int], bool],
) -> list[tuple[int, int]]:
    tiles = state.self_player.farm.tiles
    out: list[tuple[int, int]] = []
    for y, row in enumerate(tiles):
        for x, t in enumerate(row):
            if predicate(t, x, y):
                out.append((x, y))
    return out


def nearest(
    origin: tuple[int, int],
    points: Sequence[tuple[int, int]],
) -> Optional[tuple[int, int]]:
    if not points:
        return None
    return min(points, key=lambda p: (manhattan(origin, p), p[1], p[0]))


def needs_water(tile: Tile, day: int) -> bool:
    if not is_plant(tile):
        return False
    assert isinstance(tile, dict)
    return not bool(tile.get("watered_today", False))


def can_harvest_plant(tile: Tile, day: int) -> bool:
    if not is_plant(tile):
        return False
    assert isinstance(tile, dict)
    crop = tile["crop"]
    if crop not in CROPS:
        return False
    age = day - int(tile.get("planted_day", day))
    return int(tile.get("yield_units", 0)) > 0 and age >= int(CROPS[crop]["first_yield_day"])


def can_harvest_animal(tile: Tile) -> bool:
    return has_animal(tile) and isinstance(tile, dict) and int(tile.get("yield_units", 0)) > 0


def plant_needs_urgent_water(tile: Tile, day: int) -> bool:
    """True if plant will weed tonight without water."""
    if not is_plant(tile):
        return False
    assert isinstance(tile, dict)
    if tile.get("watered_today", False):
        return False
    return int(tile.get("consecutive_unwatered", 0)) >= 1
