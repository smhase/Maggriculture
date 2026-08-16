"""Tile / geometry helpers for GameState (no duplicated engine logic)."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from kaggriculture.env.state import Tile


def board_size(tiles: Sequence[Sequence[Tile]]) -> int:
    return len(tiles)


def tile_at(tiles: Sequence[Sequence[Tile]], x: int, y: int) -> Tile:
    return tiles[y][x]


def shed_access_tiles(size: int) -> list[tuple[int, int]]:
    """Official shed-adjacent standing tiles (NWSE order)."""
    half = size // 2
    return [
        (half - 1, half - 1),
        (half, half - 1),
        (half - 1, half),
        (half, half),
    ]


def is_shed_adjacent(pos: tuple[int, int], size: int) -> bool:
    return tuple(pos) in set(shed_access_tiles(size))


def is_locked(tile: Tile) -> bool:
    return tile == "LOCKED"


def is_empty(tile: Tile) -> bool:
    return tile is None


def is_plant(tile: Tile) -> bool:
    return isinstance(tile, dict) and tile.get("kind") == "PLANT"


def is_weed(tile: Tile) -> bool:
    return isinstance(tile, dict) and tile.get("kind") == "WEED"


def is_structure(tile: Tile) -> bool:
    return isinstance(tile, dict) and tile.get("kind") in ("COOP", "PASTURE")


def has_animal(tile: Tile) -> bool:
    return isinstance(tile, dict) and "animal" in tile


def empty_structure(tile: Tile) -> bool:
    return is_structure(tile) and not has_animal(tile)


def plant_crop(tile: Tile) -> Optional[str]:
    if is_plant(tile):
        assert isinstance(tile, dict)
        return str(tile["crop"])
    return None


def as_dict(tile: Tile) -> Optional[dict[str, Any]]:
    return tile if isinstance(tile, dict) else None
