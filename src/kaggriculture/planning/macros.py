"""Strategic macro actions justified by Kaggriculture mechanics.

Macros are intents, not Kaggle primitives. The tactical scheduler turns them
into one legal farmer/market action per turn.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union


class MacroKind(str, Enum):
    WATER_URGENT = "water_urgent"
    HARVEST_FIELD = "harvest_field"
    PLANT_CROP = "plant_crop"
    MAINTAIN_FIELD = "maintain_field"  # water non-urgent plants
    CLEAR_WEED = "clear_weed"
    BUY_SEEDS = "buy_seeds"
    SELL_COMMODITY = "sell_commodity"
    LIQUIDATE_INVENTORY = "liquidate_inventory"
    EXPAND_FARM = "expand_farm"
    IDLE = "idle"


@dataclass(frozen=True)
class WaterUrgent:
    kind: MacroKind = MacroKind.WATER_URGENT
    target: Optional[tuple[int, int]] = None  # tile; None = nearest


@dataclass(frozen=True)
class HarvestField:
    kind: MacroKind = MacroKind.HARVEST_FIELD
    target: Optional[tuple[int, int]] = None


@dataclass(frozen=True)
class PlantCrop:
    kind: MacroKind = MacroKind.PLANT_CROP
    crop: str = "WHEAT"
    target: Optional[tuple[int, int]] = None


@dataclass(frozen=True)
class MaintainField:
    kind: MacroKind = MacroKind.MAINTAIN_FIELD
    target: Optional[tuple[int, int]] = None


@dataclass(frozen=True)
class ClearWeed:
    kind: MacroKind = MacroKind.CLEAR_WEED
    target: Optional[tuple[int, int]] = None


@dataclass(frozen=True)
class BuySeeds:
    kind: MacroKind = MacroKind.BUY_SEEDS
    crop: str = "WHEAT"
    quantity: int = 1


@dataclass(frozen=True)
class SellCommodity:
    kind: MacroKind = MacroKind.SELL_COMMODITY
    item: str = "WHEAT"
    quantity: Optional[int] = None  # None = all in shed


@dataclass(frozen=True)
class LiquidateInventory:
    kind: MacroKind = MacroKind.LIQUIDATE_INVENTORY


@dataclass(frozen=True)
class ExpandFarm:
    kind: MacroKind = MacroKind.EXPAND_FARM


@dataclass(frozen=True)
class Idle:
    kind: MacroKind = MacroKind.IDLE


Macro = Union[
    WaterUrgent,
    HarvestField,
    PlantCrop,
    MaintainField,
    ClearWeed,
    BuySeeds,
    SellCommodity,
    LiquidateInventory,
    ExpandFarm,
    Idle,
]


def macro_kind(macro: Macro) -> MacroKind:
    return macro.kind


def macro_label(macro: Macro) -> str:
    k = macro.kind.value
    if isinstance(macro, PlantCrop):
        return f"{k}:{macro.crop}"
    if isinstance(macro, BuySeeds):
        return f"{k}:{macro.crop}x{macro.quantity}"
    if isinstance(macro, SellCommodity):
        return f"{k}:{macro.item}"
    if getattr(macro, "target", None):
        return f"{k}@{macro.target}"  # type: ignore[attr-defined]
    return k
