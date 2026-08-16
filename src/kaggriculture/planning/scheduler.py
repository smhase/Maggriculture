"""Tactical scheduler: macro intent → one Kaggle action dict.

Accounts for farmer position, movement, and same-turn market orders.
Invalidated macros degrade to IDLE / nearest maintenance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from kaggriculture.agents.navigation import (
    can_harvest_plant,
    nearest,
    needs_water,
    plant_needs_urgent_water,
    scan_tiles,
    step_toward,
)
from kaggriculture.env.actions import make_action
from kaggriculture.env.legal import next_land_cost
from kaggriculture.env.rules import CROPS, DEFAULT_TURNS_PER_DAY
from kaggriculture.env.state import GameState
from kaggriculture.env.tiles import is_empty, is_weed
if TYPE_CHECKING:
    from kaggriculture.agents.profiles import StrategyProfile

from kaggriculture.planning.macros import (
    BuySeeds,
    ClearWeed,
    ExpandFarm,
    HarvestField,
    Idle,
    LiquidateInventory,
    Macro,
    MaintainField,
    PlantCrop,
    SellCommodity,
    WaterUrgent,
)


def schedule(
    state: GameState,
    macro: Macro,
    *,
    turns_per_day: int = DEFAULT_TURNS_PER_DAY,
    extra_market: Optional[list[list[Any]]] = None,
) -> dict[str, Any]:
    """Emit one primitive action advancing ``macro``."""
    market = list(extra_market or [])
    # Always opportunistically sell shed goods unless macro is a specific sell
    if not isinstance(macro, (SellCommodity, LiquidateInventory, BuySeeds, ExpandFarm)):
        market.extend(_sell_all(state))

    farmer = _farmer_for_macro(state, macro, turns_per_day)
    # Market-only macros
    if isinstance(macro, BuySeeds):
        market = _buy_seeds(state, macro) + market
        farmer = farmer if farmer != ["PASS"] else ["PASS"]
    elif isinstance(macro, SellCommodity):
        market = _sell_one(state, macro) + market
    elif isinstance(macro, LiquidateInventory):
        market = _sell_all(state) + market
    elif isinstance(macro, ExpandFarm):
        land = next_land_cost(state.self_player.farm.unlocked_quadrants)
        if land is not None and state.self_player.farm.money >= land:
            market = [["BUY_LAND"]] + market

    # Deduplicate / cap market
    market = _cap_market(market)
    hands = [["PASS"] for _ in state.self_player.farm.hands]
    return make_action(farmer=farmer, hands=hands, market=market)


def propose_macros(
    state: GameState,
    *,
    preferred_crop: str = "WHEAT",
    turns_per_day: int = DEFAULT_TURNS_PER_DAY,
    profile: Optional["StrategyProfile"] = None,
) -> list[Macro]:
    """Generate a small set of justified candidate macros from state."""
    farm = state.self_player.farm
    private = state.self_player.private
    day = state.day
    macros: list[Macro] = []

    urgent = scan_tiles(state, lambda t, x, y: plant_needs_urgent_water(t, day))
    waters = scan_tiles(state, lambda t, x, y: needs_water(t, day))
    harvests = scan_tiles(state, lambda t, x, y: can_harvest_plant(t, day))
    weeds = scan_tiles(state, lambda t, x, y: is_weed(t))
    empties = scan_tiles(state, lambda t, x, y: is_empty(t))

    if urgent:
        macros.append(WaterUrgent(target=nearest(farm.farmer, urgent)))
    if harvests:
        macros.append(HarvestField(target=nearest(farm.farmer, harvests)))
    if waters and not urgent:
        macros.append(MaintainField(target=nearest(farm.farmer, waters)))
    if weeds:
        macros.append(ClearWeed(target=nearest(farm.farmer, weeds)))

    can_plant_hour = state.hour < turns_per_day - 1
    seeds = int(private.seeds.get(preferred_crop, 0)) if private else 0
    if can_plant_hour and preferred_crop in CROPS and empties:
        if seeds > 0:
            macros.append(PlantCrop(crop=preferred_crop, target=nearest(farm.farmer, empties)))
        cost = int(CROPS[preferred_crop]["seed"])
        if private is not None and farm.money >= cost and seeds < 3:
            macros.append(BuySeeds(crop=preferred_crop, quantity=min(3, int(farm.money // cost))))

    if private is not None:
        for item, n in private.shed.items():
            if int(n) > 0:
                macros.append(SellCommodity(item=item, quantity=int(n)))
                break
        if any(int(n) > 0 for n in private.shed.values()):
            macros.append(LiquidateInventory())

    land = next_land_cost(farm.unlocked_quadrants)
    empty_n = len(empties)
    reserve = int(profile.cash_reserve) if profile is not None else 1200
    risk = profile.risk if profile is not None else "stable"
    expand_ok = land is not None and empty_n < 5 and farm.money >= land + reserve
    if risk == "aggressive":
        expand_ok = land is not None and farm.money >= land and (empty_n < 8 or state.day <= 20)
    elif risk == "safe":
        expand_ok = (
            land is not None
            and empty_n < 3
            and farm.money >= land + max(reserve, 1500)
            and state.day <= 12
        )
    else:
        expand_ok = expand_ok and state.day <= 15
    if expand_ok:
        macros.append(ExpandFarm())

    macros.append(Idle())
    return _dedupe_macros(macros)


def _farmer_for_macro(state: GameState, macro: Macro, turns_per_day: int) -> list[Any]:
    farm = state.self_player.farm
    private = state.self_player.private
    pos = farm.farmer
    fx, fy = pos
    tile = farm.tiles[fy][fx]
    day = state.day

    def go(target: Optional[tuple[int, int]], on_arrive: list[Any]) -> list[Any]:
        if target is None:
            return ["PASS"]
        if target == pos:
            return on_arrive
        return step_toward(pos, target)

    if isinstance(macro, WaterUrgent):
        target = macro.target or nearest(
            pos, scan_tiles(state, lambda t, x, y: plant_needs_urgent_water(t, day))
        )
        if target == pos and plant_needs_urgent_water(tile, day):
            return ["WATER"]
        return go(target, ["WATER"] if needs_water(tile, day) else ["PASS"])

    if isinstance(macro, MaintainField):
        target = macro.target or nearest(
            pos, scan_tiles(state, lambda t, x, y: needs_water(t, day))
        )
        if target == pos and needs_water(tile, day):
            return ["WATER"]
        return go(target, ["WATER"] if needs_water(tile, day) else ["PASS"])

    if isinstance(macro, HarvestField):
        target = macro.target or nearest(
            pos, scan_tiles(state, lambda t, x, y: can_harvest_plant(t, day))
        )
        if target == pos and can_harvest_plant(tile, day):
            return ["HARVEST"]
        return go(target, ["HARVEST"] if can_harvest_plant(tile, day) else ["PASS"])

    if isinstance(macro, ClearWeed):
        target = macro.target or nearest(pos, scan_tiles(state, lambda t, x, y: is_weed(t)))
        if target == pos and is_weed(tile):
            return ["DIG"]
        return go(target, ["DIG"] if is_weed(tile) else ["PASS"])

    if isinstance(macro, PlantCrop):
        crop = macro.crop
        seeds = int(private.seeds.get(crop, 0)) if private else 0
        if state.hour >= turns_per_day - 1 or seeds <= 0:
            return ["PASS"]
        target = macro.target or nearest(pos, scan_tiles(state, lambda t, x, y: is_empty(t)))
        if target == pos and is_empty(tile):
            return ["PLANT", crop]
        return go(target, ["PLANT", crop] if is_empty(tile) and seeds > 0 else ["PASS"])

    # Market-only / idle
    return ["PASS"]


def _sell_all(state: GameState) -> list[list[Any]]:
    private = state.self_player.private
    if private is None:
        return []
    return [["SELL", item, int(n)] for item, n in private.shed.items() if int(n) > 0]


def _sell_one(state: GameState, macro: SellCommodity) -> list[list[Any]]:
    private = state.self_player.private
    if private is None:
        return []
    have = int(private.shed.get(macro.item, 0))
    if have <= 0:
        return []
    qty = have if macro.quantity is None else min(have, int(macro.quantity))
    return [["SELL", macro.item, qty]]


def _buy_seeds(state: GameState, macro: BuySeeds) -> list[list[Any]]:
    if macro.crop not in CROPS:
        return []
    cost = int(CROPS[macro.crop]["seed"])
    afford = int(state.self_player.farm.money // cost) if cost else 0
    qty = min(int(macro.quantity), afford)
    if qty <= 0:
        return []
    return [["BUY_SEED", macro.crop, qty]]


def _cap_market(orders: list[list[Any]], limit: int = 10) -> list[list[Any]]:
    seen: set[tuple[Any, ...]] = set()
    out: list[list[Any]] = []
    for o in orders:
        key = tuple(o)
        if key in seen:
            continue
        seen.add(key)
        out.append(o)
        if len(out) >= limit:
            break
    return out


def _dedupe_macros(macros: list[Macro]) -> list[Macro]:
    seen: set[str] = set()
    out: list[Macro] = []
    for m in macros:
        # local import to avoid cycle
        from kaggriculture.planning.macros import macro_label

        label = macro_label(m)
        if label not in seen:
            seen.add(label)
            out.append(m)
    return out
