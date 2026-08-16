"""Approximate forward model for macro search.

This is a mental model for lookahead, not the official interpreter.
Live actions still go through kaggle-environments.
"""

from __future__ import annotations

import copy
from typing import Optional

from kaggriculture.env.legal import next_land_cost
from kaggriculture.env.rules import CROPS, DEFAULT_TURNS_PER_DAY, LAND_ORDER
from kaggriculture.env.state import FarmState, GameState, PlayerState, PrivateState
from kaggriculture.env.tiles import is_empty, is_weed
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


def apply_macro(
    state: GameState,
    macro: Macro,
    *,
    turns_per_day: int = DEFAULT_TURNS_PER_DAY,
) -> GameState:
    """Return a copied state after an approximate macro effect."""
    nxt = copy.deepcopy(state)
    farm = nxt.self_player.farm
    private = nxt.self_player.private
    tiles = [list(row) for row in farm.tiles]

    def set_tile(pos: Optional[tuple[int, int]], tile: object) -> None:
        if pos is None:
            return
        x, y = pos
        if 0 <= y < len(tiles) and 0 <= x < len(tiles[y]):
            tiles[y][x] = tile

    def tile_at(pos: Optional[tuple[int, int]]):
        if pos is None:
            return None
        x, y = pos
        if 0 <= y < len(tiles) and 0 <= x < len(tiles[y]):
            return tiles[y][x]
        return None

    farmer = farm.farmer
    money = float(farm.money)
    hour = int(nxt.hour)
    day = int(nxt.day)
    seeds = dict(private.seeds) if private else {}
    shed = dict(private.shed) if private else {}
    unlocked = list(farm.unlocked_quadrants)

    target = getattr(macro, "target", None)

    if isinstance(macro, (WaterUrgent, MaintainField)):
        tile = tile_at(target)
        if isinstance(tile, dict) and tile.get("kind") == "PLANT":
            watered = dict(tile)
            watered["watered_today"] = True
            watered["consecutive_unwatered"] = 0
            set_tile(target, watered)
        if target is not None:
            farmer = target
    elif isinstance(macro, HarvestField):
        tile = tile_at(target)
        if isinstance(tile, dict) and tile.get("kind") == "PLANT":
            crop = str(tile.get("crop", "WHEAT"))
            units = int(tile.get("yield_units", 0) or 0)
            shed[crop] = int(shed.get(crop, 0)) + units
            harvested = dict(tile)
            harvested["yield_units"] = 0
            set_tile(target, harvested)
        if target is not None:
            farmer = target
    elif isinstance(macro, PlantCrop):
        crop = macro.crop
        if target is not None and is_empty(tile_at(target)) and int(seeds.get(crop, 0)) > 0:
            seeds[crop] = int(seeds.get(crop, 0)) - 1
            set_tile(
                target,
                {
                    "kind": "PLANT",
                    "crop": crop,
                    "planted_day": day,
                    "watered_today": False,
                    "yield_units": 0,
                    "consecutive_unwatered": 0,
                },
            )
            farmer = target
    elif isinstance(macro, ClearWeed):
        if target is not None and is_weed(tile_at(target)):
            set_tile(target, None)
            farmer = target
    elif isinstance(macro, BuySeeds) and private is not None:
        if macro.crop in CROPS:
            cost = int(CROPS[macro.crop]["seed"])
            qty = int(macro.quantity)
            spend = cost * qty
            if money >= spend and qty > 0:
                money -= spend
                seeds[macro.crop] = int(seeds.get(macro.crop, 0)) + qty
    elif isinstance(macro, SellCommodity) and private is not None:
        have = int(shed.get(macro.item, 0))
        qty = have if macro.quantity is None else min(have, int(macro.quantity))
        if qty > 0:
            price = float(nxt.market.prices.get(macro.item, 0))
            money += qty * price
            shed[macro.item] = have - qty
    elif isinstance(macro, LiquidateInventory) and private is not None:
        for item, n in list(shed.items()):
            qty = int(n)
            if qty <= 0:
                continue
            price = float(nxt.market.prices.get(item, 0))
            money += qty * price
            shed[item] = 0
    elif isinstance(macro, ExpandFarm):
        land = next_land_cost(unlocked)
        if land is not None and money >= land:
            money -= land
            extra = len(unlocked) - 1
            if extra < 0:
                extra = 0
            if extra < len(LAND_ORDER) and LAND_ORDER[extra] not in unlocked:
                unlocked.append(LAND_ORDER[extra])
    elif isinstance(macro, Idle):
        pass

    hour += 1
    turn = int(nxt.turn) + 1
    if hour >= turns_per_day:
        hour = 0
        day += 1

    new_private = None
    if private is not None:
        new_private = PrivateState(
            shed=shed,
            seeds=seeds,
            inventories=copy.deepcopy(list(private.inventories)),
        )
    new_farm = FarmState(
        money=money,
        tiles=tiles,
        farmer=farmer,
        hands=list(farm.hands),
        unlocked_quadrants=unlocked,
        hires_today=farm.hires_today,
    )
    new_self = PlayerState(
        player_id=nxt.self_player.player_id,
        farm=new_farm,
        private=new_private,
    )
    return GameState(
        turn=turn,
        day=day,
        hour=hour,
        turns_remaining=max(0, nxt.turns_remaining - 1),
        player_id=nxt.player_id,
        self_player=new_self,
        opponent=copy.deepcopy(nxt.opponent),
        market=copy.deepcopy(nxt.market),
        town=copy.deepcopy(nxt.town),
        episode_steps=nxt.episode_steps,
        remaining_overage_time=nxt.remaining_overage_time,
        raw=nxt.raw,
    )
