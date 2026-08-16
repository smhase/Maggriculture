"""Legal-action helpers for debugging and planning.

The official Kaggriculture interpreter treats illegal ops as silent no-ops.
These helpers mirror that logic for agent code — they are **not** authoritative.
When in doubt, trust ``kaggle_environments``.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from kaggriculture.env.rules import (
    ANIMALS,
    CROPS,
    DEFAULT_MAX_MARKET_ORDERS,
    DEFAULT_SHED_CAPACITY,
    FARM_HAND_COST_MULT,
    FARMER_MOVES,
    LAND_ORDER,
    LAND_PRICES,
    PRODUCTS,
)
from kaggriculture.env.state import GameState, PrivateState
from kaggriculture.env.tiles import (
    as_dict,
    board_size,
    empty_structure,
    has_animal,
    is_empty,
    is_locked,
    is_plant,
    is_shed_adjacent,
    is_weed,
    tile_at,
)


def fib_hire_index(n: int) -> int:
    """Official fib: fib(0)=1, fib(1)=1, fib(2)=2, ..."""
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def hire_cost(hires_today: int, mult: int = FARM_HAND_COST_MULT) -> int:
    return int(mult) * fib_hire_index(int(hires_today))


def next_land_cost(unlocked_quadrants: Sequence[str]) -> Optional[int]:
    n_extra = len(unlocked_quadrants) - 1  # NW always present
    if n_extra < 0:
        n_extra = 0
    if n_extra >= len(LAND_ORDER):
        return None
    return int(LAND_PRICES[n_extra])


def shed_used(private: PrivateState) -> int:
    return int(sum(int(v) for v in private.shed.values()))


def shed_room(private: PrivateState, capacity: int = DEFAULT_SHED_CAPACITY) -> int:
    return max(0, int(capacity) - shed_used(private))


def unit_position(state: GameState, unit_index: int) -> Optional[tuple[int, int]]:
    farm = state.self_player.farm
    if unit_index == 0:
        return farm.farmer
    if unit_index - 1 < len(farm.hands):
        return farm.hands[unit_index - 1]
    return None


def unit_inventory(state: GameState, unit_index: int) -> dict[str, int]:
    private = state.self_player.private
    if private is None:
        return {}
    if unit_index >= len(private.inventories):
        return {}
    return dict(private.inventories[unit_index])


def get_legal_unit_actions(
    state: GameState,
    unit_index: int = 0,
) -> list[list[Any]]:
    """Legal ops for one farmer/hand standing on its current tile.

    Always includes PASS. Does not model atomic multi-unit PLANT contention;
    use ``is_action_legal`` on a full action dict for that.
    """
    farm = state.self_player.farm
    private = state.self_player.private
    pos = unit_position(state, unit_index)
    if pos is None:
        return [["PASS"]]

    x, y = pos
    size = board_size(farm.tiles)
    tile = tile_at(farm.tiles, x, y)
    inv = unit_inventory(state, unit_index)
    seeds = dict(private.seeds) if private is not None else {}

    legal: list[list[Any]] = [["PASS"]]

    for op, (dx, dy) in FARMER_MOVES.items():
        nx, ny = x + dx, y + dy
        if 0 <= nx < size and 0 <= ny < size:
            legal.append([op])

    # Shed ops work even on LOCKED shed-access tiles
    if private is not None and is_shed_adjacent((x, y), size):
        legal.append(["DROP"])
        for item, n in private.shed.items():
            if int(n) > 0:
                legal.append(["PICKUP", item, 1])
                if int(n) > 1:
                    legal.append(["PICKUP", item, int(n)])
        for item, n in inv.items():
            if int(n) > 0 and shed_room(private) > 0:
                legal.append(["PLACE", item, 1])

    if is_locked(tile):
        return _dedupe(legal)

    if is_empty(tile):
        for crop, n in seeds.items():
            if int(n) > 0 and crop in CROPS:
                legal.append(["PLANT", crop])
        legal.append(["BUILD_COOP"])
        legal.append(["BUILD_PASTURE"])

    if is_weed(tile):
        legal.append(["DIG"])

    if is_plant(tile):
        td = as_dict(tile)
        assert td is not None
        crop = td["crop"]
        cd = CROPS[crop]
        age = state.day - int(td.get("planted_day", state.day))
        if not td.get("watered_today", False):
            legal.append(["WATER"])
        if int(td.get("yield_units", 0)) > 0 and age >= int(cd["first_yield_day"]):
            legal.append(["HARVEST"])
        if inv.get("FERTILIZER", 0) > 0:
            legal.append(["FERTILIZE"])
        legal.append(["DIG"])

    if has_animal(tile):
        td = as_dict(tile)
        assert td is not None
        if not td.get("fed_today", False) and inv.get("WHEAT", 0) > 0:
            legal.append(["FEED"])
        if int(td.get("yield_units", 0)) > 0:
            legal.append(["HARVEST"])
        if td.get("fertilizer_available", False):
            legal.append(["COLLECT_FERTILIZER"])
        if not td.get("cared_today", False):
            legal.append(["CARE"])

    if empty_structure(tile):
        td = as_dict(tile)
        assert td is not None
        legal.append(["DIG"])
        for animal, n in inv.items():
            if int(n) > 0 and animal in ANIMALS:
                if ANIMALS[animal]["structure"] == td.get("kind"):
                    legal.append(["PLACE", animal])

    return _dedupe(legal)


def get_legal_market_orders(
    state: GameState,
    *,
    max_n: int = 1,
    shed_capacity: int = DEFAULT_SHED_CAPACITY,
    hire_mult: int = FARM_HAND_COST_MULT,
) -> list[list[Any]]:
    """Single market orders that should succeed for at least one unit."""
    farm = state.self_player.farm
    private = state.self_player.private
    if private is None:
        return []

    money = float(farm.money)
    room = shed_room(private, shed_capacity)
    legal: list[list[Any]] = []

    for crop, data in CROPS.items():
        cost = int(data["seed"])
        if money >= cost:
            legal.append(["BUY_SEED", crop, 1])
            afford = int(money // cost)
            if afford > 1 and max_n > 1:
                legal.append(["BUY_SEED", crop, min(afford, max_n)])

    for animal, data in ANIMALS.items():
        cost = int(data["cost"])
        if money >= cost and room >= 1:
            legal.append(["BUY_ANIMAL", animal, 1])

    for item in ("WHEAT", "FERTILIZER"):
        price = int(state.market.prices.get(item, 10**9))
        if money >= price and room >= 1:
            legal.append(["BUY_PRODUCT", item, 1])

    for item, n in private.shed.items():
        if int(n) > 0 and item in PRODUCTS:
            legal.append(["SELL", item, 1])
            if int(n) > 1:
                legal.append(["SELL", item, int(n)])

    cost = hire_cost(farm.hires_today, hire_mult)
    if money >= cost:
        legal.append(["HIRE"])

    land = next_land_cost(farm.unlocked_quadrants)
    if land is not None and money >= land:
        legal.append(["BUY_LAND"])

    return _dedupe(legal)


def get_legal_actions(
    state: GameState,
    *,
    max_market_orders: int = DEFAULT_MAX_MARKET_ORDERS,
) -> dict[str, Any]:
    """Return legal unit ops and market orders (not a cartesian product).

    Shape::

        {
          "farmer": [...],
          "hands": [[...], ...],
          "market": [...],
        }
    """
    n_hands = len(state.self_player.farm.hands)
    return {
        "farmer": get_legal_unit_actions(state, 0),
        "hands": [get_legal_unit_actions(state, i + 1) for i in range(n_hands)],
        "market": get_legal_market_orders(state)[:],
        "max_market_orders": max_market_orders,
    }


def is_unit_action_legal(
    state: GameState,
    action: Sequence[Any],
    unit_index: int = 0,
) -> bool:
    if not isinstance(action, (list, tuple)) or not action:
        return False
    op = action[0]
    # Normalize comparable forms: ["PICKUP", item] vs ["PICKUP", item, 1]
    legal = get_legal_unit_actions(state, unit_index)
    act = list(action)
    if act in legal:
        return True
    # Allow PICKUP/PLACE without explicit n (defaults to 1 in engine)
    if op in ("PICKUP", "PLACE") and len(act) == 2:
        return [op, act[1], 1] in legal
    return False


def is_market_order_legal(
    state: GameState,
    order: Sequence[Any],
    *,
    shed_capacity: int = DEFAULT_SHED_CAPACITY,
    hire_mult: int = FARM_HAND_COST_MULT,
) -> bool:
    if not isinstance(order, (list, tuple)) or not order:
        return False
    op = order[0]
    private = state.self_player.private
    farm = state.self_player.farm
    if private is None:
        return False
    money = float(farm.money)
    room = shed_room(private, shed_capacity)

    if op == "HIRE":
        return money >= hire_cost(farm.hires_today, hire_mult)
    if op == "BUY_LAND":
        cost = next_land_cost(farm.unlocked_quadrants)
        return cost is not None and money >= cost
    if op in ("BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"):
        if len(order) < 3:
            return False
        item = order[1]
        try:
            n = int(order[2])
        except (TypeError, ValueError):
            return False
        if n <= 0:
            return False
        if op == "BUY_SEED":
            if item not in CROPS:
                return False
            return money >= int(CROPS[item]["seed"])  # at least first unit
        if op == "BUY_ANIMAL":
            if item not in ANIMALS:
                return False
            return money >= int(ANIMALS[item]["cost"]) and room >= 1
        if op == "BUY_PRODUCT":
            if item not in ("WHEAT", "FERTILIZER"):
                return False
            price = int(state.market.prices.get(item, 10**9))
            return money >= price and room >= 1
        if op == "SELL":
            if item not in PRODUCTS:
                return False
            return int(private.shed.get(item, 0)) >= 1
    return False


def is_action_legal(
    state: GameState,
    action: dict[str, Any],
    *,
    max_market_orders: int = DEFAULT_MAX_MARKET_ORDERS,
) -> bool:
    """Validate a full Kaggle action dict against our legality model.

    Includes atomic PLANT seed contention across farmer + hands.
    Extra market orders beyond the cap are considered illegal here (engine
    would silently drop them).
    """
    if not isinstance(action, dict):
        return False
    farmer = action.get("farmer", ["PASS"])
    hands = action.get("hands", []) or []
    market = action.get("market", []) or []

    if not isinstance(hands, list) or not isinstance(market, list):
        return False
    if len(hands) > len(state.self_player.farm.hands):
        # Extra hand actions are ignored by engine; treat as illegal for clarity
        return False
    if len(market) > max_market_orders:
        return False
    if not is_unit_action_legal(state, farmer, 0):
        return False
    for i, hand_act in enumerate(hands):
        if not is_unit_action_legal(state, hand_act, i + 1):
            return False

    # Atomic PLANT: if total demand for a crop exceeds seeds, all PLANTs for
    # that crop become PASS — treat the original action as illegal.
    private = state.self_player.private
    seeds = dict(private.seeds) if private is not None else {}
    demand: dict[str, int] = {}
    for a in [farmer, *hands]:
        if isinstance(a, (list, tuple)) and len(a) >= 2 and a[0] == "PLANT":
            demand[a[1]] = demand.get(a[1], 0) + 1
    for crop, n in demand.items():
        if n > int(seeds.get(crop, 0)):
            return False

    for order in market:
        if not is_market_order_legal(state, order):
            return False
    return True


def _dedupe(actions: list[list[Any]]) -> list[list[Any]]:
    seen: set[tuple[Any, ...]] = set()
    out: list[list[Any]] = []
    for a in actions:
        key = tuple(a)
        if key not in seen:
            seen.add(key)
            out.append(a)
    return out
