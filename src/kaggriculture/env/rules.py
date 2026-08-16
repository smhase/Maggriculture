"""Official Kaggriculture constants re-exported from kaggle-environments.

Import rules/constants from here so agent code does not depend on private
package paths. Do not duplicate formulas — the official interpreter is
authoritative.
"""

from __future__ import annotations

from kaggle_environments.envs.kaggriculture.kaggriculture import (
    ANIMALS,
    CROPS,
    FARM_HAND_COST_MULT,
    FARMER_MOVES,
    HINGE_GAIN,
    LAND_ORDER,
    LAND_PRICES,
    MARKET_I0,
    MARKET_PARAMS,
    MAX_SHOP_INSTANCES,
    PRICE_FLOOR,
    PRODUCTS,
    SHOPS,
    TOWN_CENTER_PRODUCTS,
    market_price,
)

# Default configuration values from kaggriculture.json
DEFAULT_EPISODE_STEPS = 720
DEFAULT_BOARD_SIZE = 10
DEFAULT_STARTING_MONEY = 3000
DEFAULT_TURNS_PER_DAY = 24
DEFAULT_SHED_CAPACITY = 100
DEFAULT_MAX_MARKET_ORDERS = 10
DEFAULT_WEED_SPAWN_CHANCE = 0.005
DEFAULT_ACT_TIMEOUT = 1
DEFAULT_REMAINING_OVERAGE_TIME = 60

__all__ = [
    "ANIMALS",
    "CROPS",
    "FARM_HAND_COST_MULT",
    "FARMER_MOVES",
    "HINGE_GAIN",
    "LAND_ORDER",
    "LAND_PRICES",
    "MARKET_I0",
    "MARKET_PARAMS",
    "MAX_SHOP_INSTANCES",
    "PRICE_FLOOR",
    "PRODUCTS",
    "SHOPS",
    "TOWN_CENTER_PRODUCTS",
    "market_price",
    "DEFAULT_EPISODE_STEPS",
    "DEFAULT_BOARD_SIZE",
    "DEFAULT_STARTING_MONEY",
    "DEFAULT_TURNS_PER_DAY",
    "DEFAULT_SHED_CAPACITY",
    "DEFAULT_MAX_MARKET_ORDERS",
    "DEFAULT_WEED_SPAWN_CHANCE",
    "DEFAULT_ACT_TIMEOUT",
    "DEFAULT_REMAINING_OVERAGE_TIME",
]
