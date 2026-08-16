# Kaggriculture Game Mechanics

Derived from the official environment in `kaggle-environments` **1.32.7**:

- `kaggle_environments/envs/kaggriculture/kaggriculture.py`
- `kaggle_environments/envs/kaggriculture/kaggriculture.json`
- Competition description on Kaggle (cross-checked; **source wins on conflicts**)

This document is Phase 0 of the research platform. Do not treat community notebooks or local reimplementations as authoritative.

---

## High-level rules

- Two players, each with a separate farm. Players **cannot** see each other's shed / seeds / carried inventories, but **can** see both farms (tiles, money, farmer/hand positions, unlocked quadrants, hires_today), the shared market, and town shops.
- Season length: `episodeSteps` (default **720**) = `turnsPerDay` (24) × 30 days.
- Each turn each player returns one action dict covering the main farmer, hired hands, and an ordered market queue.
- Winner: highest **bank money** at terminal. Unsold inventory / seeds / plants / animals do **not** count toward the score (`reward = farm.money`).
- Ties are possible.

### Empirically measured episode length (seeded `pass` vs `pass`)

| Quantity | Value at `episodeSteps=720` |
| --- | --- |
| `len(env.steps)` | **720** |
| Final `obs.step` | **719** |
| Final `day` / `hour` | **29 / 23** |
| Final statuses | `DONE`, `DONE` |
| `DONE` trigger in interpreter | `step >= episodeSteps - 2` (i.e. ≥ 718) |

**Surprise:** the day-29 end-of-day refresh (`(step+1) % 24 == 0` when `step == 719`) does **not** run under default 720, because the episode is marked `DONE` when `step >= 718`. The last EOD that fires is for **day 28**. Plants planted on day 29 that rely on a night refresh will not get it.

Short episodes behave similarly: `episodeSteps=48` → `len(steps)=48`, final `day=1`, `hour=23`.

---

## Observation format

Top-level fields (per agent):

| Field | Shared? | Meaning |
| --- | --- | --- |
| `player` | no | 0 or 1 |
| `step` | yes | Framework turn index (0 … episodeSteps−1 on recorded states) |
| `day` | yes | 0-indexed in-game day |
| `hour` | yes | 0-indexed turn within the day |
| `farms` | yes | Length-2 list of public farm dicts |
| `market` | yes | `{inventory, prices}` (optional `params` if overrides) |
| `town` | yes | `{unlocked_shops: [...]}` (duplicates allowed) |
| `private` | **no** | This player's shed, seeds, inventories |
| `remainingOverageTime` | no | Starts at 60; depleted when act duration > `actTimeout` |

### Public farm dict

```text
money, tiles[y][x], farmer=[x,y], hands=[[x,y],...],
unlocked_quadrants, hires_today
```

Coordinates: **x = column (EAST +x), y = row (SOUTH +y)**. Indexing is `tiles[y][x]`.

### Tile values

- `None` — empty unlocked tile
- `"LOCKED"` — unbought quadrant (passable for movement; most tile ops no-op)
- Plant dict: `kind=PLANT`, `crop`, `planted_day`, `watered_today`, `consecutive_unwatered`, `yield_units`, `max_lifespan_step`, `fertilized_until_day`
- Weed: `{kind: WEED}`
- Structure: `kind=COOP|PASTURE`, optional `animal`, `placed_day`, `yield_units`, `fed_today`, `consecutive_unfed`, `cared_today`, `fertilizer_available`, `pending_care_bonus`

### Private dict

- `shed` — non-seed items (products + animals + fertilizer); capacity `shedCapacity` (100)
- `seeds` — separate uncapped seed counts; **never** picked up into inventory
- `inventories` — `[main_farmer, *hands]` carried items

---

## Action format

```python
{
  "farmer": [op, *args],          # exactly one unit op
  "hands":  [[op, *args], ...],   # one op per current hand, in order
  "market": [[op, *args], ...],   # ordered; truncated to maxMarketOrdersPerTurn (10)
}
```

Default / PASS:

```python
{"farmer": ["PASS"], "hands": [], "market": []}
```

### Farmer / hand ops (from interpreter)

| Op | Notes |
| --- | --- |
| `NORTH/SOUTH/EAST/WEST` | Move one cell; off-board no-op; **LOCKED tiles are passable** |
| `PASS` | No-op |
| `PICKUP`, item, n? | Shed-adjacent; move up to n of item from shed → inventory. **Item arg required.** Seeds cannot be picked up. |
| `DROP` | Shed-adjacent; dump entire inventory into shed (overflow discarded) |
| `PLACE`, item, n? | Place animal on matching empty structure, else shed-drop when adjacent |
| `PLANT`, crop | Empty unlocked tile; consumes one seed |
| `WATER` | Plant not yet watered today; may add yield bonus for one-time crops |
| `HARVEST` | Requires `yield_units > 0` and maturity for plants |
| `FERTILIZE` | Consumes 1 fertilizer from inventory; bonus through `day+2` inclusive |
| `DIG` | Clears plant/weed/empty structure; **not** occupied animal tiles |
| `BUILD_COOP` / `BUILD_PASTURE` | Empty unlocked tile |
| `FEED` | Animal; consumes 1 wheat from inventory; once/day |
| `COLLECT_FERTILIZER` | If `fertilizer_available` |
| `CARE` | Once/day; banks care bonus when also fed at EOD |

### Market ops

| Op | Form | Notes |
| --- | --- | --- |
| `BUY_SEED` | `[BUY_SEED, crop, n]` | Fixed seed prices from `CROPS[*].seed` |
| `BUY_ANIMAL` | `[BUY_ANIMAL, animal, n]` | Fixed costs; lands in shed |
| `BUY_PRODUCT` | `[BUY_PRODUCT, item, n]` | **Only WHEAT and FERTILIZER** |
| `SELL` | `[SELL, item, n]` | Sells from **shed only** |
| `HIRE` | `[HIRE]` | Cost `farmHandCostMult * fib(hires_today)` |
| `BUY_LAND` | `[BUY_LAND]` | Unlocks NE → SW → SE at $1000 / $2000 / $4000 |

---

## Action legality

**Illegal / impossible unit and market operations are silent no-ops.** The environment does **not** set status `INVALID` for watering empty ground, planting without seeds, etc. Malformed market sub-ops abort that order.

Special case — **atomic PLANT**: if farmer+hands request more `PLANT crop` than available seeds for that crop, **all** PLANT requests for that crop become `PASS` this turn.

Our research validators are for debugging/planning only; the official engine remains authoritative.

---

## Turn processing order

From `interpreter` in `kaggriculture.py`:

1. Initialize on first call (`_initialize`).
2. For each player: validate atomic PLANT demand → apply farmer action → apply each hand action.
3. `_process_market` — lockstep per-unit across players; HIRE/BUY_LAND atomic in player order; then SELL/BUY_* unit-by-unit; refresh prices.
4. `_town_consume` — shops every `townShopSellInterval` (4), town center every `townCenterSellInterval` (24).
5. `_decay_plants` for both farms.
6. If `(step + 1) % turnsPerDay == 0`: `_end_of_day`.
7. Advance `day` / `hour`.
8. If `step >= episodeSteps - 2`: set both agents `DONE` and `reward = money`.

**Implication:** HIRE runs in the market phase **after** unit actions, so a newly hired hand **cannot act the same turn**.

`actTimeout` default is **1 second**; agents start with `remainingOverageTime = 60`.

---

## Movement and shed access

- Board default 10×10; NW unlocked; others `"LOCKED"`.
- Shed is **not** a tile. Shed-adjacent tiles (half = boardSize//2): `(4,4), (5,4), (4,5), (5,5)` for size 10 — one per quadrant.
- `PICKUP` / `DROP` / shed-`PLACE` work from shed-adjacent tiles **even if LOCKED**.
- Farmer respawns at first NW shed-access tile each morning; hands vanish at EOD.

---

## Crops (`CROPS`)

| Crop | Seed $ | first_yield_day | max_yield_day | interval | max_yield | ongoing |
| --- | --- | --- | --- | --- | --- | --- |
| WHEAT | 10 | 2 | 4 | 0 | 6 | no |
| CARROT | 20 | 2 | 3 | 0 | 4 | no |
| TOMATO | 50 | 8 | 8 | 1 | 4 | yes |
| STRAWBERRY | 100 | 10 | 10 | 2 | 4 | yes |
| MELON | 80 | 10 | 12 | 0 | 6 | no |

### Plant lifecycle

- New plant: `consecutive_unwatered = 1`, `watered_today = False`, one-time crops start `yield_units = 1`, ongoing start `0`.
- Must water every day. At EOD: if unwatered, counter++; if counter ≥ 2 → **WEED**.
- **Planting day counts as missed water.** Plant + no water same day → weed that night.
- One-time WATER bonus window: ages `ceil(max_yield_day/2) .. max_yield_day`; +1 yield/day (+2 if fertilized); capped at `max_yield`.
- Ongoing: scheduled production at EOD after `first_yield_day`; +1 (+2 if fertilized **and** watered that day); after `max_yield` productions, `max_lifespan_step` is set and decay begins.
- Decay: after `max_lifespan_step`, every other step `yield_units -= 1`; at ≤0 → WEED.
- `FERTILIZE`: consumes inventory fertilizer; `fertilized_until_day = max(old, day+2)` (3 days inclusive).
- Weeds also spawn on empty unlocked tiles at EOD with `weedSpawnChance` (0.005).

### Harvest

- Plants: need `yield_units > 0` and `day - planted_day >= first_yield_day`. Produce goes to unit inventory. One-time crops clear the tile; ongoing keep the plant with `yield_units=0`.

---

## Animals (`ANIMALS`)

| Animal | Cost | Structure | first_yield_day | interval | max_held | Product |
| --- | --- | --- | --- | --- | --- | --- |
| GOOSE | 300 | COOP | 4 | 1 | 4 | EGG |
| COW | 400 | PASTURE | 8 | 2 | 6 | MILK |
| SHEEP | 500 | PASTURE | 6 | 3 | 6 | WOOL |

- Build structure on empty tile → buy animal into shed → `PICKUP` → stand on structure → `PLACE animal`.
- New animal: `consecutive_unfed = 0` (survives first day unfed).
- Must feed wheat daily; ≥2 consecutive unfed → animal escapes, empty structure remains.
- CARE: if fed+cared at EOD, `pending_care_bonus += 1`. On production day, if fed, bonus added to yield then cleared; if unfed on production day, base 1 still produced but bonus discarded.
- Every surviving animal sets `fertilizer_available = True` at EOD (does not stack if uncollected).
- Animals produce indefinitely while fed; `max_held` caps unharvested product on the tile.

---

## Inventory / shed

- Seeds are separate; `PLANT` consumes seeds directly.
- Shed capacity 100 for non-seed items; overflow discarded on DROP / PLACE / EOD drop / buy into full shed.
- End of day: all inventories dump into shed (overflow lost); hands cleared; `hires_today = 0`.

---

## Market

- Products: WHEAT, CARROT, TOMATO, STRAWBERRY, MELON, EGG, MILK, WOOL, FERTILIZER.
- Start inventory `I0 = 10000`; sell price from `market_price` using per-resource shape params (`MARKET_PARAMS`). Floor `$1`.
- Shape functions: `linear`, `sq`, `sqrt`, `log`, `log10`, and **`hinge`** (`HINGE_GAIN = 8`). Hinge is linear in `x/T` until the knee `T`, then a quadratic spike; `f(T) = 1` by construction.
- 1.32.7 scarcity curves (below I0): carrot `hinge/1.00`, tomato `hinge/0.40`, egg `hinge/0.40`. Other products unchanged from 1.32.6. Starting prices at I0 are unchanged.
- Sales at $1 do **not** increase market inventory.
- Buy seed/animal: fixed catalog prices. Buy product: only wheat & fertilizer at dynamic price quoted post-buy.
- Concurrent orders: both players quote on the same pre-commit inventory for each unit, then commit.

---

## Land expansion

- Order: NE, SW, SE (`LAND_ORDER`).
- Prices: 1000, 2000, 4000.
- Unlocking converts `"LOCKED"` tiles in that quadrant to `None`.

---

## Workers / farm hands

- `HIRE` cost = `farmHandCostMult * fib(n)` with fib(0)=1, fib(1)=1, …; resets daily.
- Spawn on least-occupied shed-access tile (NWSE tie-break); may spawn on LOCKED tile.
- Hands disappear at EOD; must re-hire each day.

---

## Town

- Shop unlock every `townShopUnlockInterval` days (default 3) at EOD when `next_day % interval == 0`, drawn with replacement from `SHOPS`, capped at 8 instances.
- Each unlocked instance consumes its products every 4 turns (2× if single-product shop).
- Town center consumes one of every non-fertilizer product every 24 turns.

---

## Stochastic elements

Driven by episode `configuration.seed` (cleared from agent-visible config; stored on `env.info['seed']`):

1. Weed spawns at EOD (`Random((seed * 1_000_003) ^ day)`).
2. Shop unlock choices (same RNG).

**Official built-in `"random"` agent uses unseeded `random.Random()`** — not reproducible. Use our `RandomLegalAgent` for seeded experiments.

---

## Both-players / shared state

Affected by both players:

- Market inventory and prices (sales, buys, town drain).
- Relative win/loss via final money comparison.

Independent per player: farm tiles, money, private inventories, hires, land unlocks.

---

## Configuration defaults (`kaggriculture.json`)

| Parameter | Default |
| --- | --- |
| episodeSteps | 720 |
| actTimeout | 1 |
| boardSize | 10 |
| startingMoney | 3000 |
| maxMarketOrdersPerTurn | 10 |
| turnsPerDay | 24 |
| shedCapacity | 100 |
| weedSpawnChance | 0.005 |
| townShopUnlockInterval | 3 |
| townShopSellInterval | 4 |
| townCenterSellInterval | 24 |
| farmHandCostMult | 1 |
| seed | null |
| marketParams | {} |

---

## Unclear / surprising mechanics (flagged)

1. **`PICKUP` requires an item argument** — some competition text omits it; source requires `["PICKUP", item, n?]`.
2. **`DROP` exists** and dumps the entire inventory.
3. **Seeds never enter the shed**; `SELL` only sells shed contents (must DROP / EOD first after harvest).
4. **Hire/land after unit actions** — new hands idle until next turn.
5. **Atomic PLANT** — over-demand zeroes all PLANTs for that crop that turn.
6. **New plants start `consecutive_unwatered=1`**; animals start `consecutive_unfed=0`.
7. **Official `random` is unseeded.**
8. **Melon `max_yield_day=12` in code** vs table “time to max yield 10” (cap hit earlier under watering).
9. **Day-29 EOD does not fire** at default 720 steps (measured).
10. **Invalid actions are silent no-ops**, not hard failures — hard to detect without our own validators.
11. Competition markdown said watering ongoing crops does not increase yield; code doubles ongoing yield when fertilized **and** watered on production day.
12. **Carrot / tomato / egg scarcity hinge (1.32.7):** once town demand drains inventory past one field’s `T`, those three sell prices spike. Glut-side curves are unchanged.

---

## Built-in agents

- `pass` — always PASS
- `random` — unseeded noisy actions
- `starter` — carrot buy/plant/water/harvest/sell loop
