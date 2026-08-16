# Compact Replay Format

Phase 8 uses `compact_v2`, a JSON format designed for research inspection and
the future replay UI. The official Kaggle replay remains available separately
with `--full-replay` and remains authoritative.

## Why deltas

A full Kaggriculture state contains two 10×10 farms. Repeating both grids for
all 720 turns wastes space because most tiles do not change on most turns.
Compact v2 stores:

1. one complete normalized `initial_state`;
2. the chosen actions and headline values for every turn;
3. only state fields and tiles that changed after the initial turn; and
4. human-oriented events derived from consecutive states.

Use `reconstruct_turn()` rather than applying deltas in UI or analysis code.
That function is the stable compatibility boundary if the storage encoding
changes later.

## Top-level document

```text
meta
initial_state
turns[]
```

`meta` contains the format version, timestamp, Git commit when available,
environment version, seed, configured horizon, agents, final scores/statuses,
runtime, and state encoding.

`initial_state` contains:

- step, day, and hour;
- both public farms, including farmer/hands, unlocked land, and complete tiles;
- each player's own private shed, seeds, and unit inventories;
- shared market inventory and prices; and
- unlocked town shops.

Private state is merged from the two separate official player observations
after a local game. This is intentional for offline research; it does not imply
that a live agent can observe its opponent's private resources.

## Turn records

Every turn retains the compact v1 headline fields for compatibility:

- `step`, `day`, `hour`;
- both actions;
- both bank balances;
- market prices;
- statuses, rewards, agent durations, and shops.

It additionally contains:

- `state_delta` — changed player fields, private counts, unit inventories,
  tiles, market counts/prices, and shops;
- `events` — semantic changes useful to people and the future UI;
- `reasoning` — optional CoC traces (`coc_v1`), one per seat. Turn 0 is
  `[null, null]`. A seat may be one object or a list (crew farmer + hands).

Turn zero corresponds to `initial_state` and has an empty delta. A later turn's
delta transforms the preceding recorded state into that turn's resulting
state. Recorded actions follow the official `kaggle-environments` step layout.

## Events

Current event types are:

- `money_changed`
- `resource_changed` for shed or seed counts
- `land_unlocked`
- `worker_count_changed`
- `plant_created`
- `plant_removed`
- `weed_appeared`
- `structure_built`
- `animal_placed`
- generic `tile_changed`
- `shops_changed`

Events are derived diagnostics, not authoritative mechanics. The underlying
state delta and official full replay take precedence if an event label is
ambiguous—for example, a removed plant may have been harvested or dug up.

## Python API

```python
from kaggriculture.simulation.replay import load_compact_replay, reconstruct_turn

replay = load_compact_replay("experiments/replays/demo.compact.json")
state_at_327 = reconstruct_turn(replay, 327)

player_a = state_at_327["players"][0]
player_b = state_at_327["players"][1]
prices = state_at_327["market"]["prices"]
events = replay["turns"][327]["events"]
```

Random access currently applies deltas from the beginning, which is cheap for
720 turns. The Phase 9 viewer may cache reconstructed turns or periodic
checkpoints if profiling shows a need.

## Compatibility and limitations

- `load_compact_replay()` can read structural compact v1 files, but
  `reconstruct_turn()` rejects them because they have no initial state.
- Compact v2 may include per-turn `reasoning` (see `meta.reasoning_schema`).
  Traces are research-only and are not sent to the official engine.
- Events describe observable transitions; they do not attempt to infer every
  economic or strategic cause. Decision traces explain chosen actions.
- JSON is deliberately readable. Compression can be added at the file boundary
  later without changing the logical schema.
