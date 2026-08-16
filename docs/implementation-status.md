# Implementation Status

Audited against `docs/roadmap.md` on 2026-08-16. A phase is called complete
only when its central deliverable exists and is covered by relevant tests.
Later prototypes do not move the sequential phase boundary past an earlier
gap.

## Sequential milestone

The platform is implemented through **Phase 13**. The next sequential target is
**Phase 14 — ML dataset**.

## Evidence by phase

- **Phase 0 — complete.** `docs/game-mechanics.md` documents official 1.32.7
  mechanics and measured episode boundaries.
- **Phase 1 — complete.** `KaggricultureEnv` supports seeded games, explicit
  stepping, official `run`, timing, results, compact/full replays, statuses,
  and local-validator rejection counts.
- **Phase 2 — complete.** Typed `GameState`, player, farm, private, market, and
  town structures are produced by one observation parser and tested against
  official observations.
- **Phase 3 — complete.** Primitive action helpers and conservative legal-unit,
  market, and full-action validation exist with atomic-plant regression tests.
- **Phase 4 — complete for the baseline scope.** Crop and animal analysis,
  horizon-aware returns, land ranking, and investment ranking are tested.
  Travel and worker scheduling costs remain planner concerns rather than exact
  economic simulation.
- **Phase 5 — complete.** Random-legal, minimal economic, and deterministic
  scripted agents complete official games; the economic and scripted agents
  beat the official starter in the full-horizon regression matchup.
- **Phase 6 — complete for crop macros.** Typed macros, candidate generation,
  scheduling, navigation, interruption handling, and invalidation fallbacks
  exist. Animal-production macros are intentionally deferred until an animal
  baseline justifies them.
- **Phase 7 — complete.** Seeded batch runs now emit `experiment_v1` JSON with
  reproducibility metadata, per-game records, aggregate outcomes, scores,
  profit, runtime throughput, validator rejections, and compact private
  end-state resources.
- **Phase 8 — complete.** Compact replay v2 stores one full initial state and
  reconstructable per-turn deltas for both farms, both private inventories,
  the shared market, and town. It retains actions/headline metrics, derives
  semantic events, and exposes a loader/reconstruction API for the viewer.
- **Phase 9 — complete.** The Streamlit research console embeds the official
  Kaggriculture episode player for compact v2 / full replays, plus experiment
  control/treatment comparison and a read-only population dashboard with
  exploratory Elo, confidence/sample counts, rankings, and matchup matrix.
  Presentation transforms have tests independent of Streamlit.
- **Phase 10 — complete.** YAML-configured populations produce deterministic
  round-robin schedules with paired seat-swapped games on the same seeds.
  `tournament_v1` reports persist Elo, per-agent samples/outcomes, failures,
  every game, pair aggregates, and a full directional matchup matrix. The UI
  prefers these authoritative reports while retaining an exploratory fallback.
- **Phase 11 — complete for the planned prototype.** Configurable macro beam
  search, terminal-aware evaluator (cash, liquidation, seeds, crops,
  infrastructure, future costs, risk), CoC traces, strategy/crew variants,
  act-timeout regression, and a smoke tournament harness against pass/heuristic.
  Full-horizon quality is the existing planner-vs-starter 720-turn test.
  Search still does not clone the official engine.
- **Phase 12 — complete.** Public-farm opponent profiles cover crop/animal mix,
  expansion, capital, hire counts, harvestable dump pressure, and a money-delta
  selling proxy. The planner applies a small interpretable leaf adjustment and
  records the summary in CoC causes.
- **Phase 13 — complete.** A per-episode `MarketTracker` keeps SMA, momentum,
  and inventory/supply pressure from public prices. Falling or oversupplied
  markets nudge the planner leaf when goods sit in the shed. Unobservable
  opponent-shed sale events were dropped.
- **Phases 14–20 — not started.** Learning, self-play, hybrid,
  and submission modules are placeholders only.

## Current limitations that affect phase claims

- Local legality is diagnostic, not authoritative; official illegal actions
  are usually silent no-ops.
- Validator rejection counts evaluate recorded actions against the preceding
  observation and may need new regression cases as mechanics are discovered.
- Batch execution is serial. Parallel workers and JSONL streaming are future
  scale optimizations, not required for the initial Phase 7 contract.
- Replay random access currently reapplies at most 719 small deltas. Phase 9
  currently does this on each slider change; an in-memory state cache can be
  added if profiling shows a need.
- Tournament Elo is sequential and depends on configured schedule order; the
  paired schedule is deterministic and seat-balanced, and the matchup matrix
  remains the primary evidence for non-transitive relationships.
- Existing Phase 11 search uses an approximate `apply_macro` mental model
  rather than cloning the official engine. Live `act()` output remains an
  official farmer/hands/market dict.
