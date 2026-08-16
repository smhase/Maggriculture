# Kaggriculture Competitive Agent Roadmap

This is the repository copy of the main project plan supplied on 2026-08-14.
It is the source of truth for sequencing work. The official
`kaggle-environments` Kaggriculture implementation overrides this roadmap
whenever the two disagree.

## Goal

Build a small AI research and simulation platform whose final output is a
Kaggle-compatible agent. The platform must support reproducible local games,
many agent variants, large experiment batches, replay analysis, tournaments,
planning, opponent and market models, learning, and automatic submission
export.

The intended competitive agent is a hybrid of deterministic economics,
macro-level planning, opponent and market modelling, and learned policy/value
components. Do not jump directly to reinforcement learning.

## Design principles

- Use the official environment as the source of truth; wrap it instead of
  copying the engine.
- Preserve the real 720-turn competitive horizon in evaluation and training.
- Centralize observation parsing and legal-action diagnostics.
- Plan over macro actions, then translate them into primitive legal actions.
- Keep strategy experiments reproducible, versioned, and hypothesis-driven.
- Prefer typed, composable Python modules with minimal global state.
- Measure before optimizing, while keeping million-game workloads in mind.
- Generate submissions automatically; never maintain a hand-copied `main.py`.

The intended progression is:

```text
official environment -> GameState -> economics -> scripted baselines
-> optimized/adaptive strategy -> macro planner -> opponent/market models
-> self-play data -> imitation/value learning -> RL fine-tuning -> hybrid agent
```

## Phases

### Phase 0 — Competition understanding

Inspect official source and document observations, actions, legality, turn
order, movement, farms, crops, animals, inventory, market, land, workers,
visibility, stochastic mechanics, the terminal condition, and scoring in
`docs/game-mechanics.md`. Flag unclear and surprising behavior explicitly.

### Phase 1 — Environment wrapper

Provide deterministic seeding, head-to-head play, turn access, replay capture,
timing, status and result reporting, invalid-action diagnostics, and quiet or
configurable logging without duplicating the interpreter.

### Phase 2 — GameState abstraction

Normalize raw observations into typed player, farm, market, town, and game
structures. Keep the raw observation for debugging and test normalized fields
against official observations.

### Phase 3 — Actions and validation

Represent only real Kaggriculture actions. Provide `get_legal_actions` and
`is_action_legal` as debugging/planning helpers while retaining the official
environment as authority.

### Phase 4 — Economic analysis

Analyze crops, animals, land, and other investments for costs, production and
travel time, yield, break-even, profit, capital efficiency, worker/tile use,
market price, and terminal-horizon value. Provide ranking APIs and regression
tests.

### Phase 5 — Baseline agents

Maintain a random-legal stress agent, a minimal profitable economic agent, and
a deterministic scripted season plan with debuggable strategic reasoning.

### Phase 6 — Macro actions

Define mechanics-backed intents such as planting, maintaining, harvesting,
selling, buying inputs, expanding, animal production, and liquidation. A
tactical scheduler must account for positions, travel, dependencies,
interruptions, and invalidated plans.

### Phase 7 — Batch simulator

Run many seeded matchups from API and CLI. Record wins, losses, draws, scores,
score differential, profit, runtime, validator rejections, end-state resources,
agent/environment versions, configuration, timestamp, seed range, and Git
commit in structured experiment reports.

### Phase 8 — Replay system

Capture compact per-turn state for both players, actions, market state, and
important events. Support useful inspection without writing unnecessarily huge
files; use deltas where they materially help.

### Phase 9 — Research UI

Build a lightweight replay viewer plus experiment and tournament views. Favor
debuggability over polish. Include navigation, playback, farm/inventory/market
inspection, useful histories, run comparison, rankings, Elo, and matchup data.

### Phase 10 — Tournament system

Run configured populations and games per matchup. Store Elo (initially), sample
counts, and the full matchup matrix so rock-paper-scissors relationships remain
visible.

### Phase 11 — Model-based planner

Start with configurable beam search / model-predictive control over macros.
Use a terminal-aware hand-built evaluator covering cash, liquidation value,
production value, infrastructure, future costs, and risk. Measure runtime under
Kaggle limits.

### Phase 12 — Opponent modelling

Summarize crop and animal mix, expansion, capital, selling behavior, production
cycles, and market influence in a simple interpretable profile before
considering neural models.

### Phase 13 — Market modelling

Track history and evaluate simple moving-average, momentum, supply-pressure,
inventory, and harvest/sale-event predictors. Retain only predictors that
improve head-to-head decisions.

### Phase 14 — ML dataset

Use the strong planner as teacher. Store compact features, candidate actions,
planner scores, selections, and outcomes rather than giant raw objects.

### Phase 15 — Imitation learning

Distill search into a fast policy, then compare it against its teacher and the
scripted benchmark population. Use it as a potential RL initialization.

### Phase 16 — Value network

Predict final score differential or win probability from `GameState` features
and use the model to evaluate planner frontiers for deeper effective search.

### Phase 17 — Population self-play

Train against a diverse historical population, not only the latest self.
Preserve strong checkpoints and track Elo, per-opponent win rate,
exploitability, and strategy diversity.

### Phase 18 — Reinforcement learning

Only after strong baselines, simulation, datasets, and self-play exist,
investigate masked PPO / actor-critic or another justified method. Start from
imitation weights where useful and use structured features.

### Phase 19 — Hybrid agent

Incrementally combine `GameState`, opponent and market models, policy/strategy,
macro generation, beam search, value evaluation, and tactical scheduling.

### Phase 20 — Submission export and release validation

Generate a Kaggle-ready directory automatically. Validate the callable,
packaged files/checkpoints, imports, dependency assumptions, determinism,
runtime, size, and invalid actions. Benchmark 100–1000 games against a selected
population and emit a release report with version, Git commit, Elo, win rate,
runtime, rejection count, and known weaknesses.

## Experiment discipline

Treat a meaningful strategy change as one hypothesis. Record a control,
treatment, configuration, benchmark population, seed range, result, and
matchup-specific regressions. Avoid bundling unrelated changes into one
experiment.

Important parameters belong in configuration: agent version, beam width,
depth, horizon, model toggles, number of games, and seed range. Logging must
support `ERROR`, `WARNING`, `INFO`, `DEBUG`, and `TRACE` without flooding large
runs.

## Phase workflow

For each phase:

1. Inspect existing code and official mechanics.
2. State the smallest coherent increment.
3. Implement it with tests.
4. Run the tests and a relevant sanity simulation.
5. Report mechanics discoveries, limitations, and the recommended next phase.

Keep `docs/implementation-status.md` current as evidence changes.
