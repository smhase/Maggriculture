# Kaggriculture Research Platform

Local research and simulation platform for the Kaggle
[Kaggriculture](https://www.kaggle.com/competitions/kaggriculture) competition.

The official `kaggle-environments` engine is the source of truth. This repo wraps
it with typed state, reproducible agents, economics, planning, compact replays,
and batch experiments; tournaments and submission export come later.

> **Milestone status:** implemented through Phase 10, with a lightweight planner
> prototype (Phase 11). Opponent/market models, ML, and export remain stubs.

## Requirements

- Python **≥ 3.11** (3.12 recommended)
- Pinned: `kaggle-environments==1.32.7`

## Setup

```bash
# From repo root (uv recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh   # if needed
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
# or: uv pip install -r requirements.txt && uv pip install -e .
```

## Run one game

```bash
python -m kaggriculture.simulation.runner \
  --agent-a random_legal \
  --agent-b starter \
  --games 1 \
  --seed-start 42 \
  --replay-dir experiments/replays
```

## Run several games

```bash
python -m kaggriculture.simulation.runner \
  --agent-a random_legal \
  --agent-b starter \
  --games 3 \
  --seed-start 42 \
  --replay-dir experiments/replays \
  --experiment-json experiments/results/sanity.json
```

Experiment JSON contains Git/environment/agent versions, configuration,
aggregate outcomes and throughput, per-game scores, validator rejections, and
end-state resources.

## Run 1000 games

```bash
python -m kaggriculture.simulation.runner \
  --agent-a planner \
  --agent-b scripted \
  --games 1000 \
  --seed-start 10000 \
  --experiment-json experiments/results/planner_vs_scripted_1000.json
```

Built-in / local agents: `pass`, `random`, `starter`, `random_legal`,
`heuristic`, `scripted`, `planner`, `long_term`, `short_term`, `solo`,
`risk_taker`, `safe`, `swing`, `always_win`, `never_lose`, `crew`.

## Run a tournament

```bash
python -m kaggriculture.simulation.tournament \
  --config experiments/configs/tournament_baselines.yaml \
  --output experiments/results/baseline_tournament.json
```

Tournament configs define a named population and an even number of games per
matchup. Every seed is run twice with seats swapped. The `tournament_v1` report
persists Elo, participant sample counts, per-game outcomes, failure diagnostics,
and the full matchup matrix.

## Python API

```python
from kaggriculture.env import KaggricultureEnv, parse_observation
from kaggriculture.agents import RandomLegalAgent, OfficialAgent

env = KaggricultureEnv(seed=42)
result = env.play(
    RandomLegalAgent(seed=0),
    OfficialAgent("starter"),
    replay_path="experiments/replays/demo.compact.json",
)
print(result.rewards, result.statuses, result.duration_s)
```

Debug step loop:

```python
env = KaggricultureEnv(seed=1, episode_steps=48)
env.reset()
while not env.done:
    env.step({"farmer": ["PASS"], "hands": [], "market": []},
             {"farmer": ["PASS"], "hands": [], "market": []})
```

## Tests

```bash
source .venv/bin/activate
pytest -q
```

## Check the official engine pin

Compares the repo pin, the installed `kaggle-environments` version, PyPI latest,
and GitHub `master` `kaggriculture.py` / `.json`:

```bash
source .venv/bin/activate
python -m kaggriculture.env.version_check
```

A daily crontab in this machine's user crontab runs that check at 09:00 local
time and appends to `experiments/results/env_version_check.log`.

## Open the research UI

```bash
source .venv/bin/activate
streamlit run ui/app.py
```

The UI includes compact replay navigation, farm/inventory/action inspection,
cash and market charts, experiment comparison, and an exploratory population
ranking derived from existing experiment results.

## Docs

- [Game mechanics](docs/game-mechanics.md) — Phase 0, derived from official source
- [Main roadmap](docs/roadmap.md) — phases, principles, and working discipline
- [Implementation status](docs/implementation-status.md) — audited phase boundary
- [Compact replay format](docs/replay-format.md) — v2 deltas, events, reconstruction
- [Research UI](docs/research-ui.md) — pages, inputs, and rating limitations
- [Tournament system](docs/tournament.md) — configs, scheduling, report schema

## Not implemented yet

- Learning / RL
- Submission exporter

## Implemented through Phase 11

- Official env wrapper + reconstructable compact v2 replays
- `GameState` parsing + legal-action helpers
- Economics: `analyze_crop`, `analyze_animal`, `rank_investments`
- Agents: `random_legal`, `heuristic`, `scripted`, `planner`, strategy profiles, `crew`
- Macros + forward-model beam search + CoC traces on compact replays
- Batch experiment reports with aggregate metrics and reproducibility metadata
- Configured, seat-balanced round-robin tournaments with persisted Elo/matrix
- Streamlit replay, experiment, and population research views

## Project layout

See `src/kaggriculture/` for env wrappers, agents, and simulation. Later-phase
packages exist as docstring stubs so the roadmap stays navigable.
