# Research UI

Phase 9 provides a lightweight Streamlit console over the artifacts produced
by Phases 7 and 8. It is read-only: the UI does not mutate agents, experiments,
or official replays.

## Start the app

```bash
source .venv/bin/activate
streamlit run ui/app.py
```

The app defaults to `experiments/replays` and `experiments/results`. Override
those roots for temporary or external artifacts when needed:

```bash
KAGGRICULTURE_REPLAY_DIR=/tmp/replays \
KAGGRICULTURE_RESULTS_DIR=/tmp/results \
streamlit run ui/app.py
```

## Replay viewer

The viewer reads compact v2 replays (or a sibling official `*.full.json`) and
embeds the **official Kaggriculture episode player** — the same tile graphics,
town, market, and HUD used on the Kaggle leaderboard. That player has its own
play/step controls.

It also keeps a research inspector for:

- replay selection;
- previous/next, slider-based jumping, and play/pause;
- reconstructed farms and private resources for both players;
- farmer and hired-hand locations;
- chosen actions, **decision traces (why)**, and derived events;
- shared market inventory, prices, and town shops; and
- full-season bank-balance and commodity-price charts.

Compact v1 files retain headline charts and actions but cannot reconstruct
farms or drive the official visualizer unless a matching `*.full.json` is
present. Official `*.full.json` files are not listed in the compact-replay
selector; they are picked up automatically when named alongside a compact file.

## Experiment comparison

The experiments page accepts both `experiment_v1` reports and older
list-of-games result files. It displays outcomes, average scores, score
differential, per-game results, runtime metadata, and an optional control vs
treatment comparison.

The treatment summary reports the change in average score differential and
Player A win rate. It does not claim statistical significance; experiment
design and benchmark-population discipline still belong in the research
workflow.

## Tournament overview

The page prefers authoritative `tournament_v1` reports created by the Phase 10
runner. These contain configured, seat-balanced populations and persisted:

- Elo starting at the config's initial rating and K-factor;
- wins, losses, draws, failures, effective win rate, and average score;
- per-agent and per-matchup sample counts; and
- the complete directional matchup matrix.

When no tournament report exists, the page falls back to a read-only population
view derived from compatible experiment files:

- exploratory Elo starting at 1500, updated with K=24;
- wins, losses, draws, effective win rate, and average score;
- a Wilson 95% interval using draws as half-wins;
- sample counts; and
- a complete observed matchup matrix.

Fallback ratings depend on the set and order of experiment files and are
explicitly labelled exploratory.

## Architecture

Presentation transformations are ordinary Python functions and are tested
without starting Streamlit. Rendering stays in `ui/`, while replay parsing and
reconstruction remain in `kaggriculture.simulation`. This prevents the UI from
becoming a second simulation or data-model layer.
