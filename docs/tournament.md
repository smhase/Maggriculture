# Tournament System

Phase 10 turns a named agent population into a balanced round-robin evaluation
and one durable `tournament_v1` report. The official environment still runs
every game; the tournament layer only schedules games and aggregates outcomes.

## Run

```bash
python -m kaggriculture.simulation.tournament \
  --config experiments/configs/tournament_baselines.yaml \
  --output experiments/results/baseline_tournament.json
```

The command exits with status 2 if any game has an `ERROR`, `INVALID`, or
`TIMEOUT` status. Failed games remain in the report but do not affect Elo,
completed-game sample counts, scores, or win rates.

## Configuration

Tournament YAML uses schema `tournament_config_v1`:

```yaml
schema: tournament_config_v1
name: baseline_population_v1
population:
  - {id: starter, agent: starter, label: Official starter}
  - {id: planner_v1, agent: planner, label: Macro beam planner v1}
games_per_matchup: 2
seed_start: 10000
episode_steps: 720
initial_rating: 1500
k_factor: 24
debug: false
```

Participant `id` is the stable identity used by ratings and the matrix. `agent`
is a spec understood by `simulation.runner.resolve_agent`. Keeping these fields
separate permits multiple configured variants of an implementation once the
agent resolver supports their parameters.

`games_per_matchup` must be even and at least two. For each unordered pair and
each repetition, the scheduler runs the same environment seed twice with seats
swapped. A local random agent's seed is derived from participant ID plus the
environment seed, so it also retains the same random stream in the mirror game.
The official built-in `random` agent remains nondeterministic and should not be
used in reproducibility-sensitive populations.

## Persisted report

`tournament_v1` records:

- environment version, Git commit, timestamp, and the normalized config;
- planned/completed/failed counts and runtime;
- ranked Elo plus wins, losses, draws, failures, sample count, effective win
  rate, and average score for every participant;
- aggregate results for every configured unordered pair;
- a full directional matrix whose cells contain games, wins, losses, draws,
  failures, and effective win rate; and
- every underlying game, including seats, participant IDs, agent implementation
  names/versions, seed, reward, status, diagnostics, resources, and replay path.

Elo is updated sequentially in deterministic schedule order. It is a convenient
ranking, not a replacement for the matchup matrix: non-transitive populations
and small samples can make a single rating misleading.
