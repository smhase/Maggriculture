"""Phase 10 population scheduling, ratings, persistence, and CLI tests."""

from __future__ import annotations

import json

import pytest

from kaggriculture.simulation.population import TournamentConfig, balanced_schedule
from kaggriculture.simulation.tournament import (
    build_tournament_report,
    main,
    run_tournament,
)


def _config(**overrides):
    values = {
        "schema": "tournament_config_v1",
        "name": "test_population",
        "population": [
            {"id": "pass_a", "agent": "pass"},
            {"id": "pass_b", "agent": "pass"},
            {"id": "pass_c", "agent": "pass"},
        ],
        "games_per_matchup": 2,
        "seed_start": 50,
        "episode_steps": 24,
    }
    values.update(overrides)
    return TournamentConfig.from_mapping(values)


def test_balanced_schedule_mirrors_every_seed_and_pair():
    schedule = balanced_schedule(_config())
    assert len(schedule) == 6
    assert (schedule[0].agent_a.id, schedule[0].agent_b.id) == ("pass_a", "pass_b")
    assert (schedule[1].agent_a.id, schedule[1].agent_b.id) == ("pass_b", "pass_a")
    assert schedule[0].seed == schedule[1].seed == 50
    assert [game.seed for game in schedule] == [50, 50, 51, 51, 52, 52]


def test_config_rejects_unbalanced_matchup_count():
    with pytest.raises(ValueError, match="even integer"):
        _config(games_per_matchup=3)


def test_failed_games_are_persisted_but_excluded_from_elo_and_samples():
    config = _config(
        population=[
            {"id": "pass_a", "agent": "pass"},
            {"id": "pass_b", "agent": "pass"},
        ]
    )
    players = [{"id": "pass_a"}, {"id": "pass_b"}]
    report = build_tournament_report(
        config,
        [
            {
                "players": players,
                "statuses": ["DONE", "DONE"],
                "rewards": [3100, 3000],
                "winner_id": "pass_a",
            },
            {
                "players": players,
                "statuses": ["ERROR", "DONE"],
                "rewards": [None, 3000],
                "winner_id": None,
            },
        ],
        wall_duration_s=1.0,
    )
    ratings = {row["id"]: row for row in report["ratings"]}
    assert ratings["pass_a"]["rating"] == 1512.0
    assert ratings["pass_b"]["rating"] == 1488.0
    assert ratings["pass_a"]["games"] == 1
    assert ratings["pass_a"]["failures"] == 1
    assert report["matchups"][0]["games"] == 1
    assert report["matchups"][0]["failures"] == 1


def test_tournament_persists_ratings_samples_and_full_matrix(tmp_path):
    output = tmp_path / "tournament.json"
    report = run_tournament(_config(), output_path=output)

    assert report["schema"] == "tournament_v1"
    assert report["summary"]["planned_games"] == 6
    assert report["summary"]["completed_games"] == 6
    assert report["summary"]["failed_games"] == 0
    assert len(report["ratings"]) == 3
    assert all(row["rating"] == 1500.0 for row in report["ratings"])
    assert all(row["games"] == 4 for row in report["ratings"])
    assert len(report["matchups"]) == 3
    assert report["matrix"]["pass_a"]["pass_b"] == {
        "games": 2,
        "wins": 0,
        "losses": 0,
        "draws": 2,
        "failures": 0,
        "effective_win_rate": 0.5,
    }
    assert json.loads(output.read_text(encoding="utf-8"))["schema"] == "tournament_v1"
    first, second = report["games"][:2]
    assert first["seed"] == second["seed"]
    assert [player["id"] for player in first["players"]] == ["pass_a", "pass_b"]
    assert [player["id"] for player in second["players"]] == ["pass_b", "pass_a"]


def test_tournament_cli_loads_yaml_and_writes_report(tmp_path):
    config_path = tmp_path / "tournament.yaml"
    output_path = tmp_path / "report.json"
    config_path.write_text(
        """schema: tournament_config_v1
name: cli_test
population:
  - {id: pass_a, agent: pass}
  - {id: pass_b, agent: pass}
games_per_matchup: 2
seed_start: 70
episode_steps: 24
""",
        encoding="utf-8",
    )
    assert main(["--config", str(config_path), "--output", str(output_path)]) == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["summary"]["completed_games"] == 2
