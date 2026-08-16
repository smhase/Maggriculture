"""Phase 7 batch metrics and experiment report tests."""

from __future__ import annotations

import json

import kaggle_environments

from kaggriculture.agents import OfficialAgent
from kaggriculture.env.official_env import KaggricultureEnv
from kaggriculture.simulation.metrics import build_experiment_report, summarize_results
from kaggriculture.simulation.runner import main, play_games


def test_batch_summary_and_report_capture_reproducibility():
    results = play_games("pass", "pass", games=2, seed_start=20, episode_steps=24)

    summary = summarize_results(results)
    assert summary["games"] == 2
    assert summary["completed_games"] == 2
    assert summary["wins"] == [0, 0]
    assert summary["draws"] == 2
    assert summary["average_score"] == [3000.0, 3000.0]
    assert summary["average_score_differential_a_minus_b"] == 0.0
    assert summary["validator_rejected_actions"] == [0, 0]
    assert summary["total_steps"] == 48

    report = build_experiment_report(
        results,
        config={"games": 2, "seed_start": 20, "episode_steps": 24},
    )
    assert report["schema"] == "experiment_v1"
    assert report["kaggle_environments_version"] == kaggle_environments.__version__
    assert report["agents"][0]["version"].startswith("official@")
    assert [game["seed"] for game in report["games"]] == [20, 21]
    assert report["games"][0]["end_state_resources"][0]["money"] == 3000.0


def test_validator_rejections_count_silent_noops():
    def always_water(_obs, _configuration=None):
        return {"farmer": ["WATER"], "hands": [], "market": []}

    env = KaggricultureEnv(seed=2, episode_steps=24)
    result = env.play(always_water, OfficialAgent("pass"))

    assert result.statuses == ["DONE", "DONE"]
    assert result.invalid_action_counts[0] > 0
    assert result.invalid_action_counts[1] == 0


def test_runner_writes_experiment_json(tmp_path):
    output = tmp_path / "experiment.json"
    exit_code = main(
        [
            "--agent-a",
            "pass",
            "--agent-b",
            "pass",
            "--games",
            "2",
            "--seed-start",
            "30",
            "--episode-steps",
            "24",
            "--experiment-json",
            str(output),
        ]
    )

    assert exit_code == 0
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["config"]["seed_start"] == 30
    assert data["summary"]["games"] == 2
    assert len(data["games"]) == 2
