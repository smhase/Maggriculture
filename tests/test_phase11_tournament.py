"""Phase 11 planner vs baseline population (short horizon)."""

from __future__ import annotations

from pathlib import Path

from kaggriculture.simulation.population import load_tournament_config
from kaggriculture.simulation.tournament import run_tournament

CONFIG = Path(__file__).resolve().parents[1] / "experiments/configs/tournament_planner_smoke.yaml"


def test_planner_smoke_tournament_completes_without_failures():
    """96-turn matchups are harsh for planting; this checks the harness, not Elo."""
    config = load_tournament_config(CONFIG)
    report = run_tournament(config)
    assert report["schema"] == "tournament_v1"
    assert report["summary"]["failed_games"] == 0
    assert report["summary"]["completed_games"] == 6
    by_id = {row["id"]: row for row in report["ratings"]}
    assert by_id["planner_v1"]["games"] == 4
    assert by_id["planner_v1"]["failures"] == 0
