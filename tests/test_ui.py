"""Phase 9 UI data transformation tests (independent of Streamlit runtime)."""

from __future__ import annotations

from ui.experiments import experiment_game_rows, normalize_experiment
from ui.replay import farm_grid, replay_timeline
from ui.tournament import (
    aggregate_population,
    matchup_matrix,
    tournament_report_view,
    wilson_interval,
)


def test_replay_presentation_helpers():
    replay = {
        "turns": [
            {"step": 0, "money": [3000, 3000], "market_prices": {"WHEAT": 25}},
            {"step": 1, "money": [2990, 2980], "market_prices": {"WHEAT": 26}},
        ]
    }
    timeline = replay_timeline(replay)
    assert timeline[1] == {
        "turn": 1,
        "Player A": 2990,
        "Player B": 2980,
        "price:WHEAT": 26,
    }

    player = {
        "farmer": [0, 0],
        "hands": [[1, 1]],
        "tiles": [
            [None, {"kind": "PLANT", "crop": "WHEAT", "yield_units": 2}],
            ["LOCKED", {"kind": "WEED"}],
        ],
    }
    grid = farm_grid(player)
    assert grid[0][0] == "F··"
    assert grid[0][1] == "WHE:2"
    assert grid[1][0] == "LOCK"
    assert grid[1][1] == "H·WEED"
    marked = farm_grid(player, target=(1, 0))
    assert marked[0][1].startswith("*")


def test_reasoning_rows():
    from ui.replay import _reasoning_rows

    rows = _reasoning_rows(
        {"headline": "hi", "unit": "farmer", "source": "planner", "causes": ["a"], "plan": ["x"]}
    )
    assert rows[0]["Headline"] == "hi"


def test_normalize_legacy_and_experiment_v1():
    legacy = [
        {
            "seed": 42,
            "agents": ["scripted", "starter"],
            "rewards": [4500, 3500],
            "winner": 0,
            "duration_s": 1.5,
        }
    ]
    normalized = normalize_experiment(legacy, name="legacy")
    assert normalized["schema"] == "legacy_games_v0"
    assert normalized["agents"] == ["scripted", "starter"]
    assert normalized["summary"]["wins"] == [1, 0]
    assert experiment_game_rows(normalized)[0]["Differential"] == 1000.0

    modern = normalize_experiment(
        {
            "schema": "experiment_v1",
            "agents": [
                {"name": "planner", "version": "0.1.0"},
                {"name": "scripted", "version": "0.2.0"},
            ],
            "summary": {"games": 2, "wins": [2, 0]},
            "games": [],
            "config": {"seed_start": 10},
        },
        name="modern",
    )
    assert modern["agents"] == ["planner", "scripted"]
    assert modern["metadata"]["config"]["seed_start"] == 10


def test_population_aggregation_elo_confidence_and_matrix():
    experiment = normalize_experiment(
        [
            {
                "agents": ["planner", "scripted"],
                "rewards": [5000, 4000],
                "winner": 0,
                "statuses": ["DONE", "DONE"],
            },
            {
                "agents": ["planner", "scripted"],
                "rewards": [4200, 4200],
                "winner": None,
                "statuses": ["DONE", "DONE"],
            },
        ],
        name="population",
    )
    leaderboard, matchups = aggregate_population([experiment])
    assert [row["Agent"] for row in leaderboard] == ["planner", "scripted"]
    assert leaderboard[0]["Games"] == 2
    assert leaderboard[0]["Win rate"] == 0.75
    assert leaderboard[0]["Elo"] > leaderboard[1]["Elo"]
    lower, upper = wilson_interval(1.5, 2)
    assert 0.0 <= lower < upper <= 1.0
    matrix = matchup_matrix(leaderboard, matchups)
    assert matrix[0]["scripted"] == "75% · n=2"


def test_authoritative_tournament_report_view():
    report = {
        "schema": "tournament_v1",
        "ratings": [
            {
                "id": "planner_v1",
                "rating": 1512.5,
                "games": 2,
                "wins": 1,
                "losses": 0,
                "draws": 1,
                "effective_win_rate": 0.75,
                "average_score": 4100.0,
                "failures": 0,
            },
            {
                "id": "scripted_v1",
                "rating": 1487.5,
                "games": 2,
                "wins": 0,
                "losses": 1,
                "draws": 1,
                "effective_win_rate": 0.25,
                "average_score": 3900.0,
                "failures": 0,
            },
        ],
        "matchups": [
            {
                "agent_a": "planner_v1",
                "agent_b": "scripted_v1",
                "games": 2,
                "wins_a": 1,
                "wins_b": 0,
                "draws": 1,
            }
        ],
    }
    leaderboard, matchups = tournament_report_view(report)
    assert leaderboard[0]["Agent"] == "planner_v1"
    assert leaderboard[0]["Elo"] == 1512.5
    assert matchup_matrix(leaderboard, matchups)[0]["scripted_v1"] == "75% · n=2"
