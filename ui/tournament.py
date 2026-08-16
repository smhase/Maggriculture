"""Authoritative tournament reports with an exploratory experiment fallback."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from ui.common import discover_json_files, read_json
from ui.experiments import load_experiment


def aggregate_population(
    experiments: Sequence[Mapping[str, Any]],
    *,
    k_factor: float = 24.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Derive exploratory Elo and matchup records from experiment game rows."""
    records: dict[str, dict[str, Any]] = {}
    ratings: dict[str, float] = defaultdict(lambda: 1500.0)
    matchup: dict[tuple[str, str], dict[str, Any]] = {}

    for experiment in experiments:
        fallback_agents = list(experiment.get("agents", []))
        for game in experiment.get("games", []):
            names = _game_agent_names(game, fallback_agents)
            if len(names) < 2 or names[0] == names[1]:
                continue
            statuses = game.get("statuses", [])
            if any(status in ("ERROR", "INVALID", "TIMEOUT") for status in statuses):
                continue
            rewards = game.get("rewards", [None, None])
            if len(rewards) < 2 or rewards[0] is None or rewards[1] is None:
                continue
            winner = game.get("winner")
            outcome_a = 1.0 if winner == 0 else 0.0 if winner == 1 else 0.5
            outcome_b = 1.0 - outcome_a

            for index, name in enumerate(names[:2]):
                record = records.setdefault(
                    name,
                    {"agent": name, "games": 0, "wins": 0, "losses": 0, "draws": 0, "score_total": 0.0},
                )
                record["games"] += 1
                record["score_total"] += float(rewards[index])
                if winner == index:
                    record["wins"] += 1
                elif winner is None:
                    record["draws"] += 1
                else:
                    record["losses"] += 1

            expected_a = 1.0 / (1.0 + 10.0 ** ((ratings[names[1]] - ratings[names[0]]) / 400.0))
            change = k_factor * (outcome_a - expected_a)
            ratings[names[0]] += change
            ratings[names[1]] -= change

            left, right = sorted(names[:2])
            key = (left, right)
            match = matchup.setdefault(
                key,
                {"agent_a": left, "agent_b": right, "games": 0, "wins_a": 0, "wins_b": 0, "draws": 0},
            )
            match["games"] += 1
            if winner is None:
                match["draws"] += 1
            else:
                winning_name = names[int(winner)]
                match["wins_a" if winning_name == left else "wins_b"] += 1

    leaderboard: list[dict[str, Any]] = []
    for name, record in records.items():
        games = int(record["games"])
        effective_wins = float(record["wins"]) + 0.5 * float(record["draws"])
        lower, upper = wilson_interval(effective_wins, games)
        leaderboard.append(
            {
                "Agent": name,
                "Elo": round(ratings[name], 1),
                "Games": games,
                "Wins": record["wins"],
                "Losses": record["losses"],
                "Draws": record["draws"],
                "Win rate": effective_wins / games if games else 0.0,
                "95% interval": f"{lower:.1%}–{upper:.1%}",
                "Average score": record["score_total"] / games if games else 0.0,
            }
        )
    leaderboard.sort(key=lambda row: (row["Elo"], row["Win rate"]), reverse=True)
    matchups = sorted(matchup.values(), key=lambda row: (row["agent_a"], row["agent_b"]))
    return leaderboard, matchups


def tournament_report_view(
    report: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Adapt a persisted ``tournament_v1`` report to stable UI rows."""
    if report.get("schema") != "tournament_v1":
        raise ValueError("Expected a tournament_v1 report")
    leaderboard: list[dict[str, Any]] = []
    for rating in report.get("ratings", []):
        games = int(rating.get("games", 0))
        effective_wins = float(rating.get("wins", 0)) + 0.5 * float(rating.get("draws", 0))
        lower, upper = wilson_interval(effective_wins, games)
        leaderboard.append(
            {
                "Agent": str(rating.get("id", "unknown")),
                "Elo": round(float(rating.get("rating", 1500.0)), 1),
                "Games": games,
                "Wins": int(rating.get("wins", 0)),
                "Losses": int(rating.get("losses", 0)),
                "Draws": int(rating.get("draws", 0)),
                "Win rate": float(rating.get("effective_win_rate") or 0.0),
                "95% interval": f"{lower:.1%}–{upper:.1%}",
                "Average score": rating.get("average_score"),
                "Failures": int(rating.get("failures", 0)),
            }
        )
    leaderboard.sort(key=lambda row: (row["Elo"], row["Win rate"]), reverse=True)
    return leaderboard, [dict(matchup) for matchup in report.get("matchups", [])]


def wilson_interval(successes: float, games: int, z: float = 1.96) -> tuple[float, float]:
    if games <= 0:
        return 0.0, 0.0
    proportion = successes / games
    denominator = 1.0 + z * z / games
    center = (proportion + z * z / (2.0 * games)) / denominator
    margin = z * math.sqrt(
        (proportion * (1.0 - proportion) + z * z / (4.0 * games)) / games
    ) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def matchup_matrix(
    leaderboard: Sequence[Mapping[str, Any]],
    matchups: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    names = [str(row["Agent"]) for row in leaderboard]
    lookup = {(row["agent_a"], row["agent_b"]): row for row in matchups}
    rows: list[dict[str, Any]] = []
    for name in names:
        output: dict[str, Any] = {"Agent": name}
        for opponent in names:
            if name == opponent:
                output[opponent] = "—"
                continue
            left, right = sorted((name, opponent))
            match = lookup.get((left, right))
            if not match:
                output[opponent] = "no games"
                continue
            wins = match["wins_a"] if name == left else match["wins_b"]
            effective = float(wins) + 0.5 * float(match["draws"])
            output[opponent] = f"{effective / match['games']:.0%} · n={match['games']}"
        rows.append(output)
    return rows


def render_tournament_page(default_directory: Path) -> None:
    import streamlit as st

    st.markdown('<span class="kg-badge">Population view</span><span class="kg-badge">Persisted Elo</span>', unsafe_allow_html=True)
    st.subheader("Tournament overview")
    directory = Path(
        st.text_input("Results directory", value=str(default_directory), key="tournament_directory")
    ).expanduser()
    tournament_reports: list[tuple[Path, dict[str, Any]]] = []
    experiments: list[dict[str, Any]] = []
    for path in sorted(discover_json_files(directory), key=lambda candidate: candidate.name):
        try:
            raw = read_json(path)
            if isinstance(raw, Mapping) and raw.get("schema") == "tournament_v1":
                tournament_reports.append((path, dict(raw)))
            else:
                experiments.append(load_experiment(path))
        except (OSError, ValueError):
            continue
    if tournament_reports:
        selected = st.selectbox(
            "Tournament report",
            range(len(tournament_reports)),
            format_func=lambda index: tournament_reports[index][0].name,
        )
        report = tournament_reports[selected][1]
        leaderboard, matchups = tournament_report_view(report)
        summary = report.get("summary", {})
        st.caption(
            f"Authoritative configured tournament · {summary.get('completed_games', 0)} "
            f"completed · {summary.get('failed_games', 0)} failed · "
            f"schema {report.get('schema')}"
        )
    else:
        leaderboard, matchups = aggregate_population(experiments)
        st.caption(
            "No persisted tournament report found. Showing exploratory Elo derived "
            "from compatible experiment files; file ordering affects these ratings."
        )
    if not leaderboard:
        st.info("No tournament report or compatible head-to-head experiments found.")
        return

    total_games = sum(matchup["games"] for matchup in matchups)
    top = leaderboard[0]
    cols = st.columns(4)
    cols[0].metric("Agents", len(leaderboard))
    cols[1].metric("Recorded games", total_games)
    cols[2].metric("Current leader", top["Agent"])
    cols[3].metric("Leader Elo", top["Elo"])

    st.markdown("### Ranking")
    st.dataframe(
        leaderboard,
        hide_index=True,
        use_container_width=True,
        column_config={"Win rate": st.column_config.ProgressColumn(format="percent", min_value=0.0, max_value=1.0)},
    )
    st.markdown("### Matchup matrix")
    st.caption("Cells show effective win rate (draw = half win) and completed sample count.")
    st.dataframe(matchup_matrix(leaderboard, matchups), hide_index=True, use_container_width=True)
    with st.expander("Raw matchup records"):
        st.dataframe(matchups, hide_index=True, use_container_width=True)


def _game_agent_names(game: Mapping[str, Any], fallback: Sequence[str]) -> list[str]:
    agents = game.get("agents") or fallback
    return [
        str(agent.get("name", "unknown")) if isinstance(agent, Mapping) else str(agent)
        for agent in agents
    ]
