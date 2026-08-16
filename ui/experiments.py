"""Experiment comparison page with legacy-result compatibility."""

from __future__ import annotations

from pathlib import Path
from statistics import fmean
from typing import Any, Mapping

from ui.common import discover_json_files, read_json


def normalize_experiment(data: Any, *, name: str = "experiment") -> dict[str, Any]:
    """Normalize experiment_v1 or the older list-of-games result format."""
    if isinstance(data, Mapping) and data.get("schema") == "experiment_v1":
        games = list(data.get("games", []))
        agents = [
            str(agent.get("name", "unknown")) if isinstance(agent, Mapping) else str(agent)
            for agent in data.get("agents", [])
        ]
        summary = dict(data.get("summary", {}))
        return {
            "name": name,
            "schema": "experiment_v1",
            "agents": agents,
            "summary": summary,
            "games": games,
            "metadata": {
                "created_at": data.get("created_at"),
                "git_commit": data.get("git_commit"),
                "environment": data.get("kaggle_environments_version"),
                "config": data.get("config", {}),
            },
        }
    if isinstance(data, list):
        games = [dict(game) for game in data if isinstance(game, Mapping)]
        agents = [str(agent) for agent in games[0].get("agents", [])] if games else []
        return {
            "name": name,
            "schema": "legacy_games_v0",
            "agents": agents,
            "summary": summarize_game_rows(games),
            "games": games,
            "metadata": {},
        }
    raise ValueError("Expected experiment_v1 object or legacy list of game rows")


def summarize_game_rows(games: list[Mapping[str, Any]]) -> dict[str, Any]:
    wins = [0, 0]
    draws = 0
    scores: list[list[float]] = [[], []]
    durations: list[float] = []
    invalid = [0, 0]
    differentials: list[float] = []
    for game in games:
        winner = game.get("winner")
        if winner in (0, 1):
            wins[int(winner)] += 1
        else:
            draws += 1
        rewards = game.get("rewards", [])
        if len(rewards) >= 2 and rewards[0] is not None and rewards[1] is not None:
            scores[0].append(float(rewards[0]))
            scores[1].append(float(rewards[1]))
            differentials.append(float(rewards[0]) - float(rewards[1]))
        if game.get("duration_s") is not None:
            durations.append(float(game["duration_s"]))
        for player, count in enumerate(game.get("validator_rejected_actions", [])[:2]):
            invalid[player] += int(count)
    count = len(games)
    return {
        "games": count,
        "completed_games": count,
        "wins": wins,
        "losses": [wins[1], wins[0]],
        "draws": draws,
        "win_rate": [wins[player] / count if count else None for player in (0, 1)],
        "average_score": [fmean(values) if values else None for values in scores],
        "average_score_differential_a_minus_b": (
            fmean(differentials) if differentials else None
        ),
        "total_duration_s": sum(durations),
        "games_per_second": count / sum(durations) if durations and sum(durations) else None,
        "validator_rejected_actions": invalid,
    }


def load_experiment(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    return normalize_experiment(read_json(path), name=path.stem)


def experiment_game_rows(experiment: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Create stable rows for UI tables/charts across both schemas."""
    rows: list[dict[str, Any]] = []
    for index, game in enumerate(experiment.get("games", [])):
        rewards = game.get("rewards", [None, None])
        rows.append(
            {
                "Game": index + 1,
                "Seed": game.get("seed"),
                "Player A": rewards[0] if len(rewards) > 0 else None,
                "Player B": rewards[1] if len(rewards) > 1 else None,
                "Differential": (
                    float(rewards[0]) - float(rewards[1])
                    if len(rewards) >= 2
                    and rewards[0] is not None
                    and rewards[1] is not None
                    else None
                ),
                "Winner": game.get("winner"),
                "Runtime (s)": game.get("duration_s"),
            }
        )
    return rows


def render_experiments_page(default_directory: Path) -> None:
    import streamlit as st

    st.markdown('<span class="kg-badge">Reproducible runs</span><span class="kg-badge">Control vs treatment</span>', unsafe_allow_html=True)
    st.subheader("Experiments")
    directory = Path(
        st.text_input("Results directory", value=str(default_directory), key="results_directory")
    ).expanduser()
    paths = discover_json_files(directory)
    experiments: list[tuple[Path, dict[str, Any]]] = []
    rejected: list[str] = []
    for path in paths:
        try:
            experiments.append((path, load_experiment(path)))
        except (OSError, ValueError):
            rejected.append(path.name)
    if not experiments:
        st.info("No experiment reports found. Run the simulator with `--experiment-json`.")
        return

    options = list(range(len(experiments)))
    selector_a, selector_b = st.columns(2)
    with selector_a:
        selected_a = st.selectbox(
            "Control",
            options,
            format_func=lambda index: experiments[index][0].name,
            key="experiment_a",
        )
    with selector_b:
        compare_options: list[int | None] = [None, *options]
        selected_b = st.selectbox(
            "Treatment (optional)",
            compare_options,
            format_func=lambda index: "None" if index is None else experiments[index][0].name,
            key="experiment_b",
        )

    control = experiments[selected_a][1]
    _render_experiment_summary(control, heading="Control")
    if selected_b is not None:
        treatment = experiments[selected_b][1]
        _render_comparison(control, treatment)
        _render_experiment_summary(treatment, heading="Treatment")
    if rejected:
        st.caption(f"Ignored non-experiment JSON: {', '.join(rejected)}")


def _render_experiment_summary(experiment: Mapping[str, Any], *, heading: str) -> None:
    import streamlit as st

    summary = experiment.get("summary", {})
    agents = list(experiment.get("agents", []))
    while len(agents) < 2:
        agents.append(f"Player {len(agents) + 1}")
    st.markdown(f"### {heading} · {experiment.get('name')}")
    st.caption(f"{agents[0]} vs {agents[1]} · schema {experiment.get('schema')}")
    cols = st.columns(6)
    cols[0].metric("Games", summary.get("games", 0))
    wins = summary.get("wins", [0, 0])
    cols[1].metric(f"{agents[0]} wins", wins[0] if len(wins) else 0)
    cols[2].metric(f"{agents[1]} wins", wins[1] if len(wins) > 1 else 0)
    cols[3].metric("Draws", summary.get("draws", 0))
    scores = summary.get("average_score", [None, None])
    cols[4].metric("Average A", _number(scores[0] if len(scores) else None))
    cols[5].metric("Average diff", _number(summary.get("average_score_differential_a_minus_b")))

    rows = experiment_game_rows(experiment)
    chart_col, table_col = st.columns([3, 2])
    with chart_col:
        st.caption("Score by game")
        st.line_chart(
            {
                agents[0]: [row["Player A"] for row in rows],
                agents[1]: [row["Player B"] for row in rows],
            },
            x_label="Game",
            y_label="Final score",
        )
    with table_col:
        st.caption("Runs")
        st.dataframe(rows, hide_index=True, use_container_width=True)
    with st.expander("Reproducibility metadata"):
        st.json(experiment.get("metadata", {}))


def _render_comparison(control: Mapping[str, Any], treatment: Mapping[str, Any]) -> None:
    import streamlit as st

    control_summary = control.get("summary", {})
    treatment_summary = treatment.get("summary", {})
    control_diff = control_summary.get("average_score_differential_a_minus_b")
    treatment_diff = treatment_summary.get("average_score_differential_a_minus_b")
    delta = (
        float(treatment_diff) - float(control_diff)
        if control_diff is not None and treatment_diff is not None
        else None
    )
    control_win = _first(control_summary.get("win_rate"))
    treatment_win = _first(treatment_summary.get("win_rate"))
    win_delta = (
        float(treatment_win) - float(control_win)
        if control_win is not None and treatment_win is not None
        else None
    )
    st.markdown("### Treatment effect")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric(
        "Treatment avg differential",
        _number(treatment_diff),
        delta=_number(delta),
    )
    col_b.metric(
        "Treatment Player A win rate",
        _percent_value(treatment_win),
        delta=_percent(win_delta),
    )
    col_c.metric(
        "Additional games",
        int(treatment_summary.get("games", 0)) - int(control_summary.get("games", 0)),
    )


def _first(value: Any) -> Any:
    return value[0] if isinstance(value, list) and value else None


def _number(value: Any) -> str:
    return "—" if value is None else f"{float(value):,.1f}"


def _percent(value: Any) -> str:
    return "—" if value is None else f"{float(value):+.1%}"


def _percent_value(value: Any) -> str:
    return "—" if value is None else f"{float(value):.1%}"
