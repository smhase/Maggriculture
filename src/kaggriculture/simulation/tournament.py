"""Configured round-robin tournaments with persisted Elo and matchups."""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import kaggle_environments

from kaggriculture.env.official_env import GameResult, KaggricultureEnv
from kaggriculture.logging import configure_logging
from kaggriculture.simulation.metrics import game_record
from kaggriculture.simulation.population import (
    PopulationEntry,
    ScheduledGame,
    TournamentConfig,
    balanced_schedule,
    load_tournament_config,
)
from kaggriculture.simulation.replay import git_commit, save_json
from kaggriculture.simulation.runner import resolve_agent


SCHEMA_VERSION = "tournament_v1"
_FAILURE_STATUSES = {"ERROR", "INVALID", "TIMEOUT"}


def run_tournament(
    config: TournamentConfig,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run a configured population and optionally persist its complete report."""
    schedule = balanced_schedule(config)
    started = time.perf_counter()
    game_rows: list[dict[str, Any]] = []

    for scheduled in schedule:
        result = _play_scheduled_game(scheduled, config)
        row = game_record(result)
        row.update(
            {
                "index": scheduled.index,
                "pairing_index": scheduled.pairing_index,
                "repeat": scheduled.repeat,
                "players": [
                    _participant_record(scheduled.agent_a, result, 0),
                    _participant_record(scheduled.agent_b, result, 1),
                ],
                "winner_id": (
                    scheduled.agent_a.id if result.winner == 0
                    else scheduled.agent_b.id if result.winner == 1
                    else None
                ),
            }
        )
        game_rows.append(row)
        print(
            f"game={scheduled.index} seed={scheduled.seed} "
            f"{scheduled.agent_a.id} vs {scheduled.agent_b.id} "
            f"rewards={result.rewards} statuses={result.statuses} "
            f"winner={row['winner_id'] or 'draw'}"
        )

    report = build_tournament_report(
        config,
        game_rows,
        wall_duration_s=time.perf_counter() - started,
    )
    if output_path is not None:
        save_json(output_path, report)
    return report


def build_tournament_report(
    config: TournamentConfig,
    games: Sequence[Mapping[str, Any]],
    *,
    wall_duration_s: float,
) -> dict[str, Any]:
    """Aggregate ratings, participant samples, and every configured matchup."""
    ratings = {entry.id: config.initial_rating for entry in config.population}
    stats = {
        entry.id: {
            "id": entry.id,
            "label": entry.label,
            "agent": entry.agent,
            "rating": config.initial_rating,
            "games": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "failures": 0,
            "score_total": 0.0,
        }
        for entry in config.population
    }
    matchup_stats: dict[tuple[str, str], dict[str, Any]] = {}
    population_order = {entry.id: index for index, entry in enumerate(config.population)}
    for left_index, left in enumerate(config.population):
        for right in config.population[left_index + 1 :]:
            matchup_stats[(left.id, right.id)] = {
                "agent_a": left.id,
                "agent_b": right.id,
                "games": 0,
                "wins_a": 0,
                "wins_b": 0,
                "draws": 0,
                "failures": 0,
                "score_total_a": 0.0,
                "score_total_b": 0.0,
            }

    failed_games = 0
    for game in games:
        players = list(game.get("players", []))
        if len(players) < 2:
            continue
        ids = [str(players[0]["id"]), str(players[1]["id"])]
        statuses = list(game.get("statuses", []))
        failed = any(status in _FAILURE_STATUSES for status in statuses)
        left, right = sorted(ids, key=population_order.__getitem__)
        matchup = matchup_stats[(left, right)]
        if failed:
            failed_games += 1
            matchup["failures"] += 1
            stats[ids[0]]["failures"] += 1
            stats[ids[1]]["failures"] += 1
            continue

        rewards = list(game.get("rewards", [None, None]))
        if len(rewards) < 2 or rewards[0] is None or rewards[1] is None:
            failed_games += 1
            matchup["failures"] += 1
            stats[ids[0]]["failures"] += 1
            stats[ids[1]]["failures"] += 1
            continue

        winner_id = game.get("winner_id")
        outcome_a = 1.0 if winner_id == ids[0] else 0.0 if winner_id == ids[1] else 0.5
        expected_a = 1.0 / (1.0 + 10.0 ** ((ratings[ids[1]] - ratings[ids[0]]) / 400.0))
        change = config.k_factor * (outcome_a - expected_a)
        ratings[ids[0]] += change
        ratings[ids[1]] -= change

        for seat, participant_id in enumerate(ids):
            participant = stats[participant_id]
            participant["games"] += 1
            participant["score_total"] += float(rewards[seat])
            if winner_id == participant_id:
                participant["wins"] += 1
            elif winner_id is None:
                participant["draws"] += 1
            else:
                participant["losses"] += 1

        matchup["games"] += 1
        reward_by_id = {ids[0]: float(rewards[0]), ids[1]: float(rewards[1])}
        matchup["score_total_a"] += reward_by_id[left]
        matchup["score_total_b"] += reward_by_id[right]
        if winner_id is None:
            matchup["draws"] += 1
        elif winner_id == left:
            matchup["wins_a"] += 1
        else:
            matchup["wins_b"] += 1

    rating_rows: list[dict[str, Any]] = []
    for participant_id, participant in stats.items():
        participant["rating"] = round(ratings[participant_id], 3)
        games_played = int(participant["games"])
        participant["effective_win_rate"] = (
            (participant["wins"] + 0.5 * participant["draws"]) / games_played
            if games_played else None
        )
        participant["average_score"] = (
            participant.pop("score_total") / games_played if games_played else None
        )
        rating_rows.append(participant)
    rating_rows.sort(key=lambda row: (row["rating"], row["effective_win_rate"] or 0.0), reverse=True)
    for rank, row in enumerate(rating_rows, 1):
        row["rank"] = rank

    matchups: list[dict[str, Any]] = []
    for matchup in matchup_stats.values():
        completed = int(matchup["games"])
        matchup["effective_win_rate_a"] = (
            (matchup["wins_a"] + 0.5 * matchup["draws"]) / completed
            if completed else None
        )
        matchup["average_score_a"] = (
            matchup.pop("score_total_a") / completed if completed else None
        )
        matchup["average_score_b"] = (
            matchup.pop("score_total_b") / completed if completed else None
        )
        matchups.append(matchup)

    return {
        "schema": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "kaggle_environments_version": kaggle_environments.__version__,
        "config": config.to_dict(),
        "summary": {
            "agents": len(config.population),
            "planned_games": len(balanced_schedule(config)),
            "completed_games": len(games) - failed_games,
            "failed_games": failed_games,
            "wall_duration_s": wall_duration_s,
            "games_per_second": len(games) / wall_duration_s if wall_duration_s > 0 else None,
        },
        "ratings": rating_rows,
        "matchups": matchups,
        "matrix": _build_matrix(config.population, matchup_stats),
        "games": [dict(game) for game in games],
    }


def _play_scheduled_game(scheduled: ScheduledGame, config: TournamentConfig) -> GameResult:
    agent_a = resolve_agent(
        scheduled.agent_a.agent,
        seed=_participant_seed(scheduled.agent_a.id, scheduled.seed),
        episode_steps=config.episode_steps,
    )
    agent_b = resolve_agent(
        scheduled.agent_b.agent,
        seed=_participant_seed(scheduled.agent_b.id, scheduled.seed),
        episode_steps=config.episode_steps,
    )
    replay_path = None
    if config.replay_dir:
        replay_path = (
            Path(config.replay_dir)
            / f"{scheduled.index:04d}_{scheduled.agent_a.id}_vs_"
            f"{scheduled.agent_b.id}_seed{scheduled.seed}.compact.json"
        )
    env = KaggricultureEnv(
        seed=scheduled.seed,
        episode_steps=config.episode_steps,
        debug=config.debug,
    )
    return env.play(agent_a, agent_b, replay_path=replay_path)


def _participant_seed(participant_id: str, environment_seed: int) -> int:
    payload = f"{participant_id}\0{environment_seed}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _participant_record(
    entry: PopulationEntry,
    result: GameResult,
    seat: int,
) -> dict[str, Any]:
    return {
        "id": entry.id,
        "label": entry.label,
        "agent": entry.agent,
        "name": result.agent_names[seat],
        "version": result.agent_versions[seat],
        "seat": seat,
    }


def _build_matrix(
    population: Sequence[PopulationEntry],
    matchups: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    matrix: dict[str, dict[str, Any]] = defaultdict(dict)
    for entry in population:
        matrix[entry.id][entry.id] = None
    for (left, right), matchup in matchups.items():
        games = int(matchup["games"])
        draws = int(matchup["draws"])
        failures = int(matchup["failures"])
        matrix[left][right] = {
            "games": games,
            "wins": int(matchup["wins_a"]),
            "losses": int(matchup["wins_b"]),
            "draws": draws,
            "failures": failures,
            "effective_win_rate": ((matchup["wins_a"] + 0.5 * draws) / games if games else None),
        }
        matrix[right][left] = {
            "games": games,
            "wins": int(matchup["wins_b"]),
            "losses": int(matchup["wins_a"]),
            "draws": draws,
            "failures": failures,
            "effective_win_rate": ((matchup["wins_b"] + 0.5 * draws) / games if games else None),
        }
    return {entry.id: dict(matrix[entry.id]) for entry in population}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run a balanced Kaggriculture tournament")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    try:
        config = load_tournament_config(args.config)
        report = run_tournament(config, output_path=args.output)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    summary = report["summary"]
    print(
        f"tournament={config.name} agents={summary['agents']} "
        f"completed={summary['completed_games']} failed={summary['failed_games']} "
        f"output={args.output}"
    )
    return 2 if summary["failed_games"] else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
