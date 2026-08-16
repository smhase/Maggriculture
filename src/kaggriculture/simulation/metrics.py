"""Experiment-grade aggregation for batches of Kaggriculture games."""

from __future__ import annotations

from datetime import datetime, timezone
from statistics import fmean, pstdev
from typing import Any, Mapping, Sequence

from kaggriculture.env.official_env import GameResult
from kaggriculture.env.rules import DEFAULT_STARTING_MONEY
from kaggriculture.simulation.replay import git_commit


SCHEMA_VERSION = "experiment_v1"
_FAILURE_STATUSES = {"ERROR", "INVALID", "TIMEOUT"}


def game_record(result: GameResult) -> dict[str, Any]:
    """Convert one result into a stable JSON-safe experiment record."""
    rewards = [float(r) if r is not None else None for r in result.rewards]
    score_diff = None
    if len(rewards) >= 2 and rewards[0] is not None and rewards[1] is not None:
        score_diff = rewards[0] - rewards[1]
    return {
        "seed": result.seed,
        "agents": [
            {"name": name, "version": version}
            for name, version in zip(result.agent_names, result.agent_versions)
        ],
        "rewards": rewards,
        "score_differential_a_minus_b": score_diff,
        "profit": [
            reward - DEFAULT_STARTING_MONEY if reward is not None else None
            for reward in rewards
        ],
        "statuses": list(result.statuses),
        "winner": result.winner,
        "num_steps": result.num_steps,
        "duration_s": result.duration_s,
        "turns_per_second": (
            result.num_steps / result.duration_s if result.duration_s > 0 else None
        ),
        "validator_rejected_actions": list(result.invalid_action_counts),
        "end_state_resources": list(result.end_state_resources),
        "compact_replay_path": result.compact_replay_path,
        "full_replay_path": result.full_replay_path,
    }


def summarize_results(results: Sequence[GameResult]) -> dict[str, Any]:
    """Aggregate wins, scores, performance, and validator diagnostics."""
    if not results:
        raise ValueError("Cannot summarize an empty result set")

    wins = [0, 0]
    draws = 0
    failures = 0
    scores: list[list[float]] = [[], []]
    score_diffs: list[float] = []
    invalid = [0, 0]
    total_steps = 0
    total_duration = 0.0

    for result in results:
        failed = any(status in _FAILURE_STATUSES for status in result.statuses)
        failures += int(failed)
        if not failed:
            if result.winner in (0, 1):
                wins[result.winner] += 1
            else:
                draws += 1
        if len(result.rewards) >= 2:
            for player_id in (0, 1):
                reward = result.rewards[player_id]
                if reward is not None:
                    scores[player_id].append(float(reward))
            if result.rewards[0] is not None and result.rewards[1] is not None:
                score_diffs.append(float(result.rewards[0] - result.rewards[1]))
        for player_id, count in enumerate(result.invalid_action_counts[:2]):
            invalid[player_id] += int(count)
        total_steps += result.num_steps
        total_duration += result.duration_s

    completed = len(results) - failures

    def avg(values: Sequence[float]) -> float | None:
        return fmean(values) if values else None

    def deviation(values: Sequence[float]) -> float | None:
        return pstdev(values) if values else None

    return {
        "games": len(results),
        "completed_games": completed,
        "failed_games": failures,
        "wins": wins,
        "losses": [wins[1], wins[0]],
        "draws": draws,
        "win_rate": [
            wins[player_id] / completed if completed else None
            for player_id in (0, 1)
        ],
        "average_score": [avg(scores[0]), avg(scores[1])],
        "score_stddev": [deviation(scores[0]), deviation(scores[1])],
        "average_profit": [
            avg([score - DEFAULT_STARTING_MONEY for score in scores[0]]),
            avg([score - DEFAULT_STARTING_MONEY for score in scores[1]]),
        ],
        "average_score_differential_a_minus_b": avg(score_diffs),
        "score_differential_stddev": deviation(score_diffs),
        "validator_rejected_actions": invalid,
        "total_steps": total_steps,
        "total_duration_s": total_duration,
        "games_per_second": len(results) / total_duration if total_duration > 0 else None,
        "turns_per_second": total_steps / total_duration if total_duration > 0 else None,
    }


def build_experiment_report(
    results: Sequence[GameResult],
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a reproducible batch report with metadata and per-game records."""
    if not results:
        raise ValueError("Cannot build a report without game results")
    first = results[0]
    return {
        "schema": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "kaggle_environments_version": first.env_version,
        "config": dict(config),
        "agents": [
            {"name": name, "version": version}
            for name, version in zip(first.agent_names, first.agent_versions)
        ],
        "summary": summarize_results(results),
        "games": [game_record(result) for result in results],
    }
