"""CLI and helpers for running local Kaggriculture matches.

Example::

    python -m kaggriculture.simulation.runner \\
        --agent-a random_legal --agent-b starter \\
        --games 3 --seed-start 42 \\
        --replay-dir experiments/replays
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

from kaggriculture.agents import (
    CrewAgent,
    MinimalEconomicAgent,
    OfficialAgent,
    PlannerAgent,
    RandomLegalAgent,
    ScriptedAgent,
)
from kaggriculture.agents.profiles import NAMED_PROFILES
from kaggriculture.env.official_env import GameResult, KaggricultureEnv
from kaggriculture.logging import configure_logging, get_logger
from kaggriculture.simulation.metrics import build_experiment_report, summarize_results
from kaggriculture.simulation.replay import save_json

logger = get_logger("simulation.runner")

_SUPPORTED = (
    "pass, random, starter, random_legal, heuristic, scripted, planner, "
    "long_term, short_term, solo, risk_taker, safe, swing, always_win, never_lose, crew"
)


def resolve_agent(spec: str, *, seed: int = 0, episode_steps: int = 720) -> Any:
    """Map CLI agent names to agent instances / built-in names."""
    key = spec.strip().lower()
    if key in ("pass", "random", "starter"):
        return OfficialAgent(key)
    if key in ("random_legal", "random-legal", "legal_random"):
        return RandomLegalAgent(seed=seed, episode_steps=episode_steps)
    if key in ("heuristic", "minimal", "minimal_economic", "economic"):
        return MinimalEconomicAgent(episode_steps=episode_steps)
    if key in ("scripted", "scripted_v1", "script"):
        return ScriptedAgent(episode_steps=episode_steps)
    if key in ("planner", "planner_v1", "beam"):
        return PlannerAgent(episode_steps=episode_steps)
    if key == "crew":
        return CrewAgent(episode_steps=episode_steps)
    if key in NAMED_PROFILES:
        profile = NAMED_PROFILES[key]
        if profile.hire == "crew":
            return CrewAgent(episode_steps=episode_steps, profile=profile)
        return PlannerAgent(episode_steps=episode_steps, profile=profile)
    raise ValueError(f"Unknown agent {spec!r}. Supported: {_SUPPORTED}")


def play_games(
    agent_a_spec: str,
    agent_b_spec: str,
    *,
    games: int = 1,
    seed_start: int = 0,
    episode_steps: int = 720,
    replay_dir: Optional[Path] = None,
    full_replay: bool = False,
    debug: bool = False,
) -> list[GameResult]:
    """Run ``games`` head-to-head matches with incremental seeds."""
    results: list[GameResult] = []
    for i in range(games):
        seed = seed_start + i
        agent_a = resolve_agent(agent_a_spec, seed=seed, episode_steps=episode_steps)
        agent_b = resolve_agent(agent_b_spec, seed=seed + 10_000, episode_steps=episode_steps)
        # Reset RNG for random_legal each game (resolve creates fresh instances)

        env = KaggricultureEnv(seed=seed, episode_steps=episode_steps, debug=debug)
        replay_path = None
        full_path = None
        if replay_dir is not None:
            replay_dir.mkdir(parents=True, exist_ok=True)
            tag = f"{agent_a_spec}_vs_{agent_b_spec}_seed{seed}"
            replay_path = replay_dir / f"{tag}.compact.json"
            if full_replay:
                full_path = replay_dir / f"{tag}.full.json"

        result = env.play(
            agent_a,
            agent_b,
            replay_path=replay_path,
            full_replay_path=full_path,
        )
        results.append(result)
        print(
            f"game={i} seed={seed} {result.agent_names[0]} vs {result.agent_names[1]} "
            f"rewards={result.rewards} statuses={result.statuses} "
            f"steps={result.num_steps} time={result.duration_s:.2f}s "
            f"winner={result.winner}"
        )
    return results


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run local Kaggriculture matches")
    parser.add_argument("--agent-a", default="random_legal")
    parser.add_argument("--agent-b", default="starter")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--episode-steps", type=int, default=720)
    parser.add_argument("--replay-dir", type=Path, default=None)
    parser.add_argument("--full-replay", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--log-level",
        default="WARNING",
        help="ERROR|WARNING|INFO|DEBUG|TRACE",
    )
    parser.add_argument(
        "--experiment-json",
        type=Path,
        default=None,
        help="Write reproducible metadata, aggregate metrics, and game records",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Deprecated alias for --experiment-json",
    )
    args = parser.parse_args(argv)

    configure_logging(args.log_level)
    results = play_games(
        args.agent_a,
        args.agent_b,
        games=args.games,
        seed_start=args.seed_start,
        episode_steps=args.episode_steps,
        replay_dir=args.replay_dir,
        full_replay=args.full_replay,
        debug=args.debug,
    )

    summary = summarize_results(results)
    print(
        "summary "
        f"games={summary['games']} wins={summary['wins']} draws={summary['draws']} "
        f"avg_scores={summary['average_score']} "
        f"avg_diff={summary['average_score_differential_a_minus_b']} "
        f"games_per_second={summary['games_per_second']:.3f} "
        f"validator_rejections={summary['validator_rejected_actions']}"
    )

    output_path = args.experiment_json or args.summary_json
    if args.experiment_json is not None and args.summary_json is not None:
        parser.error("Use only one of --experiment-json or --summary-json")
    if output_path is not None:
        report = build_experiment_report(
            results,
            config={
                "agent_a": args.agent_a,
                "agent_b": args.agent_b,
                "games": args.games,
                "seed_start": args.seed_start,
                "episode_steps": args.episode_steps,
                "replay_dir": str(args.replay_dir) if args.replay_dir else None,
                "full_replay": args.full_replay,
                "debug": args.debug,
            },
        )
        save_json(output_path, report)
        print(f"Wrote experiment report to {output_path}")

    # Non-zero if any game errored
    if any(s in ("ERROR", "INVALID", "TIMEOUT") for r in results for s in r.statuses):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
