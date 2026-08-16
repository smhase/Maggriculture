"""Wrapper around the official kaggle-environments Kaggriculture engine."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

import kaggle_environments
from kaggle_environments import make

from kaggriculture.env.actions import default_action
from kaggriculture.env.legal import is_action_legal
from kaggriculture.env.observation import parse_observation
from kaggriculture.env.rules import DEFAULT_EPISODE_STEPS
from kaggriculture.logging import get_logger
from kaggriculture.simulation.replay import build_compact_replay, save_json

logger = get_logger("env")

AgentLike = Union[str, Callable[..., Any], Any]


@dataclass
class GameResult:
    """Outcome of one head-to-head episode."""

    rewards: list[Optional[float]]
    statuses: list[str]
    seed: Optional[int]
    episode_steps: int
    num_steps: int
    duration_s: float
    agent_names: list[str]
    agent_versions: list[str]
    winner: Optional[int]  # 0, 1, or None for tie / error
    invalid_action_counts: list[int] = field(default_factory=lambda: [0, 0])
    end_state_resources: list[dict[str, Any]] = field(default_factory=list)
    compact_replay_path: Optional[str] = None
    full_replay_path: Optional[str] = None
    env_version: str = field(default_factory=lambda: kaggle_environments.__version__)
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return all(s == "DONE" for s in self.statuses)


def _agent_name(agent: AgentLike) -> str:
    if isinstance(agent, str):
        return agent
    if hasattr(agent, "name"):
        return str(getattr(agent, "name"))
    if hasattr(agent, "__name__"):
        return str(agent.__name__)
    return type(agent).__name__


def _agent_version(agent: AgentLike) -> str:
    """Return a stable version label for experiment metadata."""
    if isinstance(agent, str):
        return f"official@{kaggle_environments.__version__}"
    version = getattr(agent, "version", None)
    if version == "official":
        return f"official@{kaggle_environments.__version__}"
    return str(version) if version is not None else "unversioned"


def _to_kaggle_agent(agent: AgentLike) -> AgentLike:
    """Convert our Agent objects to callables / built-in names for env.run."""
    if isinstance(agent, str):
        return agent
    if hasattr(agent, "as_kaggle_fn"):
        return agent.as_kaggle_fn()
    if callable(agent):
        return agent
    raise TypeError(f"Unsupported agent type: {type(agent)!r}")


class KaggricultureEnv:
    """Thin adapter over ``kaggle_environments.make('kaggriculture')``.

    The official interpreter remains the source of truth. This wrapper adds
    seeding helpers, structured results, and optional replay export.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        episode_steps: int = DEFAULT_EPISODE_STEPS,
        debug: bool = False,
        configuration: Optional[dict[str, Any]] = None,
        **config_overrides: Any,
    ) -> None:
        cfg: dict[str, Any] = {"episodeSteps": episode_steps}
        if seed is not None:
            cfg["seed"] = int(seed)
        if configuration:
            cfg.update(configuration)
        cfg.update(config_overrides)

        self._requested_seed = seed if seed is not None else cfg.get("seed")
        self._episode_steps = int(cfg.get("episodeSteps", DEFAULT_EPISODE_STEPS))
        self._debug = debug
        self._configuration = cfg
        self._env = make("kaggriculture", configuration=dict(cfg), debug=debug)
        self._last_result: Optional[GameResult] = None

    @property
    def raw(self) -> Any:
        """Underlying kaggle Environment instance."""
        return self._env

    @property
    def done(self) -> bool:
        return bool(self._env.done)

    @property
    def configuration(self) -> Any:
        return self._env.configuration

    @property
    def info(self) -> dict[str, Any]:
        return dict(self._env.info or {})

    @property
    def steps(self) -> list[Any]:
        return list(self._env.steps)

    @property
    def state(self) -> Any:
        return self._env.state

    @property
    def resolved_seed(self) -> Optional[int]:
        info_seed = self._env.info.get("seed") if self._env.info else None
        if info_seed is not None:
            return int(info_seed)
        return self._requested_seed if self._requested_seed is not None else None

    def reset(self, num_agents: int = 2) -> Any:
        """Reset the official environment."""
        return self._env.reset(num_agents)

    def step(
        self,
        action_a: Optional[dict[str, Any]] = None,
        action_b: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Advance one turn with explicit actions (debug / training loop)."""
        actions = [
            action_a if action_a is not None else default_action(),
            action_b if action_b is not None else default_action(),
        ]
        return self._env.step(actions)

    def play(
        self,
        agent_a: AgentLike,
        agent_b: AgentLike,
        *,
        replay_path: Optional[str | Path] = None,
        full_replay_path: Optional[str | Path] = None,
        reset: bool = True,
    ) -> GameResult:
        """Run a full head-to-head episode via official ``env.run``."""
        if reset:
            # Recreate env so configuration seed is applied fresh each game.
            self._env = make(
                "kaggriculture",
                configuration=dict(self._configuration),
                debug=self._debug,
            )

        names = [_agent_name(agent_a), _agent_name(agent_b)]
        versions = [_agent_version(agent_a), _agent_version(agent_b)]
        kaggle_agents = [_to_kaggle_agent(agent_a), _to_kaggle_agent(agent_b)]

        logger.info("Starting game %s vs %s seed=%s", names[0], names[1], self._requested_seed)
        t0 = time.perf_counter()
        steps = self._env.run(kaggle_agents)
        duration = time.perf_counter() - t0

        final = steps[-1]
        rewards = [getattr(s, "reward", None) for s in final]
        statuses = [str(getattr(s, "status", "")) for s in final]
        winner = _winner(rewards, statuses)
        seed = self.resolved_seed
        invalid_action_counts = _count_validator_rejections(
            steps, episode_steps=self._episode_steps
        )
        end_state_resources = _extract_end_state_resources(
            final, episode_steps=self._episode_steps
        )

        compact_path: Optional[str] = None
        full_path: Optional[str] = None

        if replay_path is not None:
            compact = build_compact_replay(
                steps=steps,
                logs=self._env.logs or [],
                seed=seed,
                agent_names=names,
                episode_steps=self._episode_steps,
                env_version=kaggle_environments.__version__,
                duration_s=duration,
                rewards=rewards,
                statuses=statuses,
            )
            compact_path = str(save_json(replay_path, compact.to_dict()))

        if full_replay_path is not None:
            full_path = str(save_json(full_replay_path, self._env.toJSON()))

        result = GameResult(
            rewards=rewards,
            statuses=statuses,
            seed=seed,
            episode_steps=self._episode_steps,
            num_steps=len(steps),
            duration_s=duration,
            agent_names=names,
            agent_versions=versions,
            winner=winner,
            invalid_action_counts=invalid_action_counts,
            end_state_resources=end_state_resources,
            compact_replay_path=compact_path,
            full_replay_path=full_path,
            info=dict(self._env.info or {}),
        )
        self._last_result = result
        logger.info(
            "Finished in %.2fs steps=%d rewards=%s statuses=%s",
            duration,
            len(steps),
            rewards,
            statuses,
        )
        return result


def _plain_action(action: Any) -> dict[str, Any]:
    """Normalize a recorded Kaggle action for the local validator."""
    if isinstance(action, dict):
        return dict(action)
    return {
        "farmer": list(getattr(action, "farmer", ["PASS"]) or ["PASS"]),
        "hands": [list(a) for a in (getattr(action, "hands", []) or [])],
        "market": [list(a) for a in (getattr(action, "market", []) or [])],
    }


def _count_validator_rejections(
    steps: Sequence[Any], *, episode_steps: int
) -> list[int]:
    """Count actions rejected by our debug validator for each player.

    Recorded actions on step ``i`` were selected from the observations on
    step ``i - 1``. The official engine still decides what actually happens;
    this metric only exposes likely silent no-ops or malformed actions.
    """
    counts = [0, 0]
    for i in range(1, len(steps)):
        previous = steps[i - 1]
        current = steps[i]
        for player_id in range(min(2, len(previous), len(current))):
            try:
                state = parse_observation(
                    previous[player_id].observation,
                    episode_steps=episode_steps,
                )
                action = _plain_action(getattr(current[player_id], "action", None))
                if not is_action_legal(state, action):
                    counts[player_id] += 1
            except (AttributeError, IndexError, TypeError, ValueError):
                # Metrics must never turn a completed official game into a failure.
                logger.debug(
                    "Could not audit action at step=%s player=%s", i, player_id,
                    exc_info=True,
                )
    return counts


def _extract_end_state_resources(
    final_states: Sequence[Any], *, episode_steps: int
) -> list[dict[str, Any]]:
    """Capture compact, player-private end-state resources for experiments."""
    resources: list[dict[str, Any]] = []
    for player_id, final_state in enumerate(final_states):
        try:
            state = parse_observation(
                final_state.observation,
                episode_steps=episode_steps,
            )
            farm = state.self_player.farm
            private = state.self_player.private
            tile_counts: dict[str, int] = {}
            for row in farm.tiles:
                for tile in row:
                    if tile is None:
                        kind = "EMPTY"
                    elif isinstance(tile, str):
                        kind = tile
                    else:
                        kind = str(tile.get("kind", "UNKNOWN"))
                        if kind == "PLANT" and tile.get("crop"):
                            kind = f"PLANT:{tile['crop']}"
                        elif tile.get("animal"):
                            kind = f"{kind}:{tile['animal']}"
                    tile_counts[kind] = tile_counts.get(kind, 0) + 1
            carried: dict[str, int] = {}
            if private is not None:
                for inventory in private.inventories:
                    for item, quantity in inventory.items():
                        carried[item] = carried.get(item, 0) + int(quantity)
            resources.append(
                {
                    "player_id": player_id,
                    "money": farm.money,
                    "shed": dict(private.shed) if private is not None else {},
                    "seeds": dict(private.seeds) if private is not None else {},
                    "carried": carried,
                    "tile_counts": tile_counts,
                    "unlocked_quadrants": list(farm.unlocked_quadrants),
                    "hands": len(farm.hands),
                }
            )
        except (AttributeError, IndexError, TypeError, ValueError):
            logger.debug(
                "Could not extract end state for player=%s", player_id,
                exc_info=True,
            )
            resources.append({"player_id": player_id})
    return resources


def _winner(
    rewards: Sequence[Optional[float]], statuses: Sequence[str]
) -> Optional[int]:
    if any(s in ("ERROR", "INVALID", "TIMEOUT") for s in statuses):
        return None
    if rewards[0] is None or rewards[1] is None:
        return None
    if rewards[0] > rewards[1]:
        return 0
    if rewards[1] > rewards[0]:
        return 1
    return None
