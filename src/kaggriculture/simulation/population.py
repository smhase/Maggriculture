"""Configuration and balanced scheduling for tournament populations."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


CONFIG_SCHEMA = "tournament_config_v1"


@dataclass(frozen=True)
class PopulationEntry:
    """One named tournament participant and its runner agent specification."""

    id: str
    agent: str
    label: str

    @classmethod
    def from_value(cls, value: str | Mapping[str, Any]) -> "PopulationEntry":
        if isinstance(value, str):
            participant_id = value.strip()
            if not participant_id:
                raise ValueError("Population agent names cannot be empty")
            return cls(id=participant_id, agent=participant_id, label=participant_id)
        participant_id = str(value.get("id", "")).strip()
        agent = str(value.get("agent", value.get("spec", ""))).strip()
        if not participant_id or not agent:
            raise ValueError("Each population entry needs non-empty 'id' and 'agent'")
        return cls(
            id=participant_id,
            agent=agent,
            label=str(value.get("label", participant_id)).strip() or participant_id,
        )

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "agent": self.agent, "label": self.label}


@dataclass(frozen=True)
class TournamentConfig:
    """Validated, reproducible tournament configuration."""

    name: str
    population: tuple[PopulationEntry, ...]
    games_per_matchup: int = 2
    seed_start: int = 0
    episode_steps: int = 720
    initial_rating: float = 1500.0
    k_factor: float = 24.0
    replay_dir: str | None = None
    debug: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TournamentConfig":
        schema = data.get("schema", CONFIG_SCHEMA)
        if schema != CONFIG_SCHEMA:
            raise ValueError(f"Unsupported tournament config schema: {schema!r}")
        raw_population = data.get("population", data.get("agents", []))
        if not isinstance(raw_population, Sequence) or isinstance(raw_population, (str, bytes)):
            raise ValueError("'population' must be a list of agent entries")
        population = tuple(PopulationEntry.from_value(value) for value in raw_population)
        if len(population) < 2:
            raise ValueError("A tournament population needs at least two agents")
        ids = [entry.id for entry in population]
        if len(set(ids)) != len(ids):
            raise ValueError("Tournament participant ids must be unique")

        games_per_matchup = int(data.get("games_per_matchup", 2))
        if games_per_matchup < 2 or games_per_matchup % 2:
            raise ValueError("games_per_matchup must be an even integer of at least 2")
        episode_steps = int(data.get("episode_steps", 720))
        if episode_steps < 2:
            raise ValueError("episode_steps must be at least 2")
        k_factor = float(data.get("k_factor", 24.0))
        if k_factor <= 0:
            raise ValueError("k_factor must be positive")

        return cls(
            name=str(data.get("name", "tournament")).strip() or "tournament",
            population=population,
            games_per_matchup=games_per_matchup,
            seed_start=int(data.get("seed_start", 0)),
            episode_steps=episode_steps,
            initial_rating=float(data.get("initial_rating", 1500.0)),
            k_factor=k_factor,
            replay_dir=(str(data["replay_dir"]) if data.get("replay_dir") else None),
            debug=bool(data.get("debug", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONFIG_SCHEMA,
            "name": self.name,
            "population": [entry.to_dict() for entry in self.population],
            "games_per_matchup": self.games_per_matchup,
            "seed_start": self.seed_start,
            "episode_steps": self.episode_steps,
            "initial_rating": self.initial_rating,
            "k_factor": self.k_factor,
            "replay_dir": self.replay_dir,
            "debug": self.debug,
        }


@dataclass(frozen=True)
class ScheduledGame:
    """One game in a deterministic, seat-balanced round-robin schedule."""

    index: int
    pairing_index: int
    repeat: int
    seed: int
    agent_a: PopulationEntry
    agent_b: PopulationEntry


def load_tournament_config(path: str | Path) -> TournamentConfig:
    """Load and validate a YAML tournament config."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, Mapping):
        raise ValueError("Tournament config must be a YAML mapping")
    return TournamentConfig.from_mapping(data)


def balanced_schedule(config: TournamentConfig) -> list[ScheduledGame]:
    """Generate mirrored games for every unordered population pair.

    Each seed is used exactly twice for a pair: once in each seating. This
    controls for environment randomness while balancing first/second player.
    """
    games: list[ScheduledGame] = []
    next_seed = config.seed_start
    index = 0
    for pairing_index, (left, right) in enumerate(combinations(config.population, 2)):
        for repeat in range(config.games_per_matchup // 2):
            seed = next_seed
            next_seed += 1
            games.append(ScheduledGame(index, pairing_index, repeat, seed, left, right))
            index += 1
            games.append(ScheduledGame(index, pairing_index, repeat, seed, right, left))
            index += 1
    return games
