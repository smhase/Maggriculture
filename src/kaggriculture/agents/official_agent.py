"""Wrappers for built-in official Kaggriculture agents."""

from __future__ import annotations

from typing import Any, Optional

from kaggle_environments.envs.kaggriculture.kaggriculture import agents as OFFICIAL_AGENTS

from kaggriculture.agents.base import Agent


class OfficialAgent(Agent):
    """Delegate to a built-in agent: pass | random | starter."""

    def __init__(self, builtin: str = "starter") -> None:
        if builtin not in OFFICIAL_AGENTS:
            raise ValueError(
                f"Unknown built-in agent {builtin!r}; choose from {sorted(OFFICIAL_AGENTS)}"
            )
        self.builtin = builtin
        self.name = builtin
        self.version = "official"
        self._fn = OFFICIAL_AGENTS[builtin]

    def act(self, observation: Any, configuration: Optional[Any] = None) -> dict[str, Any]:
        # Official agents only take obs
        return self._fn(observation)

    def as_kaggle_fn(self) -> Any:
        # Prefer the string name so env.run uses the registered built-in.
        return self.builtin
