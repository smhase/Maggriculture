"""Wrappers for built-in official Kaggriculture agents."""

from __future__ import annotations

from typing import Any, Optional

from kaggle_environments.envs.kaggriculture.kaggriculture import agents as OFFICIAL_AGENTS

from kaggriculture.agents.base import Agent
from kaggriculture.planning.trace import DecisionTrace, describe_action


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
        super().__init__()
        self.name = builtin

    def act(self, observation: Any, configuration: Optional[Any] = None) -> dict[str, Any]:
        raw = self._fn(observation)
        action = _plain_action(raw)
        self.last_trace = DecisionTrace(
            source=f"official:{self.builtin}",
            headline=f"official {self.builtin} emitted {describe_action(action)}",
            causes=["opaque_builtin"],
            macro=describe_action(action),
        )
        return action

    def as_kaggle_fn(self) -> Any:
        # Wrap the builtin so research replays still get CoC traces.
        return Agent.as_kaggle_fn(self)


def _plain_action(action: Any) -> dict[str, Any]:
    if isinstance(action, dict):
        return {
            "farmer": list(action.get("farmer") or ["PASS"]),
            "hands": [list(h) for h in (action.get("hands") or [])],
            "market": [list(m) for m in (action.get("market") or [])],
        }
    return {
        "farmer": list(getattr(action, "farmer", ["PASS"]) or ["PASS"]),
        "hands": [list(h) for h in (getattr(action, "hands", []) or [])],
        "market": [list(m) for m in (getattr(action, "market", []) or [])],
    }
