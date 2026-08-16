"""Base agent protocol compatible with kaggle-environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from kaggriculture.planning.trace import DecisionTrace, dump_trace, missing_trace


class Agent(ABC):
    """Research agent that can also be exported as a Kaggle callable."""

    name: str = "agent"
    version: str = "0.0.0"

    def __init__(self) -> None:
        self.last_trace: Optional[DecisionTrace] = None
        self.decision_traces: list[Any] = []

    def begin_episode(self) -> None:
        self.decision_traces = []
        self.last_trace = None

    def commit_trace(self) -> None:
        """Record last_trace (or a fallback) after act()."""
        trace = self.last_trace
        if trace is None:
            trace = missing_trace(self.name)
            self.last_trace = trace
        if isinstance(trace, DecisionTrace):
            self.decision_traces.append(trace)
        else:
            self.decision_traces.append(trace)

    @abstractmethod
    def act(self, observation: Any, configuration: Optional[Any] = None) -> dict[str, Any]:
        """Return an official Kaggriculture action dict."""

    def as_kaggle_fn(self) -> Callable[..., dict[str, Any]]:
        """Return a function ``(obs, config=None) -> action`` for ``env.run``."""

        def _fn(observation: Any, configuration: Optional[Any] = None) -> dict[str, Any]:
            action = self.act(observation, configuration)
            self.commit_trace()
            return action

        _fn.__name__ = self.name.replace(" ", "_")
        return _fn

    def __call__(self, observation: Any, configuration: Optional[Any] = None) -> dict[str, Any]:
        action = self.act(observation, configuration)
        self.commit_trace()
        return action

    def dumped_traces(self) -> list[Any]:
        return [dump_trace(item) for item in self.decision_traces]
