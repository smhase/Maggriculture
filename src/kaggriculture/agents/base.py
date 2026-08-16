"""Base agent protocol compatible with kaggle-environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional


class Agent(ABC):
    """Research agent that can also be exported as a Kaggle callable."""

    name: str = "agent"
    version: str = "0.0.0"

    @abstractmethod
    def act(self, observation: Any, configuration: Optional[Any] = None) -> dict[str, Any]:
        """Return an official Kaggriculture action dict."""

    def as_kaggle_fn(self) -> Callable[..., dict[str, Any]]:
        """Return a function ``(obs, config=None) -> action`` for ``env.run``."""

        def _fn(observation: Any, configuration: Optional[Any] = None) -> dict[str, Any]:
            return self.act(observation, configuration)

        _fn.__name__ = self.name.replace(" ", "_")
        return _fn

    def __call__(self, observation: Any, configuration: Optional[Any] = None) -> dict[str, Any]:
        return self.act(observation, configuration)
