"""Macro-level planner agent (beam search + tactical scheduler)."""

from __future__ import annotations

from typing import Any, Optional

from kaggriculture.agents.base import Agent
from kaggriculture.env.observation import parse_observation
from kaggriculture.env.rules import DEFAULT_EPISODE_STEPS, DEFAULT_TURNS_PER_DAY
from kaggriculture.logging import get_logger
from kaggriculture.planning.beam_search import beam_search, choose_crop
from kaggriculture.planning.scheduler import schedule

logger = get_logger("agents.planner")


class PlannerAgent(Agent):
    """Conservative MPC-style agent over macros.

    Parameters match the roadmap defaults but stay small for Kaggle actTimeout.
    """

    name = "planner"
    version = "0.1.0"

    def __init__(
        self,
        *,
        episode_steps: int = DEFAULT_EPISODE_STEPS,
        turns_per_day: int = DEFAULT_TURNS_PER_DAY,
        beam_width: int = 6,
        depth: int = 3,
        preferred_crop: Optional[str] = None,
    ) -> None:
        self.episode_steps = int(episode_steps)
        self.turns_per_day = int(turns_per_day)
        self.beam_width = int(beam_width)
        self.depth = int(depth)
        self.preferred_crop = preferred_crop
        self.last_reasoning: str = ""

    def act(self, observation: Any, configuration: Optional[Any] = None) -> dict[str, Any]:
        episode_steps = self.episode_steps
        turns_per_day = self.turns_per_day
        if configuration is not None:
            if isinstance(configuration, dict):
                episode_steps = int(configuration.get("episodeSteps", episode_steps))
                turns_per_day = int(configuration.get("turnsPerDay", turns_per_day))
            else:
                episode_steps = int(getattr(configuration, "episodeSteps", episode_steps))
                turns_per_day = int(getattr(configuration, "turnsPerDay", turns_per_day))

        state = parse_observation(observation, episode_steps=episode_steps)
        crop = self.preferred_crop or choose_crop(state)
        result = beam_search(
            state,
            beam_width=self.beam_width,
            depth=self.depth,
            preferred_crop=crop,
            turns_per_day=turns_per_day,
        )
        self.last_reasoning = result.reasoning
        logger.debug("%s", result.reasoning)
        return schedule(state, result.best_macro, turns_per_day=turns_per_day)
