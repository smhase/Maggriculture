"""Macro-level planner agent (beam search + tactical scheduler)."""

from __future__ import annotations

from typing import Any, Optional

from kaggriculture.agents.base import Agent
from kaggriculture.agents.profiles import DEFAULT, StrategyProfile, resolve_swing
from kaggriculture.env.observation import parse_observation
from kaggriculture.env.rules import DEFAULT_EPISODE_STEPS, DEFAULT_TURNS_PER_DAY
from kaggriculture.logging import get_logger
from kaggriculture.planning.beam_search import beam_search, choose_crop
from kaggriculture.planning.market_model import MarketTracker
from kaggriculture.planning.opponent_model import profile_opponent
from kaggriculture.planning.macros import BuySeeds, ExpandFarm, LiquidateInventory, SellCommodity, macro_label
from kaggriculture.planning.scheduler import schedule

logger = get_logger("agents.planner")


class PlannerAgent(Agent):
    """Conservative MPC-style agent over macros.

    Parameters match the roadmap defaults but stay small for Kaggle actTimeout.
    """

    name = "planner"
    version = "0.2.0"

    def __init__(
        self,
        *,
        episode_steps: int = DEFAULT_EPISODE_STEPS,
        turns_per_day: int = DEFAULT_TURNS_PER_DAY,
        beam_width: int = 6,
        depth: int = 3,
        preferred_crop: Optional[str] = None,
        profile: Optional[StrategyProfile] = None,
    ) -> None:
        super().__init__()
        self.episode_steps = int(episode_steps)
        self.turns_per_day = int(turns_per_day)
        self.profile = profile or DEFAULT
        if profile is None:
            self.profile = StrategyProfile(
                name="planner",
                beam_width=int(beam_width),
                depth=int(depth),
            )
        self.beam_width = int(self.profile.beam_width)
        self.depth = int(self.profile.search_depth)
        self.preferred_crop = preferred_crop or self.profile.preferred_crop
        self.name = self.profile.name
        self.last_reasoning: str = ""
        self._opp_money: Optional[float] = None
        self._market = MarketTracker()
        self.teacher_buffer = None

    def begin_episode(self) -> None:
        super().begin_episode()
        self._opp_money = None
        self._market.reset()

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
        profile = resolve_swing(
            self.profile,
            self_money=float(state.self_player.farm.money),
            opponent_money=float(state.opponent.farm.money),
        )
        crop = self.preferred_crop or choose_crop(state)
        opp = profile_opponent(state, previous_money=self._opp_money)
        self._opp_money = opp.money
        market = self._market.observe(state)
        result = beam_search(
            state,
            beam_width=profile.beam_width,
            depth=profile.search_depth,
            preferred_crop=crop,
            turns_per_day=turns_per_day,
            profile=profile,
            opponent=opp,
            market=market,
        )
        self.last_trace = result.reasoning
        self.last_reasoning = result.reasoning.headline
        logger.debug("%s", result.reasoning.headline)
        if self.teacher_buffer is not None:
            from kaggriculture.learning.features import compact_features

            self.teacher_buffer.record(
                features=compact_features(state),
                candidates=[macro_label(n.macros[0]) for n in result.beam if n.macros],
                scores=[n.score for n in result.beam if n.macros],
                chosen=macro_label(result.best_macro),
                headline=result.reasoning.headline,
            )
        extra: list[list[Any]] = []
        action = schedule(
            state,
            result.best_macro,
            turns_per_day=turns_per_day,
            extra_market=extra or None,
        )
        if not isinstance(
            result.best_macro, (SellCommodity, LiquidateInventory, BuySeeds, ExpandFarm)
        ):
            if any((order or [None])[0] == "SELL" for order in action.get("market", [])):
                if self.last_trace is not None:
                    self.last_trace.causes.append("opportunistic_sell")
        return action
