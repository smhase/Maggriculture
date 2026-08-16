"""Seeded random agent that prefers conservative legal-ish actions."""

from __future__ import annotations

import random
from typing import Any, Optional

from kaggriculture.agents.base import Agent
from kaggriculture.agents.legal import farmer_candidates, market_candidates
from kaggriculture.env.actions import make_action
from kaggriculture.env.observation import parse_observation
from kaggriculture.env.rules import DEFAULT_EPISODE_STEPS
from kaggriculture.planning.trace import DecisionTrace, describe_action


class RandomLegalAgent(Agent):
    """Choose uniformly among conservative candidate actions.

    Unlike the official ``random`` agent, this uses an explicit RNG seed so
    experiments are reproducible when the episode seed is also fixed.
    """

    name = "random_legal"
    version = "0.1.0"

    def __init__(self, seed: int = 0, *, episode_steps: int = DEFAULT_EPISODE_STEPS) -> None:
        super().__init__()
        self.seed = int(seed)
        self.episode_steps = int(episode_steps)
        self._rng = random.Random(self.seed)

    def reset_rng(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.seed = int(seed)
        self._rng = random.Random(self.seed)

    def act(self, observation: Any, configuration: Optional[Any] = None) -> dict[str, Any]:
        episode_steps = self.episode_steps
        if configuration is not None:
            if isinstance(configuration, dict):
                episode_steps = int(configuration.get("episodeSteps", episode_steps))
            else:
                episode_steps = int(getattr(configuration, "episodeSteps", episode_steps))

        state = parse_observation(observation, episode_steps=episode_steps)
        farmer_ops = farmer_candidates(state, unit_index=0)
        farmer = list(self._rng.choice(farmer_ops))

        hands: list[list[Any]] = []
        for i in range(len(state.self_player.farm.hands)):
            hand_ops = farmer_candidates(state, unit_index=i + 1)
            hands.append(list(self._rng.choice(hand_ops)))

        market: list[list[Any]] = []
        # Occasional market action (~20%)
        if self._rng.random() < 0.2:
            m_opts = market_candidates(state)
            if m_opts:
                market.append(list(self._rng.choice(m_opts)))

        action = make_action(farmer=farmer, hands=hands, market=market)
        self.last_trace = DecisionTrace(
            source=self.name,
            headline=f"sampled_legal {describe_action(action)}",
            causes=["sampled_legal", f"farmer_candidates={len(farmer_ops)}"],
            macro=str(farmer[0] if farmer else "PASS"),
        )
        return action
