"""Phase 13 market tracker tests."""

from __future__ import annotations

from kaggle_environments import make

from kaggriculture.env.observation import parse_observation
from kaggriculture.planning.market_model import MarketTracker, market_score_adjust


def test_market_tracker_sma_and_momentum():
    env = make("kaggriculture", configuration={"episodeSteps": 48, "seed": 3})
    env.reset(2)
    state = parse_observation(env.state[0].observation, episode_steps=48)
    tracker = MarketTracker(window=3)
    first = tracker.observe(state)
    assert "WHEAT" in first.prices
    assert first.momentum["WHEAT"] == 0.0
    second = tracker.observe(state)
    assert second.sma["WHEAT"] == first.prices["WHEAT"]
    assert second.supply_pressure["WHEAT"] > 0


def test_falling_market_penalizes_held_inventory():
    env = make("kaggriculture", configuration={"episodeSteps": 48, "seed": 3})
    env.reset(2)
    state = parse_observation(env.state[0].observation, episode_steps=48)
    tracker = MarketTracker(window=3)
    snap = tracker.observe(state)
    falling = type(snap)(
        prices=snap.prices,
        inventory=snap.inventory,
        sma={k: v + 10 for k, v in snap.prices.items()},
        momentum={k: -10.0 for k in snap.prices},
        supply_pressure={k: 1.5 for k in snap.prices},
    )
    # Give the player wheat in shed via a shallow copy of private if present
    if state.self_player.private is None:
        return
    from dataclasses import replace

    private = replace(state.self_player.private, shed={"WHEAT": 8})
    player = replace(state.self_player, private=private)
    held = replace(state, self_player=player)
    assert market_score_adjust(held, falling) < 0
