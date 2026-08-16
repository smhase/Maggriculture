"""Phase 12 opponent profiling tests."""

from __future__ import annotations

from kaggle_environments import make

from kaggriculture.env.observation import parse_observation
from kaggriculture.planning.opponent_model import opponent_score_adjust, profile_opponent


def test_profile_opponent_start_of_game():
    env = make("kaggriculture", configuration={"episodeSteps": 96, "seed": 2})
    env.reset(2)
    state = parse_observation(env.state[0].observation, episode_steps=96)
    profile = profile_opponent(state)
    assert profile.money >= 2500
    assert profile.expansion_stage == 0
    assert profile.n_hands == 0
    assert profile.money_delta is None
    again = profile_opponent(state, previous_money=profile.money)
    assert again.money_delta == 0
    assert isinstance(profile.crop_counts, dict)


def test_behind_opponent_adjusts_score_down():
    env = make("kaggriculture", configuration={"episodeSteps": 96, "seed": 2})
    env.reset(2)
    state = parse_observation(env.state[0].observation, episode_steps=96)
    rich = profile_opponent(state)
    # Force a richer opponent snapshot
    from dataclasses import replace

    rich = replace(rich, money=float(state.self_player.farm.money) + 2000)
    assert opponent_score_adjust(state, rich) < 0
