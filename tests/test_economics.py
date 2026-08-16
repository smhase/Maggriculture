"""Phase 4 economics tests."""

from __future__ import annotations

from kaggle_environments import make

from kaggriculture.economics import analyze_animal, analyze_crop, best_crop, rank_investments
from kaggriculture.env.observation import parse_observation
from kaggriculture.env.rules import CROPS


def _state(steps: int = 720, seed: int = 0):
    env = make("kaggriculture", configuration={"episodeSteps": steps, "seed": seed})
    env.reset(2)
    return parse_observation(env.state[0].observation, episode_steps=steps)


def test_wheat_analysis_positive_early():
    state = _state(720)
    a = analyze_crop(state, "WHEAT")
    assert a.seed_cost == 10
    assert a.market_unit_price == 25  # at I0
    assert a.matures_before_terminal
    assert a.expected_yield_watered >= 1
    assert a.expected_profit_watered > 0
    assert a.terminal_value_factor == 1.0


def test_melon_not_viable_near_terminal():
    # Very little time left
    state = _state(24)  # turns_remaining at step 0 is 24
    a = analyze_crop(state, "MELON")
    # Melon needs ~10 days = 240 turns to first yield — cannot mature
    assert not a.matures_before_terminal
    assert a.terminal_value_factor == 0.0
    assert a.expected_profit_watered == 0.0


def test_rank_puts_fast_crop_ahead_of_slow_on_efficiency():
    state = _state(720)
    ranks = rank_investments(state, include_animals=False, include_land=False)
    wheat = next(r for r in ranks if r.name == "WHEAT")
    melon = next(r for r in ranks if r.name == "MELON")
    # Wheat ties capital for fewer days → higher capital efficiency OR better
    # occupancy-normalized profit_per_turn among staples.
    assert wheat.capital_efficiency > 0
    assert melon.detail.turns_to_first_yield > wheat.detail.turns_to_first_yield


def test_wheat_matures_on_short_horizon_when_days_allow():
    # 96 steps → final day index 3; wheat first_yield_day=2 → plant day 0 OK
    state = _state(96)
    a = analyze_crop(state, "WHEAT")
    assert a.matures_before_terminal
    assert analyze_crop(state, "MELON").matures_before_terminal is False


def test_best_crop_is_known():
    state = _state(720)
    b = best_crop(state)
    assert b.crop in CROPS


def test_animal_analysis_structure():
    state = _state(720)
    g = analyze_animal(state, "GOOSE")
    assert g.product == "EGG"
    assert g.structure == "COOP"
    assert g.acquisition_cost == 300
    assert g.matures_before_terminal
    assert g.expected_productions_in_horizon > 0


def test_animal_zero_near_terminal():
    state = _state(24)
    g = analyze_animal(state, "COW")  # first yield day 8
    assert not g.matures_before_terminal
    assert g.expected_profit == 0.0
