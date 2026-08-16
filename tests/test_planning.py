"""Phase 6 / 11 planning tests."""

from __future__ import annotations

from kaggle_environments import make

from kaggriculture.agents import OfficialAgent, PlannerAgent
from kaggriculture.env.official_env import KaggricultureEnv
from kaggriculture.env.observation import parse_observation
from kaggriculture.planning import (
    beam_search,
    evaluate_state,
    propose_macros,
    schedule,
)
from kaggriculture.planning.macros import Idle, PlantCrop, WaterUrgent


def _state(steps: int = 720, seed: int = 1):
    env = make("kaggriculture", configuration={"episodeSteps": steps, "seed": seed})
    env.reset(2)
    return parse_observation(env.state[0].observation, episode_steps=steps)


def test_propose_macros_includes_idle_and_buy_or_plant():
    state = _state()
    macros = propose_macros(state, preferred_crop="WHEAT")
    kinds = {m.kind.value for m in macros}
    assert "idle" in kinds
    assert "buy_seeds" in kinds or "plant_crop" in kinds or "liquidate_inventory" in kinds


def test_schedule_idle_is_pass():
    state = _state()
    action = schedule(state, Idle())
    assert action["farmer"] == ["PASS"]
    assert isinstance(action["market"], list)


def test_schedule_buy_seeds():
    state = _state()
    from kaggriculture.planning.macros import BuySeeds

    action = schedule(state, BuySeeds(crop="WHEAT", quantity=2))
    assert any(o[0] == "BUY_SEED" for o in action["market"])


def test_evaluator_starts_near_cash():
    state = _state()
    v = evaluate_state(state)
    assert 2500 < v < 4000  # ~3000 cash, small penalties possible


def test_beam_search_returns_macro():
    state = _state()
    result = beam_search(state, beam_width=4, depth=2, preferred_crop="WHEAT")
    assert result.best_macro is not None
    assert result.beam
    assert "base=" in result.reasoning


def test_planner_short_game():
    env = KaggricultureEnv(seed=7, episode_steps=96)
    result = env.play(PlannerAgent(episode_steps=96, beam_width=4, depth=2), OfficialAgent("pass"))
    assert result.statuses == ["DONE", "DONE"]
    # Short horizon is harsh; require no crash and not a total wipeout
    assert result.rewards[0] is not None
    assert result.rewards[0] >= 2800


def test_planner_beats_starter_full():
    env = KaggricultureEnv(seed=42, episode_steps=720)
    result = env.play(
        PlannerAgent(episode_steps=720, beam_width=6, depth=3),
        OfficialAgent("starter"),
    )
    assert result.statuses == ["DONE", "DONE"]
    assert result.rewards[0] > result.rewards[1]
