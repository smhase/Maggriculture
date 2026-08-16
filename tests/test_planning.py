"""Phase 6 / 11 planning tests."""

from __future__ import annotations

from kaggle_environments import make

from kaggriculture.agents import OfficialAgent, PlannerAgent
from kaggriculture.env.official_env import KaggricultureEnv
from kaggriculture.env.observation import parse_observation
from kaggriculture.planning import (
    apply_macro,
    beam_search,
    evaluate_breakdown,
    evaluate_state,
    propose_macros,
    schedule,
)
from kaggriculture.planning.macros import BuySeeds, Idle, PlantCrop, WaterUrgent


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
    breakdown = evaluate_breakdown(state)
    assert 2500 < breakdown.total < 4000  # ~3000 cash, small penalties possible
    assert breakdown.cash >= 2500
    assert breakdown.seed_value >= 0
    assert breakdown.infrastructure_value >= 0
    assert breakdown.future_cost >= 0
    assert breakdown.risk_penalty >= 0


def test_planner_act_stays_under_kaggle_timeout():
    import time

    from kaggriculture.env.rules import DEFAULT_ACT_TIMEOUT

    env = make("kaggriculture", configuration={"episodeSteps": 96, "seed": 1})
    env.reset(2)
    agent = PlannerAgent(episode_steps=96, beam_width=6, depth=3)
    started = time.perf_counter()
    action = agent.act(env.state[0].observation)
    elapsed = time.perf_counter() - started
    assert set(action) <= {"farmer", "hands", "market"}
    assert elapsed < float(DEFAULT_ACT_TIMEOUT)
    assert elapsed < 0.5


def test_beam_search_returns_macro():
    state = _state()
    result = beam_search(state, beam_width=4, depth=2, preferred_crop="WHEAT")
    assert result.best_macro is not None
    assert result.beam
    assert "base=" in result.reasoning.headline
    assert result.reasoning.plan
    assert result.reasoning.schema == "coc_v1"


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


def test_apply_macro_plants_and_waters():
    from kaggriculture.agents.navigation import scan_tiles
    from kaggriculture.env.tiles import is_empty, is_plant

    state = _state()
    empties = scan_tiles(state, lambda t, x, y: is_empty(t))
    assert empties
    stocked = apply_macro(state, BuySeeds(crop="WHEAT", quantity=1))
    planted = apply_macro(
        stocked, PlantCrop(crop="WHEAT", target=empties[0])
    )
    tile = planted.self_player.farm.tiles[empties[0][1]][empties[0][0]]
    assert is_plant(tile)
    watered = apply_macro(planted, WaterUrgent(target=empties[0]))
    tile = watered.self_player.farm.tiles[empties[0][1]][empties[0][0]]
    assert tile.get("watered_today") is True
    assert watered.hour == (planted.hour + 1) % 24


def test_beam_search_respects_tiny_time_budget():
    state = _state()
    result = beam_search(state, beam_width=4, depth=6, time_budget_s=0.001)
    assert result.best_macro is not None
    assert result.reasoning.plan
