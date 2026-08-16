"""Environment integration tests against official kaggle-environments."""

from __future__ import annotations

import json
from pathlib import Path

import kaggle_environments
import pytest
from kaggle_environments import make

from kaggriculture.agents import OfficialAgent, RandomLegalAgent
from kaggriculture.env.actions import default_action, make_action
from kaggriculture.env.observation import parse_observation
from kaggriculture.env.official_env import KaggricultureEnv
from kaggriculture.env.rules import (
    CROPS,
    DEFAULT_STARTING_MONEY,
    HINGE_GAIN,
    MARKET_I0,
    MARKET_PARAMS,
    PRODUCTS,
)


def test_official_env_imports():
    assert kaggle_environments.__version__ == "1.32.7"
    env = make("kaggriculture")
    assert env.configuration.episodeSteps == 720
    assert "kaggriculture" in str(env.name) or env.name == "kaggriculture"


def test_episode_length_full():
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 42})
    steps = env.run(["pass", "pass"])
    assert len(steps) == 720
    final = steps[-1]
    assert [s.status for s in final] == ["DONE", "DONE"]
    obs = final[0].observation
    assert obs.step == 719
    assert obs.day == 29
    assert obs.hour == 23
    # Day-29 EOD does not fire: farmer still at spawn, no day-29 weed refresh path needed
    assert final[0].reward == DEFAULT_STARTING_MONEY


def test_episode_length_short():
    env = make("kaggriculture", configuration={"episodeSteps": 48, "seed": 1})
    steps = env.run(["pass", "pass"])
    assert len(steps) == 48
    obs = steps[-1][0].observation
    assert obs.day == 1
    assert obs.hour == 23


def test_seed_determinism():
    def run_once(seed: int):
        env = make("kaggriculture", configuration={"episodeSteps": 72, "seed": seed})
        steps = env.run(["pass", "pass"])
        obs = steps[-1][0].observation
        shops = list(obs.town["unlocked_shops"] if isinstance(obs.town, dict) else obs.town.unlocked_shops)
        # Collect weed positions for player 0
        farm = obs.farms[0]
        tiles = farm["tiles"] if isinstance(farm, dict) else farm.tiles
        weeds = [
            (x, y)
            for y, row in enumerate(tiles)
            for x, t in enumerate(row)
            if isinstance(t, dict) and t.get("kind") == "WEED"
        ]
        prices = dict(
            obs.market["prices"] if isinstance(obs.market, dict) else obs.market.prices
        )
        return shops, weeds, prices, env.info.get("seed")

    a = run_once(123)
    b = run_once(123)
    assert a == b


def test_seed_differs():
    def shops(seed: int):
        env = make("kaggriculture", configuration={"episodeSteps": 120, "seed": seed})
        steps = env.run(["pass", "pass"])
        obs = steps[-1][0].observation
        return list(
            obs.town["unlocked_shops"] if isinstance(obs.town, dict) else obs.town.unlocked_shops
        )

    # Different seeds should (almost always) diverge on shop draws over several unlocks
    assert shops(1) != shops(999) or shops(2) != shops(1000)


def test_parse_observation_matches_raw():
    env = make("kaggriculture", configuration={"episodeSteps": 48, "seed": 7})
    env.reset(2)
    # After reset, state[0] has initial observation
    obs0 = env.state[0].observation
    state = parse_observation(obs0, episode_steps=48)

    assert state.player_id == 0
    assert state.self_player.farm.money == DEFAULT_STARTING_MONEY
    assert state.opponent.farm.money == DEFAULT_STARTING_MONEY
    assert list(state.self_player.farm.unlocked_quadrants) == ["NW"]
    assert state.opponent.private is None
    assert state.self_player.private is not None
    assert state.self_player.private.seeds.get("WHEAT", 0) == 0
    for p in PRODUCTS:
        assert state.market.inventory[p] == MARKET_I0
        assert state.market.prices[p] > 0
    assert state.turn == 0
    assert state.day == 0
    assert state.hour == 0


def test_parse_observation_player1_private():
    env = KaggricultureEnv(seed=3, episode_steps=24)
    env.reset()
    obs1 = env.state[1].observation
    state = parse_observation(obs1, episode_steps=24)
    assert state.player_id == 1
    assert state.self_player.private is not None
    assert state.opponent.player_id == 0
    assert state.opponent.private is None


def test_action_default_pass():
    a = default_action()
    assert a == {"farmer": ["PASS"], "hands": [], "market": []}
    b = make_action(farmer=["WATER"], market=[["BUY_SEED", "WHEAT", 1]])
    assert b["farmer"] == ["WATER"]
    assert b["market"] == [["BUY_SEED", "WHEAT", 1]]
    assert b["hands"] == []


def test_illegal_action_is_noop():
    env = make("kaggriculture", configuration={"episodeSteps": 24, "seed": 5}, debug=True)
    env.reset(2)
    # WATER on empty spawn tile should no-op without ERROR
    water = {"farmer": ["WATER"], "hands": [], "market": []}
    env.step([water, default_action()])
    assert env.state[0].status == "ACTIVE"
    assert env.state[1].status == "ACTIVE"
    farm = env.state[0].observation.farms[0]
    fx, fy = farm["farmer"] if isinstance(farm, dict) else farm.farmer
    tiles = farm["tiles"] if isinstance(farm, dict) else farm.tiles
    # Still empty or unchanged — not an error state
    assert tiles[fy][fx] is None or tiles[fy][fx] == "LOCKED" or isinstance(tiles[fy][fx], dict)


def test_play_records_replay(tmp_path: Path):
    replay = tmp_path / "game.compact.json"
    env = KaggricultureEnv(seed=11, episode_steps=48)
    result = env.play(
        OfficialAgent("pass"),
        OfficialAgent("starter"),
        replay_path=replay,
    )
    assert result.num_steps == 48
    assert result.statuses == ["DONE", "DONE"]
    assert replay.exists()
    data = json.loads(replay.read_text())
    assert data["meta"]["seed"] == 11
    assert data["meta"]["kaggle_environments_version"] == kaggle_environments.__version__
    assert data["meta"]["format"] == "compact_v2"
    assert len(data["initial_state"]["players"]) == 2
    assert len(data["turns"]) == 48
    assert "money" in data["turns"][0]
    assert "actions" in data["turns"][0]
    assert "market_prices" in data["turns"][0]


def test_random_legal_short_game(tmp_path: Path):
    env = KaggricultureEnv(seed=42, episode_steps=48)
    result = env.play(
        RandomLegalAgent(seed=1, episode_steps=48),
        OfficialAgent("pass"),
        replay_path=tmp_path / "rl.compact.json",
    )
    assert result.statuses == ["DONE", "DONE"]
    assert result.num_steps == 48
    assert all(r is not None for r in result.rewards)


def test_random_legal_full_game():
    env = KaggricultureEnv(seed=42, episode_steps=720)
    result = env.play(
        RandomLegalAgent(seed=0, episode_steps=720),
        OfficialAgent("starter"),
    )
    assert result.statuses == ["DONE", "DONE"]
    assert result.num_steps == 720
    assert result.rewards[0] is not None
    assert result.rewards[1] is not None


def test_wrapper_step_loop():
    env = KaggricultureEnv(seed=2, episode_steps=24)
    env.reset()
    n = 0
    while not env.done:
        env.step(default_action(), default_action())
        n += 1
        if n > 30:
            break
    assert env.done
    assert n == 23  # initial state + 23 steps → DONE at episodeSteps-related boundary
    # Verify DONE rewards present
    assert env.state[0].status == "DONE"


def test_crops_constants_from_official():
    assert set(CROPS) == {"WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"}
    assert CROPS["WHEAT"]["seed"] == 10
    assert HINGE_GAIN == 8.0
    assert MARKET_PARAMS["CARROT"]["below_func"] == "hinge"
    assert MARKET_PARAMS["TOMATO"]["below_func"] == "hinge"
    assert MARKET_PARAMS["EGG"]["below_func"] == "hinge"
