"""Phase 3 legal-action tests."""

from __future__ import annotations

from kaggle_environments import make

from kaggriculture.env.actions import default_action, make_action
from kaggriculture.env.legal import (
    get_legal_actions,
    get_legal_market_orders,
    get_legal_unit_actions,
    hire_cost,
    is_action_legal,
    is_unit_action_legal,
)
from kaggriculture.env.observation import parse_observation


def _initial_state(seed: int = 1, steps: int = 48):
    env = make("kaggriculture", configuration={"episodeSteps": steps, "seed": seed})
    env.reset(2)
    return parse_observation(env.state[0].observation, episode_steps=steps), env


def test_pass_always_legal():
    state, _ = _initial_state()
    assert is_unit_action_legal(state, ["PASS"], 0)
    assert is_action_legal(state, default_action())


def test_initial_moves_and_builds():
    state, _ = _initial_state()
    legal = get_legal_unit_actions(state, 0)
    ops = {a[0] for a in legal}
    assert "PASS" in ops
    assert "NORTH" in ops or "SOUTH" in ops or "EAST" in ops or "WEST" in ops
    # Spawn is empty unlocked → can build
    assert ["BUILD_COOP"] in legal
    assert ["BUILD_PASTURE"] in legal


def test_buy_seed_legal_when_rich():
    state, _ = _initial_state()
    market = get_legal_market_orders(state)
    assert ["BUY_SEED", "WHEAT", 1] in market
    assert is_action_legal(
        state,
        make_action(farmer=["PASS"], market=[["BUY_SEED", "WHEAT", 1]]),
    )


def test_sell_illegal_when_shed_empty():
    state, _ = _initial_state()
    assert not is_action_legal(
        state,
        make_action(farmer=["PASS"], market=[["SELL", "WHEAT", 1]]),
    )


def test_water_on_empty_illegal():
    state, _ = _initial_state()
    assert not is_unit_action_legal(state, ["WATER"], 0)


def test_atomic_plant_overdemand_illegal(tmp_path=None):
    env = make("kaggriculture", configuration={"episodeSteps": 48, "seed": 2})
    # Buy one wheat, hire a hand, then both try to PLANT — illegal in our validator
    def buyer(obs):
        if obs.get("step", 0) == 0:
            return {
                "farmer": ["PASS"],
                "hands": [],
                "market": [["BUY_SEED", "WHEAT", 1], ["HIRE"]],
            }
        return default_action()

    env.run([buyer, "pass"])
    # After game start we need mid-episode state with 1 seed and a hand —
    # reconstruct via step loop instead.
    env2 = make("kaggriculture", configuration={"episodeSteps": 48, "seed": 3})
    env2.reset(2)
    env2.step(
        [
            {"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "WHEAT", 1], ["HIRE"]]},
            default_action(),
        ]
    )
    # Hand cannot act same turn as hire; step once more with PASS
    env2.step([default_action(), default_action()])
    state = parse_observation(env2.state[0].observation, episode_steps=48)
    assert state.self_player.private is not None
    assert state.self_player.private.seeds.get("WHEAT", 0) == 1
    # If a hand exists, dual plant is illegal
    if state.self_player.farm.hands:
        dual = make_action(
            farmer=["PLANT", "WHEAT"],
            hands=[["PLANT", "WHEAT"]],
            market=[],
        )
        assert not is_action_legal(state, dual)
        single = make_action(farmer=["PLANT", "WHEAT"], hands=[["PASS"]], market=[])
        # Only legal if standing on empty tile
        # May need to be on empty — check unit legal list
        if ["PLANT", "WHEAT"] in get_legal_unit_actions(state, 0):
            assert is_action_legal(state, single)


def test_get_legal_actions_shape():
    state, _ = _initial_state()
    bundle = get_legal_actions(state)
    assert "farmer" in bundle and "hands" in bundle and "market" in bundle
    assert isinstance(bundle["farmer"], list)
    assert ["PASS"] in bundle["farmer"]


def test_hire_cost_sequence():
    assert hire_cost(0) == 1
    assert hire_cost(1) == 1
    assert hire_cost(2) == 2
    assert hire_cost(4) == 5
