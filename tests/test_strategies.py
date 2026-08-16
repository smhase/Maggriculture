"""Strategy profiles, swing switching, solo hire ban, crew identities."""

from __future__ import annotations

from kaggle_environments import make

from kaggriculture.agents import CrewAgent
from kaggriculture.agents.crew import HAND_SPECIALTIES, _specialty_op
from kaggriculture.agents.planner_agent import PlannerAgent
from kaggriculture.agents.profiles import (
    ALWAYS_WIN,
    LONG_TERM,
    NEVER_LOSE,
    SHORT_TERM,
    SOLO,
    SWING,
    resolve_swing,
)
from kaggriculture.env.legal import is_action_legal
from kaggriculture.env.observation import parse_observation
from kaggriculture.env.official_env import KaggricultureEnv
from kaggriculture.simulation.runner import resolve_agent


def _obs(steps: int = 96, seed: int = 4):
    env = make("kaggriculture", configuration={"episodeSteps": steps, "seed": seed})
    env.reset(2)
    return env.state[0].observation


def test_solo_never_hires_short_game():
    env = KaggricultureEnv(seed=9, episode_steps=48)
    result = env.play(
        resolve_agent("solo", episode_steps=48),
        resolve_agent("pass", episode_steps=48),
        replay_path=None,
    )
    assert result.statuses == ["DONE", "DONE"]
    for step in env.steps[1:]:
        action = step[0].action
        if action is None:
            continue
        if isinstance(action, dict):
            market = action.get("market") or []
        else:
            market = getattr(action, "market", []) or []
        for order in market:
            op = order[0] if order else None
            assert op != "HIRE"


def test_swing_switches_with_scoreboard():
    behind = resolve_swing(SWING, self_money=1000, opponent_money=4000)
    ahead = resolve_swing(SWING, self_money=5000, opponent_money=3000)
    assert behind.risk == "aggressive"
    assert behind.objective == "win"
    assert ahead.risk == "safe"
    assert ahead.objective == "never_lose"


def test_named_profiles_exist():
    assert LONG_TERM.search_depth > SHORT_TERM.search_depth
    assert SOLO.hire == "never"
    assert ALWAYS_WIN.objective == "win"
    assert NEVER_LOSE.objective == "never_lose"


def test_crew_action_is_official_shaped_and_legal():
    obs = _obs()
    state = parse_observation(obs, episode_steps=96)
    agent = CrewAgent(episode_steps=96)
    action = agent.act(obs)
    assert set(action.keys()) == {"farmer", "hands", "market"}
    assert len(action["hands"]) == len(state.self_player.farm.hands)
    assert is_action_legal(state, action)
    traces = agent.last_trace
    assert isinstance(traces, list)
    assert traces[0].unit == "farmer"


def test_crew_specialties_differ_when_work_exists():
    obs = _obs()
    state = parse_observation(obs, episode_steps=96)
    claimed: set[tuple[int, int]] = set()
    ops = []
    for identity in HAND_SPECIALTIES[:2]:
        op, cause, _target = _specialty_op(
            state,
            pos=state.self_player.farm.farmer,
            identity=identity,
            claimed=claimed,
            crop="WHEAT",
            allow_plant=True,
        )
        ops.append((identity, op[0], cause))
    assert ops[0][0] != ops[1][0]


def test_planner_profile_name_on_long_term():
    agent = PlannerAgent(episode_steps=96, profile=LONG_TERM)
    assert agent.name == "long_term"
    assert agent.depth >= 4
