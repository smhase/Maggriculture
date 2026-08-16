"""CoC traces on every local agent and compact replay persistence."""

from __future__ import annotations

from kaggle_environments import make

from kaggriculture.agents import OfficialAgent, PlannerAgent
from kaggriculture.agents.profiles import LONG_TERM, SHORT_TERM
from kaggriculture.env.official_env import KaggricultureEnv
from kaggriculture.planning.trace import DecisionTrace, trace_headline, trace_target
from kaggriculture.simulation.runner import resolve_agent


def _obs(steps: int = 96, seed: int = 3):
    env = make("kaggriculture", configuration={"episodeSteps": steps, "seed": seed})
    env.reset(2)
    return env.state[0].observation


def test_every_resolve_agent_emits_trace():
    obs = _obs()
    names = [
        "pass",
        "starter",
        "random_legal",
        "heuristic",
        "scripted",
        "planner",
        "long_term",
        "short_term",
        "solo",
        "risk_taker",
        "safe",
        "swing",
        "always_win",
        "never_lose",
        "crew",
    ]
    for name in names:
        agent = resolve_agent(name, seed=1, episode_steps=96)
        agent.begin_episode()
        action = agent.act(obs)
        agent.commit_trace()
        assert set(action) <= {"farmer", "hands", "market"}
        assert agent.last_trace is not None
        headline = trace_headline(agent.last_trace)
        assert headline
        dumped = agent.dumped_traces()
        assert dumped


def test_replay_stores_reasoning_both_seats(tmp_path):
    compact_path = tmp_path / "trace.compact.json"
    env = KaggricultureEnv(seed=5, episode_steps=48)
    env.play(
        PlannerAgent(episode_steps=48, beam_width=3, depth=2),
        OfficialAgent("pass"),
        replay_path=compact_path,
    )
    from kaggriculture.simulation.replay import load_compact_replay

    replay = load_compact_replay(compact_path)
    assert replay["meta"].get("reasoning_schema") == "coc_v1"
    later = [turn for turn in replay["turns"][1:] if turn.get("reasoning")]
    assert later
    seat_a, seat_b = later[0]["reasoning"]
    assert seat_a is not None
    assert seat_b is not None
    assert "base=" in (seat_a.get("headline") or "")
    assert "opaque_builtin" in (seat_b.get("causes") or [])


def test_trace_headline_helpers():
    trace = DecisionTrace(
        source="planner",
        headline="water first",
        target=[1, 2],
        unit="farmer",
    )
    assert trace_headline(trace) == "water first"
    assert trace_target(trace) == (1, 2)
    crew = [trace.to_dict(), {"headline": "hand water", "unit": "hand:0"}]
    assert "water first" in trace_headline(crew)


def test_long_term_searches_deeper_than_short_term():
    assert LONG_TERM.search_depth > SHORT_TERM.search_depth
    assert LONG_TERM.discount > SHORT_TERM.discount
