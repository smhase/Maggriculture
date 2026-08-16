"""Phase 14 compact teacher dataset tests."""

from __future__ import annotations

from kaggle_environments import make

from kaggriculture.agents import OfficialAgent, PlannerAgent
from kaggriculture.env.observation import parse_observation
from kaggriculture.env.official_env import KaggricultureEnv
from kaggriculture.learning import TeacherBuffer, compact_features


def test_compact_features_are_small():
    env = make("kaggriculture", configuration={"episodeSteps": 48, "seed": 1})
    env.reset(2)
    state = parse_observation(env.state[0].observation, episode_steps=48)
    feats = compact_features(state)
    assert "tiles" not in feats
    assert "money" in feats
    assert feats["n_plants"] == 0


def test_planner_writes_teacher_samples(tmp_path):
    agent = PlannerAgent(episode_steps=24, beam_width=3, depth=1)
    agent.teacher_buffer = TeacherBuffer()
    env = KaggricultureEnv(seed=4, episode_steps=24)
    result = env.play(agent, OfficialAgent("pass"))
    assert result.statuses == ["DONE", "DONE"]
    assert agent.teacher_buffer.samples
    sample = agent.teacher_buffer.samples[0]
    assert "features" in sample
    assert "chosen" in sample
    assert "candidates" in sample
    assert sample.get("outcome", {}).get("seat") == 0
    path = tmp_path / "teacher.jsonl"
    agent.teacher_buffer.dump_jsonl(path)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(agent.teacher_buffer.samples)
