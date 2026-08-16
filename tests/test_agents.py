"""Phase 5 baseline agent tests."""

from __future__ import annotations

from pathlib import Path

from kaggriculture.agents import MinimalEconomicAgent, OfficialAgent, ScriptedAgent
from kaggriculture.env.official_env import KaggricultureEnv


def test_heuristic_short_game_completes():
    env = KaggricultureEnv(seed=10, episode_steps=96)
    result = env.play(MinimalEconomicAgent(episode_steps=96), OfficialAgent("pass"))
    assert result.statuses == ["DONE", "DONE"]
    assert result.rewards[0] is not None
    # Should make some profit against a pure pass bot
    assert result.rewards[0] > 3000


def test_scripted_short_game_completes():
    env = KaggricultureEnv(seed=11, episode_steps=96)
    result = env.play(ScriptedAgent(episode_steps=96), OfficialAgent("pass"))
    assert result.statuses == ["DONE", "DONE"]
    assert result.rewards[0] > 3000


def test_scripted_beats_starter_full(tmp_path: Path):
    env = KaggricultureEnv(seed=42, episode_steps=720)
    result = env.play(
        ScriptedAgent(episode_steps=720),
        OfficialAgent("starter"),
        replay_path=tmp_path / "scripted_vs_starter.compact.json",
    )
    assert result.statuses == ["DONE", "DONE"]
    assert result.num_steps == 720
    # Scripted wheat loop should beat the built-in carrot starter
    assert result.rewards[0] > result.rewards[1]


def test_heuristic_vs_starter_full():
    env = KaggricultureEnv(seed=42, episode_steps=720)
    result = env.play(
        MinimalEconomicAgent(episode_steps=720),
        OfficialAgent("starter"),
    )
    assert result.statuses == ["DONE", "DONE"]
    assert result.rewards[0] > result.rewards[1]
