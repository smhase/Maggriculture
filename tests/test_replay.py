"""Phase 8 compact replay reconstruction and event tests."""

from __future__ import annotations

import json

from kaggriculture.agents import OfficialAgent, ScriptedAgent
from kaggriculture.env.official_env import KaggricultureEnv
from kaggriculture.simulation.replay import (
    load_compact_replay,
    load_official_environment,
    official_environment_from_compact,
    reconstruct_turn,
    sibling_full_replay_path,
)


def test_compact_v2_reconstructs_both_players_and_private_state(tmp_path):
    compact_path = tmp_path / "game.compact.json"
    env = KaggricultureEnv(seed=42, episode_steps=96)
    result = env.play(
        ScriptedAgent(episode_steps=96),
        OfficialAgent("starter"),
        replay_path=compact_path,
    )

    replay = load_compact_replay(compact_path)
    assert replay["meta"]["format"] == "compact_v2"
    assert replay["meta"]["state_encoding"] == "initial_snapshot_plus_deltas"
    assert len(replay["initial_state"]["players"]) == 2
    assert all("private" in player for player in replay["initial_state"]["players"])

    final_state = reconstruct_turn(replay, len(replay["turns"]) - 1)
    assert final_state["step"] == 95
    assert [player["money"] for player in final_state["players"]] == result.rewards
    assert final_state["market"]["prices"] == replay["turns"][-1]["market_prices"]
    assert len(final_state["players"][0]["tiles"]) == 10
    assert len(final_state["players"][1]["tiles"][0]) == 10

    for player_id in (0, 1):
        expected = result.end_state_resources[player_id]
        actual = final_state["players"][player_id]
        assert actual["private"]["shed"] == expected["shed"]
        assert actual["private"]["seeds"] == expected["seeds"]


def test_replay_contains_semantic_events_and_turn_deltas(tmp_path):
    compact_path = tmp_path / "events.compact.json"
    env = KaggricultureEnv(seed=7, episode_steps=96)
    env.play(
        ScriptedAgent(episode_steps=96),
        OfficialAgent("starter"),
        replay_path=compact_path,
    )
    replay = load_compact_replay(compact_path)

    assert replay["turns"][0]["state_delta"] == {}
    assert any(turn["state_delta"].get("players") for turn in replay["turns"][1:])
    event_types = {
        event["type"]
        for turn in replay["turns"]
        for event in turn["events"]
    }
    assert "plant_created" in event_types
    assert "money_changed" in event_types
    assert "resource_changed" in event_types

    planted_turn = next(
        index
        for index, turn in enumerate(replay["turns"])
        if any(event["type"] == "plant_created" for event in turn["events"])
    )
    state = reconstruct_turn(replay, planted_turn)
    assert any(
        isinstance(tile, dict) and tile.get("kind") == "PLANT"
        for player in state["players"]
        for row in player["tiles"]
        for tile in row
    )


def test_compact_v2_is_smaller_than_full_official_replay(tmp_path):
    compact_path = tmp_path / "size.compact.json"
    full_path = tmp_path / "size.full.json"
    env = KaggricultureEnv(seed=9, episode_steps=96)
    env.play(
        ScriptedAgent(episode_steps=96),
        OfficialAgent("starter"),
        replay_path=compact_path,
        full_replay_path=full_path,
    )

    assert compact_path.stat().st_size < full_path.stat().st_size


def test_official_environment_from_compact_matches_full_replay(tmp_path):
    compact_path = tmp_path / "viz.compact.json"
    full_path = tmp_path / "viz.full.json"
    env = KaggricultureEnv(seed=11, episode_steps=48)
    env.play(
        ScriptedAgent(episode_steps=48),
        OfficialAgent("starter"),
        replay_path=compact_path,
        full_replay_path=full_path,
    )
    replay = load_compact_replay(compact_path)
    rebuilt = official_environment_from_compact(replay)
    official = json.loads(full_path.read_text(encoding="utf-8"))

    assert len(rebuilt["steps"]) == len(official["steps"]) == 48
    last_rebuilt = rebuilt["steps"][-1][0]["observation"]
    last_official = official["steps"][-1][0]["observation"]
    assert last_rebuilt["farms"][0]["tiles"] == last_official["farms"][0]["tiles"]
    assert last_rebuilt["farms"][1]["farmer"] == last_official["farms"][1]["farmer"]
    assert last_rebuilt["market"]["prices"] == last_official["market"]["prices"]
    assert last_rebuilt["private"] == last_official["private"]
    assert rebuilt["steps"][3][0]["action"] == official["steps"][3][0]["action"]
    reconstructed = reconstruct_turn(replay, 47)
    assert last_rebuilt["farms"][0]["money"] == reconstructed["players"][0]["money"]
    assert sibling_full_replay_path(compact_path) == full_path
    loaded = load_official_environment(replay, compact_path)
    assert loaded["steps"][-1][0]["observation"]["step"] == last_official["step"]
