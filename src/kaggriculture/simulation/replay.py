"""Compact and full replay serialization for simulated games."""

from __future__ import annotations

import copy
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


def git_commit() -> Optional[str]:
    """Return current HEAD SHA if this is a git checkout."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=Path.cwd(),
            text=True,
        )
        return out.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _to_plain(obj: Any) -> Any:
    """Recursively convert kaggle Struct / Mapping objects to JSON-safe types."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Mapping):
        return {str(k): _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if hasattr(obj, "items"):
        try:
            return {str(k): _to_plain(v) for k, v in obj.items()}
        except Exception:
            pass
    # Struct-like with known action keys
    if hasattr(obj, "farmer") or hasattr(obj, "market") or hasattr(obj, "hands"):
        return {
            "farmer": _to_plain(getattr(obj, "farmer", ["PASS"])),
            "hands": _to_plain(getattr(obj, "hands", []) or []),
            "market": _to_plain(getattr(obj, "market", []) or []),
        }
    return str(obj)


@dataclass
class CompactTurn:
    step: int
    day: int
    hour: int
    actions: list[Any]
    money: list[float]
    market_prices: dict[str, int]
    statuses: list[str]
    durations: list[Optional[float]] = field(default_factory=list)
    rewards: list[Optional[float]] = field(default_factory=list)
    shops: list[str] = field(default_factory=list)
    state_delta: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    reasoning: list[Any] = field(default_factory=lambda: [None, None])


@dataclass
class CompactReplay:
    meta: dict[str, Any]
    initial_state: dict[str, Any]
    turns: list[CompactTurn]

    def to_dict(self) -> dict[str, Any]:
        # Avoid dataclasses.asdict — it tries to reconstruct kaggle Struct types.
        return {
            "meta": dict(self.meta),
            "initial_state": self.initial_state,
            "turns": [
                {
                    "step": t.step,
                    "day": t.day,
                    "hour": t.hour,
                    "actions": t.actions,
                    "money": t.money,
                    "market_prices": t.market_prices,
                    "statuses": t.statuses,
                    "durations": t.durations,
                    "rewards": t.rewards,
                    "shops": t.shops,
                    "state_delta": t.state_delta,
                    "events": t.events,
                    "reasoning": t.reasoning,
                }
                for t in self.turns
            ],
        }


def build_compact_replay(
    *,
    steps: Sequence[Any],
    logs: Sequence[Any],
    seed: Optional[int],
    agent_names: Sequence[str],
    episode_steps: int,
    env_version: str,
    duration_s: float,
    rewards: Sequence[Optional[float]],
    statuses: Sequence[str],
    decision_traces: Optional[Sequence[Sequence[Any]]] = None,
) -> CompactReplay:
    """Extract a compact turn log from official env.steps / env.logs.

    Version 2 stores one full initial state and per-turn deltas so both farms
    and both players' private resources can be reconstructed efficiently.
    """
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "kaggle_environments_version": env_version,
        "seed": seed,
        "episode_steps": episode_steps,
        "agents": list(agent_names),
        "duration_s": duration_s,
        "final_rewards": list(rewards),
        "final_statuses": list(statuses),
        "num_steps": len(steps),
        "format": "compact_v2",
        "state_encoding": "initial_snapshot_plus_deltas",
        "reasoning_schema": "coc_v1",
    }

    initial_state = _build_state_snapshot(steps[0]) if steps else {}
    previous_snapshot: Optional[dict[str, Any]] = None
    turns: list[CompactTurn] = []
    for i, step_states in enumerate(steps):
        s0 = step_states[0]
        obs = s0.observation if hasattr(s0, "observation") else s0["observation"]
        day = int(getattr(obs, "day", obs.get("day", 0) if isinstance(obs, dict) else 0) or 0)
        hour = int(getattr(obs, "hour", obs.get("hour", 0) if isinstance(obs, dict) else 0) or 0)
        step_idx = int(
            getattr(obs, "step", obs.get("step", i) if isinstance(obs, dict) else i) or i
        )

        snapshot = initial_state if i == 0 else _build_state_snapshot(step_states)
        money = [float(player["money"]) for player in snapshot["players"]]
        prices = {
            str(k): int(v) for k, v in snapshot["market"]["prices"].items()
        }
        shops = list(snapshot["town"]["unlocked_shops"])

        actions = []
        statuses_t = []
        rewards_t = []
        for st in step_states:
            action = getattr(st, "action", None)
            if action is None and isinstance(st, dict):
                action = st.get("action")
            actions.append(_to_plain(action))
            status = getattr(st, "status", None)
            if status is None and isinstance(st, dict):
                status = st.get("status")
            statuses_t.append(str(status))
            reward = getattr(st, "reward", None)
            if reward is None and isinstance(st, dict):
                reward = st.get("reward")
            rewards_t.append(reward)

        durations: list[Optional[float]] = []
        if logs and i < len(logs):
            log_row = logs[i]
            if isinstance(log_row, (list, tuple)):
                for lg in log_row:
                    if isinstance(lg, dict):
                        durations.append(lg.get("duration"))
                    else:
                        durations.append(getattr(lg, "duration", None))
            else:
                durations = []

        turns.append(
            CompactTurn(
                step=step_idx,
                day=day,
                hour=hour,
                actions=actions,
                money=money,
                market_prices=prices,
                statuses=statuses_t,
                durations=durations,
                rewards=rewards_t,
                shops=shops,
                state_delta=(
                    {}
                    if previous_snapshot is None
                    else _build_state_delta(previous_snapshot, snapshot)
                ),
                events=(
                    []
                    if previous_snapshot is None
                    else _derive_events(previous_snapshot, snapshot)
                ),
                reasoning=_reasoning_for_turn(i, decision_traces),
            )
        )
        previous_snapshot = snapshot

    return CompactReplay(meta=meta, initial_state=initial_state, turns=turns)


def _reasoning_for_turn(
    turn_index: int,
    decision_traces: Optional[Sequence[Sequence[Any]]],
) -> list[Any]:
    """Map act() traces onto compact turns (turn 0 is the reset snapshot)."""
    seats = [None, None]
    if not decision_traces or turn_index <= 0:
        return seats
    act_index = turn_index - 1
    for seat, traces in enumerate(decision_traces[:2]):
        if traces is None:
            continue
        if act_index < len(traces):
            seats[seat] = traces[act_index]
    return seats


def load_compact_replay(path: Path | str) -> dict[str, Any]:
    """Load a compact replay JSON document."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or "meta" not in data or "turns" not in data:
        raise ValueError(f"Not a compact replay: {path}")
    return data


def official_environment_from_compact(
    replay: Mapping[str, Any] | CompactReplay,
) -> dict[str, Any]:
    """Rebuild an official ``env.toJSON()`` payload from compact v2.

    The Kaggle episode player reads this schema. Compact v1 cannot be converted
    because it has no farm snapshots.
    """
    data = replay.to_dict() if isinstance(replay, CompactReplay) else replay
    if data.get("initial_state") is None:
        raise ValueError("compact_v1 cannot be converted to an official environment")
    turns = data.get("turns", [])
    if not turns:
        raise ValueError("Replay has no turns")

    template = _official_environment_template()
    meta = data.get("meta", {})
    configuration = dict(template.get("configuration") or {})
    configuration["episodeSteps"] = int(meta.get("episode_steps") or len(turns))
    if meta.get("seed") is not None:
        configuration["seed"] = meta["seed"]

    steps: list[list[dict[str, Any]]] = []
    state = copy.deepcopy(data["initial_state"])
    for index, turn in enumerate(turns):
        if index:
            _apply_state_delta(state, turn.get("state_delta", {}) or {})
        steps.append(_official_step_states(state, turn))

    rewards = list(meta.get("final_rewards") or turns[-1].get("rewards") or [None, None])
    statuses = list(meta.get("final_statuses") or turns[-1].get("statuses") or ["DONE", "DONE"])
    return {
        "id": f"compact:{meta.get('seed', 'unknown')}",
        "name": template.get("name", "kaggriculture"),
        "title": template.get("title", "Kaggriculture"),
        "description": template.get("description", ""),
        "version": template.get("version", "0.1.0"),
        "module_version": meta.get("kaggle_environments_version")
        or template.get("module_version"),
        "configuration": configuration,
        "specification": copy.deepcopy(template.get("specification") or {}),
        "steps": steps,
        "rewards": rewards,
        "statuses": statuses,
        "schema_version": 1,
        "info": {"seed": meta.get("seed")},
    }


def sibling_full_replay_path(compact_path: Path | str) -> Optional[Path]:
    """Return a matching ``*.full.json`` next to a compact replay, if present."""
    path = Path(compact_path)
    name = path.name
    if name.endswith(".compact.json"):
        candidate = path.with_name(name[: -len(".compact.json")] + ".full.json")
    else:
        candidate = path.with_name(path.stem + ".full.json")
    return candidate if candidate.is_file() else None


def load_official_environment(
    compact_replay: Mapping[str, Any] | CompactReplay,
    compact_path: Optional[Path | str] = None,
) -> dict[str, Any]:
    """Prefer an official full replay file; otherwise reconstruct from compact v2."""
    if compact_path is not None:
        full_path = sibling_full_replay_path(compact_path)
        if full_path is not None:
            with full_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict) or "steps" not in payload:
                raise ValueError(f"Not an official replay: {full_path}")
            return payload
    return official_environment_from_compact(compact_replay)


def reconstruct_turn(
    replay: Mapping[str, Any] | CompactReplay,
    turn_index: int,
) -> dict[str, Any]:
    """Rebuild the normalized state at a zero-based replay turn index.

    Compact v2 stores a full initial snapshot and small subsequent deltas. This
    helper is the compatibility boundary for the future replay UI: callers do
    not need to understand the delta representation.
    """
    data = replay.to_dict() if isinstance(replay, CompactReplay) else replay
    initial = data.get("initial_state")
    turns = data.get("turns", [])
    if initial is None:
        raise ValueError("Replay has no initial_state; compact_v1 cannot be reconstructed")
    if turn_index < 0 or turn_index >= len(turns):
        raise IndexError(f"turn_index {turn_index} outside 0..{len(turns) - 1}")
    state = copy.deepcopy(initial)
    for turn in turns[1 : turn_index + 1]:
        _apply_state_delta(state, turn.get("state_delta", {}))
    return state


_OFFICIAL_ENV_TEMPLATE: Optional[dict[str, Any]] = None
_DEFAULT_ACTION = {"farmer": ["PASS"], "hands": [], "market": []}


def _official_environment_template() -> dict[str, Any]:
    global _OFFICIAL_ENV_TEMPLATE
    if _OFFICIAL_ENV_TEMPLATE is None:
        from kaggle_environments import make

        _OFFICIAL_ENV_TEMPLATE = make("kaggriculture").toJSON()
    return _OFFICIAL_ENV_TEMPLATE


def _public_farm(player: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "money": float(player.get("money", 0.0)),
        "tiles": copy.deepcopy(player.get("tiles", [])),
        "farmer": copy.deepcopy(player.get("farmer", [0, 0])),
        "hands": copy.deepcopy(player.get("hands", []) or []),
        "unlocked_quadrants": copy.deepcopy(
            player.get("unlocked_quadrants", []) or []
        ),
        "hires_today": int(player.get("hires_today", 0) or 0),
    }


def _official_observation(state: Mapping[str, Any], player_id: int) -> dict[str, Any]:
    farms = [_public_farm(player) for player in state["players"]]
    private = copy.deepcopy(state["players"][player_id].get("private") or {})
    return {
        "remainingOverageTime": 60,
        "step": int(state.get("step", 0) or 0),
        "player": player_id,
        "farms": farms,
        "private": private,
        "market": copy.deepcopy(state.get("market") or {}),
        "town": copy.deepcopy(state.get("town") or {"unlocked_shops": []}),
        "day": int(state.get("day", 0) or 0),
        "hour": int(state.get("hour", 0) or 0),
    }


def _official_action(action: Any) -> dict[str, Any]:
    if not action:
        return dict(_DEFAULT_ACTION)
    plain = _to_plain(action)
    if not isinstance(plain, dict):
        return dict(_DEFAULT_ACTION)
    return {
        "farmer": plain.get("farmer") or ["PASS"],
        "hands": plain.get("hands") or [],
        "market": plain.get("market") or [],
    }


def _official_step_states(
    state: Mapping[str, Any], turn: Mapping[str, Any]
) -> list[dict[str, Any]]:
    actions = list(turn.get("actions") or [])
    rewards = list(turn.get("rewards") or [])
    statuses = list(turn.get("statuses") or [])
    players = []
    for player_id in (0, 1):
        action = actions[player_id] if player_id < len(actions) else None
        reward = rewards[player_id] if player_id < len(rewards) else None
        status = statuses[player_id] if player_id < len(statuses) else "ACTIVE"
        players.append(
            {
                "action": _official_action(action),
                "reward": reward,
                "info": {},
                "observation": _official_observation(state, player_id),
                "status": str(status),
            }
        )
    return players


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _counts(obj: Any) -> dict[str, int]:
    plain = _to_plain(obj or {})
    if not isinstance(plain, dict):
        return {}
    return {str(key): int(value) for key, value in plain.items()}


def _build_state_snapshot(step_states: Sequence[Any]) -> dict[str, Any]:
    """Merge both agent observations into one local research snapshot."""
    if len(step_states) < 2:
        raise ValueError("Kaggriculture replay requires two player states")
    obs0 = _get(step_states[0], "observation", {})
    players: list[dict[str, Any]] = []
    for player_id in (0, 1):
        obs = _get(step_states[player_id], "observation", {})
        farms = _get(obs, "farms", []) or _get(obs0, "farms", [])
        farm = farms[player_id]
        private = _get(obs, "private", {}) or {}
        players.append(
            {
                "player_id": player_id,
                "money": float(_get(farm, "money", 0.0)),
                "farmer": _to_plain(_get(farm, "farmer", [0, 0])),
                "hands": _to_plain(_get(farm, "hands", []) or []),
                "unlocked_quadrants": _to_plain(
                    _get(farm, "unlocked_quadrants", []) or []
                ),
                "hires_today": int(_get(farm, "hires_today", 0) or 0),
                "tiles": _to_plain(_get(farm, "tiles", []) or []),
                "private": {
                    "shed": _counts(_get(private, "shed", {})),
                    "seeds": _counts(_get(private, "seeds", {})),
                    "inventories": _to_plain(
                        _get(private, "inventories", []) or []
                    ),
                },
            }
        )

    market = _get(obs0, "market", {}) or {}
    town = _get(obs0, "town", {}) or {}
    return {
        "step": int(_get(obs0, "step", 0) or 0),
        "day": int(_get(obs0, "day", 0) or 0),
        "hour": int(_get(obs0, "hour", 0) or 0),
        "players": players,
        "market": {
            "inventory": _counts(_get(market, "inventory", {})),
            "prices": _counts(_get(market, "prices", {})),
        },
        "town": {
            "unlocked_shops": _to_plain(
                _get(town, "unlocked_shops", []) or []
            )
        },
    }


def _changed_counts(
    previous: Mapping[str, Any], current: Mapping[str, Any]
) -> dict[str, int]:
    return {
        str(key): int(current.get(key, 0))
        for key in set(previous) | set(current)
        if int(previous.get(key, 0)) != int(current.get(key, 0))
    }


def _build_state_delta(
    previous: Mapping[str, Any], current: Mapping[str, Any]
) -> dict[str, Any]:
    delta: dict[str, Any] = {
        "step": current["step"],
        "day": current["day"],
        "hour": current["hour"],
        "players": [],
    }
    for old_player, new_player in zip(previous["players"], current["players"]):
        player_delta: dict[str, Any] = {}
        for key in (
            "money",
            "farmer",
            "hands",
            "unlocked_quadrants",
            "hires_today",
        ):
            if old_player[key] != new_player[key]:
                player_delta[key] = copy.deepcopy(new_player[key])

        private_delta: dict[str, Any] = {}
        for key in ("shed", "seeds"):
            changes = _changed_counts(
                old_player["private"][key], new_player["private"][key]
            )
            if changes:
                private_delta[key] = changes
        if (
            old_player["private"]["inventories"]
            != new_player["private"]["inventories"]
        ):
            private_delta["inventories"] = copy.deepcopy(
                new_player["private"]["inventories"]
            )
        if private_delta:
            player_delta["private"] = private_delta

        tile_changes: list[dict[str, Any]] = []
        for y, (old_row, new_row) in enumerate(
            zip(old_player["tiles"], new_player["tiles"])
        ):
            for x, (old_tile, new_tile) in enumerate(zip(old_row, new_row)):
                if old_tile != new_tile:
                    tile_changes.append(
                        {"x": x, "y": y, "tile": copy.deepcopy(new_tile)}
                    )
        if tile_changes:
            player_delta["tile_changes"] = tile_changes
        delta["players"].append(player_delta)

    market_delta: dict[str, Any] = {}
    for key in ("inventory", "prices"):
        changes = _changed_counts(previous["market"][key], current["market"][key])
        if changes:
            market_delta[key] = changes
    if market_delta:
        delta["market"] = market_delta
    if previous["town"] != current["town"]:
        delta["town"] = copy.deepcopy(current["town"])
    return delta


def _apply_state_delta(state: dict[str, Any], delta: Mapping[str, Any]) -> None:
    for key in ("step", "day", "hour"):
        if key in delta:
            state[key] = delta[key]
    for player_id, player_delta in enumerate(delta.get("players", [])):
        player = state["players"][player_id]
        for key in (
            "money",
            "farmer",
            "hands",
            "unlocked_quadrants",
            "hires_today",
        ):
            if key in player_delta:
                player[key] = copy.deepcopy(player_delta[key])
        private_delta = player_delta.get("private", {})
        for key in ("shed", "seeds"):
            if key in private_delta:
                player["private"][key].update(private_delta[key])
        if "inventories" in private_delta:
            player["private"]["inventories"] = copy.deepcopy(
                private_delta["inventories"]
            )
        for change in player_delta.get("tile_changes", []):
            player["tiles"][change["y"]][change["x"]] = copy.deepcopy(
                change["tile"]
            )
    market_delta = delta.get("market", {})
    for key in ("inventory", "prices"):
        if key in market_delta:
            state["market"][key].update(market_delta[key])
    if "town" in delta:
        state["town"] = copy.deepcopy(delta["town"])


def _tile_label(tile: Any) -> str:
    if tile is None:
        return "EMPTY"
    if isinstance(tile, str):
        return tile
    if not isinstance(tile, Mapping):
        return "UNKNOWN"
    kind = str(tile.get("kind", "UNKNOWN"))
    if kind == "PLANT":
        return f"PLANT:{tile.get('crop', 'UNKNOWN')}"
    if tile.get("animal"):
        return f"{kind}:{tile['animal']}"
    return kind


def _derive_events(
    previous: Mapping[str, Any], current: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Derive human-oriented events from consecutive normalized snapshots."""
    events: list[dict[str, Any]] = []
    for player_id, (old_player, new_player) in enumerate(
        zip(previous["players"], current["players"])
    ):
        money_delta = float(new_player["money"] - old_player["money"])
        if money_delta:
            events.append(
                {"type": "money_changed", "player": player_id, "delta": money_delta}
            )
        old_unlocked = set(old_player["unlocked_quadrants"])
        newly_unlocked = [
            quadrant
            for quadrant in new_player["unlocked_quadrants"]
            if quadrant not in old_unlocked
        ]
        for quadrant in newly_unlocked:
            events.append(
                {
                    "type": "land_unlocked",
                    "player": player_id,
                    "quadrant": quadrant,
                }
            )
        if len(old_player["hands"]) != len(new_player["hands"]):
            events.append(
                {
                    "type": "worker_count_changed",
                    "player": player_id,
                    "from": len(old_player["hands"]),
                    "to": len(new_player["hands"]),
                }
            )
        for storage in ("shed", "seeds"):
            old_counts = old_player["private"][storage]
            new_counts = new_player["private"][storage]
            for item in sorted(set(old_counts) | set(new_counts)):
                quantity_delta = int(new_counts.get(item, 0)) - int(
                    old_counts.get(item, 0)
                )
                if quantity_delta:
                    events.append(
                        {
                            "type": "resource_changed",
                            "player": player_id,
                            "storage": storage,
                            "item": item,
                            "delta": quantity_delta,
                        }
                    )
        for y, (old_row, new_row) in enumerate(
            zip(old_player["tiles"], new_player["tiles"])
        ):
            for x, (old_tile, new_tile) in enumerate(zip(old_row, new_row)):
                if old_tile == new_tile:
                    continue
                before = _tile_label(old_tile)
                after = _tile_label(new_tile)
                event_type = "tile_changed"
                if after == "WEED" and before != "WEED":
                    event_type = "weed_appeared"
                elif after.startswith("PLANT:") and not before.startswith("PLANT:"):
                    event_type = "plant_created"
                elif before.startswith("PLANT:") and not after.startswith("PLANT:"):
                    event_type = "plant_removed"
                elif before == "EMPTY" and after in ("COOP", "PASTURE"):
                    event_type = "structure_built"
                elif ":" in after and ":" not in before:
                    event_type = "animal_placed"
                events.append(
                    {
                        "type": event_type,
                        "player": player_id,
                        "x": x,
                        "y": y,
                        "before": before,
                        "after": after,
                    }
                )
    if previous["town"] != current["town"]:
        events.append(
            {
                "type": "shops_changed",
                "shops": copy.deepcopy(current["town"]["unlocked_shops"]),
            }
        )
    return events


def save_json(path: Path | str, data: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    return path
