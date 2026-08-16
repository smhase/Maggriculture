"""Replay viewer page and pure replay presentation helpers."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

from kaggriculture.simulation.replay import (
    load_compact_replay,
    load_official_environment,
    reconstruct_turn,
    sibling_full_replay_path,
)
from ui.common import discover_json_files, nonzero_items

_VISUALIZER_DIR = Path(__file__).resolve().parent / "static" / "replay_visualizers"


def replay_timeline(replay: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten headline turn data into chart-friendly rows."""
    rows: list[dict[str, Any]] = []
    for index, turn in enumerate(replay.get("turns", [])):
        money = turn.get("money", [None, None])
        row: dict[str, Any] = {
            "turn": int(turn.get("step", index)),
            "Player A": money[0] if len(money) > 0 else None,
            "Player B": money[1] if len(money) > 1 else None,
        }
        for item, price in turn.get("market_prices", {}).items():
            row[f"price:{item}"] = price
        rows.append(row)
    return rows


def farm_grid(player: Mapping[str, Any]) -> list[list[str]]:
    """Render a normalized farm snapshot as a compact text grid."""
    tiles = player.get("tiles", [])
    farmer = tuple(player.get("farmer", [-1, -1]))
    hands = {tuple(position) for position in player.get("hands", [])}
    grid: list[list[str]] = []
    for y, row in enumerate(tiles):
        rendered: list[str] = []
        for x, tile in enumerate(row):
            label = _tile_label(tile)
            if (x, y) == farmer:
                label = f"F·{label}"
            elif (x, y) in hands:
                label = f"H·{label}"
            rendered.append(label)
        grid.append(rendered)
    return grid


def _tile_label(tile: Any) -> str:
    if tile is None:
        return "·"
    if tile == "LOCKED":
        return "LOCK"
    if not isinstance(tile, Mapping):
        return "?"
    kind = str(tile.get("kind", "?"))
    if kind == "WEED":
        return "WEED"
    if kind == "PLANT":
        crop = str(tile.get("crop", "?"))
        return f"{crop[:3]}:{int(tile.get('yield_units', 0))}"
    if kind in ("COOP", "PASTURE"):
        animal = tile.get("animal")
        return f"{kind[:4]}:{str(animal)[:3]}" if animal else kind[:4]
    return kind[:6]


def render_replay_page(default_directory: Path) -> None:
    import streamlit as st

    st.markdown(
        '<span class="kg-badge">Official visualizer</span>'
        '<span class="kg-badge">Turn reconstruction</span>',
        unsafe_allow_html=True,
    )
    st.subheader("Replay viewer")
    directory = Path(
        st.text_input("Replay directory", value=str(default_directory), key="replay_directory")
    ).expanduser()
    paths = [
        path for path in discover_json_files(directory) if path.name.endswith(".compact.json")
    ]
    if not paths:
        st.info("No replay JSON files found. Run a game with `--replay-dir` first.")
        return

    selected = st.selectbox(
        "Replay",
        paths,
        format_func=lambda path: path.name,
        key="replay_path",
    )
    try:
        replay = load_compact_replay(selected)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        st.error(f"Could not load replay: {exc}")
        return

    turns = replay.get("turns", [])
    if not turns:
        st.warning("Replay has no turns.")
        return
    meta = replay.get("meta", {})
    names = list(meta.get("agents", ["Player A", "Player B"]))
    while len(names) < 2:
        names.append(f"Player {len(names) + 1}")

    if st.session_state.get("loaded_replay") != str(selected):
        st.session_state.loaded_replay = str(selected)
        st.session_state.replay_turn = 0
        st.session_state.replay_playing = False
        st.session_state.pop("_replay_advance", None)

    _render_official_game_view(selected, replay, names)
    _render_replay_inspector(replay, names)


def official_visualizer_html(
    environment: Mapping[str, Any],
    *,
    step: int = 0,
    playing: bool = False,
    agents: list[str] | None = None,
) -> str:
    """Inject replay JSON into the official Kaggriculture episode player."""
    from kaggle_environments.envs.kaggriculture.kaggriculture import html_renderer
    from kaggle_environments.utils import get_player

    window_kaggle = {
        "debug": False,
        "playing": playing,
        "step": int(step),
        "controls": True,
        "environment": dict(environment),
        "logs": [],
    }
    if agents:
        window_kaggle["agents"] = list(agents)
    return get_player(window_kaggle, html_renderer(None, "html"))


def _render_official_game_view(
    compact_path: Path,
    replay: Mapping[str, Any],
    names: list[str],
) -> None:
    import streamlit as st

    can_render = replay.get("meta", {}).get("format") == "compact_v2" or (
        sibling_full_replay_path(compact_path) is not None
    )
    if not can_render:
        st.info(
            "The official farm visualizer needs compact v2 or a matching `*.full.json`. "
            "Record one with `--replay-dir` (compact v2 is the default) or `--full-replay`."
        )
        return

    st.caption(
        "Official Kaggriculture player — same tiles, town, market, and HUD as the "
        "Kaggle episode viewer. Use its play/step controls to walk the season."
    )
    try:
        with st.spinner("Building official game view…"):
            html_path = _cached_visualizer_html(compact_path, replay, names)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        st.error(f"Could not build the official visualizer: {exc}")
        return

    st.iframe(html_path, width="stretch", height=980)


def _render_replay_inspector(replay: Mapping[str, Any], names: list[str]) -> None:
    import streamlit as st

    @st.fragment
    def inspector() -> None:
        turns = replay.get("turns", [])
        meta = replay.get("meta", {})

        if st.session_state.pop("_replay_advance", False):
            current = int(st.session_state.get("replay_turn", 0))
            if current >= len(turns) - 1:
                st.session_state.replay_playing = False
            else:
                st.session_state.replay_turn = current + 1

        control_a, control_b, control_c, control_d = st.columns([1, 1, 5, 1.5])
        with control_a:
            st.button(
                "◀",
                width="stretch",
                on_click=_move_turn,
                args=(-1, len(turns)),
            )
        with control_b:
            st.button(
                "▶",
                width="stretch",
                on_click=_move_turn,
                args=(1, len(turns)),
            )
        with control_c:
            st.slider(
                "Turn",
                min_value=0,
                max_value=len(turns) - 1,
                key="replay_turn",
            )
        with control_d:
            st.toggle("Play inspector", key="replay_playing")

        turn_index = int(st.session_state.replay_turn)
        turn = turns[turn_index]
        metric_cols = st.columns(6)
        metric_cols[0].metric("Turn", turn.get("step", turn_index))
        metric_cols[1].metric("Day", int(turn.get("day", 0)) + 1)
        metric_cols[2].metric("Hour", turn.get("hour", 0))
        money = turn.get("money", [0, 0])
        metric_cols[3].metric(names[0], f"${money[0]:,.0f}")
        metric_cols[4].metric(names[1], f"${money[1]:,.0f}")
        metric_cols[5].metric("Events", len(turn.get("events", [])))

        if meta.get("format") != "compact_v2":
            st.warning(
                "This is a compact v1 replay. Headline charts and actions are available, "
                "but farm reconstruction requires compact v2."
            )
            state = None
        else:
            state = reconstruct_turn(replay, turn_index)

        timeline = replay_timeline(replay)
        with st.expander("Season charts", expanded=False):
            cash_data = {
                names[0]: [row["Player A"] for row in timeline],
                names[1]: [row["Player B"] for row in timeline],
            }
            st.caption("Bank balance")
            st.line_chart(cash_data, x_label="Turn", y_label="Money")
            products = sorted(turn.get("market_prices", {}))
            if products:
                product = st.selectbox("Market commodity", products, key="replay_product")
                st.line_chart(
                    {product: [row.get(f"price:{product}") for row in timeline]},
                    x_label="Turn",
                    y_label="Price",
                )

        if state is not None:
            tab_a, tab_b, market_tab = st.tabs((names[0], names[1], "Market & events"))
            with tab_a:
                _render_player(state["players"][0], turn.get("actions", [None, None])[0])
            with tab_b:
                _render_player(state["players"][1], turn.get("actions", [None, None])[1])
            with market_tab:
                _render_market_and_events(state, turn)
        else:
            st.subheader("Chosen actions")
            st.json(turn.get("actions", []))

        if st.session_state.replay_playing:
            if turn_index < len(turns) - 1:
                time.sleep(0.18)
            st.session_state._replay_advance = True
            st.rerun(scope="fragment")

    inspector()


def _cached_visualizer_html(
    compact_path: Path,
    replay: Mapping[str, Any],
    names: list[str],
) -> Path:
    _VISUALIZER_DIR.mkdir(parents=True, exist_ok=True)
    stamp = int(compact_path.stat().st_mtime)
    dest = _VISUALIZER_DIR / f"{compact_path.stem}-{stamp}.html"
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        return dest
    environment = load_official_environment(replay, compact_path)
    dest.write_text(
        official_visualizer_html(environment, step=0, playing=False, agents=names),
        encoding="utf-8",
    )
    return dest


def _move_turn(delta: int, count: int) -> None:
    import streamlit as st

    current = int(st.session_state.get("replay_turn", 0))
    st.session_state.replay_turn = min(count - 1, max(0, current + delta))


def _render_player(player: Mapping[str, Any], action: Any) -> None:
    import streamlit as st

    left, right = st.columns([3, 2])
    with left:
        st.caption("Farm · F = farmer, H = hired hand")
        grid = farm_grid(player)
        st.dataframe(
            grid,
            width="stretch",
            hide_index=True,
            column_config={index: str(index) for index in range(len(grid[0]) if grid else 0)},
        )
    with right:
        st.markdown("#### Position & infrastructure")
        st.write(
            f"Farmer `{player.get('farmer')}` · Hands `{len(player.get('hands', []))}` · "
            f"Land `{', '.join(player.get('unlocked_quadrants', []))}`"
        )
        private = player.get("private", {})
        st.markdown("#### Shed")
        shed_rows = nonzero_items(private.get("shed", {}))
        if shed_rows:
            st.dataframe(shed_rows, hide_index=True, width="stretch")
        else:
            st.caption("Empty")
        st.markdown("#### Seeds")
        seed_rows = nonzero_items(private.get("seeds", {}))
        if seed_rows:
            st.dataframe(seed_rows, hide_index=True, width="stretch")
        else:
            st.caption("None")
        st.markdown("#### Chosen action")
        st.code(json.dumps(action, indent=2), language="json")


def _render_market_and_events(state: Mapping[str, Any], turn: Mapping[str, Any]) -> None:
    import streamlit as st

    market_col, event_col = st.columns([2, 3])
    with market_col:
        st.markdown("#### Market")
        rows = [
            {
                "Commodity": item,
                "Price": state["market"]["prices"].get(item),
                "Inventory": inventory,
            }
            for item, inventory in state["market"]["inventory"].items()
        ]
        st.dataframe(rows, hide_index=True, width="stretch")
        st.markdown("#### Town shops")
        shops = state.get("town", {}).get("unlocked_shops", [])
        st.write(", ".join(shops) if shops else "No shops unlocked")
    with event_col:
        st.markdown("#### Important events")
        events = turn.get("events", [])
        if events:
            st.dataframe(events, hide_index=True, width="stretch")
        else:
            st.caption("No derived events on this turn.")
