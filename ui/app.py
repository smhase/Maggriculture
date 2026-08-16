"""Kaggriculture research UI entry point.

Run from the repository root:

    streamlit run ui/app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import kaggle_environments
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # Streamlit puts ui/ rather than the repository root on sys.path when
    # invoked as `streamlit run ui/app.py`.
    sys.path.insert(0, str(ROOT))

from ui.experiments import render_experiments_page
from ui.replay import render_replay_page
from ui.tournament import render_tournament_page


def _apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
          --kg-ink: #17251d;
          --kg-green: #1f5b3a;
          --kg-mint: #dfeee3;
          --kg-cream: #fbf8ef;
          --kg-gold: #c89936;
        }
        .stApp { background: linear-gradient(180deg, #fbf8ef 0%, #f4f7f1 100%); }
        [data-testid="stSidebar"] { background: #173d2b; }
        [data-testid="stSidebar"] * { color: #f8f4e8; }
        [data-testid="stMetric"] {
          background: rgba(255,255,255,.72);
          border: 1px solid #d8dfd4;
          border-radius: 12px;
          padding: 10px 14px;
          box-shadow: 0 4px 18px rgba(29, 65, 45, .06);
        }
        .kg-kicker { color: #6f785f; letter-spacing: .11em; font-size: .76rem;
          font-weight: 700; text-transform: uppercase; }
        .kg-title { color: var(--kg-ink); font-family: Georgia, serif;
          font-size: 2.45rem; line-height: 1.05; margin: .2rem 0 .5rem; }
        .kg-subtitle { color: #5d6a60; max-width: 760px; margin-bottom: 1.2rem; }
        .kg-badge { display: inline-block; border-radius: 999px; padding: .25rem .65rem;
          background: var(--kg-mint); color: var(--kg-green); font-weight: 650;
          font-size: .78rem; margin-right: .35rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _header() -> None:
    st.markdown('<div class="kg-kicker">Competitive agent laboratory</div>', unsafe_allow_html=True)
    st.markdown('<div class="kg-title">Kaggriculture Research Console</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="kg-subtitle">Inspect complete games, compare controlled experiments, '
        'and understand the local agent population without touching the official engine.</div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Kaggriculture Research Console",
        page_icon="🌾",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _apply_theme()
    with st.sidebar:
        st.markdown("## 🌾 Kaggriculture")
        page = st.radio(
            "Workspace",
            ("Replay viewer", "Experiments", "Tournament"),
            label_visibility="collapsed",
        )
        st.divider()
        st.caption(
            f"Official environment {kaggle_environments.__version__} · 720-turn horizon"
        )

    _header()
    replay_dir = Path(
        os.environ.get("KAGGRICULTURE_REPLAY_DIR", ROOT / "experiments/replays")
    )
    results_dir = Path(
        os.environ.get("KAGGRICULTURE_RESULTS_DIR", ROOT / "experiments/results")
    )

    if page == "Replay viewer":
        render_replay_page(replay_dir)
    elif page == "Experiments":
        render_experiments_page(results_dir)
    else:
        render_tournament_page(results_dir)


if __name__ == "__main__":
    main()
