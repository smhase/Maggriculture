"""Replay buffer for imitation / RL — stores TeacherBuffer samples."""

from __future__ import annotations

from kaggriculture.learning.dataset import TeacherBuffer

ReplayBuffer = TeacherBuffer

__all__ = ["ReplayBuffer", "TeacherBuffer"]
