"""Teacher dataset buffer: compact samples from the planner (Phase 14)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence


class TeacherBuffer:
    """In-memory list of compact (features, candidates, choice) rows."""

    def __init__(self) -> None:
        self.samples: list[dict[str, Any]] = []
        self.outcome: Optional[dict[str, Any]] = None

    def record(
        self,
        *,
        features: dict[str, Any],
        candidates: Sequence[str],
        scores: Sequence[float],
        chosen: str,
        headline: str = "",
    ) -> None:
        self.samples.append(
            {
                "features": dict(features),
                "candidates": list(candidates),
                "scores": [float(s) for s in scores],
                "chosen": chosen,
                "headline": headline,
            }
        )

    def attach_outcome(
        self,
        *,
        reward: Optional[float],
        opponent_reward: Optional[float],
        winner: Optional[int],
        seat: int,
    ) -> None:
        self.outcome = {
            "reward": reward,
            "opponent_reward": opponent_reward,
            "winner": winner,
            "seat": seat,
        }
        for sample in self.samples:
            sample["outcome"] = dict(self.outcome)

    def dump_jsonl(self, path: str | Path, *, append: bool = False) -> Path:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with dest.open(mode, encoding="utf-8") as handle:
            for sample in self.samples:
                handle.write(json.dumps(sample, separators=(",", ":")) + "\n")
        return dest
