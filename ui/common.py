"""Shared data-loading helpers for the research UI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def discover_json_files(directory: Path | str) -> list[Path]:
    """Return readable JSON files in deterministic newest-first order."""
    root = Path(directory).expanduser()
    if not root.is_dir():
        return []
    paths = [path for path in root.glob("*.json") if path.is_file()]
    return sorted(paths, key=lambda path: (path.stat().st_mtime, path.name), reverse=True)


def read_json(path: Path | str) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def nonzero_items(counts: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"Item": item, "Quantity": int(quantity)}
        for item, quantity in sorted(counts.items())
        if int(quantity) != 0
    ]


def compact_path(path: Path | str, roots: Iterable[Path] = ()) -> str:
    candidate = Path(path)
    for root in roots:
        try:
            return str(candidate.relative_to(root))
        except ValueError:
            continue
    return str(candidate)
