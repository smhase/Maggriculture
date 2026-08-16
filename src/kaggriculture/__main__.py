"""CLI entry: ``python -m kaggriculture``."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    """Dispatch to the simulation runner CLI."""
    from kaggriculture.simulation.runner import main as runner_main

    return runner_main(argv if argv is not None else sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
