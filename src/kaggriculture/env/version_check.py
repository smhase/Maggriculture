"""Daily check that the pinned Kaggriculture engine matches PyPI and GitHub.

Compares, in order:

1. ``kaggle-environments==`` pin in ``pyproject.toml``
2. The installed package version
3. The latest version on PyPI
4. SHA-256 of installed ``kaggriculture.py`` vs GitHub ``master``

Exit 0 if everything matches, 1 if the pin/engine is behind or drifted,
2 if the check itself failed (network, missing files).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tomllib
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import kaggle_environments

PYPI_JSON_URL = "https://pypi.org/pypi/kaggle-environments/json"
GITHUB_ENGINE_URL = (
    "https://raw.githubusercontent.com/Kaggle/kaggle-environments/"
    "master/kaggle_environments/envs/kaggriculture/kaggriculture.py"
)
GITHUB_SPEC_URL = (
    "https://raw.githubusercontent.com/Kaggle/kaggle-environments/"
    "master/kaggle_environments/envs/kaggriculture/kaggriculture.json"
)
DEFAULT_TIMEOUT_S = 30


def repo_root(start: Optional[Path] = None) -> Path:
    """Walk upward until ``pyproject.toml`` + ``src/kaggriculture`` are found."""
    env = os.environ.get("KAGGRICULTURE_ROOT")
    if env:
        return Path(env).resolve()
    here = (start or Path(__file__).resolve()).absolute()
    candidates = [here, *here.parents, Path.cwd()]
    for path in candidates:
        if (path / "pyproject.toml").is_file() and (path / "src" / "kaggriculture").is_dir():
            return path
    raise FileNotFoundError("Could not locate Kaggriculture repo root (pyproject.toml)")


def pinned_kaggle_environments_version(root: Optional[Path] = None) -> str:
    data = tomllib.loads((repo_root(root) / "pyproject.toml").read_text())
    for dep in data["project"]["dependencies"]:
        if dep.startswith("kaggle-environments=="):
            return dep.split("==", 1)[1].strip()
    raise ValueError("pyproject.toml has no kaggle-environments== pin")


def installed_engine_path() -> Path:
    import kaggle_environments.envs.kaggriculture.kaggriculture as engine

    return Path(engine.__file__).resolve()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def fetch_url(url: str, *, timeout: float = DEFAULT_TIMEOUT_S) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "kaggriculture-env-check"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def pypi_latest_version(*, timeout: float = DEFAULT_TIMEOUT_S) -> str:
    payload = json.loads(fetch_url(PYPI_JSON_URL, timeout=timeout).decode("utf-8"))
    return str(payload["info"]["version"])


def run_check(*, timeout: float = DEFAULT_TIMEOUT_S, fetch_remote: bool = True) -> dict[str, Any]:
    root = repo_root()
    pin = pinned_kaggle_environments_version(root)
    installed = kaggle_environments.__version__
    engine_path = installed_engine_path()
    installed_hash = sha256_file(engine_path)

    report: dict[str, Any] = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(root),
        "pin": pin,
        "installed": installed,
        "engine_path": str(engine_path),
        "installed_engine_sha256": installed_hash,
        "pypi_latest": None,
        "github_engine_sha256": None,
        "github_spec_sha256": None,
        "findings": [],
        "status": "ok",
    }

    findings: list[str] = []
    if installed != pin:
        findings.append(
            f"Installed kaggle-environments {installed} does not match pin {pin}."
        )

    if fetch_remote:
        pypi_latest = pypi_latest_version(timeout=timeout)
        report["pypi_latest"] = pypi_latest
        if pypi_latest != pin:
            findings.append(
                f"PyPI latest is {pypi_latest}; repo pin is {pin}."
            )

        github_py = fetch_url(GITHUB_ENGINE_URL, timeout=timeout)
        github_json = fetch_url(GITHUB_SPEC_URL, timeout=timeout)
        report["github_engine_sha256"] = sha256_bytes(github_py)
        report["github_spec_sha256"] = sha256_bytes(github_json)
        if report["github_engine_sha256"] != installed_hash:
            findings.append(
                "GitHub master kaggriculture.py differs from the installed engine."
            )
        spec_path = engine_path.with_suffix(".json")
        if spec_path.is_file() and sha256_file(spec_path) != report["github_spec_sha256"]:
            findings.append(
                "GitHub master kaggriculture.json differs from the installed spec."
            )

    report["findings"] = findings
    report["status"] = "ok" if not findings else "drift"
    return report


def write_report(report: dict[str, Any], root: Optional[Path] = None) -> Path:
    out_dir = repo_root(root) / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "env_version_check.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    return path


def format_report(report: dict[str, Any]) -> str:
    lines = [
        f"Kaggriculture engine check  {report['checked_at']}",
        f"  pin        {report['pin']}",
        f"  installed  {report['installed']}",
        f"  pypi       {report.get('pypi_latest') or '(skipped)'}",
        f"  status     {report['status']}",
    ]
    if report["findings"]:
        lines.append("  findings:")
        lines.extend(f"    - {item}" for item in report["findings"])
    else:
        lines.append("  findings: none (in sync)")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Only compare the local pin to the installed package.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help="HTTP timeout in seconds (default: 30).",
    )
    args = parser.parse_args(argv)

    try:
        report = run_check(timeout=args.timeout, fetch_remote=not args.offline)
        path = write_report(report)
        print(format_report(report))
        print(f"  wrote      {path}")
    except (OSError, urllib.error.URLError, TimeoutError, ValueError, KeyError) as exc:
        print(f"Kaggriculture engine check failed: {exc}", file=sys.stderr)
        return 2
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
