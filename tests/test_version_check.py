"""Local (offline) tests for the official-engine version checker."""

from __future__ import annotations

from kaggriculture.env.version_check import (
    pinned_kaggle_environments_version,
    repo_root,
    run_check,
)


def test_pin_matches_pyproject_and_install():
    pin = pinned_kaggle_environments_version()
    assert pin == "1.32.7"
    report = run_check(fetch_remote=False)
    assert report["pin"] == pin
    assert report["installed"] == pin
    assert report["status"] == "ok"
    assert report["findings"] == []
    assert (repo_root() / "pyproject.toml").is_file()
