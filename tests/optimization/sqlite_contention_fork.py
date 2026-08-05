"""Run fork-based SQLite contention before an interpreter starts threads."""

from __future__ import annotations

import multiprocessing
import sys
import threading
from pathlib import Path

from tests.core.effects.authority_sqlite_scenarios import (
    run_spawned_same_owner_different_attempts_arbitrate_once,
)
from tests.optimization.tools.sqlite_scenarios import (
    run_sqlite_capacity_race,
    run_sqlite_global_capacity,
    run_sqlite_same_call_replay,
)


def _directory(root: Path, name: str) -> Path:
    directory = root / name
    directory.mkdir()
    return directory


def main(root: Path) -> None:
    if "fork" not in multiprocessing.get_all_start_methods():
        raise RuntimeError("fork is not supported by this platform")
    if threading.active_count() != 1:
        raise RuntimeError("fork contention must run before threads exist")

    run_spawned_same_owner_different_attempts_arbitrate_once(
        _directory(root, "effect"),
        "fork",
    )
    run_sqlite_capacity_race(
        _directory(root, "capacity"),
        "fork",
    )
    run_sqlite_global_capacity(
        _directory(root, "global"),
        "fork",
    )
    run_sqlite_same_call_replay(
        _directory(root, "replay"),
        "fork",
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: sqlite_contention_fork.py ROOT")
    main(Path(sys.argv[1]))
