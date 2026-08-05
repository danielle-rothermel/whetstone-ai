"""Run fork-based SQLite contention before an interpreter starts threads."""

from __future__ import annotations

import multiprocessing
import sys
import threading
from pathlib import Path

from tests.optimization.test_effect_authority import (
    test_spawned_same_owner_different_attempts_arbitrate_once,
)
from tests.optimization.test_tool_store import (
    test_spawned_global_capacity_has_one_process_shared_bucket,
    test_spawned_same_call_replay_has_one_ordinal,
    test_spawned_sqlite_capacity_race_is_atomic,
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

    test_spawned_same_owner_different_attempts_arbitrate_once(
        _directory(root, "effect"),
        "fork",
    )
    test_spawned_sqlite_capacity_race_is_atomic(
        _directory(root, "capacity"),
        "fork",
    )
    test_spawned_global_capacity_has_one_process_shared_bucket(
        _directory(root, "global"),
        "fork",
    )
    test_spawned_same_call_replay_has_one_ordinal(
        _directory(root, "replay"),
        "fork",
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: sqlite_contention_fork.py ROOT")
    main(Path(sys.argv[1]))
