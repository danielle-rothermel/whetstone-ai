"""Isolated fork coverage for SQLite contention contracts."""

from __future__ import annotations

import multiprocessing
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.sqlite_contention
@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="fork is unavailable on this platform",
)
@pytest.mark.process_integration
def test_fork_contention_runs_before_threads_exist(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.optimization.sqlite_contention_fork",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr
