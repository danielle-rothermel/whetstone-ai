"""SQLite contention pathway tests."""

# ruff: noqa: E402

from __future__ import annotations

from pathlib import Path

import pytest

from tests.optimization.processes import (
    in_process_start_methods,
)
from tests.optimization.tools.sqlite_scenarios import (
    run_sqlite_global_capacity,
    run_sqlite_same_call_replay,
)

FULL_A = "a" * 64
FULL_B = "b" * 64
import multiprocessing
from datetime import datetime

from tests.core.effects.authority_spawn import acquire_then_exit, spawn_result
from tests.core.effects.authority_sqlite_scenarios import (
    run_spawned_authority_contention,
    run_spawned_same_owner_different_attempts_arbitrate_once,
)
from tests.core.effects.authority_support import (
    _request,
)
from tests.optimization.processes import (
    join_processes,
    terminate_processes,
)
from tests.optimization.sqlite_time import wait_for_sqlite_authority_after
from whetstone.core.effects.authority import (
    AcquireOutcome,
)


@pytest.mark.sqlite_contention
@pytest.mark.parametrize("start_method", in_process_start_methods())
@pytest.mark.process_integration
def test_spawned_global_capacity_has_one_process_shared_bucket(
    tmp_path: Path, start_method: str
) -> None:
    run_sqlite_global_capacity(tmp_path, start_method)


@pytest.mark.sqlite_contention
@pytest.mark.parametrize("start_method", in_process_start_methods())
@pytest.mark.process_integration
def test_spawned_same_call_replay_has_one_ordinal(
    tmp_path: Path, start_method: str
) -> None:
    run_sqlite_same_call_replay(tmp_path, start_method)


@pytest.mark.sqlite_contention
@pytest.mark.parametrize("start_method", in_process_start_methods())
@pytest.mark.process_integration
def test_spawned_same_owner_different_attempts_arbitrate_once(
    tmp_path: Path,
    start_method: str,
) -> None:
    run_spawned_same_owner_different_attempts_arbitrate_once(
        tmp_path,
        start_method,
    )


@pytest.mark.sqlite_time_integration
@pytest.mark.sqlite_contention
@pytest.mark.process_integration
def test_spawned_sqlite_owner_exit_allows_authority_timed_takeover(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    database = tmp_path / "takeover.sqlite"
    request = _request()
    output = context.Queue()
    process = context.Process(
        target=acquire_then_exit,
        args=(
            str(database),
            request.model_dump(),
            "crashed-worker",
            "crashed-attempt",
            1.2,
            output,
        ),
    )
    process_started = False
    try:
        process.start()
        process_started = True
        first = spawn_result(output)
        join_processes((process,), timeout=10)
    finally:
        if process_started:
            terminate_processes((process,), timeout=10)
    assert first["lease"]["fence"] == 1
    first_expiry = datetime.fromisoformat(first["lease"]["expires_at"])
    wait_for_sqlite_authority_after(database, first_expiry)

    takeovers = run_spawned_authority_contention(
        database,
        start_method="spawn",
    )
    assert sorted(result["outcome"] for result in takeovers) == [
        AcquireOutcome.ACQUIRED.value,
        AcquireOutcome.BUSY.value,
    ]
    acquired = next(
        result
        for result in takeovers
        if result["outcome"] == AcquireOutcome.ACQUIRED
    )
    busy = next(
        result
        for result in takeovers
        if result["outcome"] == AcquireOutcome.BUSY
    )
    assert acquired["lease"]["fence"] == 2
    assert busy["busy_expires_at"] == acquired["lease"]["expires_at"]
