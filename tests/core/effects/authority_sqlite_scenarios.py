from __future__ import annotations

import multiprocessing
import sqlite3
from pathlib import Path
from typing import Any, cast

from tests.core.effects.authority_spawn import race_acquire, spawn_result
from tests.core.effects.authority_support import _request
from tests.optimization.processes import join_processes, terminate_processes
from whetstone.core.effects.authority import AcquireOutcome, EffectAuthority


def run_spawned_authority_contention(
    database: Path,
    *,
    start_method: str,
) -> list[dict[str, Any]]:
    context = cast(Any, multiprocessing.get_context(start_method))
    EffectAuthority.sqlite(database)
    payload = _request().model_dump()
    start = context.Event()
    output = context.Queue()
    ready = [context.Event() for _ in range(2)]
    attempted = [context.Event() for _ in range(2)]
    acquired = [context.Event() for _ in range(2)]
    # The process checks finish before this lease, so scheduling cannot
    # create a second owner.
    processes = [
        context.Process(
            target=race_acquire,
            args=(
                str(database),
                payload,
                "shared-worker",
                attempt,
                60.0,
                ready[index],
                start,
                attempted[index],
                acquired[index],
                output,
            ),
        )
        for index, attempt in enumerate(("attempt-1", "attempt-2"))
    ]
    coordinator = sqlite3.connect(database, isolation_level=None)
    started: list[Any] = []
    try:
        for process in processes:
            process.start()
            started.append(process)
        assert all(signal.wait(timeout=10) for signal in ready)
        coordinator.execute("BEGIN IMMEDIATE")
        start.set()
        assert all(signal.wait(timeout=10) for signal in attempted)
        assert not any(signal.is_set() for signal in acquired)
        coordinator.rollback()
        assert all(signal.wait(timeout=10) for signal in acquired)
        results = [spawn_result(output) for _ in processes]
        assert not [result for result in results if "error" in result]
        join_processes(processes, timeout=10)
        return results
    finally:
        coordinator.rollback()
        coordinator.close()
        start.set()
        terminate_processes(started, timeout=10)


def run_spawned_same_owner_different_attempts_arbitrate_once(
    root: Path,
    start_method: str,
) -> None:
    results = run_spawned_authority_contention(
        root / "race.sqlite",
        start_method=start_method,
    )
    assert sorted(result["outcome"] for result in results) == [
        AcquireOutcome.ACQUIRED.value,
        AcquireOutcome.BUSY.value,
    ]
