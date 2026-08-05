"""Explicit clean-interpreter scenarios for SQLite Tool admission."""

from __future__ import annotations

import multiprocessing
import sqlite3
from pathlib import Path
from typing import Any, cast

from tests.optimization.processes import (
    join_processes,
    terminate_processes,
)
from tests.optimization.tools.store_spawn import (
    admit_once,
)
from tests.optimization.tools.support import (
    capacity_binding,
    sqlite_store,
    tool_config,
)
from whetstone.optimization.tools.contracts import (
    ToolCapacityScope,
    ToolConfig,
)

FULL_A = "a" * 64
FULL_B = "b" * 64


def run_spawned_sqlite_admissions(
    database: Path,
    config: ToolConfig,
    calls: tuple[tuple[str, str], ...],
    *,
    start_method: str,
    hold_transaction: bool,
) -> list[dict[str, Any]]:
    context = cast(Any, multiprocessing.get_context(start_method))
    queue = context.Queue()
    start = context.Event()
    ready = [context.Event() for _ in calls]
    attempted = [context.Event() for _ in calls]
    acquired = [context.Event() for _ in calls]
    processes = [
        context.Process(
            target=admit_once,
            args=(
                str(
                    database.with_name(
                        f"{database.stem}-objects-{index}.sqlite"
                    )
                ),
                str(database),
                config.model_dump(mode="json"),
                call_id,
                template,
                ready[index],
                start,
                attempted[index],
                acquired[index],
                queue,
            ),
        )
        for index, (call_id, template) in enumerate(calls)
    ]
    started: list[Any] = []
    coordinator: sqlite3.Connection | None = None
    try:
        for process in processes:
            process.start()
            started.append(process)
        assert all(signal.wait(timeout=30) for signal in ready)
        if hold_transaction:
            coordinator = sqlite3.connect(database, isolation_level=None)
            coordinator.execute("BEGIN IMMEDIATE")
        start.set()
        assert all(signal.wait(timeout=30) for signal in attempted)
        if coordinator is not None:
            assert not any(signal.is_set() for signal in acquired)
            coordinator.rollback()
            coordinator.close()
            coordinator = None
        assert all(signal.wait(timeout=30) for signal in acquired)
        records = [queue.get(timeout=30) for _ in processes]
        join_processes(processes, timeout=30)
        return records
    finally:
        start.set()
        if coordinator is not None:
            coordinator.rollback()
            coordinator.close()
        terminate_processes(started, timeout=30)


def run_sqlite_capacity_race(
    tmp_path: Path,
    start_method: str,
) -> None:
    database = tmp_path / "race.sqlite"
    config = tool_config(capacity=4)
    # Initialize tables before processes start; each process still opens fully
    # independent ObjectStore and admission-authority instances.
    sqlite_store(database)
    records = run_spawned_sqlite_admissions(
        database,
        config,
        tuple((f"call-{index}", f"template-{index}") for index in range(12)),
        start_method=start_method,
        hold_transaction=True,
    )
    assert not [record for record in records if "error" in record]
    accepted = [record for record in records if record["state"] == "accepted"]
    refused = [record for record in records if record["state"] == "refused"]
    assert len(accepted) == 4
    assert len(refused) == 8
    assert sorted(record["ordinal"] for record in accepted) == [1, 2, 3, 4]
    assert (
        sqlite_store(database).accepted_count(
            config, capacity_binding(ToolCapacityScope.RUN)
        )
        == 4
    )


def run_sqlite_global_capacity(
    tmp_path: Path,
    start_method: str,
) -> None:
    database = tmp_path / "global-race.sqlite"
    config = tool_config(capacity=1, scope=ToolCapacityScope.GLOBAL)
    sqlite_store(database)
    records = run_spawned_sqlite_admissions(
        database,
        config,
        tuple((f"global-{index}", f"template-{index}") for index in range(8)),
        start_method=start_method,
        hold_transaction=False,
    )
    assert not [record for record in records if "error" in record]
    assert sum(record["state"] == "accepted" for record in records) == 1
    assert sum(record["state"] == "refused" for record in records) == 7
    assert (
        sqlite_store(database).accepted_count(
            config,
            capacity_binding(ToolCapacityScope.GLOBAL),
        )
        == 1
    )


def run_sqlite_same_call_replay(
    tmp_path: Path,
    start_method: str,
) -> None:
    database = tmp_path / "same.sqlite"
    config = tool_config(capacity=4)
    sqlite_store(database)
    records = run_spawned_sqlite_admissions(
        database,
        config,
        (("same", "same-template"),) * 6,
        start_method=start_method,
        hold_transaction=False,
    )
    assert records == [{"state": "accepted", "ordinal": 1} for _ in range(6)]
    assert (
        sqlite_store(database).accepted_count(
            config, capacity_binding(ToolCapacityScope.RUN)
        )
        == 1
    )
