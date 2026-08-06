from __future__ import annotations

import multiprocessing
from functools import partial
from pathlib import Path
from typing import Any, ClassVar

from dr_store import (
    ObjectStore,
    SqliteBackend,
)

from tests.optimization.processes import (
    join_processes,
    terminate_processes,
)
from tests.optimization.tools.store_spawn import (
    admit_postgresql_once,
)
from tests.postgres import (
    PostgresTestSchema,
    connect_in_postgres_schema,
    require_postgres_lock_wait,
)
from whetstone.core.effects.authority import (
    EffectAuthority,
)
from whetstone.optimization.tools import _postgres as postgres_store_module
from whetstone.optimization.tools.contracts import (
    ToolConfig,
)
from whetstone.optimization.tools.facade import (
    ToolAdmissionAuthority,
    ToolCallStore,
)

FULL_A = "a" * 64
FULL_B = "b" * 64


class PostgresCursor:
    _columns: ClassVar[dict[str, list[tuple[Any, ...]]]] = {
        "whetstone_tool_admission_schema": [
            (
                "component",
                "text",
                "NO",
                "pg_catalog",
                "C",
                "c",
                True,
                -1,
            ),
            ("version", "bigint", "NO", None, None, None, None, None),
        ],
        "whetstone_tool_admission_entry": [
            (
                "store_namespace_key",
                "text",
                "NO",
                "pg_catalog",
                "C",
                "c",
                True,
                -1,
            ),
            (
                "call_id",
                "text",
                "NO",
                "pg_catalog",
                "C",
                "c",
                True,
                -1,
            ),
            (
                "entry_json",
                "text",
                "NO",
                "pg_catalog",
                "C",
                "c",
                True,
                -1,
            ),
        ],
        "whetstone_tool_admission_capacity": [
            (
                "store_namespace_key",
                "text",
                "NO",
                "pg_catalog",
                "C",
                "c",
                True,
                -1,
            ),
            (
                "tool_config_hash",
                "text",
                "NO",
                "pg_catalog",
                "C",
                "c",
                True,
                -1,
            ),
            (
                "capacity_scope",
                "text",
                "NO",
                "pg_catalog",
                "C",
                "c",
                True,
                -1,
            ),
            (
                "capacity_scope_id",
                "text",
                "NO",
                "pg_catalog",
                "C",
                "c",
                True,
                -1,
            ),
            (
                "max_accepted_calls",
                "bigint",
                "NO",
                None,
                None,
                None,
                None,
                None,
            ),
            (
                "consumed",
                "bigint",
                "NO",
                None,
                None,
                None,
                None,
                None,
            ),
        ],
    }

    def __init__(self, recorder: PostgresRecorder) -> None:
        self._recorder = recorder
        self._rows: list[tuple[Any, ...]] = []
        self.rowcount = -1

    def execute(
        self,
        query: str,
        params: tuple[Any, ...] | None = None,
    ) -> None:
        self._recorder.queries.append((query, params))
        self._rows = []
        self.rowcount = -1
        normalized = " ".join(query.split())
        if normalized == "SHOW server_encoding":
            self._rows = [(self._recorder.server_encoding,)]
        elif "FROM information_schema.tables" in normalized:
            self._rows = [(table,) for table in sorted(self._recorder.tables)]
        elif normalized.startswith("CREATE TABLE IF NOT EXISTS"):
            self._recorder.tables.add(normalized.split()[5])
        elif "FROM information_schema.columns" in normalized:
            assert params is not None
            self._rows = list(self._recorder.columns[str(params[0])])
        elif "FROM pg_catalog.pg_constraint" in normalized:
            self._rows = [
                constraint
                for constraint in self._recorder.constraints
                if constraint[0] in self._recorder.tables
            ]
        elif normalized.startswith(
            "SELECT version FROM whetstone_tool_admission_schema"
        ):
            if self._recorder.schema_version is not None:
                self._rows = [(self._recorder.schema_version,)]
        elif normalized.startswith(
            "INSERT INTO whetstone_tool_admission_schema"
        ):
            assert params == ("tool_admission", 2)
            self._recorder.schema_version = 2
        elif normalized.startswith(
            "SELECT component, version FROM whetstone_tool_admission_schema"
        ):
            if self._recorder.schema_version is not None:
                self._rows = [
                    ("tool_admission", self._recorder.schema_version)
                ]
        elif normalized.startswith(
            "SELECT entry_json FROM whetstone_tool_admission_entry"
        ):
            assert params is not None
            entry = self._recorder.entries.get(
                (str(params[0]), str(params[1]))
            )
            if entry is not None:
                self._rows = [(entry,)]
        elif normalized.startswith(
            "INSERT INTO whetstone_tool_admission_capacity"
        ):
            assert params is not None
            scope = tuple(str(value) for value in params[:4])
            self._recorder.capacity.setdefault(
                scope,
                (int(params[4]), 0),
            )
        elif normalized.startswith(
            "SELECT max_accepted_calls, consumed "
            "FROM whetstone_tool_admission_capacity"
        ):
            assert params is not None
            scope = tuple(str(value) for value in params)
            self._rows = [self._recorder.capacity[scope]]
        elif normalized.startswith(
            "UPDATE whetstone_tool_admission_capacity SET consumed"
        ):
            assert params is not None
            scope = tuple(str(value) for value in params[1:])
            maximum, _ = self._recorder.capacity[scope]
            self._recorder.capacity[scope] = (maximum, int(params[0]))
            self.rowcount = 1
        elif normalized.startswith(
            "INSERT INTO whetstone_tool_admission_entry"
        ):
            assert params is not None
            key = (str(params[0]), str(params[1]))
            self._recorder.entries[key] = str(params[2])
            self.rowcount = 1

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._rows.pop(0) if self._rows else None

    def fetchall(self) -> list[tuple[Any, ...]]:
        rows = self._rows
        self._rows = []
        return rows

    def __enter__(self) -> PostgresCursor:
        return self

    def __exit__(self, *args: object) -> None:
        pass


class PostgresConnection:
    def __init__(self, recorder: PostgresRecorder) -> None:
        self._recorder = recorder

    def cursor(self) -> PostgresCursor:
        return PostgresCursor(self._recorder)

    def __enter__(self) -> PostgresConnection:
        self._recorder.entered += 1
        return self

    def __exit__(self, *args: object) -> None:
        self._recorder.exited += 1


class PostgresRecorder:
    def __init__(self) -> None:
        self.queries: list[tuple[str, tuple[Any, ...] | None]] = []
        self.schema_version: int | None = None
        self.tables: set[str] = set()
        self.columns = {
            table: list(columns)
            for table, columns in PostgresCursor._columns.items()
        }
        self.constraints = list(postgres_store_module._POSTGRES_CONSTRAINTS)
        self.server_encoding = "UTF8"
        self.entries: dict[tuple[str, str], str] = {}
        self.capacity: dict[
            tuple[str, ...],
            tuple[int, int],
        ] = {}
        self.entered = 0
        self.exited = 0

    def connect(self, dsn: str) -> PostgresConnection:
        assert dsn == "postgresql://tool-admission-test"
        return PostgresConnection(self)


def postgresql_store(
    object_database: Path,
    schema: PostgresTestSchema,
) -> ToolCallStore:
    connect_in_schema = partial(
        connect_in_postgres_schema,
        schema=schema.name,
    )
    return ToolCallStore(
        ObjectStore(SqliteBackend(object_database)),
        ToolAdmissionAuthority.postgresql(
            schema.dsn,
            _connect=connect_in_schema,
        ),
        EffectAuthority.postgresql(
            schema.dsn,
            _connect=connect_in_schema,
        ),
    )


def run_spawned_postgresql_admissions(
    tmp_path: Path,
    schema: PostgresTestSchema,
    config: ToolConfig,
    calls: tuple[tuple[str, str], ...],
    *,
    contender_role: str,
) -> list[dict[str, Any]]:
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    release = context.Event()
    starts = [context.Event() for _ in calls]
    ready = [context.Event() for _ in calls]
    query_reached = [context.Event() for _ in calls]
    backend_pids = [context.Value("i", 0) for _ in calls]
    roles = ("holder",) + (contender_role,) * (len(calls) - 1)
    processes = [
        context.Process(
            target=admit_postgresql_once,
            args=(
                str(tmp_path / f"postgres-objects-{index}.sqlite"),
                schema.dsn,
                schema.name,
                config.model_dump(mode="json"),
                call_id,
                template,
                roles[index],
                ready[index],
                starts[index],
                query_reached[index],
                release,
                backend_pids[index],
                queue,
            ),
        )
        for index, (call_id, template) in enumerate(calls)
    ]
    started: list[Any] = []
    try:
        for process in processes:
            process.start()
            started.append(process)
        assert all(signal.wait(timeout=30) for signal in ready)
        starts[0].set()
        assert query_reached[0].wait(timeout=30)
        for start in starts[1:]:
            start.set()
        assert all(signal.wait(timeout=30) for signal in query_reached[1:])
        for backend_pid in backend_pids[1:]:
            require_postgres_lock_wait(schema, backend_pid.value)
        release.set()
        records = [queue.get(timeout=30) for _ in processes]
        join_processes(processes, timeout=30)
        return records
    finally:
        for start in starts:
            start.set()
        release.set()
        terminate_processes(started, timeout=30)
