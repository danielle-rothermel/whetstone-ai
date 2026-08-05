from __future__ import annotations

import multiprocessing
import os
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from datetime import timedelta
from typing import Any, LiteralString, cast
from uuid import uuid4

import pytest

from tests.core.effects.authority_spawn import (
    postgresql_acquire_and_succeed_once,
    race_postgresql_acquire,
    replay_postgresql_effect_once,
    spawn_result,
)
from tests.core.effects.authority_support import (
    _NOW,
    _acquire,
    _request,
    _result_ref,
)
from tests.optimization.processes import join_processes, terminate_processes
from tests.postgres import (
    PostgresTestSchema,
    isolated_postgres_schema,
    require_postgres_lock_wait,
)
from whetstone.core.effects import _postgres as postgres_authority_module
from whetstone.core.effects.authority import (
    AcquireOutcome,
    EffectAuthority,
    EffectAuthorityError,
    EffectAuthoritySchemaMismatchError,
)


class _RecordingCursor:
    def __init__(self, recorder: _PostgresRecorder) -> None:
        self._recorder = recorder
        self._one: tuple[Any, ...] | None = None
        self._many: list[tuple[Any, ...]] = []
        self.rowcount = -1

    def execute(
        self, query: str, params: tuple[Any, ...] | None = None
    ) -> None:
        self._recorder.queries.append((query, params))
        normalized = " ".join(query.split())
        self._one = None
        self._many = []
        self.rowcount = -1
        if normalized == "SHOW server_encoding":
            self._one = (self._recorder.server_encoding,)
        elif "FROM information_schema.tables" in query:
            self._many = [(name,) for name in sorted(self._recorder.tables)]
        elif normalized.startswith("CREATE TABLE"):
            self._recorder.tables.add(normalized.split()[2])
        elif "FROM information_schema.columns" in query:
            self._many = [
                (table, *column)
                for table in sorted(self._recorder.tables)
                for column in self._recorder.columns[table]
            ]
        elif "FROM pg_catalog.pg_constraint" in query:
            self._many = [
                row
                for row in self._recorder.constraints
                if row[0] in self._recorder.tables
            ]
        elif (
            "SELECT singleton, schema_version" in query and "metadata" in query
        ):
            self._many = [(1, self._recorder.schema_version)]
        elif (
            normalized.startswith(
                "INSERT INTO whetstone_effect_authority_metadata"
            )
            and params is not None
        ):
            self._recorder.schema_version = int(params[1])
        elif "SELECT request_identity_hash" in query:
            self._one = self._recorder.row
        elif "clock_timestamp()" in query:
            self._one = (
                self._recorder.now.isoformat(timespec="microseconds"),
            )
        elif "RETURNING 1" in query:
            assert params is not None
            if self._recorder.row is None:
                self._recorder.row = tuple(params[1:])
                self._one = (1,)
        elif normalized.startswith("UPDATE"):
            assert params is not None
            if self._recorder.row == tuple(params[9:]):
                self._recorder.row = tuple(params[:8])
                self.rowcount = 1
            else:
                self.rowcount = 0

    def fetchone(self) -> tuple[Any, ...] | None:
        value = self._one
        self._one = None
        return value

    def fetchall(self) -> list[tuple[Any, ...]]:
        values = self._many
        self._many = []
        return values

    def __enter__(self) -> _RecordingCursor:
        return self

    def __exit__(self, *args: object) -> None:
        pass


class _RecordingConnection:
    def __init__(self, recorder: _PostgresRecorder) -> None:
        self._recorder = recorder

    def cursor(self) -> _RecordingCursor:
        return _RecordingCursor(self._recorder)

    def __enter__(self) -> _RecordingConnection:
        self._recorder.entered += 1
        return self

    def __exit__(self, *args: object) -> None:
        self._recorder.exited += 1


class _PostgresRecorder:
    def __init__(self) -> None:
        self.queries: list[tuple[str, tuple[Any, ...] | None]] = []
        self.tables: set[str] = set()
        self.schema_version = 2
        self.server_encoding = "UTF8"
        text_collation = ("pg_catalog", "C", "c", True, -1)
        no_collation = (None, None, None, None, None)
        self.columns: dict[str, tuple[tuple[Any, ...], ...]] = {
            "whetstone_effect_authority": (
                ("semantic_key", "text", "NO", 1, *text_collation),
                (
                    "request_identity_hash",
                    "text",
                    "NO",
                    2,
                    *text_collation,
                ),
                ("replay_policy", "text", "NO", 3, *text_collation),
                ("state", "text", "NO", 4, *text_collation),
                ("owner_id", "text", "NO", 5, *text_collation),
                ("attempt_id", "text", "NO", 6, *text_collation),
                ("fence", "bigint", "NO", 7, *no_collation),
                ("expires_at", "text", "YES", 8, *text_collation),
                ("terminal_json", "text", "YES", 9, *text_collation),
            ),
            "whetstone_effect_authority_metadata": (
                ("singleton", "integer", "NO", 1, *no_collation),
                ("schema_version", "integer", "NO", 2, *no_collation),
            ),
        }
        self.row: tuple[Any, ...] | None = None
        self.constraints: list[tuple[Any, ...]] = [
            (
                "whetstone_effect_authority",
                "c",
                ("state", "expires_at", "terminal_json"),
                "state = 'leased'::text AND expires_at IS NOT NULL AND "
                "terminal_json IS NULL OR state <> 'leased'::text AND "
                "expires_at IS NULL AND terminal_json IS NOT NULL",
                False,
                False,
                True,
                False,
            ),
            (
                "whetstone_effect_authority",
                "c",
                ("fence",),
                "fence > 0",
                False,
                False,
                True,
                False,
            ),
            (
                "whetstone_effect_authority",
                "c",
                ("replay_policy",),
                "replay_policy = ANY (ARRAY['idempotent'::text, "
                "'durable_workflow'::text, 'no_redrive'::text])",
                False,
                False,
                True,
                False,
            ),
            (
                "whetstone_effect_authority",
                "c",
                ("state",),
                "state = ANY (ARRAY['leased'::text, 'succeeded'::text, "
                "'failed'::text, 'recovery_required'::text])",
                False,
                False,
                True,
                False,
            ),
            (
                "whetstone_effect_authority",
                "p",
                ("semantic_key",),
                None,
                False,
                False,
                True,
                True,
            ),
            (
                "whetstone_effect_authority_metadata",
                "c",
                ("singleton",),
                "singleton = 1",
                False,
                False,
                True,
                False,
            ),
            (
                "whetstone_effect_authority_metadata",
                "p",
                ("singleton",),
                None,
                False,
                False,
                True,
                True,
            ),
        ]
        self.now = _NOW
        self.entered = 0
        self.exited = 0

    def connect(
        self, dsn: str
    ) -> AbstractContextManager[_RecordingConnection]:
        assert dsn == "postgresql://authority-test"
        return _RecordingConnection(self)

    def advance(self, duration: timedelta) -> None:
        self.now += duration


def test_postgresql_adapter_verifies_schema_and_uses_database_time() -> None:
    recorder = _PostgresRecorder()
    authority = EffectAuthority.postgresql(
        "postgresql://authority-test",
        _connect=recorder.connect,
    )
    acquired = _acquire(
        authority,
        _request(),
        duration=timedelta(minutes=5),
    )
    assert acquired.lease is not None
    recorder.advance(timedelta(minutes=1))
    renewed = authority.renew(
        acquired.lease,
        lease_duration=timedelta(minutes=5),
    )
    terminal = authority.succeed(
        renewed,
        result_ref=_result_ref("postgresql-result"),
    )
    replay = _acquire(
        authority,
        _request(),
        owner="worker-2",
        attempt="attempt-2",
        duration=timedelta(minutes=5),
    )
    assert replay.terminal == terminal
    assert recorder.entered == recorder.exited == 5
    statements = "\n".join(query for query, _ in recorder.queries)
    assert "CREATE TABLE whetstone_effect_authority" in statements
    assert "CREATE TABLE IF NOT EXISTS" not in statements
    assert "whetstone_effect_authority_metadata" in statements
    assert "information_schema.columns" in statements
    assert "pg_advisory_xact_lock(1465141076, 1)" in statements
    assert "clock_timestamp()" in statements
    assert "FOR UPDATE" in statements
    assert "ON CONFLICT (semantic_key) DO NOTHING" in statements
    assert "IS NOT DISTINCT FROM" in statements


def test_postgresql_adapter_requires_utf8_server_encoding() -> None:
    recorder = _PostgresRecorder()
    recorder.server_encoding = "SQL_ASCII"

    with pytest.raises(
        EffectAuthoritySchemaMismatchError,
        match=r"requires exact server_encoding 'UTF8'.*SQL_ASCII",
    ):
        EffectAuthority.postgresql(
            "postgresql://authority-test",
            _connect=recorder.connect,
        )

    assert [" ".join(query.split()) for query, _ in recorder.queries] == [
        "SHOW server_encoding"
    ]


def test_postgresql_adapter_rejects_non_c_text_collation() -> None:
    recorder = _PostgresRecorder()
    recorder.tables = {
        "whetstone_effect_authority",
        "whetstone_effect_authority_metadata",
    }
    columns = list(recorder.columns["whetstone_effect_authority"])
    columns[0] = (
        "semantic_key",
        "text",
        "NO",
        1,
        "public",
        "case_insensitive",
        "i",
        False,
        -1,
    )
    recorder.columns["whetstone_effect_authority"] = tuple(columns)

    with pytest.raises(
        EffectAuthoritySchemaMismatchError,
        match=r"incompatible PostgreSQL effect-authority columns.*"
        r"case_insensitive",
    ):
        EffectAuthority.postgresql(
            "postgresql://authority-test",
            _connect=recorder.connect,
        )


def test_postgresql_adapter_rejects_wrong_schema_version() -> None:
    recorder = _PostgresRecorder()
    recorder.tables = {
        "whetstone_effect_authority",
        "whetstone_effect_authority_metadata",
    }
    recorder.schema_version = 3
    with pytest.raises(EffectAuthorityError, match="schema version"):
        EffectAuthority.postgresql(
            "postgresql://authority-test",
            _connect=recorder.connect,
        )


def test_postgresql_adapter_rejects_missing_primary_key() -> None:
    recorder = _PostgresRecorder()
    recorder.tables = {
        "whetstone_effect_authority",
        "whetstone_effect_authority_metadata",
    }
    recorder.constraints = [
        row
        for row in recorder.constraints
        if not (row[0] == "whetstone_effect_authority" and row[1] == "p")
    ]

    with pytest.raises(
        EffectAuthoritySchemaMismatchError,
        match=r"missing .*PRIMARY KEY \(semantic_key\)",
    ):
        EffectAuthority.postgresql(
            "postgresql://authority-test",
            _connect=recorder.connect,
        )


def test_postgresql_adapter_rejects_wrong_check_constraint() -> None:
    recorder = _PostgresRecorder()
    recorder.tables = {
        "whetstone_effect_authority",
        "whetstone_effect_authority_metadata",
    }
    recorder.constraints = [
        (
            *row[:3],
            "replay_policy <> ''::text",
            *row[4:],
        )
        if row[0] == "whetstone_effect_authority"
        and row[2] == ("replay_policy",)
        else row
        for row in recorder.constraints
    ]

    with pytest.raises(
        EffectAuthoritySchemaMismatchError,
        match=r"missing .*CHECK .*replay_policy.*unexpected .*<>",
    ):
        EffectAuthority.postgresql(
            "postgresql://authority-test",
            _connect=recorder.connect,
        )


def _run_spawned_postgresql_authority_contention(
    schema: PostgresTestSchema,
) -> list[dict[str, Any]]:
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    release = context.Event()
    starts = [context.Event() for _ in range(2)]
    ready = [context.Event() for _ in range(2)]
    query_reached = [context.Event() for _ in range(2)]
    backend_pids = [context.Value("i", 0) for _ in range(2)]
    request = _request(key=f"postgres-effect:{uuid4()}")
    processes = [
        context.Process(
            target=race_postgresql_acquire,
            args=(
                schema.dsn,
                schema.name,
                request.model_dump(mode="json"),
                "shared-worker",
                attempt_id,
                role,
                ready[index],
                starts[index],
                query_reached[index],
                release,
                backend_pids[index],
                output,
            ),
        )
        for index, (attempt_id, role) in enumerate(
            (("attempt-1", "holder"), ("attempt-2", "contender"))
        )
    ]
    started: list[Any] = []
    try:
        for process in processes:
            process.start()
            started.append(process)
        assert all(signal.wait(timeout=30) for signal in ready)
        starts[0].set()
        assert query_reached[0].wait(timeout=30)
        starts[1].set()
        assert query_reached[1].wait(timeout=30)
        require_postgres_lock_wait(schema, backend_pids[1].value)
        release.set()
        results = [spawn_result(output, timeout=30) for _ in processes]
        join_processes(processes, timeout=30)
        return results
    finally:
        for start in starts:
            start.set()
        release.set()
        terminate_processes(started, timeout=30)


def test_spawned_postgresql_same_effect_arbitrates_once() -> None:
    with isolated_postgres_schema("effect_race") as schema:
        results = _run_spawned_postgresql_authority_contention(schema)

    assert not [result for result in results if "error" in result]
    assert sorted(result["outcome"] for result in results) == [
        AcquireOutcome.ACQUIRED.value,
        AcquireOutcome.BUSY.value,
    ]


def test_postgresql_terminal_replays_from_fresh_process() -> None:
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    request = _request(key=f"postgres-terminal:{uuid4()}")
    result_ref = _result_ref("postgresql-test-result")
    started: list[Any] = []
    with isolated_postgres_schema("effect_terminal") as schema:
        writer = context.Process(
            target=postgresql_acquire_and_succeed_once,
            args=(
                schema.dsn,
                schema.name,
                request.model_dump(mode="json"),
                result_ref.model_dump(mode="json"),
                output,
            ),
        )
        reader = context.Process(
            target=replay_postgresql_effect_once,
            args=(
                schema.dsn,
                schema.name,
                request.model_dump(mode="json"),
                output,
            ),
        )
        try:
            writer.start()
            started.append(writer)
            terminal = spawn_result(output, timeout=30)
            join_processes((writer,), timeout=30)
            assert "error" not in terminal

            reader.start()
            started.append(reader)
            replay = spawn_result(output, timeout=30)
            join_processes((reader,), timeout=30)
            assert "error" not in replay
            assert replay["outcome"] == AcquireOutcome.SUCCEEDED.value
            assert replay["terminal"] == terminal
        finally:
            terminate_processes(started, timeout=30)


@pytest.mark.skipif(
    "WHETSTONE_TEST_POSTGRES_DSN" not in os.environ,
    reason="WHETSTONE_TEST_POSTGRES_DSN is required for live collation checks",
)
def test_postgresql_17_rejects_case_insensitive_authority_schema() -> None:
    from psycopg import connect
    from psycopg.sql import SQL, Identifier

    dsn = os.environ["WHETSTONE_TEST_POSTGRES_DSN"]
    schema = f"effect_ci_{uuid4().hex}"

    @contextmanager
    def connect_in_schema(
        configured_dsn: str,
    ) -> Iterator[Any]:
        with connect(configured_dsn) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    SQL("SET LOCAL search_path TO {}, pg_catalog").format(
                        Identifier(schema)
                    )
                )
            yield connection

    with connect(dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SHOW server_version_num")
            version = cursor.fetchone()
            assert version is not None
            if int(version[0]) // 10_000 != 17:
                pytest.skip(
                    "live case-insensitive check requires PostgreSQL 17"
                )
            cursor.execute(SQL("CREATE SCHEMA {}").format(Identifier(schema)))
            cursor.execute(
                SQL(
                    """
                CREATE COLLATION {}.case_insensitive (
                    provider = icu,
                    locale = 'und-u-ks-level2',
                    deterministic = false
                )
                """
                ).format(Identifier(schema))
            )
            cursor.execute(
                SQL("SET LOCAL search_path TO {}, pg_catalog").format(
                    Identifier(schema)
                )
            )
            cursor.execute(
                SQL(
                    cast(
                        LiteralString,
                        postgres_authority_module._POSTGRES_CREATE_TABLE.replace(
                            'COLLATE "C"',
                            f'COLLATE "{schema}".case_insensitive',
                        ),
                    )
                )
            )
            cursor.execute(
                postgres_authority_module._POSTGRES_CREATE_METADATA_TABLE
            )
            cursor.execute(
                """
                INSERT INTO whetstone_effect_authority_metadata (
                    singleton, schema_version
                ) VALUES (%s, %s)
                """,
                (1, postgres_authority_module._SCHEMA_VERSION),
            )

    try:
        with pytest.raises(
            EffectAuthoritySchemaMismatchError,
            match=r"incompatible PostgreSQL effect-authority columns.*"
            r"case_insensitive",
        ):
            EffectAuthority.postgresql(
                dsn,
                _connect=connect_in_schema,
            )
    finally:
        with connect(dsn) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    SQL("DROP SCHEMA {} CASCADE").format(Identifier(schema))
                )
