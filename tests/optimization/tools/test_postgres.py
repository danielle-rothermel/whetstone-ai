"""PostgreSQL-specific Tool admission schema and contention tests."""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any, LiteralString, cast
from uuid import uuid4

import pytest
from dr_store import (
    ObjectStore,
    SqliteBackend,
)

from tests.optimization.processes import (
    join_processes,
    terminate_processes,
)
from tests.optimization.tools.postgres_support import (
    PostgresRecorder,
    postgresql_store,
    run_spawned_postgresql_admissions,
)
from tests.optimization.tools.store_spawn import (
    load_postgresql_terminal_result_once,
)
from tests.optimization.tools.support import (
    capacity_binding,
    successful_result,
    tool_call,
    tool_config,
)
from tests.postgres import (
    isolated_postgres_schema,
)
from whetstone.core.effects.authority import (
    EffectAuthority,
)
from whetstone.optimization.tools import _postgres as postgres_store_module
from whetstone.optimization.tools import admission as admission_store_module
from whetstone.optimization.tools.admission import (
    ToolAdmissionSchemaMismatchError,
    ToolCallState,
    tool_effect_request,
)
from whetstone.optimization.tools.contracts import (
    GLOBAL_CAPACITY_SCOPE_ID,
    ToolCapacityScope,
)
from whetstone.optimization.tools.facade import (
    ToolAdmissionAuthority,
    ToolCallStore,
)

FULL_A = "a" * 64
FULL_B = "b" * 64


def test_postgresql_initialization_rejects_truncated_table() -> None:
    recorder = PostgresRecorder()
    recorder.columns["whetstone_tool_admission_entry"].pop()

    with pytest.raises(
        RuntimeError,
        match="incompatible Tool admission table "
        "'whetstone_tool_admission_entry'",
    ):
        ToolAdmissionAuthority.postgresql(
            "postgresql://tool-admission-test",
            _connect=recorder.connect,
        )


def test_postgresql_initialization_rejects_unaudited_capacity_table() -> None:
    recorder = PostgresRecorder()
    recorder.tables = {"whetstone_tool_admission_capacity"}
    recorder.capacity[("namespace", "config", "run", "run-1")] = (2, 1)

    with pytest.raises(
        ToolAdmissionSchemaMismatchError,
        match=r"owned table inventory.*whetstone_tool_admission_capacity",
    ):
        ToolAdmissionAuthority.postgresql(
            "postgresql://tool-admission-test",
            _connect=recorder.connect,
        )

    statements = [" ".join(query.split()) for query, _ in recorder.queries]
    assert not any(
        statement.startswith("CREATE TABLE") for statement in statements
    )
    assert recorder.capacity == {
        ("namespace", "config", "run", "run-1"): (2, 1)
    }


def test_postgresql_initialization_requires_utf8_server_encoding() -> None:
    recorder = PostgresRecorder()
    recorder.server_encoding = "SQL_ASCII"

    with pytest.raises(
        ToolAdmissionSchemaMismatchError,
        match=r"server_encoding.*UTF8.*SQL_ASCII",
    ) as raised:
        ToolAdmissionAuthority.postgresql(
            "postgresql://tool-admission-test",
            _connect=recorder.connect,
        )

    assert raised.value.table == "<database>"
    assert raised.value.aspect == "server_encoding"
    assert [" ".join(query.split()) for query, _ in recorder.queries] == [
        "SHOW server_encoding"
    ]


def test_postgresql_initialization_rejects_non_c_text_collation() -> None:
    recorder = PostgresRecorder()
    recorder.columns["whetstone_tool_admission_entry"][0] = (
        "store_namespace_key",
        "text",
        "NO",
        "public",
        "case_insensitive",
        "i",
        False,
        -1,
    )

    with pytest.raises(
        ToolAdmissionSchemaMismatchError,
        match=r"whetstone_tool_admission_entry.*columns.*case_insensitive",
    ):
        ToolAdmissionAuthority.postgresql(
            "postgresql://tool-admission-test",
            _connect=recorder.connect,
        )


@pytest.mark.parametrize(
    ("table", "columns", "replacement"),
    [
        ("whetstone_tool_admission_schema", ("version",), None),
        (
            "whetstone_tool_admission_capacity",
            ("capacity_scope",),
            "capacity_scope = ANY (ARRAY['global'::text, 'run'::text])",
        ),
        (
            "whetstone_tool_admission_capacity",
            ("max_accepted_calls",),
            "max_accepted_calls > 0",
        ),
        (
            "whetstone_tool_admission_capacity",
            ("consumed", "max_accepted_calls"),
            "consumed >= 0",
        ),
    ],
    ids=(
        "missing-version-positive",
        "wrong-capacity-scope-enum",
        "wrong-maximum-nonnegative",
        "wrong-consumed-bounds",
    ),
)
def test_postgresql_initialization_rejects_wrong_check_constraints(
    table: str,
    columns: tuple[str, ...],
    replacement: str | None,
) -> None:
    recorder = PostgresRecorder()
    check_index = next(
        index
        for index, constraint in enumerate(recorder.constraints)
        if constraint[0] == table
        and constraint[1] == "c"
        and constraint[2] == columns
    )
    if replacement is None:
        recorder.constraints.pop(check_index)
    else:
        constraint = recorder.constraints[check_index]
        recorder.constraints[check_index] = (
            *constraint[:3],
            replacement,
            *constraint[4:],
        )

    with pytest.raises(
        ToolAdmissionSchemaMismatchError,
        match=(
            rf"incompatible Tool admission table {table!r}: expected exact "
            r"PRIMARY KEY and CHECK constraints"
        ),
    ) as raised:
        ToolAdmissionAuthority.postgresql(
            "postgresql://tool-admission-test",
            _connect=recorder.connect,
        )

    assert raised.value.table == table
    assert raised.value.aspect == "PRIMARY KEY and CHECK constraints"
    assert "apply the Tool admission schema migration" in str(raised.value)


@pytest.mark.parametrize(
    ("constraint_type", "columns", "flag_index", "wrong_value"),
    [
        (
            "p",
            (
                "store_namespace_key",
                "tool_config_hash",
                "capacity_scope",
                "capacity_scope_id",
            ),
            4,
            True,
        ),
        (
            "p",
            (
                "store_namespace_key",
                "tool_config_hash",
                "capacity_scope",
                "capacity_scope_id",
            ),
            5,
            True,
        ),
        (
            "p",
            (
                "store_namespace_key",
                "tool_config_hash",
                "capacity_scope",
                "capacity_scope_id",
            ),
            6,
            False,
        ),
        (
            "p",
            (
                "store_namespace_key",
                "tool_config_hash",
                "capacity_scope",
                "capacity_scope_id",
            ),
            7,
            False,
        ),
        ("c", ("consumed", "max_accepted_calls"), 4, True),
        ("c", ("consumed", "max_accepted_calls"), 5, True),
        ("c", ("consumed", "max_accepted_calls"), 6, False),
        ("c", ("consumed", "max_accepted_calls"), 7, True),
    ],
    ids=(
        "primary-key-deferrable",
        "primary-key-initially-deferred",
        "primary-key-not-validated",
        "primary-key-wrong-no-inherit",
        "check-deferrable",
        "check-initially-deferred",
        "check-not-validated",
        "check-no-inherit",
    ),
)
def test_postgresql_initialization_rejects_wrong_constraint_flags(
    constraint_type: str,
    columns: tuple[str, ...],
    flag_index: int,
    wrong_value: bool,
) -> None:
    recorder = PostgresRecorder()
    table = "whetstone_tool_admission_capacity"
    index = next(
        index
        for index, constraint in enumerate(recorder.constraints)
        if constraint[0] == table
        and constraint[1] == constraint_type
        and constraint[2] == columns
    )
    changed = list(recorder.constraints[index])
    changed[flag_index] = wrong_value
    recorder.constraints[index] = tuple(changed)

    with pytest.raises(
        ToolAdmissionSchemaMismatchError,
        match=(
            r"whetstone_tool_admission_capacity.*"
            r"PRIMARY KEY and CHECK constraints"
        ),
    ) as raised:
        ToolAdmissionAuthority.postgresql(
            "postgresql://tool-admission-test",
            _connect=recorder.connect,
        )

    assert "deferrable=" in str(raised.value)
    assert "deferred=" in str(raised.value)
    assert "validated=" in str(raised.value)
    assert "no_inherit=" in str(raised.value)


def test_postgresql_initialization_rejects_unknown_schema_version() -> None:
    recorder = PostgresRecorder()
    recorder.schema_version = 3
    recorder.tables = {
        "whetstone_tool_admission_schema",
        "whetstone_tool_admission_entry",
        "whetstone_tool_admission_capacity",
    }

    with pytest.raises(
        ToolAdmissionSchemaMismatchError,
        match="expected exact schema version 2, found 3",
    ):
        ToolAdmissionAuthority.postgresql(
            "postgresql://tool-admission-test",
            _connect=recorder.connect,
        )


def test_postgresql_adapter_uses_versioned_schema_and_nul_free_lock(
    tmp_path,
) -> None:
    recorder = PostgresRecorder()
    authority = ToolAdmissionAuthority.postgresql(
        "postgresql://tool-admission-test",
        _connect=recorder.connect,
    )
    config = tool_config(
        capacity=1,
        namespace="postgres-adapter",
        scope=ToolCapacityScope.GLOBAL,
    )
    store = ToolCallStore(
        ObjectStore(SqliteBackend(tmp_path / "objects.sqlite")),
        authority,
        EffectAuthority.memory(),
    )

    first = store.admit(
        tool_call(config, "first", scope_id=GLOBAL_CAPACITY_SCOPE_ID),
        config,
    )
    second = store.admit(
        tool_call(config, "second", scope_id=GLOBAL_CAPACITY_SCOPE_ID),
        config,
    )

    assert first.state is ToolCallState.ACCEPTED
    assert first.capacity_debit_ordinal == 1
    assert second.state is ToolCallState.REFUSED
    assert recorder.entered == recorder.exited == 3
    statements = "\n".join(query for query, _ in recorder.queries)
    assert "CREATE TABLE IF NOT EXISTS whetstone_tool_admission_schema" in (
        statements
    )
    assert "FROM information_schema.columns" in statements
    assert "FROM pg_catalog.pg_constraint AS constraint_record" in statements
    assert "constraint_record.condeferrable" in statements
    assert "constraint_record.condeferred" in statements
    assert "constraint_record.convalidated" in statements
    assert "constraint_record.connoinherit" in statements
    assert "pg_get_expr(" in statements
    assert "consrc" not in statements
    assert "FOR UPDATE" in statements
    assert "chr(0)" not in statements
    assert "\x00" not in statements
    lock_params = [
        params
        for query, params in recorder.queries
        if " ".join(query.split()) == "SELECT pg_advisory_xact_lock(%s)"
    ]
    assert lock_params == [
        (
            postgres_store_module._entry_lock_key(
                ("postgres-adapter", "first")
            ),
        ),
        (
            postgres_store_module._entry_lock_key(
                ("postgres-adapter", "second")
            ),
        ),
    ]


def test_postgresql_entry_lock_digest_is_pinned_and_unambiguous() -> None:
    assert (
        postgres_store_module._entry_lock_key(("namespace", "call"))
        == 5219561813675110560
    )
    assert postgres_store_module._entry_lock_key(
        ("a", "bc")
    ) != postgres_store_module._entry_lock_key(("ab", "c"))


def test_spawned_postgresql_global_capacity_race_accepts_once(
    tmp_path: Path,
) -> None:
    with isolated_postgres_schema("tool_capacity") as schema:
        config = tool_config(
            capacity=1,
            namespace=f"tool-{uuid4()}",
            scope=ToolCapacityScope.GLOBAL,
        )
        records = run_spawned_postgresql_admissions(
            tmp_path,
            schema,
            config,
            (("first", "template-1"), ("second", "template-2")),
            contender_role="capacity-contender",
        )
        durable_count = postgresql_store(
            tmp_path / "capacity-count.sqlite",
            schema,
        ).accepted_count(
            config,
            capacity_binding(ToolCapacityScope.GLOBAL),
        )

    assert not [record for record in records if "error" in record]
    assert (
        sum(
            record["state"] == ToolCallState.ACCEPTED.value
            for record in records
        )
        == 1
    )
    assert (
        sum(
            record["state"] == ToolCallState.REFUSED.value
            for record in records
        )
        == 1
    )
    assert durable_count == 1


def test_spawned_postgresql_same_call_replay_has_one_ordinal(
    tmp_path: Path,
) -> None:
    with isolated_postgres_schema("tool_replay") as schema:
        config = tool_config(
            capacity=4,
            namespace=f"tool-{uuid4()}",
        )
        records = run_spawned_postgresql_admissions(
            tmp_path,
            schema,
            config,
            (("same", "same-template"),) * 4,
            contender_role="entry-contender",
        )
        durable_count = postgresql_store(
            tmp_path / "replay-count.sqlite",
            schema,
        ).accepted_count(
            config,
            capacity_binding(ToolCapacityScope.RUN),
        )

    assert records == [
        {"state": ToolCallState.ACCEPTED.value, "ordinal": 1} for _ in range(4)
    ]
    assert durable_count == 1


def test_postgresql_completed_terminal_survives_fresh_process(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    object_database = tmp_path / "postgres-terminal-objects.sqlite"
    started: list[Any] = []
    with isolated_postgres_schema("tool_terminal") as schema:
        store = postgresql_store(object_database, schema)
        config = tool_config(
            capacity=1,
            namespace=f"tool-{uuid4()}",
        )
        call = tool_call(config, "terminal-restart")
        store.admit(call, config)
        result = successful_result(call, 1)
        acquisition = store.effect_authority.acquire(
            tool_effect_request(call),
            owner_id="terminal-writer",
            attempt_id="terminal-attempt",
            lease_duration=timedelta(minutes=5),
        )
        assert acquisition.lease is not None
        terminal = store.effect_authority.succeed(
            acquisition.lease,
            result_ref=store.persist_result(result),
        )
        completed = store.complete(result, terminal=terminal)

        reader = context.Process(
            target=load_postgresql_terminal_result_once,
            args=(
                str(object_database),
                schema.dsn,
                schema.name,
                call.model_dump(mode="json"),
                queue,
            ),
        )
        try:
            reader.start()
            started.append(reader)
            record = queue.get(timeout=30)
            join_processes((reader,), timeout=30)
            assert "error" not in record
            assert record["entry"] == completed.model_dump(mode="json")
            assert record["result"] == result.model_dump(mode="json")
        finally:
            terminate_processes(started, timeout=30)


@pytest.mark.skipif(
    "WHETSTONE_TEST_POSTGRES_DSN" not in os.environ,
    reason="WHETSTONE_TEST_POSTGRES_DSN is required for live collation checks",
)
def test_postgresql_17_rejects_case_insensitive_admission_schema() -> None:
    from psycopg import connect
    from psycopg.sql import SQL, Identifier

    dsn = os.environ["WHETSTONE_TEST_POSTGRES_DSN"]
    schema = f"tool_ci_{uuid4().hex}"

    @contextmanager
    def connect_in_schema(configured_dsn: str) -> Iterator[Any]:
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
            for create_sql in (
                postgres_store_module._POSTGRES_CREATE_SCHEMA,
                postgres_store_module._POSTGRES_CREATE_ENTRY,
                postgres_store_module._POSTGRES_CREATE_CAPACITY,
            ):
                cursor.execute(
                    SQL(
                        cast(
                            LiteralString,
                            create_sql.replace(
                                'COLLATE "C"',
                                f'COLLATE "{schema}".case_insensitive',
                            ),
                        )
                    )
                )
            cursor.execute(
                """
                INSERT INTO whetstone_tool_admission_schema (
                    component, version
                ) VALUES (%s, %s)
                """,
                (
                    admission_store_module._SCHEMA_COMPONENT,
                    admission_store_module._SCHEMA_VERSION,
                ),
            )

    try:
        with pytest.raises(
            ToolAdmissionSchemaMismatchError,
            match=r"whetstone_tool_admission_schema.*columns.*"
            r"case_insensitive",
        ):
            ToolAdmissionAuthority.postgresql(
                dsn,
                _connect=connect_in_schema,
            )
    finally:
        with connect(dsn) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    SQL("DROP SCHEMA {} CASCADE").format(Identifier(schema))
                )
