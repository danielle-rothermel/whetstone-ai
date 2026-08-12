from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tests.optimization.processes import (
    in_process_start_methods,
)
from tests.optimization.tools.sqlite_scenarios import (
    run_sqlite_capacity_race,
)
from tests.optimization.tools.support import (
    capacity_binding,
    sqlite_store,
    tool_call,
    tool_config,
)
from whetstone.optimization.tools import _sqlite as sqlite_store_module
from whetstone.optimization.tools.admission import (
    ToolAdmissionSchemaMismatchError,
)
from whetstone.optimization.tools.contracts import (
    ToolCapacityScope,
)
from whetstone.optimization.tools.facade import (
    ToolAdmissionAuthority,
)

FULL_A = "a" * 64
FULL_B = "b" * 64


def test_sqlite_admission_schema_pins_every_storage_class(tmp_path) -> None:
    database = tmp_path / "storage-classes.sqlite"
    ToolAdmissionAuthority.sqlite(database)
    expected_columns = {
        "whetstone_tool_admission_schema": ("component", "version"),
        "whetstone_tool_admission_entry": (
            "store_namespace_key",
            "call_id",
            "entry_json",
        ),
        "whetstone_tool_admission_capacity": (
            "store_namespace_key",
            "tool_config_hash",
            "capacity_scope",
            "capacity_scope_id",
            "max_accepted_calls",
            "consumed",
        ),
    }
    with sqlite3.connect(database) as connection:
        for table, columns in expected_columns.items():
            row = connection.execute(
                """
                SELECT sql FROM sqlite_master
                WHERE type = 'table' AND name = ?
                """,
                (table,),
            ).fetchone()
            assert row is not None
            compact_sql = "".join(row[0].split())
            for column in columns:
                assert f"typeof({column})=" in compact_sql


def test_sqlite_initialization_migrates_exact_unversioned_schema(
    tmp_path,
) -> None:
    database = tmp_path / "unversioned.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(sqlite_store_module._SQLITE_CREATE_ENTRY)
        connection.execute(sqlite_store_module._SQLITE_CREATE_CAPACITY)

    ToolAdmissionAuthority.sqlite(database)

    with sqlite3.connect(database) as connection:
        assert connection.execute(
            """
            SELECT component, version
            FROM whetstone_tool_admission_schema
            """
        ).fetchall() == [("tool_admission", 2)]


def test_sqlite_initialization_rejects_truncated_table(tmp_path) -> None:
    database = tmp_path / "truncated.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE whetstone_tool_admission_entry (
                store_namespace_key TEXT NOT NULL
            )
            """
        )

    with pytest.raises(
        RuntimeError,
        match=r"owned table inventory.*whetstone_tool_admission_entry",
    ):
        ToolAdmissionAuthority.sqlite(database)


def test_sqlite_initialization_rejects_unaudited_capacity_table(
    tmp_path,
) -> None:
    database = tmp_path / "capacity-only.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(sqlite_store_module._SQLITE_CREATE_CAPACITY)
        connection.execute(
            """
            INSERT INTO whetstone_tool_admission_capacity (
                store_namespace_key, tool_config_hash, capacity_scope,
                capacity_scope_id, max_accepted_calls, consumed
            ) VALUES ('namespace', 'config', 'run', 'run-1', 2, 1)
            """
        )

    with pytest.raises(
        ToolAdmissionSchemaMismatchError,
        match=r"owned table inventory.*whetstone_tool_admission_capacity",
    ):
        ToolAdmissionAuthority.sqlite(database)

    with sqlite3.connect(database) as connection:
        owned_tables = connection.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type = 'table' AND name LIKE 'whetstone_tool_admission_%'
            ORDER BY name
            """
        ).fetchall()
        consumed = connection.execute(
            "SELECT consumed FROM whetstone_tool_admission_capacity"
        ).fetchall()
    assert owned_tables == [("whetstone_tool_admission_capacity",)]
    assert consumed == [(1,)]


def test_sqlite_initialization_rejects_columns_with_wrong_constraints(
    tmp_path,
) -> None:
    database = tmp_path / "wrong-constraints.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(sqlite_store_module._SQLITE_CREATE_ENTRY)
        connection.execute(
            """
            CREATE TABLE whetstone_tool_admission_capacity (
                store_namespace_key TEXT NOT NULL,
                tool_config_hash TEXT NOT NULL,
                capacity_scope TEXT NOT NULL,
                capacity_scope_id TEXT NOT NULL,
                max_accepted_calls INTEGER NOT NULL,
                consumed INTEGER NOT NULL,
                PRIMARY KEY (
                    store_namespace_key, tool_config_hash, capacity_scope,
                    capacity_scope_id
                )
            )
            """
        )

    with pytest.raises(
        ToolAdmissionSchemaMismatchError,
        match=r"incompatible Tool admission table "
        r"'whetstone_tool_admission_capacity'.*table definition",
    ):
        ToolAdmissionAuthority.sqlite(database)


def test_sqlite_initialization_rejects_unknown_schema_version(
    tmp_path,
) -> None:
    database = tmp_path / "future.sqlite"
    ToolAdmissionAuthority.sqlite(database)
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            UPDATE whetstone_tool_admission_schema
            SET version = 3
            WHERE component = 'tool_admission'
            """
        )

    with pytest.raises(
        ToolAdmissionSchemaMismatchError,
        match="expected exact schema version 2, found 3",
    ):
        ToolAdmissionAuthority.sqlite(database)


def test_sqlite_initialization_rejects_real_schema_version(tmp_path) -> None:
    database = tmp_path / "real-version.sqlite"
    ToolAdmissionAuthority.sqlite(database)
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute(
            """
            UPDATE whetstone_tool_admission_schema
            SET version = 2.5
            WHERE component = 'tool_admission'
            """
        )

    with pytest.raises(
        ToolAdmissionSchemaMismatchError,
        match="schema version",
    ):
        ToolAdmissionAuthority.sqlite(database)


@pytest.mark.parametrize(
    "field",
    ["max_accepted_calls", "consumed"],
)
def test_sqlite_capacity_decode_rejects_real_counters(
    tmp_path,
    field: str,
) -> None:
    database = tmp_path / f"real-{field}.sqlite"
    config = tool_config(capacity=2)
    store = sqlite_store(database)
    store.admit(tool_call(config, "accepted"), config)
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute(
            f"""
            UPDATE whetstone_tool_admission_capacity
            SET {field} = 1.5
            """
        )

    with pytest.raises(
        RuntimeError,
        match=rf"{field} is not an integer",
    ):
        if field == "max_accepted_calls":
            store.admit(tool_call(config, "second"), config)
        else:
            store.accepted_count(
                config, capacity_binding(ToolCapacityScope.RUN)
            )


def test_sqlite_admission_decode_rejects_blob_json(tmp_path) -> None:
    database = tmp_path / "blob-entry.sqlite"
    config = tool_config(capacity=1)
    call = tool_call(config, "accepted")
    store = sqlite_store(database)
    store.admit(call, config)
    with sqlite3.connect(database) as connection:
        raw = connection.execute(
            """
            SELECT entry_json FROM whetstone_tool_admission_entry
            WHERE store_namespace_key = ? AND call_id = ?
            """,
            (str(call.store_namespace_key), str(call.call_id)),
        ).fetchone()
        assert raw is not None
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute(
            """
            UPDATE whetstone_tool_admission_entry SET entry_json = ?
            WHERE store_namespace_key = ? AND call_id = ?
            """,
            (
                sqlite3.Binary(raw[0].encode()),
                str(call.store_namespace_key),
                str(call.call_id),
            ),
        )

    with pytest.raises(RuntimeError, match="entry is not JSON text"):
        store.get(call)


@pytest.mark.sqlite_contention
@pytest.mark.parametrize("start_method", in_process_start_methods())
@pytest.mark.process_integration
def test_spawned_sqlite_capacity_race_is_atomic(
    tmp_path: Path, start_method: str
) -> None:
    run_sqlite_capacity_race(tmp_path, start_method)
