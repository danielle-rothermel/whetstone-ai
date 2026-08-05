"""Shared persisted Tool admission contract literals."""

from __future__ import annotations

from whetstone.optimization.tools import _postgres as postgres_store_module
from whetstone.optimization.tools import _sqlite as sqlite_store_module
from whetstone.optimization.tools import admission as admission_store_module
from whetstone.optimization.tools.admission import (
    ToolCallState,
)
from whetstone.optimization.tools.contracts import (
    GLOBAL_CAPACITY_SCOPE_ID,
    ToolCapacityScope,
)

FULL_A = "a" * 64
FULL_B = "b" * 64


def test_tool_admission_persisted_literals_are_pinned() -> None:
    assert GLOBAL_CAPACITY_SCOPE_ID == "global"
    assert admission_store_module._SCHEMA_TABLE == (
        "whetstone_tool_admission_schema"
    )
    assert admission_store_module._ENTRY_TABLE == (
        "whetstone_tool_admission_entry"
    )
    assert admission_store_module._CAPACITY_TABLE == (
        "whetstone_tool_admission_capacity"
    )
    assert admission_store_module._SCHEMA_COMPONENT == "tool_admission"
    assert admission_store_module._SCHEMA_VERSION == 2
    assert (
        postgres_store_module._ENTRY_LOCK_DOMAIN
        == "whetstone.tool_admission.entry_lock.v1"
    )
    assert sqlite_store_module._SQLITE_SCHEMA_COLUMNS == (
        ("component", "TEXT", True, 1),
        ("version", "INTEGER", True, 0),
    )
    assert sqlite_store_module._SQLITE_ENTRY_COLUMNS == (
        ("store_namespace_key", "TEXT", True, 1),
        ("call_id", "TEXT", True, 2),
        ("entry_json", "TEXT", True, 0),
    )
    assert sqlite_store_module._SQLITE_CAPACITY_COLUMNS == (
        ("store_namespace_key", "TEXT", True, 1),
        ("tool_config_hash", "TEXT", True, 2),
        ("capacity_scope", "TEXT", True, 3),
        ("capacity_scope_id", "TEXT", True, 4),
        ("max_accepted_calls", "INTEGER", True, 0),
        ("consumed", "INTEGER", True, 0),
    )
    assert postgres_store_module._POSTGRES_SCHEMA_COLUMNS == (
        ("component", "text", True, "pg_catalog", "C", "c", True, -1),
        ("version", "bigint", True, None, None, None, None, None),
    )
    assert postgres_store_module._POSTGRES_ENTRY_COLUMNS == (
        (
            "store_namespace_key",
            "text",
            True,
            "pg_catalog",
            "C",
            "c",
            True,
            -1,
        ),
        ("call_id", "text", True, "pg_catalog", "C", "c", True, -1),
        ("entry_json", "text", True, "pg_catalog", "C", "c", True, -1),
    )
    assert postgres_store_module._POSTGRES_CAPACITY_COLUMNS == (
        (
            "store_namespace_key",
            "text",
            True,
            "pg_catalog",
            "C",
            "c",
            True,
            -1,
        ),
        (
            "tool_config_hash",
            "text",
            True,
            "pg_catalog",
            "C",
            "c",
            True,
            -1,
        ),
        (
            "capacity_scope",
            "text",
            True,
            "pg_catalog",
            "C",
            "c",
            True,
            -1,
        ),
        (
            "capacity_scope_id",
            "text",
            True,
            "pg_catalog",
            "C",
            "c",
            True,
            -1,
        ),
        ("max_accepted_calls", "bigint", True, None, None, None, None, None),
        ("consumed", "bigint", True, None, None, None, None, None),
    )
    assert [state.value for state in ToolCallState] == [
        "accepted",
        "refused",
        "completed",
    ]
    assert [scope.value for scope in ToolCapacityScope] == [
        "global",
        "run",
        "step",
    ]
