"""Mis-shaped owned tables raise the domain schema-mismatch errors.

Both durability trees verify their owned tables through
``dr_store.relational``. These tests pin that a deliberately mis-shaped
table surfaces the whetstone domain error with its structured
``table``/``aspect``/``expected``/``actual`` fields populated, rather
than leaking ``RelationalContractMismatchError``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from whetstone.core.effects._sqlite import _SQLiteStore
from whetstone.core.effects.models import EffectAuthoritySchemaMismatchError
from whetstone.optim.tools._sqlite import _SQLiteAdmissionBackend
from whetstone.optim.tools.admission import ToolAdmissionSchemaMismatchError

_EFFECT_TABLE = "whetstone_effect_authority"
_EFFECT_METADATA_TABLE = "whetstone_effect_authority_metadata"
_TOOL_SCHEMA_TABLE = "whetstone_tool_admission_schema"
_TOOL_ENTRY_TABLE = "whetstone_tool_admission_entry"
_TOOL_CAPACITY_TABLE = "whetstone_tool_admission_capacity"


def _assert_structured(
    error: EffectAuthoritySchemaMismatchError | ToolAdmissionSchemaMismatchError,
    *,
    table: str,
) -> None:
    assert error.table == table
    assert error.aspect
    assert error.expected != error.actual
    assert repr(error.expected) in str(error)
    assert repr(error.actual) in str(error)


def test_effect_authority_sqlite_wrong_column_raises_domain_error(
    tmp_path: Path,
) -> None:
    """A wrong effect-authority column reports the domain error."""
    path = tmp_path / "effects-wrong-column.sqlite"
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            f"""
            CREATE TABLE {_EFFECT_TABLE} (
                semantic_key TEXT PRIMARY KEY,
                unexpected_column TEXT NOT NULL
            )
            """
        )
        connection.execute(
            f"""
            CREATE TABLE {_EFFECT_METADATA_TABLE} (
                singleton INTEGER PRIMARY KEY CHECK (
                    typeof(singleton) = 'integer' AND singleton = 1
                ),
                schema_version INTEGER NOT NULL CHECK (
                    typeof(schema_version) = 'integer'
                )
            )
            """
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(EffectAuthoritySchemaMismatchError) as caught:
        _SQLiteStore(path).initialize()

    _assert_structured(caught.value, table=_EFFECT_TABLE)
    assert caught.value.aspect == "columns"


def test_tool_admission_sqlite_missing_constraint_raises_domain_error(
    tmp_path: Path,
) -> None:
    """A dropped tool-admission CHECK reports the domain error."""
    path = tmp_path / "tools-missing-constraint.sqlite"
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_TOOL_SCHEMA_TABLE} (
                component TEXT NOT NULL PRIMARY KEY CHECK (
                    typeof(component) = 'text'
                ),
                version INTEGER NOT NULL CHECK (
                    typeof(version) = 'integer' AND version > 0
                )
            )
            """
        )
        connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_TOOL_ENTRY_TABLE} (
                store_namespace_key TEXT NOT NULL CHECK (
                    typeof(store_namespace_key) = 'text'
                ),
                call_id TEXT NOT NULL CHECK (typeof(call_id) = 'text'),
                entry_json TEXT NOT NULL,
                PRIMARY KEY (store_namespace_key, call_id)
            )
            """
        )
        connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_TOOL_CAPACITY_TABLE} (
                store_namespace_key TEXT NOT NULL CHECK (
                    typeof(store_namespace_key) = 'text'
                ),
                tool_config_hash TEXT NOT NULL CHECK (
                    typeof(tool_config_hash) = 'text'
                ),
                capacity_scope TEXT NOT NULL CHECK (
                    typeof(capacity_scope) = 'text'
                    AND capacity_scope IN ('global', 'run', 'step')
                ),
                capacity_scope_id TEXT NOT NULL CHECK (
                    typeof(capacity_scope_id) = 'text'
                ),
                max_accepted_calls INTEGER NOT NULL CHECK (
                    typeof(max_accepted_calls) = 'integer'
                    AND max_accepted_calls >= 0
                ),
                consumed INTEGER NOT NULL CHECK (
                    typeof(consumed) = 'integer'
                    AND consumed >= 0 AND consumed <= max_accepted_calls
                ),
                PRIMARY KEY (
                    store_namespace_key, tool_config_hash, capacity_scope,
                    capacity_scope_id
                )
            )
            """
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(ToolAdmissionSchemaMismatchError) as caught:
        _SQLiteAdmissionBackend(path).initialize()

    _assert_structured(caught.value, table=_TOOL_ENTRY_TABLE)
    assert caught.value.aspect == "table definition"
    assert (
        "apply the Tool admission schema migration" in str(caught.value)
    )
