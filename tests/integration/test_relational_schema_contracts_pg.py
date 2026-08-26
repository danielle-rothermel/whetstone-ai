"""PostgreSQL mis-shaped owned tables raise the domain errors.

Whetstone's Tool admission tree verifies its owned PostgreSQL tables through
``dr_store.relational``. These tests pin that a deliberately mis-shaped
table surfaces the whetstone domain error with its structured
``table``/``aspect``/``expected``/``actual`` fields populated.

Run locally:
    uv run pytest -m integration tests/integration/
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, make_url, text

from whetstone.optim.tools._postgres import _PostgreSQLAdmissionBackend
from whetstone.optim.tools.admission import ToolAdmissionSchemaMismatchError

pytestmark = pytest.mark.integration

_TOOL_SCHEMA_TABLE = "whetstone_tool_admission_schema"
_TOOL_ENTRY_TABLE = "whetstone_tool_admission_entry"
_TOOL_CAPACITY_TABLE = "whetstone_tool_admission_capacity"


def _libpq_dsn(database_url: str) -> str:
    """Strip the SQLAlchemy driver suffix to get a plain libpq DSN."""
    url = make_url(database_url).set(drivername="postgresql")
    return url.render_as_string(hide_password=False)


def _execute(database_url: str, *statements: str) -> None:
    engine = create_engine(database_url)
    try:
        with engine.begin() as connection:
            for statement in statements:
                connection.execute(text(statement))
    finally:
        engine.dispose()


def _assert_structured(error: object, *, table: str, aspect: str) -> None:
    assert error.table == table
    assert error.aspect == aspect
    assert error.expected != error.actual
    assert repr(error.expected) in str(error)
    assert repr(error.actual) in str(error)


def test_tool_admission_postgres_wrong_column_raises_domain_error(
    clean_pg: str,
) -> None:
    """A wrong tool-admission column reports the domain error."""
    _execute(
        clean_pg,
        f"""
        CREATE TABLE {_TOOL_SCHEMA_TABLE} (
            component TEXT COLLATE "C" NOT NULL PRIMARY KEY,
            version BIGINT NOT NULL CHECK (version > 0)
        )
        """,
        f"""
        CREATE TABLE {_TOOL_ENTRY_TABLE} (
            store_namespace_key TEXT COLLATE "C" NOT NULL,
            call_id TEXT COLLATE "C" NOT NULL,
            unexpected_column TEXT COLLATE "C" NOT NULL,
            PRIMARY KEY (store_namespace_key, call_id)
        )
        """,
        f"""
        CREATE TABLE {_TOOL_CAPACITY_TABLE} (
            store_namespace_key TEXT COLLATE "C" NOT NULL,
            tool_config_hash TEXT COLLATE "C" NOT NULL,
            capacity_scope TEXT COLLATE "C" NOT NULL CHECK (
                capacity_scope IN ('global', 'run', 'step')
            ),
            capacity_scope_id TEXT COLLATE "C" NOT NULL,
            max_accepted_calls BIGINT NOT NULL CHECK (
                max_accepted_calls >= 0
            ),
            consumed BIGINT NOT NULL CHECK (
                consumed >= 0 AND consumed <= max_accepted_calls
            ),
            PRIMARY KEY (
                store_namespace_key, tool_config_hash, capacity_scope,
                capacity_scope_id
            )
        )
        """,
    )

    with pytest.raises(ToolAdmissionSchemaMismatchError) as caught:
        _PostgreSQLAdmissionBackend(_libpq_dsn(clean_pg)).initialize()

    _assert_structured(
        caught.value, table=_TOOL_ENTRY_TABLE, aspect="columns"
    )
    assert "apply the Tool admission schema migration" in str(caught.value)


def test_tool_admission_postgres_missing_constraint_raises_domain_error(
    clean_pg: str,
) -> None:
    """A dropped tool-admission CHECK reports the domain error."""
    _execute(
        clean_pg,
        f"""
        CREATE TABLE {_TOOL_SCHEMA_TABLE} (
            component TEXT COLLATE "C" NOT NULL PRIMARY KEY,
            version BIGINT NOT NULL CHECK (version > 0)
        )
        """,
        f"""
        CREATE TABLE {_TOOL_ENTRY_TABLE} (
            store_namespace_key TEXT COLLATE "C" NOT NULL,
            call_id TEXT COLLATE "C" NOT NULL,
            entry_json TEXT COLLATE "C" NOT NULL,
            PRIMARY KEY (store_namespace_key, call_id)
        )
        """,
        f"""
        CREATE TABLE {_TOOL_CAPACITY_TABLE} (
            store_namespace_key TEXT COLLATE "C" NOT NULL,
            tool_config_hash TEXT COLLATE "C" NOT NULL,
            capacity_scope TEXT COLLATE "C" NOT NULL CHECK (
                capacity_scope IN ('global', 'run', 'step')
            ),
            capacity_scope_id TEXT COLLATE "C" NOT NULL,
            max_accepted_calls BIGINT NOT NULL CHECK (
                max_accepted_calls >= 0
            ),
            consumed BIGINT NOT NULL,
            PRIMARY KEY (
                store_namespace_key, tool_config_hash, capacity_scope,
                capacity_scope_id
            )
        )
        """,
    )

    with pytest.raises(ToolAdmissionSchemaMismatchError) as caught:
        _PostgreSQLAdmissionBackend(_libpq_dsn(clean_pg)).initialize()

    _assert_structured(
        caught.value,
        table=_TOOL_CAPACITY_TABLE,
        aspect="PRIMARY KEY and CHECK constraints",
    )
    assert "apply the Tool admission schema migration" in str(caught.value)
