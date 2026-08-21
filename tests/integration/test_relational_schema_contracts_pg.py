"""PostgreSQL mis-shaped owned tables raise the domain errors.

Both durability trees verify their owned PostgreSQL tables through
``dr_store.relational``. These tests pin that a deliberately mis-shaped
table surfaces the whetstone domain error with its structured
``table``/``aspect``/``expected``/``actual`` fields populated.

Run locally:
    uv run pytest -m integration tests/integration/
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, make_url, text

from whetstone.core.effects._postgres import _PostgreSQLStore
from whetstone.core.effects.models import EffectAuthoritySchemaMismatchError
from whetstone.optim.tools._postgres import _PostgreSQLAdmissionBackend
from whetstone.optim.tools.admission import ToolAdmissionSchemaMismatchError

pytestmark = pytest.mark.integration

_EFFECT_TABLE = "whetstone_effect_authority"
_EFFECT_METADATA_TABLE = "whetstone_effect_authority_metadata"
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
    assert getattr(error, "table") == table
    assert getattr(error, "aspect") == aspect
    assert getattr(error, "expected") != getattr(error, "actual")
    assert repr(getattr(error, "expected")) in str(error)
    assert repr(getattr(error, "actual")) in str(error)


def test_effect_authority_postgres_wrong_column_raises_domain_error(
    clean_pg: str,
) -> None:
    """A wrong effect-authority column reports the domain error."""
    _execute(
        clean_pg,
        f"""
        CREATE TABLE {_EFFECT_TABLE} (
            semantic_key TEXT COLLATE "C" PRIMARY KEY,
            unexpected_column TEXT COLLATE "C" NOT NULL
        )
        """,
        f"""
        CREATE TABLE {_EFFECT_METADATA_TABLE} (
            singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
            schema_version INTEGER NOT NULL
        )
        """,
    )

    with pytest.raises(EffectAuthoritySchemaMismatchError) as caught:
        _PostgreSQLStore(_libpq_dsn(clean_pg)).initialize()

    _assert_structured(caught.value, table=_EFFECT_TABLE, aspect="columns")


def test_effect_authority_postgres_missing_constraint_raises_domain_error(
    clean_pg: str,
) -> None:
    """A dropped effect-authority CHECK reports the domain error."""
    _execute(
        clean_pg,
        f"""
        CREATE TABLE {_EFFECT_TABLE} (
            semantic_key TEXT COLLATE "C" PRIMARY KEY,
            request_hash TEXT COLLATE "C" NOT NULL,
            replay_policy TEXT COLLATE "C" NOT NULL CHECK (
                replay_policy IN (
                    'idempotent', 'durable_workflow', 'no_redrive'
                )
            ),
            state TEXT COLLATE "C" NOT NULL CHECK (
                state IN (
                    'leased', 'succeeded', 'failed', 'recovery_required'
                )
            ),
            owner_id TEXT COLLATE "C" NOT NULL,
            attempt_id TEXT COLLATE "C" NOT NULL,
            fence BIGINT NOT NULL,
            expires_at TEXT COLLATE "C",
            terminal_json TEXT COLLATE "C",
            CHECK (
                (state = 'leased' AND expires_at IS NOT NULL
                    AND terminal_json IS NULL)
                OR
                (state != 'leased' AND expires_at IS NULL
                    AND terminal_json IS NOT NULL)
            )
        )
        """,
        f"""
        CREATE TABLE {_EFFECT_METADATA_TABLE} (
            singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
            schema_version INTEGER NOT NULL
        )
        """,
    )

    with pytest.raises(EffectAuthoritySchemaMismatchError) as caught:
        _PostgreSQLStore(_libpq_dsn(clean_pg)).initialize()

    _assert_structured(
        caught.value,
        table=_EFFECT_TABLE,
        aspect="PRIMARY KEY and CHECK constraints",
    )
    assert "fence > 0" in str(caught.value)


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
