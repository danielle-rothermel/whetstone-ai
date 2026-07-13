"""Whetstone's fresh platform schema bootstrap."""

from __future__ import annotations

import importlib
from typing import Any, cast

from alembic.migration import MigrationContext
from alembic.operations import Operations
from dr_platform.db import PlatformSchema, upgrade_platform_schema
from dr_platform.dbos_config import normalize_postgresql_driver_url
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, Connection

from whetstone.db import schema
from whetstone.platform.connections import (
    bind_schema_strict,
    render_connection_url,
)

PLATFORM_SCHEMA = PlatformSchema(prefix="whetstone")
PLATFORM_VERSION_TABLE = "whetstone_platform_alembic_version"

_WHETSTONE_MIGRATION_MODULES = (
    "20260712_0001_whetstone_baseline",
    "20260713_0002_generation_manifest_shards",
)


def required_run_schema_tables() -> tuple[str, ...]:
    """Every relation the run schema must own after migration."""
    return (
        tuple(sorted(schema.metadata.tables))
        + tuple(sorted(PLATFORM_SCHEMA.metadata.tables))
        + (PLATFORM_VERSION_TABLE,)
    )


def _assert_strict_migration_connection(
    connection: Connection, expected_schema: str
) -> None:
    """Fail before any DDL unless this connection is strictly run-bound."""
    schema_exists = connection.execute(
        text(
            "SELECT EXISTS (SELECT 1 FROM pg_namespace "
            "WHERE nspname = :expected)"
        ),
        {"expected": expected_schema},
    ).scalar_one()
    if not schema_exists:
        raise ValueError(
            f"expected run schema does not exist: {expected_schema}; "
            "refusing to migrate"
        )
    current = connection.execute(text("SELECT current_schema()")).scalar_one()
    if current != expected_schema:
        raise ValueError(
            f"migration connection selected {current!r}, "
            f"not run schema {expected_schema!r}"
        )
    effective = connection.execute(
        text("SELECT current_schemas(false)")
    ).scalar_one()
    if list(effective) != [expected_schema]:
        raise ValueError(
            "migration search path must contain exactly the run schema; "
            f"found {list(effective)!r}"
        )
    pre_existing = (
        connection.execute(
            text(
                "SELECT c.relname FROM pg_class c "
                "JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = :expected AND c.relname = ANY(:names)"
            ),
            {
                "expected": expected_schema,
                "names": list(required_run_schema_tables()),
            },
        )
        .scalars()
        .all()
    )
    if pre_existing:
        raise ValueError(
            "run schema is not fresh; pre-existing Whetstone application "
            f"or kernel tables: {sorted(pre_existing)}"
        )


def assert_run_schema_owns_objects(
    connection: Connection, expected_schema: str
) -> None:
    """Prove required objects resolve to the run schema, never public.

    The connection's search path must lead with the run schema; unqualified
    resolution then proves both presence and that no required object is
    satisfied by a same-named public relation.
    """
    for table_name in required_run_schema_tables():
        namespace = connection.execute(
            text(
                "SELECT n.nspname FROM pg_class c "
                "JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE c.oid = to_regclass(:name)"
            ),
            {"name": table_name},
        ).scalar()
        if namespace != expected_schema:
            raise ValueError(
                f"required table {table_name} resolves to "
                f"{namespace!r}, not run schema {expected_schema!r}"
            )
    for table_name in schema.APPEND_ONLY_OUTCOME_TABLE_NAMES:
        trigger_namespace = connection.execute(
            text(
                "SELECT n.nspname FROM pg_trigger t "
                "JOIN pg_class c ON c.oid = t.tgrelid "
                "JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE t.tgname = :trigger "
                "AND t.tgrelid = to_regclass(:table)"
            ),
            {
                "trigger": f"tr_{table_name}_append_only",
                "table": table_name,
            },
        ).scalar()
        if trigger_namespace != expected_schema:
            raise ValueError(
                f"append-only trigger for {table_name} attaches to "
                f"{trigger_namespace!r}, not run schema {expected_schema!r}"
            )
    function_namespace = connection.execute(
        text(
            "SELECT n.nspname FROM pg_proc p "
            "JOIN pg_namespace n ON n.oid = p.pronamespace "
            "WHERE p.oid = to_regproc(:name)"
        ),
        {"name": schema.APPEND_ONLY_OUTCOME_REJECT_FUNCTION},
    ).scalar()
    if function_namespace != expected_schema:
        raise ValueError(
            "append-only reject function resolves to "
            f"{function_namespace!r}, not run schema {expected_schema!r}"
        )


def ensure_platform_schema(database_url: str) -> None:
    """Upgrade the independent kernel schema using its default naming."""
    resolved = normalize_postgresql_driver_url(database_url)
    engine = create_engine(resolved)
    try:
        upgrade_platform_schema(resolved, prefix="whetstone")
    finally:
        engine.dispose()


def ensure_whetstone_application_schema(
    database_url: str | URL, *, expected_schema: str
) -> None:
    """Migrate a fresh run schema through a verified strict binding.

    The base URL is bound to exactly ``expected_schema`` (no public
    fallback): conditional ``create_all`` discovery and unqualified trigger
    DDL must never see same-named public relations. The binding is asserted
    on the actual migration connection before any DDL, and ownership of
    every required object is proven afterward.
    """
    strict_url = bind_schema_strict(database_url, expected_schema)
    engine = create_engine(strict_url)
    try:
        with engine.begin() as connection:
            _assert_strict_migration_connection(connection, expected_schema)
            for module_name in _WHETSTONE_MIGRATION_MODULES:
                migration = cast(
                    Any,
                    importlib.import_module(
                        f"whetstone.db.migrations.versions.{module_name}"
                    ),
                )
                migration.op = Operations(
                    MigrationContext.configure(connection)
                )
                migration.upgrade()
        upgrade_platform_schema(
            render_connection_url(strict_url), prefix="whetstone"
        )
        with engine.connect() as connection:
            assert_run_schema_owns_objects(connection, expected_schema)
    finally:
        engine.dispose()
