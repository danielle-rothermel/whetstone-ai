from __future__ import annotations

import importlib
import os
import uuid
from contextlib import contextmanager
from typing import Any, cast

import pytest
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.operations import Operations
from alembic.script import ScriptDirectory
from sqlalchemy import (
    Constraint,
    Table,
    create_engine,
    create_mock_engine,
    text,
)

from whetstone.db import schema
from whetstone.db.migrations.head import (
    V1_MIGRATION_BASE,
    V1_MIGRATION_HEAD,
    V1_MIGRATION_REVISION_COUNT,
)

INITIAL_MIGRATION_MODULE = (
    "whetstone.db.migrations.versions.20260708_0001_initial_schema"
)


def test_alembic_env_normalizes_database_url_driver() -> None:
    from whetstone.db.migrations.url import normalize_postgresql_driver_url

    assert normalize_postgresql_driver_url(
        "postgresql://localhost/dr_dspy"
    ) == "postgresql+psycopg://localhost/dr_dspy"
    assert normalize_postgresql_driver_url(
        "postgresql+psycopg:///dr_dspy"
    ) == "postgresql+psycopg:///dr_dspy"


def test_alembic_discovers_initial_schema_revision() -> None:
    config = Config("alembic.ini")
    script = ScriptDirectory.from_config(config)

    assert script.get_current_head() == V1_MIGRATION_HEAD
    assert script.get_heads() == [V1_MIGRATION_HEAD]
    assert script.get_bases() == [V1_MIGRATION_BASE]
    assert len(list(script.walk_revisions())) == V1_MIGRATION_REVISION_COUNT


def test_alembic_initial_schema_revision_renders_current_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    migration, statements = _render_upgrade(monkeypatch)
    migration.downgrade()

    rendered = "\n".join(statements)
    for table in schema.v1_tables:
        assert f"CREATE TABLE {table.name}" in rendered
        for column in table.columns:
            assert column.name in rendered
        for constraint_name in _named_constraint_names(table):
            assert constraint_name in rendered
        for index in table.indexes:
            assert index.name is not None
            assert index.name in rendered

    assert "submission_outcome" in rendered
    assert schema.SCORE_HARNESS_FAILURES_TABLE in rendered
    assert "generated_code_outcome" not in rendered
    assert "raw_generation" not in rendered

    for table_name in schema.APPEND_ONLY_OUTCOME_TABLE_NAMES:
        assert f"tr_{table_name}_append_only" in rendered
    assert schema.APPEND_ONLY_OUTCOME_REJECT_FUNCTION in rendered
    assert f"DROP TABLE {schema.EXPERIMENTS_TABLE}" in rendered


def test_alembic_initial_schema_revision_metadata_is_pristine() -> None:
    migration = importlib.import_module(INITIAL_MIGRATION_MODULE)

    assert migration.revision == V1_MIGRATION_BASE
    assert migration.down_revision is None


def test_alembic_initial_schema_revision_applies_to_postgres(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _require_postgres_engine()

    with _isolated_schema(engine) as schema_name:
        migration = importlib.import_module(INITIAL_MIGRATION_MODULE)
        with engine.begin() as conn:
            conn.execute(text(f"SET search_path TO {schema_name}"))
            context = MigrationContext.configure(conn)
            monkeypatch.setattr(migration, "op", Operations(context))
            migration.upgrade()

        with engine.begin() as conn:
            conn.execute(text(f"SET search_path TO {schema_name}"))
            table_names = {
                row[0]
                for row in conn.execute(
                    text(
                        "SELECT tablename FROM pg_tables "
                        "WHERE schemaname = :schema_name"
                    ),
                    {"schema_name": schema_name},
                )
            }
            assert schema.SCORE_ATTEMPTS_TABLE in table_names
            assert schema.SCORE_HARNESS_FAILURES_TABLE in table_names
            assert _trigger_exists(
                conn,
                schema_name=schema_name,
                trigger_name=f"tr_{schema.SCORE_ATTEMPTS_TABLE}_append_only",
            )

            context = MigrationContext.configure(conn)
            monkeypatch.setattr(migration, "op", Operations(context))
            migration.downgrade()
            remaining_tables = {
                row[0]
                for row in conn.execute(
                    text(
                        "SELECT tablename FROM pg_tables "
                        "WHERE schemaname = :schema_name"
                    ),
                    {"schema_name": schema_name},
                )
            }
            assert schema.EXPERIMENTS_TABLE not in remaining_tables


def _render_upgrade(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, list[str]]:
    migration = importlib.import_module(INITIAL_MIGRATION_MODULE)
    statements: list[str] = []

    def append_statement(sql: Any, *args: Any, **kwargs: Any) -> None:
        if hasattr(sql, "compile"):
            statements.append(str(sql.compile(dialect=engine.dialect)))
            return
        statements.append(str(sql))

    engine = create_mock_engine("postgresql+psycopg://", append_statement)
    context = MigrationContext.configure(cast(Any, engine.connect()))
    monkeypatch.setattr(migration, "op", Operations(context))

    migration.upgrade()
    return migration, statements


def _named_constraint_names(table: Table) -> set[str]:
    return {
        str(constraint.name)
        for constraint in table.constraints
        if _has_name(constraint)
    }


def _has_name(constraint: Constraint) -> bool:
    return constraint.name is not None


def _normalized_database_url() -> str:
    database_url = os.environ.get(
        "DATABASE_URL",
        "postgresql+psycopg:///dr_dspy",
    )
    if database_url.startswith("postgresql://"):
        return database_url.replace(
            "postgresql://",
            "postgresql+psycopg://",
            1,
        )
    return database_url


def _require_postgres_engine() -> Any:
    database_url = _normalized_database_url()
    try:
        engine = create_engine(database_url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        pytest.skip(f"PostgreSQL unavailable: {exc}")
    return engine


@contextmanager
def _isolated_schema(engine: Any) -> Any:
    schema_name = f"dr_dspy_migration_test_{uuid.uuid4().hex}"
    try:
        with engine.begin() as conn:
            conn.execute(text(f"CREATE SCHEMA {schema_name}"))
        yield schema_name
    finally:
        with engine.begin() as conn:
            conn.execute(text(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE"))
        engine.dispose()


def _trigger_exists(
    conn: Any,
    *,
    schema_name: str,
    trigger_name: str,
) -> bool:
    return bool(
        conn.execute(
            text(
                "SELECT 1 FROM information_schema.triggers "
                "WHERE trigger_schema = :schema_name "
                "AND trigger_name = :trigger_name"
            ),
            {
                "schema_name": schema_name,
                "trigger_name": trigger_name,
            },
        ).first()
    )
