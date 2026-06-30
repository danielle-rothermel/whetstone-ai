from __future__ import annotations

import importlib
import os
import uuid
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
from sqlalchemy.exc import IntegrityError

from dr_dspy.db import schema


def test_alembic_env_normalizes_database_url_driver() -> None:
    from dr_dspy.db.migrations.url import normalize_postgresql_driver_url

    assert normalize_postgresql_driver_url(
        "postgresql://localhost/dr_dspy"
    ) == "postgresql+psycopg://localhost/dr_dspy"
    assert normalize_postgresql_driver_url(
        "postgresql+psycopg:///dr_dspy"
    ) == "postgresql+psycopg:///dr_dspy"


def test_alembic_discovers_v1_schema_revision() -> None:
    config = Config("alembic.ini")
    script = ScriptDirectory.from_config(config)

    assert script.get_current_head() == "20260630_0001"


def test_alembic_v1_schema_revision_renders_upgrade_and_downgrade(
    monkeypatch: Any,
) -> None:
    migration, statements = _render_upgrade(monkeypatch)
    migration.downgrade()

    rendered = "\n".join(statements)
    assert "CREATE TABLE dr_dspy_prediction_specs" in rendered
    assert "CREATE TABLE dr_dspy_prediction_projection" in rendered
    assert "DROP TABLE dr_dspy_prediction_specs" in rendered
    assert "DROP TABLE dr_dspy_experiments" in rendered


def test_alembic_v1_schema_revision_matches_live_named_contracts(
    monkeypatch: Any,
) -> None:
    _, statements = _render_upgrade(monkeypatch)
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


def test_alembic_append_only_outcome_revision_renders_triggers(
    monkeypatch: Any,
) -> None:
    migration = importlib.import_module(
        "dr_dspy.db.migrations.versions."
        "20260630_0001_append_only_outcome_triggers"
    )
    statements: list[str] = []
    engine = create_mock_engine(
        "postgresql+psycopg://",
        lambda sql, *args, **kwargs: statements.append(
            str(sql.compile(dialect=engine.dialect))
        ),
    )
    context = MigrationContext.configure(cast(Any, engine.connect()))
    monkeypatch.setattr(migration, "op", Operations(context))

    migration.upgrade()
    rendered = "\n".join(statements)

    for table_name in schema.APPEND_ONLY_OUTCOME_TABLE_NAMES:
        assert f"tr_{table_name}_append_only" in rendered
    assert schema.APPEND_ONLY_OUTCOME_REJECT_FUNCTION in rendered


def test_alembic_v1_schema_revision_applies_to_postgres(
    monkeypatch: Any,
) -> None:
    database_url = os.environ.get(
        "DATABASE_URL",
        "postgresql+psycopg:///dr_dspy",
    )
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace(
            "postgresql://",
            "postgresql+psycopg://",
            1,
        )
    schema_name = f"dr_dspy_migration_test_{uuid.uuid4().hex}"

    try:
        engine = create_engine(database_url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        pytest.skip(f"PostgreSQL unavailable: {exc}")  # ty: ignore[too-many-positional-arguments]

    migration = importlib.import_module(
        "dr_dspy.db.migrations.versions.20260629_0001_v1_domain_schema"
    )
    append_only_migration = importlib.import_module(
        "dr_dspy.db.migrations.versions."
        "20260630_0001_append_only_outcome_triggers"
    )

    try:
        with engine.begin() as conn:
            conn.execute(text(f"CREATE SCHEMA {schema_name}"))

        with engine.begin() as conn:
            conn.execute(text(f"SET search_path TO {schema_name}, public"))
            context = MigrationContext.configure(cast(Any, conn))
            monkeypatch.setattr(migration, "op", Operations(context))
            migration.upgrade()

        with engine.begin() as conn:
            conn.execute(text(f"SET search_path TO {schema_name}, public"))
            context = MigrationContext.configure(cast(Any, conn))
            monkeypatch.setattr(
                append_only_migration,
                "op",
                Operations(context),
            )
            append_only_migration.upgrade()

        with engine.begin() as conn:
            conn.execute(text(f"SET search_path TO {schema_name}, public"))
            tables = {
                row[0]
                for row in conn.execute(
                    text(
                        "SELECT tablename FROM pg_tables "
                        "WHERE schemaname = :schema_name"
                    ),
                    {"schema_name": schema_name},
                )
            }
            assert schema.EXPERIMENTS_TABLE in tables
            assert schema.NODE_ATTEMPTS_TABLE in tables

        with engine.begin() as conn:
            conn.execute(text(f"SET search_path TO {schema_name}, public"))
            _seed_generation_run_chain(conn)

        with engine.begin() as conn:
            conn.execute(text(f"SET search_path TO {schema_name}, public"))
            with pytest.raises(IntegrityError):
                conn.execute(
                    text(
                        "INSERT INTO dr_dspy_node_attempts ("
                        "node_attempt_id, generation_run_id, prediction_id, "
                        "node_id, attempt_index, status, usage_cost, "
                        "response_metadata, started_at, completed_at"
                        ") VALUES ("
                        "'node-bad', 'run-1', 'prediction-2', 'direct', 0, "
                        "'success', '{}'::jsonb, '{}'::jsonb, "
                        "TIMESTAMPTZ '2026-06-29 12:00:00+00', "
                        "TIMESTAMPTZ '2026-06-29 12:00:00+00'"
                        ")"
                    )
                )

        with engine.begin() as conn:
            conn.execute(text(f"SET search_path TO {schema_name}, public"))
            with pytest.raises(Exception, match="append-only table"):
                conn.execute(
                    text(
                        "UPDATE dr_dspy_generation_runs "
                        "SET status = 'error' "
                        "WHERE generation_run_id = 'run-1'"
                    )
                )

        with engine.begin() as conn:
            conn.execute(text(f"SET search_path TO {schema_name}, public"))
            context = MigrationContext.configure(cast(Any, conn))
            monkeypatch.setattr(
                append_only_migration,
                "op",
                Operations(context),
            )
            append_only_migration.downgrade()
            context = MigrationContext.configure(cast(Any, conn))
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
    finally:
        with engine.begin() as conn:
            conn.execute(text(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE"))
        engine.dispose()


def _seed_generation_run_chain(conn: Any) -> None:
    conn.execute(
        text(
            "INSERT INTO dr_dspy_experiments ("
            "experiment_name, config_metadata, created_at"
            ") VALUES ("
            "'exp', '{}'::jsonb, "
            "TIMESTAMPTZ '2026-06-29 12:00:00+00'"
            ")"
        )
    )
    conn.execute(
        text(
            "INSERT INTO dr_dspy_prediction_specs ("
            "prediction_id, experiment_name, task_id, repetition_seed, "
            "graph_digest, dimensions_digest, graph_layout, provider_kind, "
            "endpoint_kind, model, throttle_key, fair_order_seed, "
            "fair_order_key, task_snapshot, graph_snapshot, dimensions, "
            "provider_configs, provider_axis_config_id, created_at"
            ") VALUES ("
            "'prediction-1', 'exp', 'HumanEval/0', 0, 'graph', 'dims', "
            "'direct', 'openai', 'responses', 'model', "
            "'openai:responses:model', 'seed', 'fair', "
            "'{}'::jsonb, '{}'::jsonb, '{}'::jsonb, "
            "'[{\"provider_kind\": \"openai\", \"endpoint_kind\": "
            "\"responses\", \"model\": \"model\", \"throttle_key\": "
            "\"openai:responses:model\"}]'::jsonb, NULL, "
            "TIMESTAMPTZ '2026-06-29 12:00:00+00'"
            ")"
        )
    )
    conn.execute(
        text(
            "INSERT INTO dr_dspy_prediction_specs ("
            "prediction_id, experiment_name, task_id, repetition_seed, "
            "graph_digest, dimensions_digest, graph_layout, provider_kind, "
            "endpoint_kind, model, throttle_key, fair_order_seed, "
            "fair_order_key, task_snapshot, graph_snapshot, dimensions, "
            "provider_configs, provider_axis_config_id, created_at"
            ") VALUES ("
            "'prediction-2', 'exp', 'HumanEval/1', 0, 'graph', 'dims', "
            "'direct', 'openai', 'responses', 'model', "
            "'openai:responses:model', 'seed', 'fair', "
            "'{}'::jsonb, '{}'::jsonb, '{}'::jsonb, "
            "'[{\"provider_kind\": \"openai\", \"endpoint_kind\": "
            "\"responses\", \"model\": \"model\", \"throttle_key\": "
            "\"openai:responses:model\"}]'::jsonb, NULL, "
            "TIMESTAMPTZ '2026-06-29 12:00:00+00'"
            ")"
        )
    )
    conn.execute(
        text(
            "INSERT INTO dr_dspy_generation_runs ("
            "generation_run_id, prediction_id, attempt_index, status, "
            "terminal_node_id, summary, started_at, completed_at"
            ") VALUES ("
            "'run-1', 'prediction-1', 0, 'success', 'direct', "
            "'{\"execution_order\": [\"direct\"], "
            "\"terminal_node_id\": \"direct\"}'::jsonb, "
            "TIMESTAMPTZ '2026-06-29 12:00:00+00', "
            "TIMESTAMPTZ '2026-06-29 12:00:00+00'"
            ")"
        )
    )


def _render_upgrade(monkeypatch: Any) -> tuple[Any, list[str]]:
    migration = importlib.import_module(
        "dr_dspy.db.migrations.versions.20260629_0001_v1_domain_schema"
    )
    statements: list[str] = []
    engine = create_mock_engine(
        "postgresql+psycopg://",
        lambda sql, *args, **kwargs: statements.append(
            str(sql.compile(dialect=engine.dialect))
        ),
    )
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
