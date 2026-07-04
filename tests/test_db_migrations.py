from __future__ import annotations

import importlib
import os
import subprocess
import sys
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
from sqlalchemy.exc import IntegrityError

from whetstone.db import schema
from whetstone.db.migrations.head import (
    V1_MIGRATION_BASE,
    V1_MIGRATION_HEAD,
    V1_MIGRATION_REVISION_COUNT,
)

V1_MIGRATION_MODULES = (
    "whetstone.db.migrations.versions.20260629_0001_v1_domain_schema",
    "whetstone.db.migrations.versions.20260629_0002_throttle_backoff",
    "whetstone.db.migrations.versions."
    "20260629_0003_batch_submit_already_scheduled_count",
    "whetstone.db.migrations.versions."
    "20260629_0004_batch_submit_enqueuing_status",
    "whetstone.db.migrations.versions."
    "20260630_0001_append_only_outcome_triggers",
    "whetstone.db.migrations.versions."
    "20260630_0002_batch_submit_terminal_enqueue_accounting",
    "whetstone.db.migrations.versions."
    "20260630_0003_batch_submit_claiming_status",
    "whetstone.db.migrations.versions."
    "20260630_0004_batch_submit_remove_prepared_status",
    "whetstone.db.migrations.versions."
    "20260630_0005_score_attempt_dataset_axes",
    "whetstone.db.migrations.versions."
    "20260630_0006_score_attempt_evaluation_incomplete_outcome",
)


def test_alembic_env_normalizes_database_url_driver() -> None:
    from whetstone.db.migrations.url import normalize_postgresql_driver_url

    assert normalize_postgresql_driver_url(
        "postgresql://localhost/dr_dspy"
    ) == "postgresql+psycopg://localhost/dr_dspy"
    assert normalize_postgresql_driver_url(
        "postgresql+psycopg:///dr_dspy"
    ) == "postgresql+psycopg:///dr_dspy"


def test_alembic_discovers_v1_schema_revision() -> None:
    config = Config("alembic.ini")
    script = ScriptDirectory.from_config(config)

    assert script.get_current_head() == V1_MIGRATION_HEAD


def test_alembic_v1_migration_chain_is_linear() -> None:
    config = Config("alembic.ini")
    script = ScriptDirectory.from_config(config)

    heads = script.get_heads()
    assert len(heads) == 1
    assert heads[0] == V1_MIGRATION_HEAD

    bases = script.get_bases()
    assert len(bases) == 1
    assert bases[0] == V1_MIGRATION_BASE

    assert len(list(script.walk_revisions())) == V1_MIGRATION_REVISION_COUNT


def test_alembic_v1_schema_revision_renders_upgrade_and_downgrade(
    monkeypatch: Any,
) -> None:
    migrations, statements = _render_upgrade(monkeypatch)
    migrations[-1].downgrade()
    migrations[-2].downgrade()
    migrations[-3].downgrade()

    rendered = "\n".join(statements)
    assert "CREATE TABLE dr_dspy_prediction_specs" in rendered
    assert "CREATE TABLE dr_dspy_prediction_projection" in rendered
    assert "CREATE TABLE dr_dspy_throttle_backoff" in rendered
    assert "already_scheduled_count" in rendered
    assert "enqueuing" in rendered
    assert "DROP TABLE dr_dspy_throttle_backoff" in rendered


def test_alembic_score_attempt_dataset_revision_renders_constraint(
    monkeypatch: Any,
) -> None:
    migration = importlib.import_module(
        "whetstone.db.migrations.versions."
        "20260630_0005_score_attempt_dataset_axes"
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

    assert "dataset_name" in rendered
    assert "dataset_split" in rendered
    assert "uq_dr_dspy_score_attempts_profile" in rendered
    assert "UPDATE dr_dspy_score_attempts" in rendered


def test_alembic_evaluation_incomplete_outcome_revision_renders_constraint(
    monkeypatch: Any,
) -> None:
    migration = importlib.import_module(
        "whetstone.db.migrations.versions."
        "20260630_0006_score_attempt_evaluation_incomplete_outcome"
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

    assert "ck_dr_dspy_score_attempts_generated_code_outcome" in rendered
    assert "evaluation_incomplete" in rendered


def test_alembic_v1_schema_revision_matches_live_named_contracts(
    monkeypatch: Any,
) -> None:
    _, statements = _render_upgrade(monkeypatch)
    dataset_statements: list[str] = []
    dataset_migration = importlib.import_module(
        "whetstone.db.migrations.versions."
        "20260630_0005_score_attempt_dataset_axes"
    )
    engine = create_mock_engine(
        "postgresql+psycopg://",
        lambda sql, *args, **kwargs: dataset_statements.append(
            str(sql.compile(dialect=engine.dialect))
        ),
    )
    context = MigrationContext.configure(cast(Any, engine.connect()))
    monkeypatch.setattr(dataset_migration, "op", Operations(context))
    dataset_migration.upgrade()
    rendered = "\n".join([*statements, *dataset_statements])

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
        "whetstone.db.migrations.versions."
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


def test_alembic_terminal_enqueue_accounting_revision_renders_constraints(
    monkeypatch: Any,
) -> None:
    migration = importlib.import_module(
        "whetstone.db.migrations.versions."
        "20260630_0002_batch_submit_terminal_enqueue_accounting"
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

    assert "already_scheduled_count" in rendered
    assert "ck_dr_dspy_batch_ops_count_bounds" in rendered
    assert "ck_dr_dspy_batch_ops_completed" in rendered


def test_alembic_claiming_status_revision_renders_constraint_and_heals_rows(
    monkeypatch: Any,
) -> None:
    migration = importlib.import_module(
        "whetstone.db.migrations.versions."
        "20260630_0003_batch_submit_claiming_status"
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

    assert "enqueue_metadata = '{}'::jsonb" in rendered
    assert "'claiming'" in rendered
    assert "ck_dr_dspy_batch_items_enqueue_status" in rendered


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
        pytest.skip(f"PostgreSQL unavailable: {exc}")

    migration = importlib.import_module(
        "whetstone.db.migrations.versions.20260629_0001_v1_domain_schema"
    )
    throttle_migration = importlib.import_module(
        "whetstone.db.migrations.versions.20260629_0002_throttle_backoff"
    )
    batch_migration = importlib.import_module(
        "whetstone.db.migrations.versions."
        "20260629_0003_batch_submit_already_scheduled_count"
    )
    enqueuing_migration = importlib.import_module(
        "whetstone.db.migrations.versions."
        "20260629_0004_batch_submit_enqueuing_status"
    )
    append_only_migration = importlib.import_module(
        "whetstone.db.migrations.versions."
        "20260630_0001_append_only_outcome_triggers"
    )
    terminal_enqueue_migration = importlib.import_module(
        "whetstone.db.migrations.versions."
        "20260630_0002_batch_submit_terminal_enqueue_accounting"
    )
    claiming_migration = importlib.import_module(
        "whetstone.db.migrations.versions."
        "20260630_0003_batch_submit_claiming_status"
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
            monkeypatch.setattr(throttle_migration, "op", Operations(context))
            throttle_migration.upgrade()

        with engine.begin() as conn:
            conn.execute(text(f"SET search_path TO {schema_name}, public"))
            context = MigrationContext.configure(cast(Any, conn))
            monkeypatch.setattr(batch_migration, "op", Operations(context))
            batch_migration.upgrade()

        with engine.begin() as conn:
            conn.execute(text(f"SET search_path TO {schema_name}, public"))
            context = MigrationContext.configure(cast(Any, conn))
            monkeypatch.setattr(
                enqueuing_migration,
                "op",
                Operations(context),
            )
            enqueuing_migration.upgrade()

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
            context = MigrationContext.configure(cast(Any, conn))
            monkeypatch.setattr(
                terminal_enqueue_migration,
                "op",
                Operations(context),
            )
            terminal_enqueue_migration.upgrade()

        with engine.begin() as conn:
            conn.execute(text(f"SET search_path TO {schema_name}, public"))
            conn.execute(
                text(
                    "INSERT INTO dr_dspy_experiments ("
                    "experiment_name, config_metadata, created_at"
                    ") VALUES ("
                    "'exp-heal', '{}'::jsonb, "
                    "TIMESTAMPTZ '2026-06-29 12:00:00+00'"
                    ")"
                )
            )
            conn.execute(
                text(
                    "INSERT INTO dr_dspy_prediction_specs ("
                    "prediction_id, experiment_name, task_id, "
                    "repetition_seed, graph_digest, dimensions_digest, "
                    "graph_layout, provider_kind, endpoint_kind, model, "
                    "throttle_key, fair_order_seed, fair_order_key, "
                    "task_snapshot, graph_snapshot, dimensions, "
                    "provider_configs, provider_axis_config_id, created_at"
                    ") VALUES ("
                    "'prediction-heal', 'exp-heal', 'HumanEval/0', 0, "
                    "'graph', 'dims', 'direct', 'openai', 'responses', "
                    "'model', 'openai:responses:model', 'seed', 'fair', "
                    "'{}'::jsonb, '{}'::jsonb, '{}'::jsonb, "
                    "'[{\"provider_kind\": \"openai\", "
                    "\"endpoint_kind\": \"responses\", \"model\": "
                    "\"model\", \"throttle_key\": "
                    "\"openai:responses:model\"}]'::jsonb, NULL, "
                    "TIMESTAMPTZ '2026-06-29 12:00:00+00'"
                    ")"
                )
            )
            conn.execute(
                text(
                    "INSERT INTO dr_dspy_batch_submit_operations ("
                    "operation_key, experiment_name, status, requested_count, "
                    "inserted_count, already_present_count, enqueued_count, "
                    "already_scheduled_count, failed_count, spec, metadata, "
                    "created_at"
                    ") VALUES ("
                    "'op-claim-heal', 'exp-heal', 'enqueuing', 1, 1, 0, 0, "
                    "0, 0, '{}'::jsonb, '{}'::jsonb, "
                    "TIMESTAMPTZ '2026-06-29 12:00:00+00'"
                    ")"
                )
            )
            conn.execute(
                text(
                    "INSERT INTO dr_dspy_batch_submit_items ("
                    "batch_submit_item_id, operation_key, item_index, "
                    "prediction_id, fair_order_key, insert_status, "
                    "enqueue_status, enqueue_metadata, failure, created_at"
                    ") VALUES ("
                    "'item-heal', 'op-claim-heal', 0, 'prediction-heal', "
                    "'fair', 'inserted', 'pending', "
                    "'{\"enqueue_claim_id\": \"stale\"}'::jsonb, NULL, "
                    "TIMESTAMPTZ '2026-06-29 12:00:00+00'"
                    ")"
                )
            )

        with engine.begin() as conn:
            conn.execute(text(f"SET search_path TO {schema_name}, public"))
            context = MigrationContext.configure(cast(Any, conn))
            monkeypatch.setattr(
                claiming_migration,
                "op",
                Operations(context),
            )
            claiming_migration.upgrade()

        with engine.begin() as conn:
            conn.execute(text(f"SET search_path TO {schema_name}, public"))
            healed = conn.execute(
                text(
                    "SELECT enqueue_metadata "
                    "FROM dr_dspy_batch_submit_items "
                    "WHERE batch_submit_item_id = 'item-heal'"
                )
            ).scalar_one()
            assert healed == {}

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
            conn.execute(
                text(
                    "INSERT INTO dr_dspy_batch_submit_operations ("
                    "operation_key, experiment_name, status, requested_count, "
                    "inserted_count, already_present_count, enqueued_count, "
                    "already_scheduled_count, failed_count, spec, metadata, "
                    "created_at"
                    ") VALUES ("
                    "'op-1', 'exp', 'enqueuing', 2, 2, 0, 0, 0, 0, "
                    "'{}'::jsonb, '{}'::jsonb, "
                    "TIMESTAMPTZ '2026-06-29 12:00:00+00'"
                    ")"
                )
            )
            conn.execute(
                text(
                    "UPDATE dr_dspy_batch_submit_operations SET "
                    "status = 'completed', "
                    "enqueued_count = 0, "
                    "already_scheduled_count = 2, "
                    "failed_count = 0, "
                    "completed_at = TIMESTAMPTZ '2026-06-29 12:00:00+00' "
                    "WHERE operation_key = 'op-1'"
                )
            )
            conn.execute(
                text(
                    "DELETE FROM dr_dspy_batch_submit_operations "
                    "WHERE operation_key = 'op-1'"
                )
            )

        with engine.begin() as conn:
            conn.execute(text(f"SET search_path TO {schema_name}, public"))
            context = MigrationContext.configure(cast(Any, conn))
            monkeypatch.setattr(
                claiming_migration,
                "op",
                Operations(context),
            )
            claiming_migration.downgrade()
            context = MigrationContext.configure(cast(Any, conn))
            monkeypatch.setattr(
                terminal_enqueue_migration,
                "op",
                Operations(context),
            )
            terminal_enqueue_migration.downgrade()
            context = MigrationContext.configure(cast(Any, conn))
            monkeypatch.setattr(
                append_only_migration,
                "op",
                Operations(context),
            )
            append_only_migration.downgrade()
            context = MigrationContext.configure(cast(Any, conn))
            monkeypatch.setattr(
                enqueuing_migration,
                "op",
                Operations(context),
            )
            enqueuing_migration.downgrade()
            context = MigrationContext.configure(cast(Any, conn))
            monkeypatch.setattr(batch_migration, "op", Operations(context))
            batch_migration.downgrade()
            context = MigrationContext.configure(cast(Any, conn))
            monkeypatch.setattr(throttle_migration, "op", Operations(context))
            throttle_migration.downgrade()
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


def test_alembic_score_attempt_dataset_revision_allows_dual_rows(
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
    schema_name = f"dr_dspy_score_dataset_test_{uuid.uuid4().hex}"

    try:
        engine = create_engine(database_url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        pytest.skip(f"PostgreSQL unavailable: {exc}")

    migration_modules = (
        "whetstone.db.migrations.versions.20260629_0001_v1_domain_schema",
        "whetstone.db.migrations.versions.20260629_0002_throttle_backoff",
        "whetstone.db.migrations.versions."
        "20260629_0003_batch_submit_already_scheduled_count",
        "whetstone.db.migrations.versions."
        "20260629_0004_batch_submit_enqueuing_status",
        "whetstone.db.migrations.versions."
        "20260630_0001_append_only_outcome_triggers",
        "whetstone.db.migrations.versions."
        "20260630_0002_batch_submit_terminal_enqueue_accounting",
        "whetstone.db.migrations.versions."
        "20260630_0003_batch_submit_claiming_status",
        "whetstone.db.migrations.versions."
        "20260630_0004_batch_submit_remove_prepared_status",
        "whetstone.db.migrations.versions."
        "20260630_0005_score_attempt_dataset_axes",
        "whetstone.db.migrations.versions."
        "20260630_0006_score_attempt_evaluation_incomplete_outcome",
    )

    try:
        with engine.begin() as conn:
            conn.execute(text(f"CREATE SCHEMA {schema_name}"))

        for module_path in migration_modules:
            migration = importlib.import_module(module_path)
            with engine.begin() as conn:
                conn.execute(text(f"SET search_path TO {schema_name}, public"))
                context = MigrationContext.configure(cast(Any, conn))
                monkeypatch.setattr(migration, "op", Operations(context))
                migration.upgrade()

        with engine.begin() as conn:
            conn.execute(text(f"SET search_path TO {schema_name}, public"))
            _seed_generation_run_chain(conn)
            conn.execute(
                text(
                    "INSERT INTO dr_dspy_score_attempts ("
                    "score_attempt_id, prediction_id, generation_run_id, "
                    "scoring_profile_id, scoring_profile_version, "
                    "parser_profile_id, parser_version, attempt_index, "
                    "dataset_name, dataset_split, status, score, "
                    "per_test_results, started_at, completed_at"
                    ") VALUES ("
                    "'score-default', 'prediction-1', 'run-1', "
                    "'humaneval', 'v1', 'humaneval-best-effort', 'v1', 0, "
                    "'evalplus/humanevalplus', 'test', 'success', 1.0, "
                    "'[]'::jsonb, "
                    "TIMESTAMPTZ '2026-06-29 12:00:00+00', "
                    "TIMESTAMPTZ '2026-06-29 12:00:00+00'"
                    ")"
                )
            )
            conn.execute(
                text(
                    "INSERT INTO dr_dspy_score_attempts ("
                    "score_attempt_id, prediction_id, generation_run_id, "
                    "scoring_profile_id, scoring_profile_version, "
                    "parser_profile_id, parser_version, attempt_index, "
                    "dataset_name, dataset_split, status, score, "
                    "per_test_results, started_at, completed_at"
                    ") VALUES ("
                    "'score-other', 'prediction-1', 'run-1', "
                    "'humaneval', 'v1', 'humaneval-best-effort', 'v1', 0, "
                    "'other/dataset', 'test', 'success', 1.0, "
                    "'[]'::jsonb, "
                    "TIMESTAMPTZ '2026-06-29 12:00:00+00', "
                    "TIMESTAMPTZ '2026-06-29 12:00:00+00'"
                    ")"
                )
            )
            count = conn.execute(
                text("SELECT COUNT(*) FROM dr_dspy_score_attempts")
            ).scalar_one()
            assert count == 2

        with engine.begin() as conn:
            conn.execute(text(f"SET search_path TO {schema_name}, public"))
            with pytest.raises(IntegrityError):
                conn.execute(
                    text(
                        "INSERT INTO dr_dspy_score_attempts ("
                        "score_attempt_id, prediction_id, generation_run_id, "
                        "scoring_profile_id, scoring_profile_version, "
                        "parser_profile_id, parser_version, attempt_index, "
                        "dataset_name, dataset_split, status, score, "
                        "per_test_results, started_at, completed_at"
                        ") VALUES ("
                        "'score-dup', 'prediction-1', 'run-1', "
                        "'humaneval', 'v1', 'humaneval-best-effort', 'v1', 0, "
                        "'evalplus/humanevalplus', 'test', 'success', 1.0, "
                        "'[]'::jsonb, "
                        "TIMESTAMPTZ '2026-06-29 12:00:00+00', "
                        "TIMESTAMPTZ '2026-06-29 12:00:00+00'"
                        ")"
                    )
                )
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


def _render_upgrade(monkeypatch: Any) -> tuple[tuple[Any, ...], list[str]]:
    first_migration = importlib.import_module(
        "whetstone.db.migrations.versions.20260629_0001_v1_domain_schema"
    )
    second_migration = importlib.import_module(
        "whetstone.db.migrations.versions.20260629_0002_throttle_backoff"
    )
    third_migration = importlib.import_module(
        "whetstone.db.migrations.versions."
        "20260629_0003_batch_submit_already_scheduled_count"
    )
    fourth_migration = importlib.import_module(
        "whetstone.db.migrations.versions."
        "20260629_0004_batch_submit_enqueuing_status"
    )
    statements: list[str] = []
    engine = create_mock_engine(
        "postgresql+psycopg://",
        lambda sql, *args, **kwargs: statements.append(
            str(sql.compile(dialect=engine.dialect))
        ),
    )
    context = MigrationContext.configure(cast(Any, engine.connect()))
    monkeypatch.setattr(first_migration, "op", Operations(context))
    monkeypatch.setattr(second_migration, "op", Operations(context))
    monkeypatch.setattr(third_migration, "op", Operations(context))
    monkeypatch.setattr(fourth_migration, "op", Operations(context))

    first_migration.upgrade()
    second_migration.upgrade()
    third_migration.upgrade()
    fourth_migration.upgrade()
    return (
        first_migration,
        second_migration,
        third_migration,
        fourth_migration,
    ), statements


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
def _isolated_schema(engine: Any):
    schema_name = f"dr_dspy_migration_test_{uuid.uuid4().hex}"
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA {schema_name}"))
    try:
        yield schema_name
    finally:
        with engine.begin() as conn:
            conn.execute(text(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE"))
        engine.dispose()


def _apply_migration_modules(
    conn: Any,
    monkeypatch: Any,
    module_paths: tuple[str, ...],
) -> None:
    for module_path in module_paths:
        migration = importlib.import_module(module_path)
        context = MigrationContext.configure(conn)
        monkeypatch.setattr(migration, "op", Operations(context))
        migration.upgrade()


def _run_migration_downgrade(
    conn: Any,
    monkeypatch: Any,
    module_path: str,
) -> None:
    migration = importlib.import_module(module_path)
    context = MigrationContext.configure(conn)
    monkeypatch.setattr(migration, "op", Operations(context))
    migration.downgrade()


def test_alembic_env_offline_mode_renders_upgrade_sql() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head", "--sql"],
        check=False,
        capture_output=True,
        text=True,
        cwd=os.getcwd(),
    )

    assert result.returncode == 0, result.stderr
    assert "CREATE TABLE dr_dspy_experiments" in result.stdout
    assert "20260630_0006" in result.stdout


def test_alembic_claiming_status_downgrade_resets_claiming_rows(
    monkeypatch: Any,
) -> None:
    engine = _require_postgres_engine()
    claiming_migration = (
        "whetstone.db.migrations.versions."
        "20260630_0003_batch_submit_claiming_status"
    )

    with _isolated_schema(engine) as schema_name:
        with engine.begin() as conn:
            conn.execute(text(f"SET search_path TO {schema_name}, public"))
            _apply_migration_modules(
                conn,
                monkeypatch,
                V1_MIGRATION_MODULES[:7],
            )
            conn.execute(
                text(
                    "INSERT INTO dr_dspy_experiments ("
                    "experiment_name, config_metadata, created_at"
                    ") VALUES ('exp', '{}'::jsonb, NOW())"
                )
            )
            conn.execute(
                text(
                    "INSERT INTO dr_dspy_prediction_specs ("
                    "prediction_id, experiment_name, task_id, "
                    "repetition_seed, graph_digest, dimensions_digest, "
                    "graph_layout, provider_kind, endpoint_kind, model, "
                    "throttle_key, fair_order_seed, fair_order_key, "
                    "task_snapshot, graph_snapshot, dimensions, "
                    "provider_configs, provider_axis_config_id, created_at"
                    ") VALUES ("
                    "'prediction-1', 'exp', 'HumanEval/0', 0, "
                    "'graph', 'dims', 'direct', 'openai', 'responses', "
                    "'model', 'openai:responses:model', 'seed', 'fair', "
                    "'{}'::jsonb, '{}'::jsonb, '{}'::jsonb, "
                    "'[{\"provider_kind\": \"openai\", "
                    "\"endpoint_kind\": \"responses\", \"model\": "
                    "\"model\", \"throttle_key\": "
                    "\"openai:responses:model\"}]'::jsonb, NULL, NOW()"
                    ")"
                )
            )
            conn.execute(
                text(
                    "INSERT INTO dr_dspy_batch_submit_operations ("
                    "operation_key, experiment_name, status, requested_count, "
                    "inserted_count, already_present_count, enqueued_count, "
                    "already_scheduled_count, failed_count, spec, metadata, "
                    "created_at"
                    ") VALUES ("
                    "'op-claim', 'exp', 'enqueuing', 1, 1, 0, 0, 0, 0, "
                    "'{}'::jsonb, '{}'::jsonb, NOW()"
                    ")"
                )
            )
            conn.execute(
                text(
                    "INSERT INTO dr_dspy_batch_submit_items ("
                    "batch_submit_item_id, operation_key, item_index, "
                    "prediction_id, fair_order_key, insert_status, "
                    "enqueue_status, enqueue_metadata, failure, created_at"
                    ") VALUES ("
                    "'item-claim', 'op-claim', 0, 'prediction-1', 'fair', "
                    "'inserted', 'claiming', "
                    "'{\"enqueue_claim_id\": \"c1\", "
                    "\"claimed_at\": \"2026-06-29T12:00:00+00:00\"}'::jsonb, "
                    "NULL, NOW()"
                    ")"
                )
            )

        with engine.begin() as conn:
            conn.execute(text(f"SET search_path TO {schema_name}, public"))
            _run_migration_downgrade(conn, monkeypatch, claiming_migration)

        with engine.connect() as conn:
            conn.execute(text(f"SET search_path TO {schema_name}, public"))
            row = conn.execute(
                text(
                    "SELECT enqueue_status, enqueue_metadata "
                    "FROM dr_dspy_batch_submit_items "
                    "WHERE batch_submit_item_id = 'item-claim'"
                )
            ).one()
        assert row[0] == "pending"
        assert row[1] == {}


def test_alembic_remove_prepared_status_upgrade_backfills_rows(
    monkeypatch: Any,
) -> None:
    engine = _require_postgres_engine()
    prepared_migration = (
        "whetstone.db.migrations.versions."
        "20260630_0004_batch_submit_remove_prepared_status"
    )

    with _isolated_schema(engine) as schema_name:
        with engine.begin() as conn:
            conn.execute(text(f"SET search_path TO {schema_name}, public"))
            _apply_migration_modules(
                conn,
                monkeypatch,
                V1_MIGRATION_MODULES[:7],
            )
            conn.execute(
                text(
                    "INSERT INTO dr_dspy_experiments ("
                    "experiment_name, config_metadata, created_at"
                    ") VALUES ('exp', '{}'::jsonb, NOW())"
                )
            )
            conn.execute(
                text(
                    "INSERT INTO dr_dspy_batch_submit_operations ("
                    "operation_key, experiment_name, status, requested_count, "
                    "inserted_count, already_present_count, enqueued_count, "
                    "already_scheduled_count, failed_count, spec, metadata, "
                    "created_at"
                    ") VALUES ("
                    "'op-prepared', 'exp', 'prepared', 1, 1, 0, 0, 0, 0, "
                    "'{}'::jsonb, '{}'::jsonb, NOW()"
                    ")"
                )
            )

        with engine.begin() as conn:
            conn.execute(text(f"SET search_path TO {schema_name}, public"))
            migration = importlib.import_module(prepared_migration)
            context = MigrationContext.configure(conn)
            monkeypatch.setattr(migration, "op", Operations(context))
            migration.upgrade()

        with engine.connect() as conn:
            conn.execute(text(f"SET search_path TO {schema_name}, public"))
            status = conn.execute(
                text(
                    "SELECT status FROM dr_dspy_batch_submit_operations "
                    "WHERE operation_key = 'op-prepared'"
                )
            ).scalar_one()
        assert status == "enqueuing"


def test_alembic_enqueuing_status_downgrade_maps_to_prepared(
    monkeypatch: Any,
) -> None:
    engine = _require_postgres_engine()
    enqueuing_migration = (
        "whetstone.db.migrations.versions."
        "20260629_0004_batch_submit_enqueuing_status"
    )

    with _isolated_schema(engine) as schema_name:
        with engine.begin() as conn:
            conn.execute(text(f"SET search_path TO {schema_name}, public"))
            _apply_migration_modules(
                conn,
                monkeypatch,
                V1_MIGRATION_MODULES[:4],
            )
            conn.execute(
                text(
                    "INSERT INTO dr_dspy_experiments ("
                    "experiment_name, config_metadata, created_at"
                    ") VALUES ('exp', '{}'::jsonb, NOW())"
                )
            )
            conn.execute(
                text(
                    "INSERT INTO dr_dspy_batch_submit_operations ("
                    "operation_key, experiment_name, status, requested_count, "
                    "inserted_count, already_present_count, enqueued_count, "
                    "already_scheduled_count, failed_count, spec, metadata, "
                    "created_at"
                    ") VALUES ("
                    "'op-enq', 'exp', 'enqueuing', 1, 1, 0, 0, 0, 0, "
                    "'{}'::jsonb, '{}'::jsonb, NOW()"
                    ")"
                )
            )

        with engine.begin() as conn:
            conn.execute(text(f"SET search_path TO {schema_name}, public"))
            _run_migration_downgrade(conn, monkeypatch, enqueuing_migration)

        with engine.connect() as conn:
            conn.execute(text(f"SET search_path TO {schema_name}, public"))
            status = conn.execute(
                text(
                    "SELECT status FROM dr_dspy_batch_submit_operations "
                    "WHERE operation_key = 'op-enq'"
                )
            ).scalar_one()
        assert status == "prepared"


def test_alembic_score_dataset_downgrade_rejects_dual_profile_rows(
    monkeypatch: Any,
) -> None:
    engine = _require_postgres_engine()
    dataset_migration = (
        "whetstone.db.migrations.versions."
        "20260630_0005_score_attempt_dataset_axes"
    )

    with _isolated_schema(engine) as schema_name:
        with engine.begin() as conn:
            conn.execute(text(f"SET search_path TO {schema_name}, public"))
            _apply_migration_modules(conn, monkeypatch, V1_MIGRATION_MODULES)
            _seed_generation_run_chain(conn)
            conn.execute(
                text(
                    "INSERT INTO dr_dspy_score_attempts ("
                    "score_attempt_id, prediction_id, generation_run_id, "
                    "scoring_profile_id, scoring_profile_version, "
                    "parser_profile_id, parser_version, attempt_index, "
                    "dataset_name, dataset_split, status, score, "
                    "per_test_results, started_at, completed_at"
                    ") VALUES ("
                    "'score-default', 'prediction-1', 'run-1', "
                    "'humaneval', 'v1', 'humaneval-best-effort', 'v1', 0, "
                    "'evalplus/humanevalplus', 'test', 'success', 1.0, "
                    "'[]'::jsonb, NOW(), NOW()"
                    "), ("
                    "'score-other', 'prediction-1', 'run-1', "
                    "'humaneval', 'v1', 'humaneval-best-effort', 'v1', 0, "
                    "'other-dataset', 'dev', 'success', 1.0, "
                    "'[]'::jsonb, NOW(), NOW()"
                    ")"
                )
            )

        with engine.begin() as conn:
            conn.execute(text(f"SET search_path TO {schema_name}, public"))
            with pytest.raises(IntegrityError):
                _run_migration_downgrade(
                    conn,
                    monkeypatch,
                    dataset_migration,
                )
