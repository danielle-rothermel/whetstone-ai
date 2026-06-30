from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy import (
    CheckConstraint,
    Constraint,
    Table,
    UniqueConstraint,
    create_engine,
    text,
)
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.schema import CreateIndex, CreateTable

from dr_dspy.db import schema


def test_schema_contains_expected_v1_table_names() -> None:
    assert set(schema.metadata.tables) == set(schema.V1_TABLE_NAMES)


def test_schema_declares_append_only_outcome_tables() -> None:
    assert schema.APPEND_ONLY_OUTCOME_TABLE_NAMES == (
        schema.GENERATION_RUNS_TABLE,
        schema.NODE_ATTEMPTS_TABLE,
        schema.SCORE_ATTEMPTS_TABLE,
    )


def test_schema_primary_keys_match_contract() -> None:
    expected = {
        schema.EXPERIMENTS_TABLE: ("experiment_name",),
        schema.PREDICTION_SPECS_TABLE: ("prediction_id",),
        schema.GENERATION_RUNS_TABLE: ("generation_run_id",),
        schema.NODE_ATTEMPTS_TABLE: ("node_attempt_id",),
        schema.SCORE_ATTEMPTS_TABLE: ("score_attempt_id",),
        schema.PREDICTION_PROJECTION_TABLE: (
            "prediction_id",
            "projection_profile_id",
            "projection_version",
        ),
        schema.BATCH_SUBMIT_OPERATIONS_TABLE: ("operation_key",),
        schema.BATCH_SUBMIT_ITEMS_TABLE: ("batch_submit_item_id",),
    }

    for table_name, primary_key in expected.items():
        table = schema.metadata.tables[table_name]
        actual = tuple(column.name for column in table.primary_key)
        assert actual == primary_key


def test_schema_foreign_keys_connect_record_families() -> None:
    assert _foreign_key_targets(schema.prediction_specs) == {
        "dr_dspy_experiments.experiment_name"
    }
    assert _foreign_key_targets(schema.generation_runs) == {
        "dr_dspy_prediction_specs.prediction_id"
    }
    assert _foreign_key_targets(schema.node_attempts) == {
        "dr_dspy_generation_runs.generation_run_id",
        "dr_dspy_generation_runs.prediction_id",
        "dr_dspy_prediction_specs.prediction_id",
    }
    assert _foreign_key_targets(schema.score_attempts) == {
        "dr_dspy_generation_runs.generation_run_id",
        "dr_dspy_generation_runs.prediction_id",
        "dr_dspy_prediction_specs.prediction_id",
    }
    assert _foreign_key_targets(schema.prediction_projection) == {
        "dr_dspy_generation_runs.generation_run_id",
        "dr_dspy_generation_runs.prediction_id",
        "dr_dspy_prediction_specs.prediction_id",
        "dr_dspy_score_attempts.score_attempt_id",
        "dr_dspy_score_attempts.prediction_id",
        "dr_dspy_score_attempts.generation_run_id",
    }
    assert _foreign_key_targets(schema.batch_submit_items) == {
        "dr_dspy_batch_submit_operations.operation_key",
        "dr_dspy_prediction_specs.prediction_id",
    }


def test_schema_has_core_unique_constraints_and_checks() -> None:
    assert "uq_dr_dspy_prediction_specs_identity" in _constraint_names(
        schema.prediction_specs,
        UniqueConstraint,
    )
    assert "uq_dr_dspy_node_attempts_run_node_attempt" in _constraint_names(
        schema.node_attempts,
        UniqueConstraint,
    )
    assert "uq_dr_dspy_score_attempts_profile" in _constraint_names(
        schema.score_attempts,
        UniqueConstraint,
    )
    assert "uq_dr_dspy_generation_runs_id_prediction" in _constraint_names(
        schema.generation_runs,
        UniqueConstraint,
    )
    assert "uq_dr_dspy_score_attempts_id_prediction" in _constraint_names(
        schema.score_attempts,
        UniqueConstraint,
    )
    assert "uq_dr_dspy_score_attempts_id_run" in _constraint_names(
        schema.score_attempts,
        UniqueConstraint,
    )
    assert "ck_dr_dspy_node_attempts_status" in _constraint_names(
        schema.node_attempts,
        CheckConstraint,
    )
    assert "ck_dr_dspy_score_attempts_generated_code_outcome" in (
        _constraint_names(schema.score_attempts, CheckConstraint)
    )
    assert "ck_dr_dspy_score_attempts_attempt_index" in _constraint_names(
        schema.score_attempts,
        CheckConstraint,
    )
    assert "ck_dr_dspy_node_attempts_status_payload" in _constraint_names(
        schema.node_attempts,
        CheckConstraint,
    )
    assert "ck_dr_dspy_prediction_specs_provider_axis" in _constraint_names(
        schema.prediction_specs,
        CheckConstraint,
    )
    assert "ck_dr_dspy_node_attempts_provider_config" in _constraint_names(
        schema.node_attempts,
        CheckConstraint,
    )
    assert "ck_dr_dspy_score_attempts_status_payload" in _constraint_names(
        schema.score_attempts,
        CheckConstraint,
    )
    assert "ck_dr_dspy_projection_has_selection" in _constraint_names(
        schema.prediction_projection,
        CheckConstraint,
    )
    assert "ck_dr_dspy_batch_items_status_payload" in _constraint_names(
        schema.batch_submit_items,
        CheckConstraint,
    )
    assert _unique_constraint_columns(
        schema.score_attempts,
        "uq_dr_dspy_score_attempts_profile",
    ) == (
        "prediction_id",
        "generation_run_id",
        "scoring_profile_id",
        "scoring_profile_version",
        "parser_profile_id",
        "parser_version",
        "attempt_index",
    )
    node_status_check = next(
        constraint
        for constraint in schema.node_attempts.constraints
        if (
            isinstance(constraint, CheckConstraint)
            and constraint.name == "ck_dr_dspy_node_attempts_status"
        )
    )
    assert "blocked" not in str(node_status_check.sqltext)
    assert "'success'" in str(node_status_check.sqltext)
    assert "'error'" in str(node_status_check.sqltext)


def test_schema_has_indexes_for_common_reads() -> None:
    index_names = {
        index.name
        for table in schema.v1_tables
        for index in table.indexes
    }

    assert {
        "ix_dr_dspy_prediction_specs_experiment",
        "ix_dr_dspy_prediction_specs_task",
        "ix_dr_dspy_prediction_specs_model",
        "ix_dr_dspy_prediction_specs_graph",
        "ix_dr_dspy_prediction_specs_fair_order",
        "ix_dr_dspy_generation_runs_prediction",
        "ix_dr_dspy_node_attempts_run",
        "ix_dr_dspy_node_attempts_node",
        "ix_dr_dspy_score_attempts_profile",
        "ix_dr_dspy_score_attempts_parser",
        "ix_dr_dspy_score_attempts_generated_code_outcome",
        "ix_dr_dspy_projection_score",
        "ix_dr_dspy_batch_items_fair_order",
    } <= index_names


def test_schema_payload_columns_are_postgres_jsonb() -> None:
    jsonb_columns = {
        (table.name, column.name)
        for table in schema.v1_tables
        for column in table.columns
        if isinstance(column.type, JSONB)
    }

    assert (
        schema.PREDICTION_SPECS_TABLE,
        "graph_snapshot",
    ) in jsonb_columns
    assert (schema.NODE_ATTEMPTS_TABLE, "provider_config") in jsonb_columns
    assert (schema.NODE_ATTEMPTS_TABLE, "output") in jsonb_columns
    assert (schema.SCORE_ATTEMPTS_TABLE, "per_test_results") in jsonb_columns
    assert (
        schema.BATCH_SUBMIT_OPERATIONS_TABLE,
        "spec",
    ) in jsonb_columns


def test_postgresql_ddl_compiles_for_all_tables_and_indexes() -> None:
    dialect = postgresql.dialect()

    for table in schema.v1_tables:
        table_sql = str(CreateTable(table).compile(dialect=dialect))
        assert table.name in table_sql
        for index in table.indexes:
            index_sql = str(CreateIndex(index).compile(dialect=dialect))
            assert index.name is not None
            assert str(index.name) in index_sql


def test_postgresql_ddl_applies_composite_foreign_keys() -> None:
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
    schema_name = f"dr_dspy_schema_test_{uuid.uuid4().hex}"

    try:
        engine = create_engine(database_url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        pytest.skip(f"PostgreSQL unavailable: {exc}")  # ty: ignore[too-many-positional-arguments]

    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA {schema_name}"))
        conn = conn.execution_options(schema_translate_map={None: schema_name})
        schema.metadata.create_all(conn)

    try:
        with engine.begin() as conn:
            conn.execute(text(f"DROP SCHEMA {schema_name} CASCADE"))
    finally:
        engine.dispose()


def _foreign_key_targets(table: Table) -> set[str]:
    return {
        f"{foreign_key.column.table.name}.{foreign_key.column.name}"
        for foreign_key in table.foreign_keys
    }


def _unique_constraint_columns(
    table: Table,
    constraint_name: str,
) -> tuple[str, ...]:
    constraint = next(
        constraint
        for constraint in table.constraints
        if (
            isinstance(constraint, UniqueConstraint)
            and constraint.name == constraint_name
        )
    )
    return tuple(column.name for column in constraint.columns)


def _constraint_names(
    table: Table,
    constraint_type: type[Constraint],
) -> set[str | None]:
    return {
        str(constraint.name) if constraint.name is not None else None
        for constraint in table.constraints
        if isinstance(constraint, constraint_type)
    }
