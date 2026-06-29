from __future__ import annotations

from enum import StrEnum

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Table,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB

from dr_dspy.humaneval.scoring import GeneratedCodeOutcome
from dr_dspy.records import (
    BatchSubmitItemStatus,
    BatchSubmitOperationStatus,
    GenerationRunStatus,
    NodeAttemptStatus,
    ScoreAttemptStatus,
)

EXPERIMENTS_TABLE = "dr_dspy_experiments"
PREDICTION_SPECS_TABLE = "dr_dspy_prediction_specs"
GENERATION_RUNS_TABLE = "dr_dspy_generation_runs"
NODE_ATTEMPTS_TABLE = "dr_dspy_node_attempts"
SCORE_ATTEMPTS_TABLE = "dr_dspy_score_attempts"
PREDICTION_PROJECTION_TABLE = "dr_dspy_prediction_projection"
BATCH_SUBMIT_OPERATIONS_TABLE = "dr_dspy_batch_submit_operations"
BATCH_SUBMIT_ITEMS_TABLE = "dr_dspy_batch_submit_items"

V1_TABLE_NAMES = (
    EXPERIMENTS_TABLE,
    PREDICTION_SPECS_TABLE,
    GENERATION_RUNS_TABLE,
    NODE_ATTEMPTS_TABLE,
    SCORE_ATTEMPTS_TABLE,
    PREDICTION_PROJECTION_TABLE,
    BATCH_SUBMIT_OPERATIONS_TABLE,
    BATCH_SUBMIT_ITEMS_TABLE,
)

metadata = MetaData()


def enum_check(column_name: str, enum_type: type[StrEnum]) -> str:
    values = ", ".join(f"'{value.value}'" for value in enum_type)
    return f"{column_name} IN ({values})"


experiments = Table(
    EXPERIMENTS_TABLE,
    metadata,
    Column("experiment_name", Text, primary_key=True),
    Column("description", Text),
    Column("config_metadata", JSONB, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

prediction_specs = Table(
    PREDICTION_SPECS_TABLE,
    metadata,
    Column("prediction_id", Text, primary_key=True),
    Column(
        "experiment_name",
        Text,
        ForeignKey(f"{EXPERIMENTS_TABLE}.experiment_name"),
        nullable=False,
    ),
    Column("task_id", Text, nullable=False),
    Column("repetition_seed", Integer, nullable=False),
    Column("graph_digest", Text, nullable=False),
    Column("dimensions_digest", Text, nullable=False),
    Column("graph_layout", Text, nullable=False),
    Column("provider_kind", Text, nullable=False),
    Column("endpoint_kind", Text, nullable=False),
    Column("model", Text, nullable=False),
    Column("throttle_key", Text, nullable=False),
    Column("fair_order_key", Text, nullable=False),
    Column("task_snapshot", JSONB, nullable=False),
    Column("graph_snapshot", JSONB, nullable=False),
    Column("dimensions", JSONB, nullable=False),
    Column("provider_configs", JSONB, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    CheckConstraint(
        "repetition_seed >= 0",
        name="ck_dr_dspy_prediction_specs_repetition_seed",
    ),
    UniqueConstraint(
        "experiment_name",
        "task_id",
        "repetition_seed",
        "graph_digest",
        "dimensions_digest",
        name="uq_dr_dspy_prediction_specs_identity",
    ),
)

generation_runs = Table(
    GENERATION_RUNS_TABLE,
    metadata,
    Column("generation_run_id", Text, primary_key=True),
    Column(
        "prediction_id",
        Text,
        ForeignKey(f"{PREDICTION_SPECS_TABLE}.prediction_id"),
        nullable=False,
    ),
    Column("attempt_index", Integer, nullable=False),
    Column("status", Text, nullable=False),
    Column("terminal_node_id", Text, nullable=False),
    Column("terminal_output_node_id", Text),
    Column("summary", JSONB, nullable=False),
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("completed_at", DateTime(timezone=True), nullable=False),
    CheckConstraint(
        "attempt_index >= 0",
        name="ck_dr_dspy_generation_runs_attempt_index",
    ),
    CheckConstraint(
        enum_check("status", GenerationRunStatus),
        name="ck_dr_dspy_generation_runs_status",
    ),
    CheckConstraint(
        "completed_at >= started_at",
        name="ck_dr_dspy_generation_runs_time_order",
    ),
    UniqueConstraint(
        "prediction_id",
        "attempt_index",
        name="uq_dr_dspy_generation_runs_attempt",
    ),
)

node_attempts = Table(
    NODE_ATTEMPTS_TABLE,
    metadata,
    Column("node_attempt_id", Text, primary_key=True),
    Column(
        "generation_run_id",
        Text,
        ForeignKey(f"{GENERATION_RUNS_TABLE}.generation_run_id"),
        nullable=False,
    ),
    Column(
        "prediction_id",
        Text,
        ForeignKey(f"{PREDICTION_SPECS_TABLE}.prediction_id"),
        nullable=False,
    ),
    Column("node_id", Text, nullable=False),
    Column("attempt_index", Integer, nullable=False),
    Column("status", Text, nullable=False),
    Column("provider_kind", Text),
    Column("endpoint_kind", Text),
    Column("model", Text),
    Column("throttle_key", Text),
    Column("provider_config", JSONB),
    Column("output", JSONB),
    Column("usage_cost", JSONB, nullable=False),
    Column("response_metadata", JSONB, nullable=False),
    Column("failure", JSONB),
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("completed_at", DateTime(timezone=True), nullable=False),
    CheckConstraint(
        "attempt_index >= 0",
        name="ck_dr_dspy_node_attempts_attempt_index",
    ),
    CheckConstraint(
        enum_check("status", NodeAttemptStatus),
        name="ck_dr_dspy_node_attempts_status",
    ),
    CheckConstraint(
        "(status != 'success' OR output IS NOT NULL) "
        "AND (status != 'error' OR failure IS NOT NULL)",
        name="ck_dr_dspy_node_attempts_status_payload",
    ),
    CheckConstraint(
        "completed_at >= started_at",
        name="ck_dr_dspy_node_attempts_time_order",
    ),
    UniqueConstraint(
        "generation_run_id",
        "node_id",
        "attempt_index",
        name="uq_dr_dspy_node_attempts_run_node_attempt",
    ),
)

score_attempts = Table(
    SCORE_ATTEMPTS_TABLE,
    metadata,
    Column("score_attempt_id", Text, primary_key=True),
    Column(
        "prediction_id",
        Text,
        ForeignKey(f"{PREDICTION_SPECS_TABLE}.prediction_id"),
        nullable=False,
    ),
    Column(
        "generation_run_id",
        Text,
        ForeignKey(f"{GENERATION_RUNS_TABLE}.generation_run_id"),
        nullable=False,
    ),
    Column("scoring_profile_id", Text, nullable=False),
    Column("scoring_profile_version", Text, nullable=False),
    Column("parser_profile_id", Text, nullable=False),
    Column("parser_version", Text, nullable=False),
    Column("status", Text, nullable=False),
    Column("generated_code_outcome", Text),
    Column("score", Float),
    Column("extracted_code", JSONB),
    Column("metrics", JSONB),
    Column("per_test_results", JSONB, nullable=False),
    Column("failure", JSONB),
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("completed_at", DateTime(timezone=True), nullable=False),
    CheckConstraint(
        enum_check("status", ScoreAttemptStatus),
        name="ck_dr_dspy_score_attempts_status",
    ),
    CheckConstraint(
        "(status != 'success' OR score IS NOT NULL) "
        "AND (status != 'error' OR failure IS NOT NULL)",
        name="ck_dr_dspy_score_attempts_status_payload",
    ),
    CheckConstraint(
        "generated_code_outcome IS NULL OR "
        f"({enum_check('generated_code_outcome', GeneratedCodeOutcome)})",
        name="ck_dr_dspy_score_attempts_generated_code_outcome",
    ),
    CheckConstraint(
        "score IS NULL OR (score >= 0 AND score <= 1)",
        name="ck_dr_dspy_score_attempts_score_range",
    ),
    CheckConstraint(
        "completed_at >= started_at",
        name="ck_dr_dspy_score_attempts_time_order",
    ),
    UniqueConstraint(
        "prediction_id",
        "generation_run_id",
        "scoring_profile_id",
        "scoring_profile_version",
        "parser_profile_id",
        "parser_version",
        name="uq_dr_dspy_score_attempts_profile",
    ),
)

prediction_projection = Table(
    PREDICTION_PROJECTION_TABLE,
    metadata,
    Column(
        "prediction_id",
        Text,
        ForeignKey(f"{PREDICTION_SPECS_TABLE}.prediction_id"),
        primary_key=True,
    ),
    Column(
        "generation_run_id",
        Text,
        ForeignKey(f"{GENERATION_RUNS_TABLE}.generation_run_id"),
    ),
    Column(
        "score_attempt_id",
        Text,
        ForeignKey(f"{SCORE_ATTEMPTS_TABLE}.score_attempt_id"),
    ),
    Column("projection_profile_id", Text, primary_key=True),
    Column("projection_version", Text, primary_key=True),
    Column("selected_at", DateTime(timezone=True), nullable=False),
    Column("selection_reason", Text),
    CheckConstraint(
        "generation_run_id IS NOT NULL OR score_attempt_id IS NOT NULL",
        name="ck_dr_dspy_projection_has_selection",
    ),
)

batch_submit_operations = Table(
    BATCH_SUBMIT_OPERATIONS_TABLE,
    metadata,
    Column("operation_key", Text, primary_key=True),
    Column(
        "experiment_name",
        Text,
        ForeignKey(f"{EXPERIMENTS_TABLE}.experiment_name"),
        nullable=False,
    ),
    Column("status", Text, nullable=False),
    Column("requested_count", Integer, nullable=False),
    Column("inserted_count", Integer, nullable=False),
    Column("already_present_count", Integer, nullable=False),
    Column("enqueued_count", Integer, nullable=False),
    Column("failed_count", Integer, nullable=False),
    Column("spec", JSONB, nullable=False),
    Column("metadata", JSONB, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("completed_at", DateTime(timezone=True)),
    CheckConstraint(
        enum_check("status", BatchSubmitOperationStatus),
        name="ck_dr_dspy_batch_ops_status",
    ),
    CheckConstraint(
        "requested_count >= 0 AND inserted_count >= 0 "
        "AND already_present_count >= 0 AND enqueued_count >= 0 "
        "AND failed_count >= 0",
        name="ck_dr_dspy_batch_ops_counts",
    ),
)

batch_submit_items = Table(
    BATCH_SUBMIT_ITEMS_TABLE,
    metadata,
    Column("batch_submit_item_id", Text, primary_key=True),
    Column(
        "operation_key",
        Text,
        ForeignKey(f"{BATCH_SUBMIT_OPERATIONS_TABLE}.operation_key"),
        nullable=False,
    ),
    Column("item_index", Integer, nullable=False),
    Column(
        "prediction_id",
        Text,
        ForeignKey(f"{PREDICTION_SPECS_TABLE}.prediction_id"),
        nullable=False,
    ),
    Column("fair_order_key", Text, nullable=False),
    Column("status", Text, nullable=False),
    Column("enqueue_metadata", JSONB, nullable=False),
    Column("failure", JSONB),
    Column("created_at", DateTime(timezone=True), nullable=False),
    CheckConstraint(
        "item_index >= 0",
        name="ck_dr_dspy_batch_items_item_index",
    ),
    CheckConstraint(
        enum_check("status", BatchSubmitItemStatus),
        name="ck_dr_dspy_batch_items_status",
    ),
    UniqueConstraint(
        "operation_key",
        "item_index",
        name="uq_dr_dspy_batch_items_operation_index",
    ),
    UniqueConstraint(
        "operation_key",
        "prediction_id",
        name="uq_dr_dspy_batch_items_operation_prediction",
    ),
)

Index(
    "ix_dr_dspy_prediction_specs_experiment",
    prediction_specs.c.experiment_name,
)
Index("ix_dr_dspy_prediction_specs_task", prediction_specs.c.task_id)
Index("ix_dr_dspy_prediction_specs_model", prediction_specs.c.provider_kind,
      prediction_specs.c.endpoint_kind, prediction_specs.c.model)
Index("ix_dr_dspy_prediction_specs_graph", prediction_specs.c.graph_layout,
      prediction_specs.c.graph_digest)
Index(
    "ix_dr_dspy_prediction_specs_fair_order",
    prediction_specs.c.fair_order_key,
)
Index("ix_dr_dspy_generation_runs_prediction", generation_runs.c.prediction_id)
Index("ix_dr_dspy_generation_runs_status", generation_runs.c.status)
Index("ix_dr_dspy_node_attempts_prediction", node_attempts.c.prediction_id)
Index("ix_dr_dspy_node_attempts_run", node_attempts.c.generation_run_id)
Index("ix_dr_dspy_node_attempts_node", node_attempts.c.node_id)
Index("ix_dr_dspy_node_attempts_model", node_attempts.c.provider_kind,
      node_attempts.c.endpoint_kind, node_attempts.c.model)
Index("ix_dr_dspy_score_attempts_prediction", score_attempts.c.prediction_id)
Index("ix_dr_dspy_score_attempts_run", score_attempts.c.generation_run_id)
Index("ix_dr_dspy_score_attempts_profile", score_attempts.c.scoring_profile_id,
      score_attempts.c.scoring_profile_version)
Index("ix_dr_dspy_score_attempts_parser", score_attempts.c.parser_profile_id,
      score_attempts.c.parser_version)
Index("ix_dr_dspy_score_attempts_generated_code_outcome",
      score_attempts.c.generated_code_outcome)
Index("ix_dr_dspy_projection_generation",
      prediction_projection.c.generation_run_id)
Index("ix_dr_dspy_projection_score", prediction_projection.c.score_attempt_id)
Index("ix_dr_dspy_projection_profile",
      prediction_projection.c.projection_profile_id,
      prediction_projection.c.projection_version)
Index("ix_dr_dspy_batch_ops_experiment",
      batch_submit_operations.c.experiment_name)
Index("ix_dr_dspy_batch_ops_status", batch_submit_operations.c.status)
Index("ix_dr_dspy_batch_items_operation", batch_submit_items.c.operation_key)
Index("ix_dr_dspy_batch_items_prediction", batch_submit_items.c.prediction_id)
Index("ix_dr_dspy_batch_items_fair_order", batch_submit_items.c.fair_order_key)

v1_tables: tuple[Table, ...] = (
    experiments,
    prediction_specs,
    generation_runs,
    node_attempts,
    score_attempts,
    prediction_projection,
    batch_submit_operations,
    batch_submit_items,
)

__all__ = [
    "BATCH_SUBMIT_ITEMS_TABLE",
    "BATCH_SUBMIT_OPERATIONS_TABLE",
    "EXPERIMENTS_TABLE",
    "GENERATION_RUNS_TABLE",
    "NODE_ATTEMPTS_TABLE",
    "PREDICTION_PROJECTION_TABLE",
    "PREDICTION_SPECS_TABLE",
    "SCORE_ATTEMPTS_TABLE",
    "V1_TABLE_NAMES",
    "batch_submit_items",
    "batch_submit_operations",
    "experiments",
    "generation_runs",
    "metadata",
    "node_attempts",
    "prediction_projection",
    "prediction_specs",
    "score_attempts",
    "v1_tables",
]
