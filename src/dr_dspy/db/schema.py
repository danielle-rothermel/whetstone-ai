from __future__ import annotations

from enum import StrEnum

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
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
    BatchSubmitItemEnqueueStatus,
    BatchSubmitItemInsertStatus,
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
THROTTLE_BACKOFF_TABLE = "dr_dspy_throttle_backoff"

V1_TABLE_NAMES = (
    EXPERIMENTS_TABLE,
    PREDICTION_SPECS_TABLE,
    GENERATION_RUNS_TABLE,
    NODE_ATTEMPTS_TABLE,
    SCORE_ATTEMPTS_TABLE,
    PREDICTION_PROJECTION_TABLE,
    BATCH_SUBMIT_OPERATIONS_TABLE,
    BATCH_SUBMIT_ITEMS_TABLE,
    THROTTLE_BACKOFF_TABLE,
)

# Outcome facts are append-only at the DB layer; projection remains mutable.
APPEND_ONLY_OUTCOME_REJECT_FUNCTION = (
    "dr_dspy_reject_append_only_outcome_mutation"
)
APPEND_ONLY_OUTCOME_TABLE_NAMES = (
    GENERATION_RUNS_TABLE,
    NODE_ATTEMPTS_TABLE,
    SCORE_ATTEMPTS_TABLE,
)

metadata = MetaData()


def enum_check(column_name: str, enum_type: type[StrEnum]) -> str:
    values = ", ".join(f"'{value.value}'" for value in enum_type)
    return f"{column_name} IN ({values})"


PREDICTION_SPECS_PROVIDER_AXIS_CHECK = """
provider_configs @> jsonb_build_array(
  jsonb_strip_nulls(
    jsonb_build_object(
      'provider_kind', provider_kind,
      'endpoint_kind', endpoint_kind,
      'model', model,
      'throttle_key', throttle_key,
      'config_id', provider_axis_config_id
    )
  )
)
""".strip()


NODE_ATTEMPTS_PROVIDER_CONFIG_CHECK = """
(
  provider_config IS NULL
  AND provider_kind IS NULL
  AND endpoint_kind IS NULL
  AND model IS NULL
  AND throttle_key IS NULL
  AND config_id IS NULL
) OR (
  provider_config IS NOT NULL
  AND provider_kind = provider_config->>'provider_kind'
  AND endpoint_kind = provider_config->>'endpoint_kind'
  AND model = provider_config->>'model'
  AND throttle_key = provider_config->>'throttle_key'
  AND (
    (config_id IS NULL AND provider_config->>'config_id' IS NULL)
    OR config_id = provider_config->>'config_id'
  )
)
""".strip()


BATCH_SUBMIT_OPS_COUNT_BOUNDS_CHECK = """
inserted_count <= requested_count
AND already_present_count <= requested_count
AND enqueued_count <= requested_count
AND failed_count <= requested_count
AND inserted_count + already_present_count <= requested_count
AND enqueued_count + failed_count <= requested_count
""".strip()


BATCH_SUBMIT_OPS_COMPLETED_CHECK = """
status != 'completed'
OR (
  completed_at IS NOT NULL
  AND enqueued_count + failed_count = requested_count
)
""".strip()


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
    Column("provider_axis_config_id", Text),
    Column("fair_order_seed", Text, nullable=False),
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
    CheckConstraint(
        PREDICTION_SPECS_PROVIDER_AXIS_CHECK,
        name="ck_dr_dspy_prediction_specs_provider_axis",
    ),
    UniqueConstraint(
        "experiment_name",
        "task_id",
        "repetition_seed",
        "graph_digest",
        "dimensions_digest",
        "provider_kind",
        "endpoint_kind",
        "model",
        "throttle_key",
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
    UniqueConstraint(
        "generation_run_id",
        "prediction_id",
        name="uq_dr_dspy_generation_runs_id_prediction",
    ),
)

node_attempts = Table(
    NODE_ATTEMPTS_TABLE,
    metadata,
    Column("node_attempt_id", Text, primary_key=True),
    Column("generation_run_id", Text, nullable=False),
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
    Column("config_id", Text),
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
        "(status != 'success' OR (output IS NOT NULL AND failure IS NULL)) "
        "AND (status != 'error' OR (failure IS NOT NULL AND output IS NULL))",
        name="ck_dr_dspy_node_attempts_status_payload",
    ),
    CheckConstraint(
        "completed_at >= started_at",
        name="ck_dr_dspy_node_attempts_time_order",
    ),
    CheckConstraint(
        NODE_ATTEMPTS_PROVIDER_CONFIG_CHECK,
        name="ck_dr_dspy_node_attempts_provider_config",
    ),
    UniqueConstraint(
        "generation_run_id",
        "node_id",
        "attempt_index",
        name="uq_dr_dspy_node_attempts_run_node_attempt",
    ),
    ForeignKeyConstraint(
        ["generation_run_id", "prediction_id"],
        [
            f"{GENERATION_RUNS_TABLE}.generation_run_id",
            f"{GENERATION_RUNS_TABLE}.prediction_id",
        ],
        name="fk_dr_dspy_node_attempts_generation_run",
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
    Column("generation_run_id", Text, nullable=False),
    Column("scoring_profile_id", Text, nullable=False),
    Column("scoring_profile_version", Text, nullable=False),
    Column("parser_profile_id", Text, nullable=False),
    Column("parser_version", Text, nullable=False),
    Column("attempt_index", Integer, nullable=False),
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
        "attempt_index >= 0",
        name="ck_dr_dspy_score_attempts_attempt_index",
    ),
    CheckConstraint(
        "(status != 'success' OR (score IS NOT NULL AND failure IS NULL)) "
        "AND (status != 'error' OR ("
        "failure IS NOT NULL "
        "AND score IS NULL "
        "AND per_test_results = '[]'::jsonb"
        "))",
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
        "attempt_index",
        name="uq_dr_dspy_score_attempts_profile",
    ),
    UniqueConstraint(
        "score_attempt_id",
        "prediction_id",
        name="uq_dr_dspy_score_attempts_id_prediction",
    ),
    UniqueConstraint(
        "score_attempt_id",
        "generation_run_id",
        name="uq_dr_dspy_score_attempts_id_run",
    ),
    ForeignKeyConstraint(
        ["generation_run_id", "prediction_id"],
        [
            f"{GENERATION_RUNS_TABLE}.generation_run_id",
            f"{GENERATION_RUNS_TABLE}.prediction_id",
        ],
        name="fk_dr_dspy_score_attempts_generation_run",
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
    Column("generation_run_id", Text),
    Column("score_attempt_id", Text),
    Column("projection_profile_id", Text, primary_key=True),
    Column("projection_version", Text, primary_key=True),
    Column("selected_at", DateTime(timezone=True), nullable=False),
    Column("selection_reason", Text),
    CheckConstraint(
        "generation_run_id IS NOT NULL OR score_attempt_id IS NOT NULL",
        name="ck_dr_dspy_projection_has_selection",
    ),
    ForeignKeyConstraint(
        ["generation_run_id", "prediction_id"],
        [
            f"{GENERATION_RUNS_TABLE}.generation_run_id",
            f"{GENERATION_RUNS_TABLE}.prediction_id",
        ],
        name="fk_dr_dspy_projection_generation_run",
    ),
    ForeignKeyConstraint(
        ["score_attempt_id", "prediction_id"],
        [
            f"{SCORE_ATTEMPTS_TABLE}.score_attempt_id",
            f"{SCORE_ATTEMPTS_TABLE}.prediction_id",
        ],
        name="fk_dr_dspy_projection_score_attempt",
    ),
    ForeignKeyConstraint(
        ["score_attempt_id", "generation_run_id"],
        [
            f"{SCORE_ATTEMPTS_TABLE}.score_attempt_id",
            f"{SCORE_ATTEMPTS_TABLE}.generation_run_id",
        ],
        name="fk_dr_dspy_projection_score_run",
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
    CheckConstraint(
        BATCH_SUBMIT_OPS_COUNT_BOUNDS_CHECK,
        name="ck_dr_dspy_batch_ops_count_bounds",
    ),
    CheckConstraint(
        BATCH_SUBMIT_OPS_COMPLETED_CHECK,
        name="ck_dr_dspy_batch_ops_completed",
    ),
    CheckConstraint(
        "completed_at IS NULL OR completed_at >= created_at",
        name="ck_dr_dspy_batch_ops_time_order",
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
    Column("insert_status", Text, nullable=False),
    Column("enqueue_status", Text, nullable=False),
    Column("enqueue_metadata", JSONB, nullable=False),
    Column("failure", JSONB),
    Column("created_at", DateTime(timezone=True), nullable=False),
    CheckConstraint(
        "item_index >= 0",
        name="ck_dr_dspy_batch_items_item_index",
    ),
    CheckConstraint(
        enum_check("insert_status", BatchSubmitItemInsertStatus),
        name="ck_dr_dspy_batch_items_insert_status",
    ),
    CheckConstraint(
        enum_check("enqueue_status", BatchSubmitItemEnqueueStatus),
        name="ck_dr_dspy_batch_items_enqueue_status",
    ),
    CheckConstraint(
        "(enqueue_status = 'failed' OR failure IS NULL) "
        "AND (enqueue_status != 'failed' OR failure IS NOT NULL)",
        name="ck_dr_dspy_batch_items_enqueue_status_payload",
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

throttle_backoff = Table(
    THROTTLE_BACKOFF_TABLE,
    metadata,
    Column("throttle_key", Text, primary_key=True),
    Column("blocked_until", DateTime(timezone=True)),
    Column("consecutive_failures", Integer, nullable=False),
    Column("failure_class", Text),
    Column("last_error_type", Text),
    Column("last_message", Text),
    Column("metadata", JSONB, nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    CheckConstraint(
        "consecutive_failures >= 0",
        name="ck_dr_dspy_throttle_backoff_failures",
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
Index("ix_dr_dspy_throttle_backoff_blocked_until",
      throttle_backoff.c.blocked_until)

v1_tables: tuple[Table, ...] = (
    experiments,
    prediction_specs,
    generation_runs,
    node_attempts,
    score_attempts,
    prediction_projection,
    batch_submit_operations,
    batch_submit_items,
    throttle_backoff,
)

__all__ = [
    "APPEND_ONLY_OUTCOME_REJECT_FUNCTION",
    "APPEND_ONLY_OUTCOME_TABLE_NAMES",
    "BATCH_SUBMIT_ITEMS_TABLE",
    "BATCH_SUBMIT_OPERATIONS_TABLE",
    "EXPERIMENTS_TABLE",
    "GENERATION_RUNS_TABLE",
    "NODE_ATTEMPTS_TABLE",
    "PREDICTION_PROJECTION_TABLE",
    "PREDICTION_SPECS_TABLE",
    "SCORE_ATTEMPTS_TABLE",
    "THROTTLE_BACKOFF_TABLE",
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
    "throttle_backoff",
    "v1_tables",
]
