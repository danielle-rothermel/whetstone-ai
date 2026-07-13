"""Fresh v6 Whetstone-owned domain schema.

Kernel lifecycle tables are owned by dr-platform under the ``whetstone``
prefix.  These tables deliberately contain only durable domain facts.
"""

from __future__ import annotations

from enum import StrEnum

from dr_code.humaneval import SubmissionOutcome
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

from whetstone.records import (
    GenerationRunStatus,
    NodeAttemptStatus,
    ScoreAttemptStatus,
)

EXPERIMENTS_TABLE = "whetstone_experiments"
PREDICTION_SPECS_TABLE = "whetstone_prediction_specs"
GENERATION_RUNS_TABLE = "whetstone_generation_runs"
NODE_ATTEMPTS_TABLE = "whetstone_node_attempts"
SCORE_ATTEMPTS_TABLE = "whetstone_score_attempts"
SCORE_HARNESS_FAILURES_TABLE = "whetstone_score_harness_failures"
EXPERIMENT_OPERATION_MANIFESTS_TABLE = (
    "whetstone_experiment_operation_manifests"
)
EXPERIMENT_ACCEPTANCE_EVALUATIONS_TABLE = (
    "whetstone_experiment_acceptance_evaluations"
)
EXPERIMENT_ACCEPTANCE_GENERATION_MEMBERS_TABLE = (
    "whetstone_experiment_acceptance_generation_members"
)
EXPERIMENT_ACCEPTANCE_GENERATION_CANDIDATES_TABLE = (
    "whetstone_experiment_acceptance_generation_candidates"
)
EXPERIMENT_ACCEPTANCE_SCORING_MEMBERS_TABLE = (
    "whetstone_experiment_acceptance_scoring_members"
)
EXPERIMENT_ACCEPTANCE_SCORING_CANDIDATES_TABLE = (
    "whetstone_experiment_acceptance_scoring_candidates"
)

V6_TABLE_NAMES = (
    EXPERIMENTS_TABLE,
    PREDICTION_SPECS_TABLE,
    GENERATION_RUNS_TABLE,
    NODE_ATTEMPTS_TABLE,
    SCORE_ATTEMPTS_TABLE,
    SCORE_HARNESS_FAILURES_TABLE,
    EXPERIMENT_OPERATION_MANIFESTS_TABLE,
    EXPERIMENT_ACCEPTANCE_EVALUATIONS_TABLE,
    EXPERIMENT_ACCEPTANCE_GENERATION_MEMBERS_TABLE,
    EXPERIMENT_ACCEPTANCE_GENERATION_CANDIDATES_TABLE,
    EXPERIMENT_ACCEPTANCE_SCORING_MEMBERS_TABLE,
    EXPERIMENT_ACCEPTANCE_SCORING_CANDIDATES_TABLE,
)

APPEND_ONLY_OUTCOME_REJECT_FUNCTION = (
    "whetstone_reject_append_only_outcome_mutation"
)
APPEND_ONLY_OUTCOME_TABLE_NAMES = (
    GENERATION_RUNS_TABLE,
    NODE_ATTEMPTS_TABLE,
    SCORE_ATTEMPTS_TABLE,
    SCORE_HARNESS_FAILURES_TABLE,
    EXPERIMENT_ACCEPTANCE_EVALUATIONS_TABLE,
    EXPERIMENT_ACCEPTANCE_GENERATION_MEMBERS_TABLE,
    EXPERIMENT_ACCEPTANCE_GENERATION_CANDIDATES_TABLE,
    EXPERIMENT_ACCEPTANCE_SCORING_MEMBERS_TABLE,
    EXPERIMENT_ACCEPTANCE_SCORING_CANDIDATES_TABLE,
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
    Column(
        "acceptance_source_version",
        Integer,
        nullable=False,
        server_default="1",
    ),
    Column("current_acceptance_id", Text),
    Column("acceptance_updated_at", DateTime(timezone=True)),
    Column("created_at", DateTime(timezone=True), nullable=False),
    CheckConstraint(
        "acceptance_source_version > 0",
        name="ck_whetstone_experiments_source_version",
    ),
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
    Column("task_snapshot", JSONB, nullable=False),
    Column("graph_snapshot", JSONB, nullable=False),
    Column("dimensions", JSONB, nullable=False),
    Column("provider_configs", JSONB, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    CheckConstraint(
        "repetition_seed >= 0",
        name="ck_whetstone_prediction_specs_repetition_seed",
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
        name="uq_whetstone_prediction_specs_identity",
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
    Column("execution_recipe_digest", Text, nullable=False),
    Column("platform_item_id", Text, nullable=False),
    Column("platform_attempt", Integer, nullable=False),
    Column("status", Text, nullable=False),
    Column("terminal_node_id", Text, nullable=False),
    Column("terminal_output_node_id", Text),
    Column("summary", JSONB, nullable=False),
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("completed_at", DateTime(timezone=True), nullable=False),
    CheckConstraint(
        "attempt_index >= 0 AND platform_attempt >= 0",
        name="ck_whetstone_generation_runs_attempt",
    ),
    CheckConstraint(
        enum_check("status", GenerationRunStatus),
        name="ck_whetstone_generation_runs_status",
    ),
    CheckConstraint(
        "completed_at >= started_at",
        name="ck_whetstone_generation_runs_time_order",
    ),
    UniqueConstraint(
        "prediction_id",
        "attempt_index",
        name="uq_whetstone_generation_runs_prediction_attempt",
    ),
    UniqueConstraint(
        "platform_item_id",
        "platform_attempt",
        name="uq_whetstone_generation_runs_platform_attempt",
    ),
    UniqueConstraint(
        "generation_run_id",
        "prediction_id",
        name="uq_whetstone_generation_runs_id_prediction",
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
        "attempt_index >= 0", name="ck_whetstone_node_attempts_attempt"
    ),
    CheckConstraint(
        enum_check("status", NodeAttemptStatus),
        name="ck_whetstone_node_attempts_status",
    ),
    CheckConstraint(
        "completed_at >= started_at",
        name="ck_whetstone_node_attempts_time_order",
    ),
    UniqueConstraint(
        "generation_run_id",
        "node_id",
        "attempt_index",
        name="uq_whetstone_node_attempts_run_node_attempt",
    ),
    ForeignKeyConstraint(
        ["generation_run_id", "prediction_id"],
        [
            f"{GENERATION_RUNS_TABLE}.generation_run_id",
            f"{GENERATION_RUNS_TABLE}.prediction_id",
        ],
        name="fk_whetstone_node_attempts_generation_run",
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
    Column("attempt_index", Integer, nullable=False),
    Column("execution_recipe_digest", Text, nullable=False),
    Column("platform_item_id", Text, nullable=False),
    Column("platform_attempt", Integer, nullable=False),
    Column("scoring_profile_id", Text, nullable=False),
    Column("scoring_profile_version", Text, nullable=False),
    Column("parser_profile_id", Text, nullable=False),
    Column("parser_version", Text, nullable=False),
    Column("dataset_name", Text, nullable=False),
    Column("dataset_split", Text, nullable=False),
    Column("dataset_snapshot", JSONB, nullable=False),
    Column("status", Text, nullable=False),
    Column("submission_outcome", Text, nullable=False),
    Column("score", Float, nullable=False),
    Column("extracted_submission", JSONB, nullable=False),
    Column("metrics", JSONB),
    Column("per_test_results", JSONB, nullable=False),
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("completed_at", DateTime(timezone=True), nullable=False),
    CheckConstraint(
        "attempt_index >= 0 AND platform_attempt >= 0",
        name="ck_whetstone_score_attempts_attempt",
    ),
    CheckConstraint(
        enum_check("status", ScoreAttemptStatus),
        name="ck_whetstone_score_attempts_status",
    ),
    CheckConstraint(
        enum_check("submission_outcome", SubmissionOutcome),
        name="ck_whetstone_score_attempts_submission_outcome",
    ),
    CheckConstraint(
        "completed_at >= started_at",
        name="ck_whetstone_score_attempts_time_order",
    ),
    UniqueConstraint(
        "generation_run_id",
        "scoring_profile_id",
        "scoring_profile_version",
        "parser_profile_id",
        "parser_version",
        "dataset_name",
        "dataset_split",
        "execution_recipe_digest",
        "attempt_index",
        name="uq_whetstone_score_attempts_profile",
    ),
    UniqueConstraint(
        "platform_item_id",
        "platform_attempt",
        name="uq_whetstone_score_attempts_platform_attempt",
    ),
    ForeignKeyConstraint(
        ["generation_run_id", "prediction_id"],
        [
            f"{GENERATION_RUNS_TABLE}.generation_run_id",
            f"{GENERATION_RUNS_TABLE}.prediction_id",
        ],
        name="fk_whetstone_score_attempts_generation_run",
    ),
)

score_harness_failures = Table(
    SCORE_HARNESS_FAILURES_TABLE,
    metadata,
    Column("score_harness_failure_id", Text, primary_key=True),
    Column(
        "prediction_id",
        Text,
        ForeignKey(f"{PREDICTION_SPECS_TABLE}.prediction_id"),
        nullable=False,
    ),
    Column("generation_run_id", Text, nullable=False),
    Column("attempt_index", Integer, nullable=False),
    Column("execution_recipe_digest", Text, nullable=False),
    Column("platform_item_id", Text, nullable=False),
    Column("platform_attempt", Integer, nullable=False),
    Column("score_attempt_id", Text, nullable=False),
    Column("scoring_profile_id", Text, nullable=False),
    Column("scoring_profile_version", Text, nullable=False),
    Column("parser_profile_id", Text, nullable=False),
    Column("parser_version", Text, nullable=False),
    Column("dataset_name", Text, nullable=False),
    Column("dataset_split", Text, nullable=False),
    Column("failure", JSONB, nullable=False),
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("completed_at", DateTime(timezone=True), nullable=False),
    CheckConstraint(
        "attempt_index >= 0 AND platform_attempt >= 0",
        name="ck_whetstone_score_harness_failures_attempt",
    ),
    CheckConstraint(
        "completed_at >= started_at",
        name="ck_whetstone_score_harness_failures_time_order",
    ),
    UniqueConstraint(
        "generation_run_id",
        "scoring_profile_id",
        "scoring_profile_version",
        "parser_profile_id",
        "parser_version",
        "dataset_name",
        "dataset_split",
        "attempt_index",
        name="uq_whetstone_score_harness_failures_profile",
    ),
    UniqueConstraint(
        "platform_item_id",
        "platform_attempt",
        name="uq_whetstone_score_harness_failures_platform_attempt",
    ),
    ForeignKeyConstraint(
        ["generation_run_id", "prediction_id"],
        [
            f"{GENERATION_RUNS_TABLE}.generation_run_id",
            f"{GENERATION_RUNS_TABLE}.prediction_id",
        ],
        name="fk_whetstone_score_harness_failures_generation_run",
    ),
)

experiment_operation_manifests = Table(
    EXPERIMENT_OPERATION_MANIFESTS_TABLE,
    metadata,
    Column(
        "experiment_name",
        Text,
        ForeignKey(f"{EXPERIMENTS_TABLE}.experiment_name"),
        nullable=False,
    ),
    Column("workflow_role", Text, nullable=False),
    Column("operation_key", Text, nullable=False),
    Column("manifest_digest", Text, nullable=False),
    Column("selection_digest", Text),
    Column("target_ref", JSONB, nullable=False),
    Column("accepted_at", DateTime(timezone=True), nullable=False),
    Column("accepted_scoring_ordinal", Integer),
    UniqueConstraint(
        "experiment_name",
        "workflow_role",
        "operation_key",
        "manifest_digest",
        name="uq_whetstone_experiment_operation_manifest",
    ),
)
Index(
    "uq_whetstone_one_generation_manifest",
    experiment_operation_manifests.c.experiment_name,
    unique=True,
    postgresql_where=experiment_operation_manifests.c.workflow_role
    == "generation",
)
Index(
    "uq_whetstone_scoring_manifest_ordinal",
    experiment_operation_manifests.c.experiment_name,
    experiment_operation_manifests.c.accepted_scoring_ordinal,
    unique=True,
    postgresql_where=experiment_operation_manifests.c.workflow_role
    == "scoring",
)
Index(
    "ix_whetstone_generation_runs_prediction", generation_runs.c.prediction_id
)
Index("ix_whetstone_score_attempts_run", score_attempts.c.generation_run_id)

experiment_acceptance_evaluations = Table(
    EXPERIMENT_ACCEPTANCE_EVALUATIONS_TABLE,
    metadata,
    Column("acceptance_id", Text, primary_key=True),
    Column(
        "experiment_name",
        Text,
        ForeignKey(
            f"{EXPERIMENTS_TABLE}.experiment_name", ondelete="RESTRICT"
        ),
        nullable=False,
    ),
    Column("acceptance_source_version", Integer, nullable=False),
    Column("status", Text, nullable=False),
    Column("generation_operation_key", Text, nullable=False),
    Column("generation_manifest_digest", Text, nullable=False),
    Column("scoring_relationships", JSONB, nullable=False),
    Column("scoring_relationships_digest", Text, nullable=False),
    Column("selected_scoring_candidates", JSONB, nullable=False),
    Column("selected_scoring_candidates_digest", Text, nullable=False),
    Column("domain_cut", JSONB, nullable=False),
    Column("domain_cut_digest", Text, nullable=False),
    Column("platform_cut", JSONB, nullable=False),
    Column("platform_cut_digest", Text, nullable=False),
    Column("required_profiles", JSONB, nullable=False),
    Column("required_profiles_digest", Text, nullable=False),
    Column("policy", JSONB, nullable=False),
    Column("policy_digest", Text, nullable=False),
    Column("observed_matrix", JSONB, nullable=False),
    Column("observed_matrix_digest", Text, nullable=False),
    Column("expected_count", Integer, nullable=False),
    Column("accepted_count", Integer, nullable=False),
    Column("missing_count", Integer, nullable=False),
    Column("rejected_count", Integer, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    CheckConstraint(
        "acceptance_source_version > 0",
        name="ck_whetstone_acceptance_source_version",
    ),
)

experiment_acceptance_generation_members = Table(
    EXPERIMENT_ACCEPTANCE_GENERATION_MEMBERS_TABLE,
    metadata,
    Column(
        "acceptance_id",
        Text,
        ForeignKey(
            f"{EXPERIMENT_ACCEPTANCE_EVALUATIONS_TABLE}.acceptance_id",
            ondelete="RESTRICT",
        ),
        primary_key=True,
    ),
    Column(
        "prediction_id",
        Text,
        ForeignKey(
            f"{PREDICTION_SPECS_TABLE}.prediction_id", ondelete="RESTRICT"
        ),
        primary_key=True,
    ),
    Column("disposition", Text, nullable=False),
    Column("generation_run_id", Text, nullable=True),
    Column("generation_operation_key", Text),
    Column("platform_item_id", Text),
    Column("platform_attempt", Integer),
)
experiment_acceptance_generation_candidates = Table(
    EXPERIMENT_ACCEPTANCE_GENERATION_CANDIDATES_TABLE,
    metadata,
    Column(
        "acceptance_id",
        Text,
        ForeignKey(
            f"{EXPERIMENT_ACCEPTANCE_EVALUATIONS_TABLE}.acceptance_id",
            ondelete="RESTRICT",
        ),
        primary_key=True,
    ),
    Column(
        "prediction_id",
        Text,
        ForeignKey(
            f"{PREDICTION_SPECS_TABLE}.prediction_id", ondelete="RESTRICT"
        ),
        primary_key=True,
    ),
    Column(
        "generation_run_id",
        Text,
        ForeignKey(
            f"{GENERATION_RUNS_TABLE}.generation_run_id", ondelete="RESTRICT"
        ),
        primary_key=True,
    ),
    Column("disposition", Text, nullable=False),
    Column("platform_item_id", Text, nullable=False),
    Column("platform_attempt", Integer, nullable=False),
)
experiment_acceptance_scoring_members = Table(
    EXPERIMENT_ACCEPTANCE_SCORING_MEMBERS_TABLE,
    metadata,
    Column(
        "acceptance_id",
        Text,
        ForeignKey(
            f"{EXPERIMENT_ACCEPTANCE_EVALUATIONS_TABLE}.acceptance_id",
            ondelete="RESTRICT",
        ),
        primary_key=True,
    ),
    Column(
        "prediction_id",
        Text,
        ForeignKey(
            f"{PREDICTION_SPECS_TABLE}.prediction_id", ondelete="RESTRICT"
        ),
        primary_key=True,
    ),
    Column("scoring_profile_id", Text, primary_key=True),
    Column("scoring_profile_version", Text, primary_key=True),
    Column("parser_profile_id", Text, primary_key=True),
    Column("parser_version", Text, primary_key=True),
    Column("dataset_name", Text, primary_key=True),
    Column("dataset_split", Text, primary_key=True),
    Column("disposition", Text, nullable=False),
    Column("generation_run_id", Text),
    Column("score_attempt_id", Text),
    Column("accepted_scoring_ordinal", Integer),
    Column("scoring_operation_key", Text),
    Column("platform_item_id", Text),
    Column("platform_attempt", Integer),
    Column("manifest_digest", Text),
)
experiment_acceptance_scoring_candidates = Table(
    EXPERIMENT_ACCEPTANCE_SCORING_CANDIDATES_TABLE,
    metadata,
    Column(
        "acceptance_id",
        Text,
        ForeignKey(
            f"{EXPERIMENT_ACCEPTANCE_EVALUATIONS_TABLE}.acceptance_id",
            ondelete="RESTRICT",
        ),
        primary_key=True,
    ),
    Column("prediction_id", Text, primary_key=True),
    Column("scoring_profile_id", Text, primary_key=True),
    Column("scoring_profile_version", Text, primary_key=True),
    Column("parser_profile_id", Text, primary_key=True),
    Column("parser_version", Text, primary_key=True),
    Column("dataset_name", Text, primary_key=True),
    Column("dataset_split", Text, primary_key=True),
    Column("accepted_scoring_ordinal", Integer, primary_key=True),
    Column("score_attempt_id", Text, primary_key=True),
    Column("generation_run_id", Text, nullable=False),
    Column("disposition", Text, nullable=False),
    Column("operation_key", Text, nullable=False),
    Column("manifest_digest", Text, nullable=False),
    Column("platform_item_id", Text, nullable=False),
    Column("platform_attempt", Integer, nullable=False),
    Column("status", Text, nullable=False),
)

v6_tables: tuple[Table, ...] = (
    experiments,
    prediction_specs,
    generation_runs,
    node_attempts,
    score_attempts,
    score_harness_failures,
    experiment_operation_manifests,
    experiment_acceptance_evaluations,
    experiment_acceptance_generation_members,
    experiment_acceptance_generation_candidates,
    experiment_acceptance_scoring_members,
    experiment_acceptance_scoring_candidates,
)

__all__ = [name for name in globals() if name.isupper()] + [
    "experiments",
    "prediction_specs",
    "generation_runs",
    "node_attempts",
    "score_attempts",
    "score_harness_failures",
    "experiment_operation_manifests",
    "experiment_acceptance_evaluations",
    "experiment_acceptance_generation_members",
    "experiment_acceptance_generation_candidates",
    "experiment_acceptance_scoring_members",
    "experiment_acceptance_scoring_candidates",
    "metadata",
    "v6_tables",
]
