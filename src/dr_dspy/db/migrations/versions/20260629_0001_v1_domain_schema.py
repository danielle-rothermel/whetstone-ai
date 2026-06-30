from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

from dr_dspy.db.schema import (
    BATCH_SUBMIT_OPS_COMPLETED_CHECK,
    BATCH_SUBMIT_OPS_COUNT_BOUNDS_CHECK,
    NODE_ATTEMPTS_PROVIDER_CONFIG_CHECK,
    PREDICTION_SPECS_PROVIDER_AXIS_CHECK,
)

revision = "20260629_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "dr_dspy_experiments",
        sa.Column("experiment_name", sa.Text(), primary_key=True),
        sa.Column("description", sa.Text()),
        sa.Column("config_metadata", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "dr_dspy_prediction_specs",
        sa.Column("prediction_id", sa.Text(), primary_key=True),
        sa.Column("experiment_name", sa.Text(), nullable=False),
        sa.Column("task_id", sa.Text(), nullable=False),
        sa.Column("repetition_seed", sa.Integer(), nullable=False),
        sa.Column("graph_digest", sa.Text(), nullable=False),
        sa.Column("dimensions_digest", sa.Text(), nullable=False),
        sa.Column("graph_layout", sa.Text(), nullable=False),
        sa.Column("provider_kind", sa.Text(), nullable=False),
        sa.Column("endpoint_kind", sa.Text(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("throttle_key", sa.Text(), nullable=False),
        sa.Column("provider_axis_config_id", sa.Text()),
        sa.Column("fair_order_seed", sa.Text(), nullable=False),
        sa.Column("fair_order_key", sa.Text(), nullable=False),
        sa.Column("task_snapshot", postgresql.JSONB(), nullable=False),
        sa.Column("graph_snapshot", postgresql.JSONB(), nullable=False),
        sa.Column("dimensions", postgresql.JSONB(), nullable=False),
        sa.Column("provider_configs", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["experiment_name"],
            ["dr_dspy_experiments.experiment_name"],
        ),
        sa.CheckConstraint(
            "repetition_seed >= 0",
            name="ck_dr_dspy_prediction_specs_repetition_seed",
        ),
        sa.CheckConstraint(
            PREDICTION_SPECS_PROVIDER_AXIS_CHECK,
            name="ck_dr_dspy_prediction_specs_provider_axis",
        ),
        sa.UniqueConstraint(
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
    op.create_table(
        "dr_dspy_generation_runs",
        sa.Column("generation_run_id", sa.Text(), primary_key=True),
        sa.Column("prediction_id", sa.Text(), nullable=False),
        sa.Column("attempt_index", sa.Integer(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("terminal_node_id", sa.Text(), nullable=False),
        sa.Column("terminal_output_node_id", sa.Text()),
        sa.Column("summary", postgresql.JSONB(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["prediction_id"],
            ["dr_dspy_prediction_specs.prediction_id"],
        ),
        sa.CheckConstraint(
            "attempt_index >= 0",
            name="ck_dr_dspy_generation_runs_attempt_index",
        ),
        sa.CheckConstraint(
            "status IN ('success', 'error', 'blocked', 'partial')",
            name="ck_dr_dspy_generation_runs_status",
        ),
        sa.CheckConstraint(
            "completed_at >= started_at",
            name="ck_dr_dspy_generation_runs_time_order",
        ),
        sa.UniqueConstraint(
            "prediction_id",
            "attempt_index",
            name="uq_dr_dspy_generation_runs_attempt",
        ),
        sa.UniqueConstraint(
            "generation_run_id",
            "prediction_id",
            name="uq_dr_dspy_generation_runs_id_prediction",
        ),
    )
    op.create_table(
        "dr_dspy_node_attempts",
        sa.Column("node_attempt_id", sa.Text(), primary_key=True),
        sa.Column("generation_run_id", sa.Text(), nullable=False),
        sa.Column("prediction_id", sa.Text(), nullable=False),
        sa.Column("node_id", sa.Text(), nullable=False),
        sa.Column("attempt_index", sa.Integer(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("provider_kind", sa.Text()),
        sa.Column("endpoint_kind", sa.Text()),
        sa.Column("model", sa.Text()),
        sa.Column("throttle_key", sa.Text()),
        sa.Column("config_id", sa.Text()),
        sa.Column("provider_config", postgresql.JSONB()),
        sa.Column("output", postgresql.JSONB()),
        sa.Column("usage_cost", postgresql.JSONB(), nullable=False),
        sa.Column("response_metadata", postgresql.JSONB(), nullable=False),
        sa.Column("failure", postgresql.JSONB()),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["prediction_id"],
            ["dr_dspy_prediction_specs.prediction_id"],
        ),
        sa.ForeignKeyConstraint(
            ["generation_run_id", "prediction_id"],
            [
                "dr_dspy_generation_runs.generation_run_id",
                "dr_dspy_generation_runs.prediction_id",
            ],
            name="fk_dr_dspy_node_attempts_generation_run",
        ),
        sa.CheckConstraint(
            "attempt_index >= 0",
            name="ck_dr_dspy_node_attempts_attempt_index",
        ),
        sa.CheckConstraint(
            "status IN ('success', 'error')",
            name="ck_dr_dspy_node_attempts_status",
        ),
        sa.CheckConstraint(
            "(status != 'success' OR "
            "(output IS NOT NULL AND failure IS NULL)) "
            "AND (status != 'error' OR "
            "(failure IS NOT NULL AND output IS NULL))",
            name="ck_dr_dspy_node_attempts_status_payload",
        ),
        sa.CheckConstraint(
            "completed_at >= started_at",
            name="ck_dr_dspy_node_attempts_time_order",
        ),
        sa.CheckConstraint(
            NODE_ATTEMPTS_PROVIDER_CONFIG_CHECK,
            name="ck_dr_dspy_node_attempts_provider_config",
        ),
        sa.UniqueConstraint(
            "generation_run_id",
            "node_id",
            "attempt_index",
            name="uq_dr_dspy_node_attempts_run_node_attempt",
        ),
    )
    op.create_table(
        "dr_dspy_score_attempts",
        sa.Column("score_attempt_id", sa.Text(), primary_key=True),
        sa.Column("prediction_id", sa.Text(), nullable=False),
        sa.Column("generation_run_id", sa.Text(), nullable=False),
        sa.Column("scoring_profile_id", sa.Text(), nullable=False),
        sa.Column("scoring_profile_version", sa.Text(), nullable=False),
        sa.Column("parser_profile_id", sa.Text(), nullable=False),
        sa.Column("parser_version", sa.Text(), nullable=False),
        sa.Column("attempt_index", sa.Integer(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("generated_code_outcome", sa.Text()),
        sa.Column("score", sa.Float()),
        sa.Column("extracted_code", postgresql.JSONB()),
        sa.Column("metrics", postgresql.JSONB()),
        sa.Column("per_test_results", postgresql.JSONB(), nullable=False),
        sa.Column("failure", postgresql.JSONB()),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["prediction_id"],
            ["dr_dspy_prediction_specs.prediction_id"],
        ),
        sa.ForeignKeyConstraint(
            ["generation_run_id", "prediction_id"],
            [
                "dr_dspy_generation_runs.generation_run_id",
                "dr_dspy_generation_runs.prediction_id",
            ],
            name="fk_dr_dspy_score_attempts_generation_run",
        ),
        sa.CheckConstraint(
            "generated_code_outcome IS NULL OR "
            "(generated_code_outcome IN "
            "('passed', 'tests_failed', 'empty_generation', "
            "'extraction_failed', 'no_top_level_functions'))",
            name="ck_dr_dspy_score_attempts_generated_code_outcome",
        ),
        sa.CheckConstraint(
            "score IS NULL OR (score >= 0 AND score <= 1)",
            name="ck_dr_dspy_score_attempts_score_range",
        ),
        sa.CheckConstraint(
            "attempt_index >= 0",
            name="ck_dr_dspy_score_attempts_attempt_index",
        ),
        sa.CheckConstraint(
            "status IN ('success', 'error')",
            name="ck_dr_dspy_score_attempts_status",
        ),
        sa.CheckConstraint(
            "(status != 'success' OR "
            "(score IS NOT NULL AND failure IS NULL)) "
            "AND (status != 'error' OR "
            "(failure IS NOT NULL "
            "AND score IS NULL "
            "AND per_test_results = '[]'::jsonb))",
            name="ck_dr_dspy_score_attempts_status_payload",
        ),
        sa.CheckConstraint(
            "completed_at >= started_at",
            name="ck_dr_dspy_score_attempts_time_order",
        ),
        sa.UniqueConstraint(
            "prediction_id",
            "generation_run_id",
            "scoring_profile_id",
            "scoring_profile_version",
            "parser_profile_id",
            "parser_version",
            "attempt_index",
            name="uq_dr_dspy_score_attempts_profile",
        ),
        sa.UniqueConstraint(
            "score_attempt_id",
            "prediction_id",
            name="uq_dr_dspy_score_attempts_id_prediction",
        ),
        sa.UniqueConstraint(
            "score_attempt_id",
            "generation_run_id",
            name="uq_dr_dspy_score_attempts_id_run",
        ),
    )
    op.create_table(
        "dr_dspy_prediction_projection",
        sa.Column("prediction_id", sa.Text(), primary_key=True),
        sa.Column("generation_run_id", sa.Text()),
        sa.Column("score_attempt_id", sa.Text()),
        sa.Column("projection_profile_id", sa.Text(), primary_key=True),
        sa.Column("projection_version", sa.Text(), primary_key=True),
        sa.Column("selected_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("selection_reason", sa.Text()),
        sa.ForeignKeyConstraint(
            ["prediction_id"],
            ["dr_dspy_prediction_specs.prediction_id"],
        ),
        sa.ForeignKeyConstraint(
            ["generation_run_id", "prediction_id"],
            [
                "dr_dspy_generation_runs.generation_run_id",
                "dr_dspy_generation_runs.prediction_id",
            ],
            name="fk_dr_dspy_projection_generation_run",
        ),
        sa.ForeignKeyConstraint(
            ["score_attempt_id", "prediction_id"],
            [
                "dr_dspy_score_attempts.score_attempt_id",
                "dr_dspy_score_attempts.prediction_id",
            ],
            name="fk_dr_dspy_projection_score_attempt",
        ),
        sa.ForeignKeyConstraint(
            ["score_attempt_id", "generation_run_id"],
            [
                "dr_dspy_score_attempts.score_attempt_id",
                "dr_dspy_score_attempts.generation_run_id",
            ],
            name="fk_dr_dspy_projection_score_run",
        ),
        sa.CheckConstraint(
            "generation_run_id IS NOT NULL OR score_attempt_id IS NOT NULL",
            name="ck_dr_dspy_projection_has_selection",
        ),
    )
    op.create_table(
        "dr_dspy_batch_submit_operations",
        sa.Column("operation_key", sa.Text(), primary_key=True),
        sa.Column("experiment_name", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("requested_count", sa.Integer(), nullable=False),
        sa.Column("inserted_count", sa.Integer(), nullable=False),
        sa.Column("already_present_count", sa.Integer(), nullable=False),
        sa.Column("enqueued_count", sa.Integer(), nullable=False),
        sa.Column("failed_count", sa.Integer(), nullable=False),
        sa.Column("spec", postgresql.JSONB(), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.ForeignKeyConstraint(
            ["experiment_name"],
            ["dr_dspy_experiments.experiment_name"],
        ),
        sa.CheckConstraint(
            "requested_count >= 0 "
            "AND inserted_count >= 0 "
            "AND already_present_count >= 0 "
            "AND enqueued_count >= 0 "
            "AND failed_count >= 0",
            name="ck_dr_dspy_batch_ops_counts",
        ),
        sa.CheckConstraint(
            BATCH_SUBMIT_OPS_COUNT_BOUNDS_CHECK,
            name="ck_dr_dspy_batch_ops_count_bounds",
        ),
        sa.CheckConstraint(
            BATCH_SUBMIT_OPS_COMPLETED_CHECK,
            name="ck_dr_dspy_batch_ops_completed",
        ),
        sa.CheckConstraint(
            "status IN ('prepared', 'completed', 'partial', 'error')",
            name="ck_dr_dspy_batch_ops_status",
        ),
        sa.CheckConstraint(
            "completed_at IS NULL OR completed_at >= created_at",
            name="ck_dr_dspy_batch_ops_time_order",
        ),
    )
    op.create_table(
        "dr_dspy_batch_submit_items",
        sa.Column("batch_submit_item_id", sa.Text(), primary_key=True),
        sa.Column("operation_key", sa.Text(), nullable=False),
        sa.Column("item_index", sa.Integer(), nullable=False),
        sa.Column("prediction_id", sa.Text(), nullable=False),
        sa.Column("fair_order_key", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("enqueue_metadata", postgresql.JSONB(), nullable=False),
        sa.Column("failure", postgresql.JSONB()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["prediction_id"],
            ["dr_dspy_prediction_specs.prediction_id"],
        ),
        sa.ForeignKeyConstraint(
            ["operation_key"],
            ["dr_dspy_batch_submit_operations.operation_key"],
        ),
        sa.CheckConstraint(
            "item_index >= 0",
            name="ck_dr_dspy_batch_items_item_index",
        ),
        sa.CheckConstraint(
            "status IN ('inserted', 'already_present', 'enqueued', 'failed')",
            name="ck_dr_dspy_batch_items_status",
        ),
        sa.CheckConstraint(
            "(status = 'failed' OR failure IS NULL) "
            "AND (status != 'failed' OR failure IS NOT NULL)",
            name="ck_dr_dspy_batch_items_status_payload",
        ),
        sa.UniqueConstraint(
            "operation_key",
            "item_index",
            name="uq_dr_dspy_batch_items_operation_index",
        ),
        sa.UniqueConstraint(
            "operation_key",
            "prediction_id",
            name="uq_dr_dspy_batch_items_operation_prediction",
        ),
    )
    op.create_index(
        "ix_dr_dspy_prediction_specs_experiment",
        "dr_dspy_prediction_specs",
        ["experiment_name"],
    )
    op.create_index(
        "ix_dr_dspy_prediction_specs_fair_order",
        "dr_dspy_prediction_specs",
        ["fair_order_key"],
    )
    op.create_index(
        "ix_dr_dspy_prediction_specs_graph",
        "dr_dspy_prediction_specs",
        ["graph_layout", "graph_digest"],
    )
    op.create_index(
        "ix_dr_dspy_prediction_specs_model",
        "dr_dspy_prediction_specs",
        ["provider_kind", "endpoint_kind", "model"],
    )
    op.create_index(
        "ix_dr_dspy_prediction_specs_task",
        "dr_dspy_prediction_specs",
        ["task_id"],
    )
    op.create_index(
        "ix_dr_dspy_generation_runs_prediction",
        "dr_dspy_generation_runs",
        ["prediction_id"],
    )
    op.create_index(
        "ix_dr_dspy_generation_runs_status",
        "dr_dspy_generation_runs",
        ["status"],
    )
    op.create_index(
        "ix_dr_dspy_node_attempts_model",
        "dr_dspy_node_attempts",
        ["provider_kind", "endpoint_kind", "model"],
    )
    op.create_index(
        "ix_dr_dspy_node_attempts_node",
        "dr_dspy_node_attempts",
        ["node_id"],
    )
    op.create_index(
        "ix_dr_dspy_node_attempts_prediction",
        "dr_dspy_node_attempts",
        ["prediction_id"],
    )
    op.create_index(
        "ix_dr_dspy_node_attempts_run",
        "dr_dspy_node_attempts",
        ["generation_run_id"],
    )
    op.create_index(
        "ix_dr_dspy_score_attempts_generated_code_outcome",
        "dr_dspy_score_attempts",
        ["generated_code_outcome"],
    )
    op.create_index(
        "ix_dr_dspy_score_attempts_parser",
        "dr_dspy_score_attempts",
        ["parser_profile_id", "parser_version"],
    )
    op.create_index(
        "ix_dr_dspy_score_attempts_prediction",
        "dr_dspy_score_attempts",
        ["prediction_id"],
    )
    op.create_index(
        "ix_dr_dspy_score_attempts_profile",
        "dr_dspy_score_attempts",
        ["scoring_profile_id", "scoring_profile_version"],
    )
    op.create_index(
        "ix_dr_dspy_score_attempts_run",
        "dr_dspy_score_attempts",
        ["generation_run_id"],
    )
    op.create_index(
        "ix_dr_dspy_projection_generation",
        "dr_dspy_prediction_projection",
        ["generation_run_id"],
    )
    op.create_index(
        "ix_dr_dspy_projection_profile",
        "dr_dspy_prediction_projection",
        ["projection_profile_id", "projection_version"],
    )
    op.create_index(
        "ix_dr_dspy_projection_score",
        "dr_dspy_prediction_projection",
        ["score_attempt_id"],
    )
    op.create_index(
        "ix_dr_dspy_batch_ops_experiment",
        "dr_dspy_batch_submit_operations",
        ["experiment_name"],
    )
    op.create_index(
        "ix_dr_dspy_batch_ops_status",
        "dr_dspy_batch_submit_operations",
        ["status"],
    )
    op.create_index(
        "ix_dr_dspy_batch_items_fair_order",
        "dr_dspy_batch_submit_items",
        ["fair_order_key"],
    )
    op.create_index(
        "ix_dr_dspy_batch_items_operation",
        "dr_dspy_batch_submit_items",
        ["operation_key"],
    )
    op.create_index(
        "ix_dr_dspy_batch_items_prediction",
        "dr_dspy_batch_submit_items",
        ["prediction_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_dr_dspy_batch_items_prediction",
        table_name="dr_dspy_batch_submit_items",
    )
    op.drop_index(
        "ix_dr_dspy_batch_items_operation",
        table_name="dr_dspy_batch_submit_items",
    )
    op.drop_index(
        "ix_dr_dspy_batch_items_fair_order",
        table_name="dr_dspy_batch_submit_items",
    )
    op.drop_index(
        "ix_dr_dspy_batch_ops_status",
        table_name="dr_dspy_batch_submit_operations",
    )
    op.drop_index(
        "ix_dr_dspy_batch_ops_experiment",
        table_name="dr_dspy_batch_submit_operations",
    )
    op.drop_index(
        "ix_dr_dspy_projection_score",
        table_name="dr_dspy_prediction_projection",
    )
    op.drop_index(
        "ix_dr_dspy_projection_profile",
        table_name="dr_dspy_prediction_projection",
    )
    op.drop_index(
        "ix_dr_dspy_projection_generation",
        table_name="dr_dspy_prediction_projection",
    )
    op.drop_index(
        "ix_dr_dspy_score_attempts_run",
        table_name="dr_dspy_score_attempts",
    )
    op.drop_index(
        "ix_dr_dspy_score_attempts_profile",
        table_name="dr_dspy_score_attempts",
    )
    op.drop_index(
        "ix_dr_dspy_score_attempts_prediction",
        table_name="dr_dspy_score_attempts",
    )
    op.drop_index(
        "ix_dr_dspy_score_attempts_parser",
        table_name="dr_dspy_score_attempts",
    )
    op.drop_index(
        "ix_dr_dspy_score_attempts_generated_code_outcome",
        table_name="dr_dspy_score_attempts",
    )
    op.drop_index(
        "ix_dr_dspy_node_attempts_run",
        table_name="dr_dspy_node_attempts",
    )
    op.drop_index(
        "ix_dr_dspy_node_attempts_prediction",
        table_name="dr_dspy_node_attempts",
    )
    op.drop_index(
        "ix_dr_dspy_node_attempts_node",
        table_name="dr_dspy_node_attempts",
    )
    op.drop_index(
        "ix_dr_dspy_node_attempts_model",
        table_name="dr_dspy_node_attempts",
    )
    op.drop_index(
        "ix_dr_dspy_generation_runs_status",
        table_name="dr_dspy_generation_runs",
    )
    op.drop_index(
        "ix_dr_dspy_generation_runs_prediction",
        table_name="dr_dspy_generation_runs",
    )
    op.drop_index(
        "ix_dr_dspy_prediction_specs_task",
        table_name="dr_dspy_prediction_specs",
    )
    op.drop_index(
        "ix_dr_dspy_prediction_specs_model",
        table_name="dr_dspy_prediction_specs",
    )
    op.drop_index(
        "ix_dr_dspy_prediction_specs_graph",
        table_name="dr_dspy_prediction_specs",
    )
    op.drop_index(
        "ix_dr_dspy_prediction_specs_fair_order",
        table_name="dr_dspy_prediction_specs",
    )
    op.drop_index(
        "ix_dr_dspy_prediction_specs_experiment",
        table_name="dr_dspy_prediction_specs",
    )
    op.drop_table("dr_dspy_batch_submit_items")
    op.drop_table("dr_dspy_batch_submit_operations")
    op.drop_table("dr_dspy_prediction_projection")
    op.drop_table("dr_dspy_score_attempts")
    op.drop_table("dr_dspy_node_attempts")
    op.drop_table("dr_dspy_generation_runs")
    op.drop_table("dr_dspy_prediction_specs")
    op.drop_table("dr_dspy_experiments")
