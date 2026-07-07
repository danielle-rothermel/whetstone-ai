from __future__ import annotations

from alembic import op

from whetstone.db.schema import (
    BATCH_SUBMIT_OPS_COMPLETED_CHECK,
    BATCH_SUBMIT_OPS_COUNT_BOUNDS_CHECK,
)

revision = "20260630_0002"
down_revision = "20260630_0001"
branch_labels = None
depends_on = None

_OLD_BATCH_SUBMIT_OPS_COUNT_BOUNDS_CHECK = """
inserted_count <= requested_count
AND already_present_count <= requested_count
AND enqueued_count <= requested_count
AND failed_count <= requested_count
AND inserted_count + already_present_count <= requested_count
AND enqueued_count + failed_count <= requested_count
""".strip()

_OLD_BATCH_SUBMIT_OPS_COMPLETED_CHECK = """
status != 'completed'
OR (
  completed_at IS NOT NULL
  AND enqueued_count + failed_count = requested_count
)
""".strip()


def upgrade() -> None:
    op.drop_constraint(
        "ck_dr_dspy_batch_ops_count_bounds",
        "dr_dspy_batch_submit_operations",
        type_="check",
    )
    op.create_check_constraint(
        "ck_dr_dspy_batch_ops_count_bounds",
        "dr_dspy_batch_submit_operations",
        BATCH_SUBMIT_OPS_COUNT_BOUNDS_CHECK,
    )
    op.drop_constraint(
        "ck_dr_dspy_batch_ops_completed",
        "dr_dspy_batch_submit_operations",
        type_="check",
    )
    op.create_check_constraint(
        "ck_dr_dspy_batch_ops_completed",
        "dr_dspy_batch_submit_operations",
        BATCH_SUBMIT_OPS_COMPLETED_CHECK,
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_dr_dspy_batch_ops_count_bounds",
        "dr_dspy_batch_submit_operations",
        type_="check",
    )
    op.create_check_constraint(
        "ck_dr_dspy_batch_ops_count_bounds",
        "dr_dspy_batch_submit_operations",
        _OLD_BATCH_SUBMIT_OPS_COUNT_BOUNDS_CHECK,
    )
    op.drop_constraint(
        "ck_dr_dspy_batch_ops_completed",
        "dr_dspy_batch_submit_operations",
        type_="check",
    )
    op.create_check_constraint(
        "ck_dr_dspy_batch_ops_completed",
        "dr_dspy_batch_submit_operations",
        _OLD_BATCH_SUBMIT_OPS_COMPLETED_CHECK,
    )
