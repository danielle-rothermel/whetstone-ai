from __future__ import annotations

from alembic import op

revision = "20260630_0004"
down_revision = "20260630_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "UPDATE dr_dspy_batch_submit_operations "
        "SET status = 'enqueuing' "
        "WHERE status = 'prepared'"
    )
    op.drop_constraint(
        "ck_dr_dspy_batch_ops_status",
        "dr_dspy_batch_submit_operations",
        type_="check",
    )
    op.create_check_constraint(
        "ck_dr_dspy_batch_ops_status",
        "dr_dspy_batch_submit_operations",
        "status IN ('enqueuing', 'completed', 'partial', 'error')",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_dr_dspy_batch_ops_status",
        "dr_dspy_batch_submit_operations",
        type_="check",
    )
    op.create_check_constraint(
        "ck_dr_dspy_batch_ops_status",
        "dr_dspy_batch_submit_operations",
        "status IN ('prepared', 'enqueuing', 'completed', 'partial', 'error')",
    )
