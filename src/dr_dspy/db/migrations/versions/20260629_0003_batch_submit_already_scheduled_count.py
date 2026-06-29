from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260629_0003"
down_revision = "20260629_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "dr_dspy_batch_submit_operations",
        sa.Column(
            "already_scheduled_count",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )
    op.alter_column(
        "dr_dspy_batch_submit_operations",
        "already_scheduled_count",
        server_default=None,
    )
    op.drop_constraint(
        "ck_dr_dspy_batch_ops_counts",
        "dr_dspy_batch_submit_operations",
        type_="check",
    )
    op.create_check_constraint(
        "ck_dr_dspy_batch_ops_counts",
        "dr_dspy_batch_submit_operations",
        "requested_count >= 0 "
        "AND inserted_count >= 0 "
        "AND already_present_count >= 0 "
        "AND enqueued_count >= 0 "
        "AND already_scheduled_count >= 0 "
        "AND failed_count >= 0",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_dr_dspy_batch_ops_counts",
        "dr_dspy_batch_submit_operations",
        type_="check",
    )
    op.create_check_constraint(
        "ck_dr_dspy_batch_ops_counts",
        "dr_dspy_batch_submit_operations",
        "requested_count >= 0 "
        "AND inserted_count >= 0 "
        "AND already_present_count >= 0 "
        "AND enqueued_count >= 0 "
        "AND failed_count >= 0",
    )
    op.drop_column(
        "dr_dspy_batch_submit_operations",
        "already_scheduled_count",
    )
