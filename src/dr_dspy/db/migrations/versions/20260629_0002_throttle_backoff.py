from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20260629_0002"
down_revision = "20260629_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "dr_dspy_throttle_backoff",
        sa.Column("throttle_key", sa.Text(), primary_key=True),
        sa.Column("blocked_until", sa.DateTime(timezone=True)),
        sa.Column("consecutive_failures", sa.Integer(), nullable=False),
        sa.Column("failure_class", sa.Text()),
        sa.Column("last_error_type", sa.Text()),
        sa.Column("last_message", sa.Text()),
        sa.Column("metadata", postgresql.JSONB(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "consecutive_failures >= 0",
            name="ck_dr_dspy_throttle_backoff_failures",
        ),
    )
    op.create_index(
        "ix_dr_dspy_throttle_backoff_blocked_until",
        "dr_dspy_throttle_backoff",
        ["blocked_until"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_dr_dspy_throttle_backoff_blocked_until",
        table_name="dr_dspy_throttle_backoff",
    )
    op.drop_table("dr_dspy_throttle_backoff")
