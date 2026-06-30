from __future__ import annotations

from alembic import op

revision = "20260630_0003"
down_revision = "20260630_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        UPDATE dr_dspy_batch_submit_items
        SET enqueue_metadata = '{}'::jsonb
        WHERE enqueue_status = 'pending'
          AND enqueue_metadata != '{}'::jsonb
        """
    )
    op.drop_constraint(
        "ck_dr_dspy_batch_items_enqueue_status",
        "dr_dspy_batch_submit_items",
        type_="check",
    )
    op.create_check_constraint(
        "ck_dr_dspy_batch_items_enqueue_status",
        "dr_dspy_batch_submit_items",
        "enqueue_status IN "
        "('pending', 'claiming', 'enqueued', "
        "'workflow_already_present', 'failed')",
    )


def downgrade() -> None:
    op.execute(
        """
        UPDATE dr_dspy_batch_submit_items
        SET enqueue_status = 'pending',
            enqueue_metadata = '{}'::jsonb
        WHERE enqueue_status = 'claiming'
        """
    )
    op.drop_constraint(
        "ck_dr_dspy_batch_items_enqueue_status",
        "dr_dspy_batch_submit_items",
        type_="check",
    )
    op.create_check_constraint(
        "ck_dr_dspy_batch_items_enqueue_status",
        "dr_dspy_batch_submit_items",
        "enqueue_status IN "
        "('pending', 'enqueued', 'workflow_already_present', 'failed')",
    )
