from __future__ import annotations

from alembic import op

from dr_dspy.db.schema import (
    APPEND_ONLY_OUTCOME_REJECT_FUNCTION,
    APPEND_ONLY_OUTCOME_TABLE_NAMES,
)

revision = "20260630_0001"
down_revision = "20260629_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION {APPEND_ONLY_OUTCOME_REJECT_FUNCTION}()
        RETURNS trigger AS $$
        BEGIN
          RAISE EXCEPTION
            'append-only table % does not allow UPDATE or DELETE',
            TG_TABLE_NAME;
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    for table_name in APPEND_ONLY_OUTCOME_TABLE_NAMES:
        trigger_name = f"tr_{table_name}_append_only"
        op.execute(
            f"""
            CREATE TRIGGER {trigger_name}
            BEFORE UPDATE OR DELETE ON {table_name}
            FOR EACH ROW
            EXECUTE PROCEDURE {APPEND_ONLY_OUTCOME_REJECT_FUNCTION}();
            """
        )


def downgrade() -> None:
    for table_name in APPEND_ONLY_OUTCOME_TABLE_NAMES:
        trigger_name = f"tr_{table_name}_append_only"
        op.execute(
            f"DROP TRIGGER IF EXISTS {trigger_name} ON {table_name}"
        )
    op.execute(
        f"DROP FUNCTION IF EXISTS {APPEND_ONLY_OUTCOME_REJECT_FUNCTION}()"
    )
