"""Fresh v6 Whetstone baseline."""

from __future__ import annotations

from alembic import op

from whetstone.db import schema

revision = "20260712_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    schema.metadata.create_all(bind=bind)
    reject_function = schema.APPEND_ONLY_OUTCOME_REJECT_FUNCTION
    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION {reject_function}()
        RETURNS trigger AS $$
        BEGIN
            RAISE EXCEPTION 'append-only table % cannot be updated or deleted',
                TG_TABLE_NAME;
        END;
        $$ LANGUAGE plpgsql
        """
    )
    for table_name in schema.APPEND_ONLY_OUTCOME_TABLE_NAMES:
        op.execute(
            f"""
            CREATE TRIGGER tr_{table_name}_append_only
            BEFORE UPDATE OR DELETE ON {table_name}
            FOR EACH ROW EXECUTE FUNCTION {reject_function}()
            """
        )


def downgrade() -> None:
    for table_name in reversed(schema.APPEND_ONLY_OUTCOME_TABLE_NAMES):
        op.execute(
            "DROP TRIGGER IF EXISTS "
            f"tr_{table_name}_append_only ON {table_name}"
        )
    schema.metadata.drop_all(bind=op.get_bind())
    op.execute(
        "DROP FUNCTION IF EXISTS "
        f"{schema.APPEND_ONLY_OUTCOME_REJECT_FUNCTION}()"
    )
