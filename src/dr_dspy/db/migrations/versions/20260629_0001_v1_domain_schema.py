from __future__ import annotations

from collections.abc import Sequence

from alembic import op
from sqlalchemy import Table

from dr_dspy.db.schema import metadata, v1_tables

revision = "20260629_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    metadata.create_all(bind, tables=v1_tables)


def downgrade() -> None:
    bind = op.get_bind()
    metadata.drop_all(bind, tables=reversed_v1_tables())


def reversed_v1_tables() -> Sequence[Table]:
    return tuple(reversed(v1_tables))
