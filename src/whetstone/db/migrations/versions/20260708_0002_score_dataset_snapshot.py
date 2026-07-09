from __future__ import annotations

from alembic import op
from sqlalchemy import Column, inspect, text
from sqlalchemy.dialects.postgresql import JSONB

revision = "20260708_0002"
down_revision = "20260708_0001"
branch_labels = None
depends_on = None

BACKFILL_SNAPSHOT = """
{
  "source_path": "unknown-pre-snapshot-migration",
  "sha256": "",
  "header": {
    "schema_version": 1,
    "dataset_id": "unknown",
    "hf_revision": "unknown",
    "overrides_digest": "unknown"
  }
}
""".strip()


def upgrade() -> None:
    _add_dataset_snapshot_column("dr_dspy_score_attempts")
    _add_dataset_snapshot_column("dr_dspy_score_harness_failures")


def downgrade() -> None:
    if _column_exists("dr_dspy_score_harness_failures", "dataset_snapshot"):
        op.drop_column("dr_dspy_score_harness_failures", "dataset_snapshot")
    if _column_exists("dr_dspy_score_attempts", "dataset_snapshot"):
        op.drop_column("dr_dspy_score_attempts", "dataset_snapshot")


def _add_dataset_snapshot_column(table_name: str) -> None:
    if _column_exists(table_name, "dataset_snapshot"):
        return
    op.add_column(
        table_name,
        Column(
            "dataset_snapshot",
            JSONB,
            nullable=False,
            server_default=text(f"'{BACKFILL_SNAPSHOT}'::jsonb"),
        ),
    )
    op.alter_column(table_name, "dataset_snapshot", server_default=None)


def _column_exists(table_name: str, column_name: str) -> bool:
    columns = inspect(op.get_bind()).get_columns(table_name)
    return any(column["name"] == column_name for column in columns)
