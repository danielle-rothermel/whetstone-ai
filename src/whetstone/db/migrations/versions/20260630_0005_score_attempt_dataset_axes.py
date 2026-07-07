from __future__ import annotations

import sqlalchemy as sa
from alembic import op

from whetstone.records.hashing import (
    DEFAULT_SCORE_DATASET_NAME,
    DEFAULT_SCORE_DATASET_SPLIT,
)

revision = "20260630_0005"
down_revision = "20260630_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "dr_dspy_score_attempts",
        sa.Column("dataset_name", sa.Text(), nullable=True),
    )
    op.add_column(
        "dr_dspy_score_attempts",
        sa.Column("dataset_split", sa.Text(), nullable=True),
    )
    op.execute(
        sa.text(
            "UPDATE dr_dspy_score_attempts "
            "SET dataset_name = :dataset_name, "
            "dataset_split = :dataset_split"
        ).bindparams(
            dataset_name=DEFAULT_SCORE_DATASET_NAME,
            dataset_split=DEFAULT_SCORE_DATASET_SPLIT,
        )
    )
    op.alter_column(
        "dr_dspy_score_attempts",
        "dataset_name",
        nullable=False,
    )
    op.alter_column(
        "dr_dspy_score_attempts",
        "dataset_split",
        nullable=False,
    )
    op.drop_constraint(
        "uq_dr_dspy_score_attempts_profile",
        "dr_dspy_score_attempts",
        type_="unique",
    )
    op.create_unique_constraint(
        "uq_dr_dspy_score_attempts_profile",
        "dr_dspy_score_attempts",
        [
            "prediction_id",
            "generation_run_id",
            "scoring_profile_id",
            "scoring_profile_version",
            "parser_profile_id",
            "parser_version",
            "attempt_index",
            "dataset_name",
            "dataset_split",
        ],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_dr_dspy_score_attempts_profile",
        "dr_dspy_score_attempts",
        type_="unique",
    )
    op.create_unique_constraint(
        "uq_dr_dspy_score_attempts_profile",
        "dr_dspy_score_attempts",
        [
            "prediction_id",
            "generation_run_id",
            "scoring_profile_id",
            "scoring_profile_version",
            "parser_profile_id",
            "parser_version",
            "attempt_index",
        ],
    )
    op.drop_column("dr_dspy_score_attempts", "dataset_split")
    op.drop_column("dr_dspy_score_attempts", "dataset_name")
