from __future__ import annotations

from alembic import op

revision = "20260630_0006"
down_revision = "20260630_0005"
branch_labels = None
depends_on = None

_OUTCOME_CHECK = (
    "generated_code_outcome IS NULL OR "
    "(generated_code_outcome IN "
    "('passed', 'tests_failed', 'evaluation_incomplete', "
    "'empty_generation', 'extraction_failed', 'no_top_level_functions'))"
)


def upgrade() -> None:
    op.drop_constraint(
        "ck_dr_dspy_score_attempts_generated_code_outcome",
        "dr_dspy_score_attempts",
        type_="check",
    )
    op.create_check_constraint(
        "ck_dr_dspy_score_attempts_generated_code_outcome",
        "dr_dspy_score_attempts",
        _OUTCOME_CHECK,
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_dr_dspy_score_attempts_generated_code_outcome",
        "dr_dspy_score_attempts",
        type_="check",
    )
    op.create_check_constraint(
        "ck_dr_dspy_score_attempts_generated_code_outcome",
        "dr_dspy_score_attempts",
        (
            "generated_code_outcome IS NULL OR "
            "(generated_code_outcome IN "
            "('passed', 'tests_failed', 'empty_generation', "
            "'extraction_failed', 'no_top_level_functions'))"
        ),
    )
