from __future__ import annotations

from enum import UNIQUE, StrEnum, verify

__all__ = ["MatrixTreatmentState"]


@verify(UNIQUE)
class MatrixTreatmentState(StrEnum):
    """Durably logged matrix lifecycle states."""

    RUN_STARTED = "run_started"
    TREATMENT_STARTED = "treatment_started"
    TREATMENT_SKIPPED = "treatment_skipped"
    TREATMENT_COMPLETED = "treatment_completed"
    TREATMENT_FAILED = "treatment_failed"
    RUN_COMPLETED = "run_completed"
