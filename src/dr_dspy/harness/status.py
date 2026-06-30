"""Legacy v0 mutable prediction-row generation and scoring statuses."""

from __future__ import annotations

from enum import StrEnum


class GenerationStatus(StrEnum):
    PENDING = "pending"
    STARTED = "started"
    GENERATED = "generated"
    ERROR = "generation_error"
    RECOVERABLE_ERROR = "generation_recoverable_error"


class ScoringStatus(StrEnum):
    PENDING = "pending"
    QUEUED = "queued"
    STARTED = "started"
    SCORED = "scored"
    ERROR = "score_error"
    RECOVERABLE_ERROR = "score_recoverable_error"


GENERATION_RETRY_STATUSES = (
    GenerationStatus.ERROR.value,
    GenerationStatus.RECOVERABLE_ERROR.value,
)
SCORING_RETRY_STATUSES = (
    ScoringStatus.ERROR.value,
    ScoringStatus.RECOVERABLE_ERROR.value,
)
SCORING_QUEUEABLE_STATUSES = (
    ScoringStatus.PENDING.value,
    ScoringStatus.ERROR.value,
    ScoringStatus.RECOVERABLE_ERROR.value,
)
STRANDED_SCORING_STATUSES = (
    ScoringStatus.STARTED.value,
    ScoringStatus.QUEUED.value,
)
