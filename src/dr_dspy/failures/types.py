from __future__ import annotations

from enum import StrEnum


class FailureClass(StrEnum):
    PERMANENT = "permanent"
    TRANSIENT = "transient"
    RATE_LIMITED = "rate_limited"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    UNKNOWN = "unknown"


RECOVERABLE_FAILURE_CLASSES = frozenset(
    {
        FailureClass.TRANSIENT,
        FailureClass.RATE_LIMITED,
        FailureClass.RESOURCE_EXHAUSTION,
    }
)
RETRYABLE_STEP_FAILURE_CLASSES = frozenset(
    {
        FailureClass.TRANSIENT,
        FailureClass.RATE_LIMITED,
    }
)
