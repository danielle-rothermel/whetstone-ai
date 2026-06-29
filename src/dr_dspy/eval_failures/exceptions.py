from __future__ import annotations

from typing import Any, ClassVar

from dr_dspy.eval_failures.types import FailureClass


class EvalFailureError(Exception):
    """Base for dr-dspy classified eval worker failures."""

    failure_class: ClassVar[FailureClass]

    def __init__(
        self,
        message: str,
        *,
        underlying: BaseException | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.underlying = underlying
        self.metadata = dict(metadata or {})


class PermanentFailureError(EvalFailureError):
    failure_class = FailureClass.PERMANENT


class TransientFailureError(EvalFailureError):
    failure_class = FailureClass.TRANSIENT


class RateLimitedFailureError(EvalFailureError):
    failure_class = FailureClass.RATE_LIMITED


class ResourceExhaustionFailureError(EvalFailureError):
    failure_class = FailureClass.RESOURCE_EXHAUSTION


class UnknownFailureError(EvalFailureError):
    failure_class = FailureClass.UNKNOWN


class RecordingFailureError(PermanentFailureError):
    """Worker could not produce a storable record of what it did."""


class EmptyGenerationError(PermanentFailureError):
    """Predictor returned no usable text for a required output field."""


class PredictionParseError(PermanentFailureError):
    """Predictor failed to parse structured output from an LM response."""


class ProviderResponseParseError(PermanentFailureError):
    """Provider response could not be parsed into an LM result."""


class StrandedGenerationError(TransientFailureError):
    pass


class StrandedScoringError(TransientFailureError):
    pass


DEFAULT_FAILURE_EXCEPTION_TYPES: dict[FailureClass, type[EvalFailureError]] = {
    FailureClass.PERMANENT: PermanentFailureError,
    FailureClass.TRANSIENT: TransientFailureError,
    FailureClass.RATE_LIMITED: RateLimitedFailureError,
    FailureClass.RESOURCE_EXHAUSTION: ResourceExhaustionFailureError,
    FailureClass.UNKNOWN: UnknownFailureError,
}


def failure_exception_type_for_class(
    failure_class: FailureClass,
) -> type[EvalFailureError]:
    return DEFAULT_FAILURE_EXCEPTION_TYPES[failure_class]
