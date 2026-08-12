from __future__ import annotations

from typing import Any, ClassVar

from dr_providers import RecoverabilityClass

FailureClass = RecoverabilityClass

RETRYABLE_FAILURE_CLASSES = frozenset(
    {
        RecoverabilityClass.TRANSIENT,
        RecoverabilityClass.RATE_LIMITED,
        RecoverabilityClass.RESOURCE_EXHAUSTION,
    }
)

RECOVERABLE_FAILURE_CLASSES = frozenset(
    {
        *RETRYABLE_FAILURE_CLASSES,
        RecoverabilityClass.UNKNOWN,
    }
)


class EvalFailureError(Exception):
    failure_class: ClassVar[RecoverabilityClass]

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
    failure_class = RecoverabilityClass.PERMANENT


class TransientFailureError(EvalFailureError):
    failure_class = RecoverabilityClass.TRANSIENT


class RateLimitedFailureError(EvalFailureError):
    failure_class = RecoverabilityClass.RATE_LIMITED


class ResourceExhaustionFailureError(EvalFailureError):
    failure_class = RecoverabilityClass.RESOURCE_EXHAUSTION


class UnknownFailureError(EvalFailureError):
    failure_class = RecoverabilityClass.UNKNOWN


class RecordingFailureError(PermanentFailureError):
    pass


class EmptyProviderGenerationError(PermanentFailureError):
    pass


DEFAULT_FAILURE_EXCEPTION_TYPES: dict[
    RecoverabilityClass, type[EvalFailureError]
] = {
    RecoverabilityClass.PERMANENT: PermanentFailureError,
    RecoverabilityClass.TRANSIENT: TransientFailureError,
    RecoverabilityClass.RATE_LIMITED: RateLimitedFailureError,
    RecoverabilityClass.RESOURCE_EXHAUSTION: ResourceExhaustionFailureError,
    RecoverabilityClass.UNKNOWN: UnknownFailureError,
}


def failure_exception_type_for_class(
    failure_class: RecoverabilityClass,
) -> type[EvalFailureError]:
    return DEFAULT_FAILURE_EXCEPTION_TYPES[failure_class]
