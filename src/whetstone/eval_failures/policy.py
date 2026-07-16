from __future__ import annotations

from typing import Any

from dr_providers import (
    RECOVERABLE_FAILURE_CLASSES,
    RETRYABLE_FAILURE_CLASSES,
    FailureClass,
    ProviderFailureError,
)
from pydantic import BaseModel, ConfigDict, Field, StrictStr

from whetstone.eval_failures.exceptions import EvalFailureError

__all__ = [
    "RECOVERABLE_FAILURE_CLASSES",
    "RETRYABLE_FAILURE_CLASSES",
    "FailureClass",
    "FailureSummary",
    "classify_exception",
    "error_text",
    "exception_type_name",
    "failure_summary_payload",
    "find_classified_exception",
    "should_retry_step",
    "summarize_exception",
    "unwrap_exception",
]


class FailureSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    failure_class: FailureClass
    failure_exception_type: StrictStr
    underlying_exception_type: StrictStr
    message: StrictStr
    failure_metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_recoverable(self) -> bool:
        return self.failure_class in RECOVERABLE_FAILURE_CLASSES


def unwrap_exception(error: BaseException) -> BaseException:
    if (
        isinstance(error, EvalFailureError | ProviderFailureError)
        and error.underlying is not None
    ):
        return unwrap_exception(error.underlying)
    if error.__cause__ is not None:
        return unwrap_exception(error.__cause__)
    if error.__context__ is not None:
        return unwrap_exception(error.__context__)
    return error


def exception_type_name(error: BaseException) -> str:
    error_type = type(error)
    return f"{error_type.__module__}.{error_type.__qualname__}"


def _iter_exception_chain(error: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        if (
            isinstance(current, EvalFailureError | ProviderFailureError)
            and current.underlying is not None
        ):
            current = current.underlying
            continue
        if current.__cause__ is not None:
            current = current.__cause__
            continue
        if current.__context__ is not None:
            current = current.__context__
            continue
        break
    return chain


def find_classified_exception(
    error: BaseException,
) -> EvalFailureError | ProviderFailureError | None:
    for exc in _iter_exception_chain(error):
        if isinstance(exc, EvalFailureError | ProviderFailureError):
            return exc
    return None


def _failure_class(
    error: EvalFailureError | ProviderFailureError,
) -> FailureClass:
    if isinstance(error, EvalFailureError):
        return error.failure_class
    return error.failure.failure_class


def classify_exception(error: BaseException) -> FailureClass | None:
    classified = find_classified_exception(error)
    return None if classified is None else _failure_class(classified)


def failure_exception_type_name(
    classified: EvalFailureError | ProviderFailureError,
) -> str:
    return exception_type_name(classified)


def underlying_exception_type_name(error: BaseException) -> str:
    root = unwrap_exception(error)
    if isinstance(root, EvalFailureError):
        if root.underlying is not None:
            return exception_type_name(unwrap_exception(root.underlying))
        return exception_type_name(root)
    return exception_type_name(root)


def summarize_exception(error: BaseException) -> FailureSummary:
    from whetstone.eval_failures.recording import (
        failure_metadata_dict_from_exception,
    )

    classified = find_classified_exception(error)
    if classified is None:
        raise error
    return FailureSummary(
        failure_class=_failure_class(classified),
        failure_exception_type=failure_exception_type_name(classified),
        underlying_exception_type=underlying_exception_type_name(error),
        message=str(error),
        failure_metadata=failure_metadata_dict_from_exception(error),
    )


def should_retry_step(error: BaseException) -> bool:
    failure_class = classify_exception(error)
    return (
        failure_class is not None
        and failure_class in RETRYABLE_FAILURE_CLASSES
    )


def error_text(summary: FailureSummary) -> str:
    return (
        f"{summary.failure_class.value}: "
        f"{summary.failure_exception_type}: "
        f"{summary.underlying_exception_type}: {summary.message}"
    )


def failure_summary_payload(summary: FailureSummary) -> dict[str, Any]:
    return summary.model_dump(mode="json")
