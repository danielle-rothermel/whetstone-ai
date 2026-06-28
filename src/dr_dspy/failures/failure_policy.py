from __future__ import annotations

import errno
from typing import Any

import httpx
import psycopg
from dbos._error import DBOSMaxStepRetriesExceeded
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)
from pydantic import BaseModel, ConfigDict, StrictStr

from dr_dspy.failures.exceptions import (
    EvalFailureError,
    failure_exception_type_for_class,
)
from dr_dspy.failures.types import (
    RECOVERABLE_FAILURE_CLASSES,
    RETRYABLE_STEP_FAILURE_CLASSES,
    FailureClass,
)

__all__ = [
    "RECOVERABLE_FAILURE_CLASSES",
    "RETRYABLE_STEP_FAILURE_CLASSES",
    "FailureClass",
    "FailureSummary",
    "classify_exception",
    "error_text",
    "exception_type_name",
    "failure_summary_payload",
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

    @property
    def is_recoverable(self) -> bool:
        return self.failure_class in RECOVERABLE_FAILURE_CLASSES


def unwrap_exception(error: BaseException) -> BaseException:
    if isinstance(error, DBOSMaxStepRetriesExceeded) and error.errors:
        return unwrap_exception(error.errors[-1])
    if isinstance(error, EvalFailureError) and error.underlying is not None:
        return unwrap_exception(error.underlying)
    if error.__cause__ is not None:
        return unwrap_exception(error.__cause__)
    if error.__context__ is not None:
        return unwrap_exception(error.__context__)
    return error


def exception_type_name(error: BaseException) -> str:
    error_type = type(error)
    return f"{error_type.__module__}.{error_type.__qualname__}"


def _explicit_failure_class(error: BaseException) -> FailureClass | None:
    failure_class = getattr(type(error), "failure_class", None)
    if isinstance(failure_class, FailureClass):
        return failure_class
    return None


def _iter_exception_chain(error: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        if isinstance(current, DBOSMaxStepRetriesExceeded) and current.errors:
            current = current.errors[-1]
            continue
        if (
            isinstance(current, EvalFailureError)
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
) -> EvalFailureError | None:
    for exc in _iter_exception_chain(error):
        if isinstance(exc, EvalFailureError):
            return exc
    return None


def is_open_file_exhaustion(error: BaseException) -> bool:
    if isinstance(error, OSError) and error.errno in (
        errno.EMFILE,
        errno.ENFILE,
    ):
        return True
    return "too many open files" in str(error).lower()


def _classify_third_party_exception(error: BaseException) -> FailureClass:
    if is_open_file_exhaustion(error):
        return FailureClass.RESOURCE_EXHAUSTION
    if isinstance(error, RateLimitError):
        return FailureClass.RATE_LIMITED
    if isinstance(error, (APIConnectionError, APITimeoutError)):
        return FailureClass.TRANSIENT
    if isinstance(error, APIStatusError):
        if error.status_code == 429:
            return FailureClass.RATE_LIMITED
        if error.status_code >= 500 or error.status_code in {408, 409, 425}:
            return FailureClass.TRANSIENT
        return FailureClass.PERMANENT
    if isinstance(error, httpx.HTTPStatusError):
        status_code = error.response.status_code
        if status_code == 429:
            return FailureClass.RATE_LIMITED
        if status_code >= 500 or status_code in {408, 409, 425}:
            return FailureClass.TRANSIENT
        return FailureClass.PERMANENT
    if isinstance(
        error,
        (
            AuthenticationError,
            BadRequestError,
            NotFoundError,
            PermissionDeniedError,
            UnprocessableEntityError,
            ValueError,
        ),
    ):
        return FailureClass.PERMANENT
    if isinstance(error, psycopg.OperationalError):
        return FailureClass.TRANSIENT
    if isinstance(error, TimeoutError):
        return FailureClass.TRANSIENT
    return FailureClass.UNKNOWN


def classify_exception(error: BaseException) -> FailureClass:
    for exc in _iter_exception_chain(error):
        explicit = _explicit_failure_class(exc)
        if explicit is not None:
            return explicit
    root = unwrap_exception(error)
    return _classify_third_party_exception(root)


def failure_exception_type_name(
    error: BaseException,
    failure_class: FailureClass,
) -> str:
    classified = find_classified_exception(error)
    if classified is not None:
        return exception_type_name(classified)
    default_type = failure_exception_type_for_class(failure_class)
    return exception_type_name(default_type(""))


def underlying_exception_type_name(error: BaseException) -> str:
    root = unwrap_exception(error)
    if isinstance(root, EvalFailureError):
        if root.underlying is not None:
            return exception_type_name(unwrap_exception(root.underlying))
        return exception_type_name(root)
    return exception_type_name(root)


def summarize_exception(error: BaseException) -> FailureSummary:
    failure_class = classify_exception(error)
    failure_type = failure_exception_type_name(error, failure_class)
    return FailureSummary(
        failure_class=failure_class,
        failure_exception_type=failure_type,
        underlying_exception_type=underlying_exception_type_name(error),
        message=str(error),
    )


def should_retry_step(error: BaseException) -> bool:
    return classify_exception(error) in RETRYABLE_STEP_FAILURE_CLASSES


def error_text(summary: FailureSummary) -> str:
    return (
        f"{summary.failure_class.value}: "
        f"{summary.failure_exception_type}: "
        f"{summary.underlying_exception_type}: {summary.message}"
    )


def failure_summary_payload(summary: FailureSummary) -> dict[str, Any]:
    return summary.model_dump(mode="json")
