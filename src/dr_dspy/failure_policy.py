from __future__ import annotations

import errno
from enum import StrEnum
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


class FailureSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    failure_class: FailureClass
    exception_type: StrictStr
    message: StrictStr

    @property
    def is_recoverable(self) -> bool:
        return self.failure_class in RECOVERABLE_FAILURE_CLASSES


def unwrap_exception(error: BaseException) -> BaseException:
    if isinstance(error, DBOSMaxStepRetriesExceeded) and error.errors:
        return unwrap_exception(error.errors[-1])
    if error.__cause__ is not None:
        return unwrap_exception(error.__cause__)
    if error.__context__ is not None:
        return unwrap_exception(error.__context__)
    return error


def exception_type_name(error: BaseException) -> str:
    error_type = type(error)
    return f"{error_type.__module__}.{error_type.__qualname__}"


def is_open_file_exhaustion(error: BaseException) -> bool:
    if isinstance(error, OSError) and error.errno in (
        errno.EMFILE,
        errno.ENFILE,
    ):
        return True
    return "too many open files" in str(error).lower()


def classify_exception(error: BaseException) -> FailureClass:
    root = unwrap_exception(error)
    if is_open_file_exhaustion(root):
        return FailureClass.RESOURCE_EXHAUSTION
    if isinstance(root, RateLimitError):
        return FailureClass.RATE_LIMITED
    if isinstance(root, (APIConnectionError, APITimeoutError)):
        return FailureClass.TRANSIENT
    if isinstance(root, APIStatusError):
        if root.status_code == 429:
            return FailureClass.RATE_LIMITED
        if root.status_code >= 500 or root.status_code in {408, 409, 425}:
            return FailureClass.TRANSIENT
        return FailureClass.PERMANENT
    if isinstance(root, httpx.HTTPStatusError):
        status_code = root.response.status_code
        if status_code == 429:
            return FailureClass.RATE_LIMITED
        if status_code >= 500 or status_code in {408, 409, 425}:
            return FailureClass.TRANSIENT
        return FailureClass.PERMANENT
    if isinstance(
        root,
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
    if isinstance(root, psycopg.OperationalError):
        return FailureClass.TRANSIENT
    if isinstance(root, TimeoutError):
        return FailureClass.TRANSIENT
    return FailureClass.UNKNOWN


def summarize_exception(error: BaseException) -> FailureSummary:
    root = unwrap_exception(error)
    return FailureSummary(
        failure_class=classify_exception(root),
        exception_type=exception_type_name(root),
        message=str(root),
    )


def should_retry_step(error: BaseException) -> bool:
    return classify_exception(error) in RETRYABLE_STEP_FAILURE_CLASSES


def error_text(summary: FailureSummary) -> str:
    return (
        f"{summary.failure_class.value}: "
        f"{summary.exception_type}: {summary.message}"
    )


def failure_summary_payload(summary: FailureSummary) -> dict[str, Any]:
    return summary.model_dump(mode="json")
