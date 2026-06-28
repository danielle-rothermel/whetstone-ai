from __future__ import annotations

import errno

import psycopg
from dbos._error import DBOSMaxStepRetriesExceeded

from dr_dspy.failure_policy import (
    FailureClass,
    classify_exception,
    should_retry_step,
    summarize_exception,
)


def test_open_file_exhaustion_is_recoverable_not_step_retry() -> None:
    error = OSError(errno.EMFILE, "Too many open files")

    summary = summarize_exception(error)

    assert summary.failure_class is FailureClass.RESOURCE_EXHAUSTION
    assert summary.is_recoverable
    assert not should_retry_step(error)


def test_unwraps_dbos_retry_wrapper_to_real_exception() -> None:
    wrapped = DBOSMaxStepRetriesExceeded(
        "generate",
        3,
        [OSError(errno.EMFILE, "Too many open files")],
    )

    summary = summarize_exception(wrapped)

    assert summary.failure_class is FailureClass.RESOURCE_EXHAUSTION
    assert summary.exception_type == "builtins.OSError"


def test_classifies_db_operational_error_as_retryable_transient() -> None:
    error = psycopg.OperationalError("connection is bad")

    assert classify_exception(error) is FailureClass.TRANSIENT
    assert should_retry_step(error)


def test_unknown_exceptions_are_not_recoverable_by_default() -> None:
    summary = summarize_exception(RuntimeError("bug"))

    assert summary.failure_class is FailureClass.UNKNOWN
    assert not summary.is_recoverable
    assert not should_retry_step(RuntimeError("bug"))
