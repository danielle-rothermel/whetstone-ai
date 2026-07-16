from __future__ import annotations

import pytest
from dbos._error import DBOSMaxStepRetriesExceeded
from dr_providers import (
    FailureClass,
    ProviderFailureError,
    failure_record,
)
from psycopg import OperationalError

from whetstone.eval_failures import (
    PermanentFailureError,
    RateLimitedFailureError,
    ResourceExhaustionFailureError,
    TransientFailureError,
    UnknownFailureError,
    classify_exception,
    find_classified_exception,
    should_retry_step,
    summarize_exception,
    unwrap_exception,
)


@pytest.mark.parametrize(
    ("error", "expected_class", "expected_retry"),
    [
        (
            PermanentFailureError("permanent"),
            FailureClass.PERMANENT,
            False,
        ),
        (
            TransientFailureError("transient"),
            FailureClass.TRANSIENT,
            True,
        ),
        (
            RateLimitedFailureError("rate limited"),
            FailureClass.RATE_LIMITED,
            True,
        ),
        (
            ResourceExhaustionFailureError("resource exhausted"),
            FailureClass.RESOURCE_EXHAUSTION,
            False,
        ),
    ],
)
def test_whetstone_failure_classes_define_policy(
    error: BaseException,
    expected_class: FailureClass,
    expected_retry: bool,
) -> None:
    summary = summarize_exception(error)

    assert classify_exception(error) is expected_class
    assert summary.failure_class is expected_class
    assert summary.is_recoverable is (
        expected_class
        in {
            FailureClass.TRANSIENT,
            FailureClass.RATE_LIMITED,
            FailureClass.RESOURCE_EXHAUSTION,
        }
    )
    assert should_retry_step(error) is expected_retry


@pytest.mark.parametrize(
    ("failure_class", "expected_retry"),
    [
        (FailureClass.PERMANENT, False),
        (FailureClass.TRANSIENT, True),
        (FailureClass.RATE_LIMITED, True),
        (FailureClass.RESOURCE_EXHAUSTION, False),
        (FailureClass.UNKNOWN, False),
    ],
)
def test_provider_failure_record_defines_policy(
    failure_class: FailureClass,
    expected_retry: bool,
) -> None:
    error = ProviderFailureError(
        failure_record(
            failure_class=failure_class,
            message="provider failed",
        )
    )

    summary = summarize_exception(error)

    assert classify_exception(error) is failure_class
    assert summary.failure_class is failure_class
    assert summary.failure_exception_type.endswith("ProviderFailureError")
    assert should_retry_step(error) is expected_retry


def test_explicit_failure_summary_preserves_metadata() -> None:
    error = PermanentFailureError("bad input", metadata={"task_id": "x"})

    summary = summarize_exception(error)

    assert summary.failure_metadata == {"task_id": "x"}


def test_cause_chain_uses_explicit_failure_and_underlying_root() -> None:
    underlying = TimeoutError("provider deadline")
    classified = TransientFailureError(
        "provider failed",
        underlying=underlying,
        metadata={"provider": "test"},
    )
    wrapper = RuntimeError("workflow wrapper")
    wrapper.__cause__ = classified

    summary = summarize_exception(wrapper)

    assert find_classified_exception(wrapper) is classified
    assert summary.failure_class is FailureClass.TRANSIENT
    assert summary.failure_exception_type.endswith("TransientFailureError")
    assert summary.underlying_exception_type == "builtins.TimeoutError"
    assert summary.message == "workflow wrapper"
    assert summary.failure_metadata == {"provider": "test"}


def test_explicit_unknown_failure_remains_classified() -> None:
    error = UnknownFailureError("intentionally unknown")

    summary = summarize_exception(error)

    assert classify_exception(error) is FailureClass.UNKNOWN
    assert summary.failure_class is FailureClass.UNKNOWN
    assert summary.failure_exception_type.endswith("UnknownFailureError")
    assert summary.is_recoverable is False
    assert should_retry_step(error) is False


class FailureClassLookalike(Exception):
    failure_class = FailureClass.TRANSIENT


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("unexpected"),
        ValueError("bad input"),
        TimeoutError("deadline exceeded"),
        OSError("too many open files"),
        FailureClassLookalike("looks transient"),
    ],
)
def test_unclassified_builtin_and_duck_typed_errors_propagate(
    error: BaseException,
) -> None:
    assert classify_exception(error) is None
    assert should_retry_step(error) is False

    with pytest.raises(type(error)) as exc_info:
        summarize_exception(error)

    assert exc_info.value is error


def test_real_psycopg_failure_propagates() -> None:
    error = OperationalError("connection lost")

    assert classify_exception(error) is None
    assert should_retry_step(error) is False

    with pytest.raises(OperationalError) as exc_info:
        summarize_exception(error)

    assert exc_info.value is error


def test_real_dbos_retry_wrapper_does_not_unwrap_errors_list() -> None:
    error = DBOSMaxStepRetriesExceeded(
        "score_prediction_step",
        3,
        [PermanentFailureError("first"), TransientFailureError("last")],
    )

    assert unwrap_exception(error) is error
    assert classify_exception(error) is None
    assert should_retry_step(error) is False

    with pytest.raises(DBOSMaxStepRetriesExceeded) as exc_info:
        summarize_exception(error)

    assert exc_info.value is error
