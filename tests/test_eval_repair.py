from __future__ import annotations

from dr_dspy.eval_repair import (
    RepairRetryCandidate,
    RepairRetrySelection,
    is_retryable_failure_class,
    unique_prediction_ids,
)
from dr_dspy.failure_policy import FailureClass


def test_retryable_failure_class_includes_legacy_and_recoverable() -> None:
    assert is_retryable_failure_class(None)
    assert is_retryable_failure_class(FailureClass.TRANSIENT.value)
    assert is_retryable_failure_class(FailureClass.RATE_LIMITED.value)
    assert is_retryable_failure_class(FailureClass.RESOURCE_EXHAUSTION.value)


def test_retryable_failure_class_excludes_permanent_and_unknown() -> None:
    assert not is_retryable_failure_class(FailureClass.PERMANENT.value)
    assert not is_retryable_failure_class(FailureClass.UNKNOWN.value)


def test_retry_selection_summary_separates_retry_categories() -> None:
    selection = RepairRetrySelection(
        candidates=[
            RepairRetryCandidate(prediction_id="legacy"),
            RepairRetryCandidate(
                prediction_id="recoverable",
                failure_class=FailureClass.TRANSIENT.value,
            ),
        ],
        excluded_count=3,
    )

    assert selection.prediction_ids == ["legacy", "recoverable"]
    assert selection.summary.legacy_count == 1
    assert selection.summary.recoverable_count == 1
    assert selection.summary.excluded_count == 3
    assert selection.summary.retryable_count == 2


def test_unique_prediction_ids_preserves_order() -> None:
    assert unique_prediction_ids(["p1", "p2", "p1", "p3"]) == [
        "p1",
        "p2",
        "p3",
    ]
