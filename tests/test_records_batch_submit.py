from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

import pytest
from pydantic import ValidationError

from dr_dspy.records import (
    BATCH_SUBMIT_SPEC_MAX_BYTES,
    BatchSubmitItemEnqueueStatus,
    BatchSubmitItemInsertStatus,
    BatchSubmitItemRecord,
    BatchSubmitOperationRecord,
    BatchSubmitOperationStatus,
    FailureMetadataPayload,
    batch_submit_operation_counts_from_items,
    build_batch_submit_operation_record,
    insert_outcome_from_rowcount,
    is_terminal_enqueue_status,
    operation_status_from_counts,
)

NOW = datetime(2026, 6, 29, 12, 0, tzinfo=UTC)


def _item(
    *,
    item_index: int,
    insert_status: BatchSubmitItemInsertStatus,
    enqueue_status: BatchSubmitItemEnqueueStatus,
) -> BatchSubmitItemRecord:
    failure = None
    if enqueue_status is BatchSubmitItemEnqueueStatus.FAILED:
        failure = FailureMetadataPayload(
            error_type="builtins.RuntimeError",
            message="enqueue failed",
        )
    return BatchSubmitItemRecord(
        batch_submit_item_id=f"item-{item_index}",
        operation_key="op-1",
        item_index=item_index,
        prediction_id=f"prediction-{item_index}",
        fair_order_key=f"fair-{item_index}",
        insert_status=insert_status,
        enqueue_status=enqueue_status,
        created_at=NOW,
        failure=failure,
    )


def test_insert_outcome_from_rowcount() -> None:
    assert insert_outcome_from_rowcount(1).value == "inserted"
    assert insert_outcome_from_rowcount(0).value == "already_present"
    with pytest.raises(ValueError, match="unexpected insert rowcount"):
        insert_outcome_from_rowcount(2)


def test_batch_submit_operation_counts_from_items() -> None:
    items = (
        _item(
            item_index=0,
            insert_status=BatchSubmitItemInsertStatus.INSERTED,
            enqueue_status=BatchSubmitItemEnqueueStatus.ENQUEUED,
        ),
        _item(
            item_index=1,
            insert_status=BatchSubmitItemInsertStatus.ALREADY_PRESENT,
            enqueue_status=BatchSubmitItemEnqueueStatus.WORKFLOW_ALREADY_PRESENT,
        ),
        _item(
            item_index=2,
            insert_status=BatchSubmitItemInsertStatus.INSERTED,
            enqueue_status=BatchSubmitItemEnqueueStatus.FAILED,
        ),
    )

    counts = batch_submit_operation_counts_from_items(items)

    assert counts.inserted_count == 2
    assert counts.already_present_count == 1
    assert counts.enqueued_count == 1
    assert counts.already_scheduled_count == 1
    assert counts.failed_count == 1


def test_build_batch_submit_operation_record_derives_counts() -> None:
    items = (
        _item(
            item_index=0,
            insert_status=BatchSubmitItemInsertStatus.INSERTED,
            enqueue_status=BatchSubmitItemEnqueueStatus.ENQUEUED,
        ),
        _item(
            item_index=1,
            insert_status=BatchSubmitItemInsertStatus.INSERTED,
            enqueue_status=BatchSubmitItemEnqueueStatus.FAILED,
        ),
    )
    record = build_batch_submit_operation_record(
        operation_key="op-1",
        experiment_name="exp",
        status=BatchSubmitOperationStatus.COMPLETED,
        requested_count=2,
        items=items,
        created_at=NOW,
        completed_at=NOW,
    )

    assert record.inserted_count == 2
    assert record.enqueued_count == 1
    assert record.failed_count == 1
    assert record.already_present_count == 0


def test_completed_batch_operation_requires_full_enqueue_accounting() -> None:
    with pytest.raises(
        ValidationError,
        match="already_scheduled_count",
    ):
        BatchSubmitOperationRecord(
            operation_key="op-1",
            experiment_name="exp",
            status=BatchSubmitOperationStatus.COMPLETED,
            requested_count=2,
            inserted_count=2,
            enqueued_count=1,
            failed_count=0,
            created_at=NOW,
            completed_at=NOW,
        )


def test_completed_batch_allows_already_scheduled_accounting() -> None:
    operation = BatchSubmitOperationRecord(
        operation_key="op-1",
        experiment_name="exp",
        status=BatchSubmitOperationStatus.COMPLETED,
        requested_count=2,
        inserted_count=2,
        enqueued_count=0,
        already_scheduled_count=2,
        failed_count=0,
        created_at=NOW,
        completed_at=NOW,
    )

    assert operation.already_scheduled_count == 2


def test_operation_status_from_counts_all_already_scheduled() -> None:
    status = operation_status_from_counts(
        requested_count=2,
        enqueued_count=0,
        already_scheduled_count=2,
        failed_count=0,
    )

    assert status is BatchSubmitOperationStatus.COMPLETED


def test_operation_status_from_counts_partial_with_mixed_outcomes() -> None:
    status = operation_status_from_counts(
        requested_count=3,
        enqueued_count=1,
        already_scheduled_count=1,
        failed_count=1,
    )

    assert status is BatchSubmitOperationStatus.PARTIAL


def test_operation_status_from_counts_incomplete_enqueue() -> None:
    status = operation_status_from_counts(
        requested_count=3,
        enqueued_count=1,
        already_scheduled_count=0,
        failed_count=0,
    )

    assert status is BatchSubmitOperationStatus.ENQUEUING


def test_terminal_batch_operation_requires_completed_at() -> None:
    with pytest.raises(ValidationError, match="completed_at"):
        BatchSubmitOperationRecord(
            operation_key="op-1",
            experiment_name="exp",
            status=BatchSubmitOperationStatus.PARTIAL,
            requested_count=1,
            failed_count=1,
            created_at=NOW,
        )


def _operation(**overrides: object) -> BatchSubmitOperationRecord:
    payload = {
        "operation_key": "op-1",
        "experiment_name": "exp",
        "status": BatchSubmitOperationStatus.ENQUEUING,
        "requested_count": 2,
        "created_at": NOW,
    }
    payload.update(overrides)
    return BatchSubmitOperationRecord(**cast(Any, payload))


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"requested_count": -1}, "non-negative"),
        ({"enqueued_count": 3}, "cannot exceed requested_count"),
        (
            {
                "inserted_count": 2,
                "already_present_count": 1,
            },
            "already_present_count cannot exceed",
        ),
        (
            {
                "enqueued_count": 2,
                "already_scheduled_count": 1,
                "failed_count": 1,
            },
            "failed_count cannot exceed",
        ),
        (
            {
                "spec": {"x": "y" * BATCH_SUBMIT_SPEC_MAX_BYTES},
            },
            "batch submit spec",
        ),
        (
            {
                "status": BatchSubmitOperationStatus.COMPLETED,
                "enqueued_count": 2,
                "completed_at": NOW,
                "created_at": NOW.replace(year=NOW.year + 1),
            },
            "completed_at must not precede created_at",
        ),
    ],
)
def test_batch_submit_operation_rejects_invalid_counts(
    overrides: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValidationError, match=match):
        _operation(**overrides)


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (BatchSubmitItemEnqueueStatus.ENQUEUED, True),
        (BatchSubmitItemEnqueueStatus.WORKFLOW_ALREADY_PRESENT, True),
        (BatchSubmitItemEnqueueStatus.FAILED, True),
        (BatchSubmitItemEnqueueStatus.PENDING, False),
        (BatchSubmitItemEnqueueStatus.CLAIMING, False),
    ],
)
def test_is_terminal_enqueue_status(
    status: BatchSubmitItemEnqueueStatus,
    expected: bool,
) -> None:
    assert is_terminal_enqueue_status(status) is expected


def _batch_item(**overrides: object) -> BatchSubmitItemRecord:
    payload = {
        "batch_submit_item_id": "item-1",
        "operation_key": "op-1",
        "item_index": 0,
        "prediction_id": "prediction-1",
        "fair_order_key": "abc",
        "insert_status": BatchSubmitItemInsertStatus.INSERTED,
        "enqueue_status": BatchSubmitItemEnqueueStatus.PENDING,
        "created_at": NOW,
    }
    payload.update(overrides)
    return BatchSubmitItemRecord(**cast(Any, payload))


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"item_index": -1}, "item_index must be non-negative"),
        (
            {
                "enqueue_status": BatchSubmitItemEnqueueStatus.FAILED,
                "failure": None,
            },
            "failed batch submit items require failure",
        ),
        (
            {
                "enqueue_status": BatchSubmitItemEnqueueStatus.CLAIMING,
                "enqueue_metadata": {"enqueue_claim_id": "claim-1"},
            },
            "require claimed_at",
        ),
        (
            {
                "enqueue_status": BatchSubmitItemEnqueueStatus.CLAIMING,
                "enqueue_metadata": {
                    "enqueue_claim_id": "",
                    "claimed_at": NOW.isoformat(),
                },
            },
            "require enqueue_claim_id",
        ),
    ],
)
def test_batch_submit_item_rejects_invalid_shape(
    overrides: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValidationError, match=match):
        _batch_item(**overrides)
