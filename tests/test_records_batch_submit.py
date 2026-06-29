from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from dr_dspy.records import (
    BatchSubmitItemEnqueueStatus,
    BatchSubmitItemInsertStatus,
    BatchSubmitItemRecord,
    BatchSubmitOperationRecord,
    BatchSubmitOperationStatus,
    FailureMetadataPayload,
    batch_submit_operation_counts_from_items,
    build_batch_submit_operation_record,
    insert_outcome_from_rowcount,
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
            enqueue_status=BatchSubmitItemEnqueueStatus.ENQUEUED,
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
    assert counts.enqueued_count == 2
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
        match="enqueued_count or failed_count",
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
