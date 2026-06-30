from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

from dr_dspy.records.models import (
    BatchSubmitItemEnqueueStatus,
    BatchSubmitItemInsertStatus,
    BatchSubmitItemRecord,
    BatchSubmitOperationRecord,
    BatchSubmitOperationStatus,
)


class InsertOutcome(StrEnum):
    INSERTED = "inserted"
    ALREADY_PRESENT = "already_present"


def insert_outcome_from_rowcount(rowcount: int) -> InsertOutcome:
    if rowcount == 1:
        return InsertOutcome.INSERTED
    if rowcount == 0:
        return InsertOutcome.ALREADY_PRESENT
    raise ValueError(f"unexpected insert rowcount: {rowcount}")


@dataclass(frozen=True)
class BatchSubmitOperationCounts:
    inserted_count: int
    already_present_count: int
    enqueued_count: int
    already_scheduled_count: int
    failed_count: int


def terminal_enqueue_total(
    *,
    enqueued_count: int,
    already_scheduled_count: int,
    failed_count: int,
) -> int:
    return enqueued_count + already_scheduled_count + failed_count


def terminal_enqueue_total_from_counts(
    counts: BatchSubmitOperationCounts,
) -> int:
    return terminal_enqueue_total(
        enqueued_count=counts.enqueued_count,
        already_scheduled_count=counts.already_scheduled_count,
        failed_count=counts.failed_count,
    )


def operation_status_from_counts(
    *,
    requested_count: int,
    enqueued_count: int,
    already_scheduled_count: int,
    failed_count: int,
) -> BatchSubmitOperationStatus:
    terminal_total = terminal_enqueue_total(
        enqueued_count=enqueued_count,
        already_scheduled_count=already_scheduled_count,
        failed_count=failed_count,
    )
    if terminal_total < requested_count:
        return BatchSubmitOperationStatus.ENQUEUING
    if failed_count >= requested_count:
        return BatchSubmitOperationStatus.ERROR
    if failed_count > 0:
        return BatchSubmitOperationStatus.PARTIAL
    return BatchSubmitOperationStatus.COMPLETED


def batch_submit_operation_counts_from_items(
    items: tuple[BatchSubmitItemRecord, ...] | list[BatchSubmitItemRecord],
) -> BatchSubmitOperationCounts:
    inserted_count = sum(
        item.insert_status is BatchSubmitItemInsertStatus.INSERTED
        for item in items
    )
    already_present_count = sum(
        item.insert_status is BatchSubmitItemInsertStatus.ALREADY_PRESENT
        for item in items
    )
    enqueued_count = sum(
        item.enqueue_status is BatchSubmitItemEnqueueStatus.ENQUEUED
        for item in items
    )
    already_scheduled_count = sum(
        (
            item.enqueue_status
            is BatchSubmitItemEnqueueStatus.WORKFLOW_ALREADY_PRESENT
        )
        for item in items
    )
    failed_count = sum(
        item.enqueue_status is BatchSubmitItemEnqueueStatus.FAILED
        for item in items
    )
    return BatchSubmitOperationCounts(
        inserted_count=inserted_count,
        already_present_count=already_present_count,
        enqueued_count=enqueued_count,
        already_scheduled_count=already_scheduled_count,
        failed_count=failed_count,
    )


def build_batch_submit_operation_record(
    *,
    operation_key: str,
    experiment_name: str,
    status: BatchSubmitOperationStatus,
    requested_count: int,
    items: tuple[BatchSubmitItemRecord, ...] | list[BatchSubmitItemRecord],
    spec: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    created_at: datetime,
    completed_at: datetime | None = None,
) -> BatchSubmitOperationRecord:
    counts = batch_submit_operation_counts_from_items(items)
    return BatchSubmitOperationRecord(
        operation_key=operation_key,
        experiment_name=experiment_name,
        status=status,
        requested_count=requested_count,
        inserted_count=counts.inserted_count,
        already_present_count=counts.already_present_count,
        enqueued_count=counts.enqueued_count,
        already_scheduled_count=counts.already_scheduled_count,
        failed_count=counts.failed_count,
        spec=dict(spec or {}),
        metadata=dict(metadata or {}),
        created_at=created_at,
        completed_at=completed_at,
    )
