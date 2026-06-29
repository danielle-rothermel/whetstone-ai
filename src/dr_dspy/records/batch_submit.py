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

# Legacy metadata key; prefer insert_status on BatchSubmitItemRecord.
SPEC_OUTCOME_METADATA_KEY = "spec_outcome"


class SpecInsertOutcome(StrEnum):
    INSERTED = "inserted"
    ALREADY_PRESENT = "already_present"


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
    failed_count: int


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
    failed_count = sum(
        item.enqueue_status is BatchSubmitItemEnqueueStatus.FAILED
        for item in items
    )
    return BatchSubmitOperationCounts(
        inserted_count=inserted_count,
        already_present_count=already_present_count,
        enqueued_count=enqueued_count,
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
        failed_count=counts.failed_count,
        spec=dict(spec or {}),
        metadata=dict(metadata or {}),
        created_at=created_at,
        completed_at=completed_at,
    )
