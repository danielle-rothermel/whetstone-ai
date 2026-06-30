from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

from dr_dspy.records.models import (
    BatchSubmitItemRecord,
    BatchSubmitItemStatus,
    BatchSubmitOperationRecord,
    BatchSubmitOperationStatus,
)

# Enqueued items record whether the spec row was new via enqueue_metadata.
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
    inserted_count = 0
    already_present_count = 0
    enqueued_count = 0
    failed_count = 0
    for item in items:
        if item.status is BatchSubmitItemStatus.FAILED:
            failed_count += 1
            continue
        if item.status is BatchSubmitItemStatus.ENQUEUED:
            enqueued_count += 1
            spec_outcome = item.enqueue_metadata.get(SPEC_OUTCOME_METADATA_KEY)
            if spec_outcome == SpecInsertOutcome.INSERTED.value:
                inserted_count += 1
            elif spec_outcome == SpecInsertOutcome.ALREADY_PRESENT.value:
                already_present_count += 1
            else:
                raise ValueError(
                    "enqueued batch submit items require "
                    f"{SPEC_OUTCOME_METADATA_KEY} metadata"
                )
            continue
        if item.status is BatchSubmitItemStatus.INSERTED:
            inserted_count += 1
            continue
        if item.status is BatchSubmitItemStatus.ALREADY_PRESENT:
            already_present_count += 1
            continue
        raise ValueError(
            f"unsupported batch submit item status: {item.status}"
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
