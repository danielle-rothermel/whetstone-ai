from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from whetstone.execution.partials import PartialCallRecord

type PartialRecordIndex = dict[tuple[str, int, str], PartialCallRecord]


@dataclass(frozen=True, slots=True)
class ExactResumeDecision:
    """One row's exact restored record and any required next attempt."""

    record: PartialCallRecord | None
    drive_ordinal: Literal[0, 1] | None


def index_partial_records(
    records: Iterable[PartialCallRecord],
    *,
    phase: str,
    unit: str,
) -> PartialRecordIndex:
    """Index only records in the current semantic phase and unit."""
    return {
        (record.task_id, record.sample_index, record.request_hash): record
        for record in records
        if record.phase == phase and record.unit == unit
    }


def resolve_exact_resume(
    records: PartialRecordIndex,
    *,
    task_id: str,
    sample_index: int,
    ordinal_0_request_hash: str,
    ordinal_1_request_hash: str,
) -> ExactResumeDecision:
    """Choose the exact terminal record or next request for one row.

    Ordinal 1 is terminal and wins when both attempts exist. An ordinal-0
    record marked pending is provisional evidence and requires the exact
    ordinal-1 request. A nonpending ordinal-0 record is terminal. Records from
    any other request identity, including another Evaluation Binding, are not
    eligible for restoration.
    """
    ordinal_1 = records.get((task_id, sample_index, ordinal_1_request_hash))
    if ordinal_1 is not None:
        if ordinal_1.redrive_pending:
            raise ValueError("an ordinal-1 partial record cannot be pending")
        return ExactResumeDecision(record=ordinal_1, drive_ordinal=None)

    ordinal_0 = records.get((task_id, sample_index, ordinal_0_request_hash))
    if ordinal_0 is None:
        return ExactResumeDecision(record=None, drive_ordinal=0)
    if ordinal_0.redrive_pending:
        return ExactResumeDecision(record=ordinal_0, drive_ordinal=1)
    return ExactResumeDecision(record=ordinal_0, drive_ordinal=None)


__all__ = [
    "ExactResumeDecision",
    "PartialRecordIndex",
    "index_partial_records",
    "resolve_exact_resume",
]
