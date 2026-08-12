from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from whetstone.execution.partials import PartialCallRecord

type PartialRecordIndex = dict[tuple[str, int, str], PartialCallRecord]


@dataclass(frozen=True, slots=True)
class ExactResumeDecision:
    record: PartialCallRecord | None
    drive_ordinal: Literal[0, 1] | None


def index_partial_records(
    records: Iterable[PartialCallRecord],
    *,
    phase: str,
    unit: str,
) -> PartialRecordIndex:
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
