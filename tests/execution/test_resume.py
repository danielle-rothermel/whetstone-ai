from __future__ import annotations

import pytest

from whetstone.execution.partials import PartialCallRecord
from whetstone.execution.resume import (
    index_partial_records,
    resolve_exact_resume,
)


def _record(request_hash: str, *, pending: bool) -> PartialCallRecord:
    return PartialCallRecord(
        phase="internal_eval",
        task_id="task-1",
        unit="candidate-1",
        sample_index=0,
        request_hash=request_hash,
        redrive_pending=pending,
        score=None if pending else 1.0,
        failed=pending,
    )


def test_other_request_hash_is_not_restorable() -> None:
    records = index_partial_records(
        (_record("c" * 64, pending=False),),
        phase="internal_eval",
        unit="candidate-1",
    )

    decision = resolve_exact_resume(
        records,
        task_id="task-1",
        sample_index=0,
        ordinal_0_request_hash="a" * 64,
        ordinal_1_request_hash="b" * 64,
    )

    assert decision.record is None
    assert decision.drive_ordinal == 0


def test_terminal_ordinal_one_wins_over_pending_ordinal_zero() -> None:
    ordinal_0 = _record("a" * 64, pending=True)
    ordinal_1 = _record("b" * 64, pending=False)
    records = index_partial_records(
        (ordinal_0, ordinal_1),
        phase="internal_eval",
        unit="candidate-1",
    )

    decision = resolve_exact_resume(
        records,
        task_id="task-1",
        sample_index=0,
        ordinal_0_request_hash=ordinal_0.request_hash,
        ordinal_1_request_hash=ordinal_1.request_hash,
    )

    assert decision.record == ordinal_1
    assert decision.drive_ordinal is None


@pytest.mark.precheck
def test_pending_ordinal_zero_requires_exact_ordinal_one() -> None:
    ordinal_0 = _record("a" * 64, pending=True)
    records = index_partial_records(
        (ordinal_0,), phase="internal_eval", unit="candidate-1"
    )

    decision = resolve_exact_resume(
        records,
        task_id="task-1",
        sample_index=0,
        ordinal_0_request_hash=ordinal_0.request_hash,
        ordinal_1_request_hash="b" * 64,
    )

    assert decision.record == ordinal_0
    assert decision.drive_ordinal == 1


def test_pending_ordinal_one_is_rejected() -> None:
    ordinal_1 = _record("b" * 64, pending=True)
    records = index_partial_records(
        (ordinal_1,), phase="internal_eval", unit="candidate-1"
    )

    with pytest.raises(ValueError, match="ordinal-1 partial record"):
        resolve_exact_resume(
            records,
            task_id="task-1",
            sample_index=0,
            ordinal_0_request_hash="a" * 64,
            ordinal_1_request_hash=ordinal_1.request_hash,
        )
