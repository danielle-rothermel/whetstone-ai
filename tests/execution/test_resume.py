"""Exact partial-row request selection shared by environment evaluators."""

from __future__ import annotations

import pytest

from whetstone.execution.partials import PartialCallRecord
from whetstone.execution.resume import (
    index_partial_records,
    resolve_exact_resume,
)


def _record(request_identity: str, *, pending: bool) -> PartialCallRecord:
    return PartialCallRecord(
        phase="internal_eval",
        instance_id="task-1",
        unit="candidate-1",
        repeat_id=0,
        request_identity=request_identity,
        redrive_pending=pending,
        score=None if pending else 1.0,
        failed=pending,
    )


def test_other_request_identity_is_not_restorable() -> None:
    records = index_partial_records(
        (_record("c" * 64, pending=False),),
        phase="internal_eval",
        unit="candidate-1",
    )

    decision = resolve_exact_resume(
        records,
        instance_id="task-1",
        repeat_id=0,
        ordinal_0_request_identity="a" * 64,
        ordinal_1_request_identity="b" * 64,
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
        instance_id="task-1",
        repeat_id=0,
        ordinal_0_request_identity=ordinal_0.request_identity,
        ordinal_1_request_identity=ordinal_1.request_identity,
    )

    assert decision.record == ordinal_1
    assert decision.drive_ordinal is None


def test_pending_ordinal_zero_requires_exact_ordinal_one() -> None:
    ordinal_0 = _record("a" * 64, pending=True)
    records = index_partial_records(
        (ordinal_0,), phase="internal_eval", unit="candidate-1"
    )

    decision = resolve_exact_resume(
        records,
        instance_id="task-1",
        repeat_id=0,
        ordinal_0_request_identity=ordinal_0.request_identity,
        ordinal_1_request_identity="b" * 64,
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
            instance_id="task-1",
            repeat_id=0,
            ordinal_0_request_identity="a" * 64,
            ordinal_1_request_identity=ordinal_1.request_identity,
        )
