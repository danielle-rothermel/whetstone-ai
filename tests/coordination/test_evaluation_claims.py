from __future__ import annotations

import pytest
from dr_store import ObjectStore, SqliteBackend

from tests.evaluation.support import _engine, _intent
from whetstone.coordination.evaluation_service import EngineEvaluationService
from whetstone.evaluation.drivers.internal import InternalRowRequest
from whetstone.execution.fanout import ProcessJob


def test_expired_owner_cannot_renew_after_new_generation_claims(
    tmp_path,
) -> None:
    database = tmp_path / "claim-fence.sqlite"
    now = [100.0]
    first_store = ObjectStore(SqliteBackend(database))
    second_store = ObjectStore(SqliteBackend(database))

    def reject_submission(_request: InternalRowRequest) -> ProcessJob:
        raise AssertionError("claim arbitration must not create process jobs")

    first_engine = _engine(
        tmp_path,
        store=first_store,
        row_job_factory=reject_submission,
    )
    second_engine = _engine(
        tmp_path,
        store=second_store,
        row_job_factory=reject_submission,
    )
    intent = _intent(
        first_engine,
        intent_id="fenced-intent",
        purpose="fence",
    )
    first = EngineEvaluationService(
        store=first_store,
        engine=first_engine,
        claim_lease_seconds=1.0,
        clock=lambda: now[0],
        sleep=lambda _seconds: None,
    )
    second = EngineEvaluationService(
        store=second_store,
        engine=second_engine,
        claim_lease_seconds=1.0,
        clock=lambda: now[0],
        sleep=lambda _seconds: None,
    )
    first_claim = first._claim(intent)
    assert first_claim is not None
    now[0] = 102.0
    second_claim = second._claim(intent)
    assert second_claim is not None
    assert second_claim.generation == 1

    with pytest.raises(RuntimeError, match="not owned"):
        first._renew_claim(intent, first_claim)
