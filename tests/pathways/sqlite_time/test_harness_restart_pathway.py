"""SQLite harness restart pathway test."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

import pytest

from tests.optimization.sqlite_time import wait_for_sqlite_authority_after
from tests.optimization.support import (
    CountingProposalAdapter,
    RecordingEvaluationService,
    make_harness,
    make_store,
    proposal_request,
    registry,
)
from whetstone.core.effects.authority import EffectAuthority
from whetstone.core.effects.models import ReplayPolicy
from whetstone.optimization.contracts import (
    EvaluationIntent,
    IntentOutcome,
    IntentResolution,
)


class CrashOnceEvaluationService:
    def __init__(self) -> None:
        self.calls = 0
        self.validation_calls: list[IntentResolution] = []

    @property
    def replay_policy(self) -> ReplayPolicy:
        return ReplayPolicy.IDEMPOTENT

    def resolve_evaluation_intent(
        self, intent: EvaluationIntent
    ) -> IntentResolution:
        del intent
        self.calls += 1
        raise RuntimeError("crash during external evaluation")

    def validate_resolution_graph(self, resolution: IntentResolution) -> None:
        self.validation_calls.append(resolution)


@pytest.mark.sqlite_time_integration
def test_fresh_sqlite_restart_reuses_adapter_checkpoint(tmp_path) -> None:
    adapter = CountingProposalAdapter()
    request = proposal_request()
    effect_database = tmp_path / "effects.sqlite"
    lease_duration = timedelta(milliseconds=200)
    crashed_store = make_store(tmp_path)
    crashed = make_harness(
        store=crashed_store,
        adapter_registry=registry(adapter),
        run=request.run,
        effect_authority=EffectAuthority.sqlite(effect_database),
        evaluation_service=CrashOnceEvaluationService(),
        lease_duration=lease_duration,
    )
    with pytest.raises(RuntimeError, match="crash during"):
        crashed.run_step(request)
    assert adapter.invocations == 1
    assert crashed.resolve_step_result(request.run_id, 0) is None

    with sqlite3.connect(effect_database) as connection:
        active = connection.execute(
            """
            SELECT semantic_key, fence, expires_at
            FROM whetstone_effect_authority
            WHERE state = 'leased'
            """
        ).fetchall()
    assert len(active) == 1
    crashed_effect_key, crashed_fence, crashed_expiry_text = active[0]
    assert type(crashed_effect_key) is str
    assert crashed_fence == 1
    assert type(crashed_expiry_text) is str

    fresh_store = make_store(tmp_path)
    wait_for_sqlite_authority_after(
        effect_database,
        datetime.fromisoformat(crashed_expiry_text),
    )
    fresh = make_harness(
        store=fresh_store,
        adapter_registry=registry(adapter),
        run=request.run,
        effect_authority=EffectAuthority.sqlite(effect_database),
        evaluation_service=RecordingEvaluationService(fresh_store),
        lease_duration=lease_duration,
    )
    result, result_ref = fresh.run_step(request)
    assert adapter.invocations == 1
    assert result.resolved_intents[0].outcome is IntentOutcome.COMPLETED
    assert fresh.resolve_step_result(request.run_id, 0) == result_ref
    with sqlite3.connect(effect_database) as connection:
        terminal = connection.execute(
            """
            SELECT state, fence FROM whetstone_effect_authority
            WHERE semantic_key = ?
            """,
            (crashed_effect_key,),
        ).fetchone()
    assert terminal == ("succeeded", 2)
