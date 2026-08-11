"""SQLite real-time lease renewal pathway tests."""

# ruff: noqa: E402, F811

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from threading import Event

import pytest

from tests.core.effects.authority_support import (
    _HASH_A,
    _HASH_B,
    _LEASE_DURATION,
    _acquire,
    _FakeClock,
    _request,
    _result_ref,
)
from tests.optimization.sqlite_time import wait_for_sqlite_authority_after
from whetstone.core.effects.authority import (
    AcquireOutcome,
    EffectAuthority,
    EffectLease,
    EffectTerminal,
    ReplayPolicy,
    StaleLeaseError,
    TerminalConflictError,
    TerminalFailure,
)
from whetstone.core.identity import TypedRef


@dataclass(frozen=True, slots=True)
class _Backend:
    authority: EffectAuthority
    clock: _FakeClock | None
    database: Path | None

    def advance_past(self, instant: datetime) -> None:
        if self.clock is not None:
            now = self.clock()
            if now <= instant:
                self.clock.advance(instant - now + timedelta(microseconds=1))
            return
        assert self.database is not None
        wait_for_sqlite_authority_after(self.database, instant)


class _CoordinatedAuthority(EffectAuthority):
    def __init__(self, authority: EffectAuthority) -> None:
        self._authority = authority
        self._renewal_wait_strategy = authority._renewal_wait_strategy
        self.release_renewal = Event()
        self.renewal_entered = Event()
        self.terminal_entered = Event()
        self.renew_calls = 0
        self.terminal_lease: EffectLease | None = None

    def renew(
        self,
        lease: EffectLease,
        *,
        lease_duration: timedelta,
    ) -> EffectLease:
        self.renew_calls += 1
        self.renewal_entered.set()
        if not self.release_renewal.wait(timeout=2):
            raise TimeoutError("test did not release coordinated renewal")
        return self._authority.renew(
            lease,
            lease_duration=lease_duration,
        )

    def _validate_lease_duration(self, value: timedelta) -> timedelta:
        return self._authority._validate_lease_duration(value)

    def succeed(
        self,
        lease: EffectLease,
        *,
        result_ref: TypedRef,
    ) -> EffectTerminal:
        self.terminal_lease = lease
        self.terminal_entered.set()
        return self._authority.succeed(lease, result_ref=result_ref)

    def fail(
        self,
        lease: EffectLease,
        *,
        result_ref: TypedRef,
        failure: TerminalFailure,
    ) -> EffectTerminal:
        self.terminal_lease = lease
        self.terminal_entered.set()
        return self._authority.fail(
            lease,
            result_ref=result_ref,
            failure=failure,
        )


@pytest.fixture(
    name="backend",
    params=(
        "memory",
        "sqlite",
    ),
)
def backend_fixture(
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> _Backend:
    if request.param == "memory":
        clock = _FakeClock()
        return _Backend(EffectAuthority.memory(clock=clock), clock, None)
    database = tmp_path / "authority.sqlite"
    return _Backend(EffectAuthority.sqlite(database), None, database)


@pytest.fixture(
    name="timed_backend",
    params=(
        "memory",
        pytest.param("sqlite", marks=pytest.mark.sqlite_time_integration),
    ),
)
def timed_backend_fixture(
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> _Backend:
    if request.param == "memory":
        clock = _FakeClock()
        return _Backend(EffectAuthority.memory(clock=clock), clock, None)
    database = tmp_path / "timed-authority.sqlite"
    return _Backend(EffectAuthority.sqlite(database), None, database)


import sqlite3
from dataclasses import dataclass
from typing import Any

import pytest

from tests.optimization.support import (
    CountingProposalAdapter,
    RecordingEvaluationService,
    base_ref,
    eval_config,
    evaluation_binding,
    internal_reward_policy,
    make_harness,
    make_intent,
    make_store,
    proposal_request,
    proposed_candidate,
    registry,
)
from whetstone.core.effects.authority import (
    EffectAuthority,
)
from whetstone.core.effects.models import ReplayPolicy
from whetstone.core.identity import (
    ImmutableJsonObject,
    TerminalFailure,
    typed_ref_for_record,
)
from whetstone.experiment.binding import eval_config_reference
from whetstone.experiment.candidate import (
    Candidate,
    candidate_reference,
)
from whetstone.optimization.adapters import AdapterOutput
from whetstone.optimization.contracts import (
    BudgetDelta,
    EvaluationIntent,
    IntentOutcome,
    IntentResolution,
    StepMode,
    StepStatus,
)
from whetstone.optimization.tools.contracts import RuntimeToolHandle


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


NESTED_EVALUATION_RESULT_SCHEMA = "whetstone.test.nested_evaluation_result"


class NestedGraphEvaluationService(RecordingEvaluationService):
    def __init__(
        self,
        *args,
        persist_nested_result: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._persist_nested_result = persist_nested_result

    def resolve_evaluation_intent(
        self, intent: EvaluationIntent
    ) -> IntentResolution:
        resolution = super().resolve_evaluation_intent(intent)
        result_ref = resolution.evaluation_result_ref
        if result_ref is None:
            return resolution
        nested_result: dict[str, Any] = {
            "intent_id": intent.intent_id,
            "score": 1.0,
        }
        nested_ref = typed_ref_for_record(
            NESTED_EVALUATION_RESULT_SCHEMA,
            nested_result,
        )
        if self._persist_nested_result:
            self._store.put(NESTED_EVALUATION_RESULT_SCHEMA, nested_result)
        stored_result = self._store.get(result_ref.reference)
        if not isinstance(stored_result, dict):
            raise AssertionError("test Evaluation Result must be an object")
        evaluation_result = dict(stored_result)
        evaluation_result["nested_result_ref"] = nested_ref.model_dump(
            mode="json"
        )
        self._store.put(result_ref.schema_name, evaluation_result)
        return resolution.model_copy(
            update={
                "evaluation_result_ref": typed_ref_for_record(
                    result_ref.schema_name,
                    evaluation_result,
                )
            }
        )

    def validate_resolution_graph(self, resolution: IntentResolution) -> None:
        self.validation_calls.append(resolution)
        result_ref = resolution.evaluation_result_ref
        if result_ref is None:
            return
        evaluation_result = self._store.get(result_ref.reference)
        if not isinstance(evaluation_result, dict):
            raise AssertionError("test Evaluation Result must be an object")
        nested_ref = TypedRef.model_validate(
            evaluation_result["nested_result_ref"]
        )
        nested_result = self._store.get(nested_ref.reference)
        if (
            typed_ref_for_record(nested_ref.schema_name, nested_result)
            != nested_ref
        ):
            raise ValueError("nested Evaluation Result ref is not exact")


@dataclass
class MutableClock:
    current: datetime

    def __call__(self) -> datetime:
        return self.current


class PoisonThenValidAdapter:
    def __init__(self, poison: str) -> None:
        self.poison = poison
        self.invocations = 0

    @property
    def key(self) -> str:
        return "proposal-test"

    @property
    def mode(self) -> StepMode:
        return StepMode.PROPOSAL_ONLY

    @property
    def required_replay_policy(self) -> ReplayPolicy:
        return ReplayPolicy.IDEMPOTENT

    def invoke(
        self,
        request,
        handles: tuple[RuntimeToolHandle, ...],
    ) -> AdapterOutput:
        assert handles == ()
        self.invocations += 1
        proposed = proposed_candidate(
            request.candidates[0],
            "poison-retry",
            text="valid-retry",
        )
        intent = make_intent(
            proposed,
            run_id=request.run_id,
            step_index=request.step_index,
            reward_policy=request.run.record.reward_policy,
        )
        if self.invocations == 1:
            proposed, intent = self._poison(request, proposed, intent)
        return AdapterOutput(
            proposed_candidates=(proposed,),
            accepted_candidates=(proposed,),
            evaluation_intents=(intent,),
            budget_delta=BudgetDelta(consumed={"generations": 1}),
            proposed_status=StepStatus.COMPLETE,
        )

    def _poison(self, request, proposed, intent):
        if self.poison == "template":
            proposed = proposed_candidate(
                request.candidates[0],
                "poison-retry",
                text="{unavailable}",
            )
            intent = make_intent(
                proposed,
                run_id=request.run_id,
                step_index=request.step_index,
                reward_policy=request.run.record.reward_policy,
            )
        elif self.poison == "base":
            proposed = Candidate(
                candidate_id=proposed.candidate_id,
                base_ref=base_ref("foreign"),
                payload=proposed.payload,
            )
            intent = intent.model_copy(
                update={"candidate": candidate_reference(proposed)}
            )
        elif self.poison == "diff":
            payload = proposed.payload.to_json()
            payload["fixed"] = "poisoned"
            proposed = Candidate(
                candidate_id=proposed.candidate_id,
                base_ref=proposed.base_ref,
                payload=ImmutableJsonObject(payload),
            )
            intent = intent.model_copy(
                update={"candidate": candidate_reference(proposed)}
            )
        elif self.poison == "run":
            intent = intent.model_copy(update={"run_id": "foreign-run"})
        elif self.poison == "step":
            intent = intent.model_copy(
                update={"step_index": request.step_index + 1}
            )
        elif self.poison == "candidate":
            outsider = proposed_candidate(
                request.candidates[0],
                "outsider",
                text="outsider",
            )
            intent = intent.model_copy(
                update={"candidate": candidate_reference(outsider)}
            )
        elif self.poison == "binding":
            other_config = eval_config_reference(eval_config("e" * 64))
            intent = intent.model_copy(
                update={"evaluation_binding": evaluation_binding(other_config)}
            )
        elif self.poison == "policy":
            other_policy = internal_reward_policy().model_copy(
                update={"policy_name": "other-policy/v1"}
            )
            intent = make_intent(
                proposed,
                run_id=request.run_id,
                step_index=request.step_index,
                reward_policy=other_policy,
            )
        else:  # pragma: no cover - closed test parameter
            raise AssertionError(f"unknown poison {self.poison!r}")
        return proposed, intent


def test_renew_and_takeover_share_backend_authority_time(
    timed_backend: _Backend,
) -> None:
    request = _request()
    first = _acquire(timed_backend.authority, request)
    assert first.lease is not None

    timed_backend.advance_past(first.lease.expires_at - _LEASE_DURATION)
    renewed = timed_backend.authority.renew(
        first.lease,
        lease_duration=_LEASE_DURATION,
    )
    assert renewed.fence == first.lease.fence
    assert renewed.expires_at > first.lease.expires_at
    with pytest.raises(StaleLeaseError):
        timed_backend.authority.renew(
            first.lease,
            lease_duration=_LEASE_DURATION,
        )

    timed_backend.advance_past(renewed.expires_at)
    takeover = _acquire(
        timed_backend.authority,
        request,
        owner="worker-2",
        attempt="attempt-2",
    )
    assert takeover.outcome is AcquireOutcome.ACQUIRED
    assert takeover.lease is not None
    assert takeover.lease.owner_id == "worker-2"
    assert takeover.lease.attempt_id == "attempt-2"
    assert takeover.lease.fence == 2


@pytest.mark.parametrize(
    "policy",
    (ReplayPolicy.IDEMPOTENT, ReplayPolicy.DURABLE_WORKFLOW),
)
def test_stale_attempt_cannot_terminalize_after_takeover(
    timed_backend: _Backend,
    policy: ReplayPolicy,
) -> None:
    request = _request(policy=policy)
    first = _acquire(timed_backend.authority, request)
    assert first.lease is not None
    timed_backend.advance_past(first.lease.expires_at)
    takeover = _acquire(
        timed_backend.authority,
        request,
        owner="worker-2",
        attempt="attempt-2",
    )
    assert takeover.lease is not None

    with pytest.raises(StaleLeaseError):
        timed_backend.authority.succeed(
            first.lease,
            result_ref=_result_ref("first"),
        )
    terminal = timed_backend.authority.succeed(
        takeover.lease,
        result_ref=_result_ref("second"),
    )
    assert terminal.attempt_id == "attempt-2"
    assert terminal.result_ref == _result_ref("second")


def test_no_redrive_expiry_becomes_immutable_recovery_required(
    timed_backend: _Backend,
) -> None:
    request = _request(policy=ReplayPolicy.NO_REDRIVE)
    first = _acquire(timed_backend.authority, request)
    assert first.lease is not None
    timed_backend.advance_past(first.lease.expires_at)

    recovered = _acquire(
        timed_backend.authority,
        request,
        owner="worker-2",
        attempt="attempt-2",
    )
    assert recovered.outcome is AcquireOutcome.RECOVERY_REQUIRED
    assert recovered.terminal is not None
    assert recovered.terminal.failure is not None
    assert recovered.terminal.failure.code.startswith("effect-recovery:")
    assert recovered.terminal.failure.details["owner_id"] == "worker-1"
    assert recovered.terminal.failure.details["attempt_id"] == "attempt-1"

    replay = _acquire(
        timed_backend.authority,
        request,
        owner="worker-3",
        attempt="attempt-3",
    )
    assert replay == recovered
    with pytest.raises(TerminalConflictError):
        timed_backend.authority.succeed(
            first.lease,
            result_ref=_result_ref("late"),
        )


def test_request_hash_and_policy_are_immutable(
    backend: _Backend,
) -> None:
    original = _request()
    _acquire(backend.authority, original)

    divergent_identity = _acquire(
        backend.authority,
        _request(identity_hash=_HASH_B),
        owner="worker-2",
        attempt="attempt-2",
    )
    assert divergent_identity.outcome is AcquireOutcome.REQUEST_CONFLICT
    assert divergent_identity.existing_request_hash == _HASH_A
    assert divergent_identity.existing_replay_policy is ReplayPolicy.IDEMPOTENT

    divergent_policy = _acquire(
        backend.authority,
        _request(policy=ReplayPolicy.DURABLE_WORKFLOW),
        owner="worker-2",
        attempt="attempt-2",
    )
    assert divergent_policy.outcome is AcquireOutcome.REQUEST_CONFLICT


def test_success_and_failure_are_exact_immutable_and_replayed(
    backend: _Backend,
) -> None:
    success_request = _request(key="effect:success")
    success = _acquire(backend.authority, success_request)
    assert success.lease is not None
    terminal = backend.authority.succeed(
        success.lease,
        result_ref=_result_ref("result-a"),
    )
    assert (
        backend.authority.succeed(
            success.lease,
            result_ref=_result_ref("result-a"),
        )
        == terminal
    )
    with pytest.raises(TerminalConflictError):
        backend.authority.succeed(
            success.lease,
            result_ref=_result_ref("result-b"),
        )
    replay = _acquire(
        backend.authority,
        success_request,
        owner="worker-2",
        attempt="attempt-2",
    )
    assert replay.outcome is AcquireOutcome.SUCCEEDED
    assert replay.terminal == terminal

    failure_request = _request(key="effect:failure")
    failed = _acquire(backend.authority, failure_request)
    assert failed.lease is not None
    failure = TerminalFailure(
        code="provider_timeout",
        message="provider timed out",
        details={"provider": "test"},
    )
    failed_terminal = backend.authority.fail(
        failed.lease,
        result_ref=_result_ref("failed-result"),
        failure=failure,
    )
    failed_replay = _acquire(
        backend.authority,
        failure_request,
        owner="worker-2",
        attempt="attempt-2",
    )
    assert failed_replay.outcome is AcquireOutcome.FAILED
    assert failed_replay.terminal == failed_terminal
    with pytest.raises(TerminalConflictError):
        backend.authority.fail(
            failed.lease,
            result_ref=_result_ref("different"),
            failure=failure,
        )


def test_sqlite_maintenance_terminalization_surfaces_renewal_loss(
    tmp_path: Path,
) -> None:
    database = tmp_path / "renewal-loss.sqlite"
    authority = EffectAuthority.sqlite(database)
    acquired = _acquire(
        authority,
        _request(),
        duration=timedelta(milliseconds=1),
    )
    assert acquired.lease is not None
    wait_for_sqlite_authority_after(database, acquired.lease.expires_at)

    with pytest.raises(StaleLeaseError, match="effect lease is stale"):
        with authority.maintain(
            acquired.lease,
            lease_duration=timedelta(milliseconds=1),
        ) as maintenance:
            assert maintenance._stop.wait(timeout=1)
            maintenance.succeed(result_ref=_result_ref("too-late"))


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
