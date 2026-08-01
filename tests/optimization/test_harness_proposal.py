"""Proposal evaluation is outside the checkpointed adapter invocation."""

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from dr_store import ObjectNotFoundError
from pydantic import ValidationError

from whetstone.optimization import (
    EVALUATION_EVIDENCE_SCHEMA,
    INTENT_RESOLUTION_SCHEMA,
    REWARD_SCHEMA,
    AdapterOutput,
    BudgetDelta,
    Candidate,
    EvaluationIntent,
    IntentOutcome,
    IntentResolution,
    ReplayPolicy,
    RuntimeToolHandle,
    StepMode,
    StepStatus,
    TerminalFailure,
    TypedRef,
    candidate_reference,
    step_result_reference,
    typed_ref_for_record,
)
from whetstone.optimization.effect_authority import (
    AcquireOutcome,
    EffectAuthority,
)
from whetstone.optimization.harness import (
    ADAPTER_CHECKPOINT_SCHEMA,
    INTENT_EFFECT_KEY_PREFIX,
)
from whetstone.optimization.identity import ImmutableJsonObject
from whetstone.optimization.schema import eval_config_reference

from .sqlite_time import wait_for_sqlite_authority_after
from .support import (
    CountingProposalAdapter,
    RecordingEvaluationService,
    base_ref,
    candidate,
    eval_config,
    evaluation_binding,
    internal_reward_policy,
    make_harness,
    make_intent,
    make_store,
    output_contract,
    proposal_request,
    proposed_candidate,
    registry,
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
            budget_delta=BudgetDelta(consumed={"rollouts": 1}),
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


@pytest.mark.parametrize(
    ("poison", "error"),
    [
        ("template", "exact run template contract"),
        ("base", "bind an exact request candidate"),
        ("diff", "canonical run mutation diff"),
        ("run", "another optimization run"),
        ("step", "another optimization step"),
        ("candidate", "not an exact Step output candidate"),
        ("binding", "target Eval Config must match"),
        ("policy", "exact run Reward Policy"),
    ],
)
def test_invalid_adapter_output_is_not_checkpointed_and_can_retry(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    poison: str,
    error: str,
) -> None:
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    authority = EffectAuthority.memory(clock=clock)
    acquired_keys: list[str] = []
    real_acquire = authority.acquire

    def record_acquire(effect_request, **kwargs):
        acquired_keys.append(str(effect_request.semantic_key))
        return real_acquire(effect_request, **kwargs)

    monkeypatch.setattr(authority, "acquire", record_acquire)
    store = make_store(tmp_path)
    service = RecordingEvaluationService(store)
    adapter = PoisonThenValidAdapter(poison)
    request = proposal_request()
    harness = make_harness(
        store=store,
        adapter_registry=registry(adapter),
        run=request.run,
        effect_authority=authority,
        evaluation_service=service,
        lease_duration=timedelta(seconds=1),
    )
    persisted_schemas: list[str] = []
    real_put = harness._put

    def record_put(schema, content):
        persisted_schemas.append(schema)
        return real_put(schema, content)

    monkeypatch.setattr(harness, "_put", record_put)

    with pytest.raises(ValueError, match=error):
        harness.run_step(request)

    assert ADAPTER_CHECKPOINT_SCHEMA not in persisted_schemas
    assert service.calls == []
    assert not any(
        key.startswith(INTENT_EFFECT_KEY_PREFIX) for key in acquired_keys
    )
    assert harness.resolve_step_result(request.run_id, 0) is None

    clock.current += timedelta(seconds=2)
    result, _ = harness.run_step(request)

    assert adapter.invocations == 2
    assert len(service.calls) == 1
    assert len(result.resolved_intents) == 1
    assert ADAPTER_CHECKPOINT_SCHEMA in persisted_schemas
    assert persisted_schemas.count(INTENT_RESOLUTION_SCHEMA) == 1
    assert "whetstone.intent_resolution" not in persisted_schemas


@pytest.mark.sqlite_time_integration
def test_fresh_sqlite_restart_reuses_adapter_checkpoint(tmp_path) -> None:
    adapter = CountingProposalAdapter()
    request = proposal_request()
    effect_database = tmp_path / "effects.sqlite"
    lease_duration = timedelta(seconds=1.2)
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


def test_restart_reuses_terminal_intent_prefix_after_later_crash(
    tmp_path,
) -> None:
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    authority = EffectAuthority.memory(clock=clock)
    store = make_store(tmp_path)
    adapter = CountingProposalAdapter(
        candidates=(candidate("first"), candidate("second")),
        budget_delta=BudgetDelta(consumed={"rollouts": 2}),
    )
    request = proposal_request(contract=output_contract(2))
    service = RecordingEvaluationService(store, crash_on_call=2)
    harness = make_harness(
        store=store,
        adapter_registry=registry(adapter),
        run=request.run,
        effect_authority=authority,
        evaluation_service=service,
        lease_duration=timedelta(seconds=1),
    )

    with pytest.raises(RuntimeError, match="crash during evaluation"):
        harness.run_step(request)
    first_intent, crashed_intent = service.calls
    clock.current += timedelta(seconds=2)

    result, _ = harness.run_step(request)

    assert adapter.invocations == 1
    assert service.calls == [first_intent, crashed_intent, crashed_intent]
    assert tuple(
        resolution.intent for resolution in result.resolved_intents
    ) == (first_intent, crashed_intent)


def test_fresh_resolution_runs_service_graph_validation_with_exact_value(
    tmp_path,
) -> None:
    store = make_store(tmp_path)
    service = NestedGraphEvaluationService(store)
    request = proposal_request()
    harness = make_harness(
        store=store,
        adapter_registry=registry(CountingProposalAdapter()),
        run=request.run,
        evaluation_service=service,
    )

    result, _ = harness.run_step(request)

    assert len(service.calls) == 1
    assert service.validation_calls == list(result.resolved_intents)


def test_graph_validation_failure_is_atomic_before_fresh_terminalization(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = make_store(tmp_path)
    authority = EffectAuthority.memory()
    service = NestedGraphEvaluationService(
        store,
        persist_nested_result=False,
    )
    request = proposal_request()
    harness = make_harness(
        store=store,
        adapter_registry=registry(CountingProposalAdapter()),
        run=request.run,
        effect_authority=authority,
        evaluation_service=service,
    )
    persisted_schemas: list[str] = []
    real_put = harness._put

    def record_put(schema, content):
        persisted_schemas.append(schema)
        return real_put(schema, content)

    monkeypatch.setattr(harness, "_put", record_put)

    with pytest.raises(ObjectNotFoundError):
        harness.run_step(request)

    assert len(service.calls) == 1
    assert len(service.validation_calls) == 1
    assert service.validation_calls[0].intent == service.calls[0]
    assert INTENT_RESOLUTION_SCHEMA not in persisted_schemas
    assert harness.resolve_step_result(request.run_id, 0) is None
    acquisition = authority.acquire(
        harness._intent_effect_request(request, service.calls[0]),
        owner_id="terminalization-probe",
        attempt_id="terminalization-probe",
        lease_duration=timedelta(seconds=1),
    )
    assert acquisition.outcome is AcquireOutcome.BUSY
    assert acquisition.terminal is None


def test_terminal_replay_graph_loss_blocks_binding_without_reexecution(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    authority = EffectAuthority.memory(clock=clock)
    store = make_store(tmp_path)
    service = NestedGraphEvaluationService(store, crash_on_call=2)
    adapter = CountingProposalAdapter(
        candidates=(candidate("first"), candidate("second")),
        budget_delta=BudgetDelta(consumed={"rollouts": 2}),
    )
    request = proposal_request(contract=output_contract(2))
    harness = make_harness(
        store=store,
        adapter_registry=registry(adapter),
        run=request.run,
        effect_authority=authority,
        evaluation_service=service,
        lease_duration=timedelta(seconds=1),
    )

    with pytest.raises(RuntimeError, match="crash during evaluation"):
        harness.run_step(request)

    terminal_resolution = service.validation_calls[0]
    result_ref = terminal_resolution.evaluation_result_ref
    assert result_ref is not None
    evaluation_result = store.get(result_ref.reference)
    assert isinstance(evaluation_result, dict)
    missing_nested_ref = TypedRef.model_validate(
        evaluation_result["nested_result_ref"]
    )
    real_get = store.get

    def get_with_nested_result_loss(reference):
        if reference == missing_nested_ref.reference:
            raise ObjectNotFoundError(reference=reference)
        return real_get(reference)

    monkeypatch.setattr(store, "get", get_with_nested_result_loss)
    calls_before_replay = list(service.calls)
    clock.current += timedelta(seconds=2)

    with pytest.raises(ObjectNotFoundError):
        harness.run_step(request)

    assert service.calls == calls_before_replay
    assert service.validation_calls == [
        terminal_resolution,
        terminal_resolution,
    ]
    assert adapter.invocations == 1
    assert harness.resolve_step_result(request.run_id, 0) is None


def test_terminal_intent_replay_rechecks_missing_primary_result(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    authority = EffectAuthority.memory(clock=clock)
    store = make_store(tmp_path)
    adapter = CountingProposalAdapter(
        candidates=(candidate("first"), candidate("second")),
        budget_delta=BudgetDelta(consumed={"rollouts": 2}),
    )
    request = proposal_request(contract=output_contract(2))
    crashed_service = RecordingEvaluationService(store, crash_on_call=2)
    crashed = make_harness(
        store=store,
        adapter_registry=registry(adapter),
        run=request.run,
        effect_authority=authority,
        evaluation_service=crashed_service,
        lease_duration=timedelta(seconds=1),
    )

    with pytest.raises(RuntimeError, match="crash during evaluation"):
        crashed.run_step(request)
    first_intent, _ = crashed_service.calls
    missing_ref = typed_ref_for_record(
        EVALUATION_EVIDENCE_SCHEMA,
        {
            "intent_id": first_intent.intent_id,
            "candidate_identity_hash": first_intent.candidate.identity_hash,
            "outcome": IntentOutcome.COMPLETED.value,
        },
    )
    clock.current += timedelta(seconds=2)

    fresh_store = make_store(tmp_path)
    fresh_service = RecordingEvaluationService(fresh_store)
    fresh = make_harness(
        store=fresh_store,
        adapter_registry=registry(adapter),
        run=request.run,
        effect_authority=authority,
        evaluation_service=fresh_service,
        lease_duration=timedelta(seconds=1),
    )
    real_get = fresh_store.get
    missing_reads = 0

    def get_with_missing_primary(reference):
        nonlocal missing_reads
        if reference == missing_ref.reference:
            missing_reads += 1
            raise ObjectNotFoundError(reference=reference)
        return real_get(reference)

    monkeypatch.setattr(fresh_store, "get", get_with_missing_primary)
    maintenance_calls = 0
    real_maintain = authority.maintain

    def record_maintenance(*args, **kwargs):
        nonlocal maintenance_calls
        maintenance_calls += 1
        return real_maintain(*args, **kwargs)

    monkeypatch.setattr(authority, "maintain", record_maintenance)

    with pytest.raises(ObjectNotFoundError):
        fresh.run_step(request)

    assert missing_reads == 1
    assert maintenance_calls == 0
    assert adapter.invocations == 1
    assert fresh_service.calls == []
    assert fresh.resolve_step_result(request.run_id, 0) is None


@pytest.mark.parametrize(
    ("missing", "outcome"),
    [
        ("evaluation_result", IntentOutcome.COMPLETED),
        ("evaluation_result", IntentOutcome.FAILED),
        ("reward_evidence", IntentOutcome.COMPLETED),
    ],
)
def test_missing_referenced_object_stops_before_resolution_terminalization(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
    outcome: IntentOutcome,
) -> None:
    store = make_store(tmp_path)
    request = proposal_request()
    service = RecordingEvaluationService(
        store,
        outcome=outcome,
        persist_evaluation_result=missing != "evaluation_result",
        persist_reward_evidence=missing != "reward_evidence",
    )
    harness = make_harness(
        store=store,
        adapter_registry=registry(CountingProposalAdapter()),
        run=request.run,
        evaluation_service=service,
    )
    persisted_schemas: list[str] = []
    real_put = harness._put

    def record_put(schema, content):
        persisted_schemas.append(schema)
        return real_put(schema, content)

    monkeypatch.setattr(harness, "_put", record_put)

    with pytest.raises(ObjectNotFoundError):
        harness.run_step(request)

    assert len(service.calls) == 1
    assert INTENT_RESOLUTION_SCHEMA not in persisted_schemas
    assert harness.resolve_step_result(request.run_id, 0) is None


@pytest.mark.parametrize("bypass", ["outcome_schema", "duplicate_reward"])
def test_model_copy_bypass_is_revalidated_before_terminalization(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    bypass: str,
) -> None:
    store = make_store(tmp_path)
    request = proposal_request()
    service = RecordingEvaluationService(store)
    real_resolve = service.resolve_evaluation_intent
    authority = EffectAuthority.memory()

    def resolve_with_bypass(intent):
        resolution = real_resolve(intent)
        if bypass == "outcome_schema":
            return resolution.model_copy(
                update={
                    "outcome": IntentOutcome.FAILED,
                    "terminal_failure": TerminalFailure(
                        code="forged_failure",
                        message="forged failure",
                    ),
                }
            )
        reward_ref = resolution.reward_ref
        assert reward_ref is not None
        duplicated = reward_ref.record.model_copy(
            update={
                "evidence_refs": (
                    *reward_ref.record.evidence_refs,
                    reward_ref.record.evidence_refs[0],
                )
            }
        )
        duplicated_ref = reward_ref.model_copy(
            update={
                "record": duplicated,
                "record_ref": typed_ref_for_record(
                    REWARD_SCHEMA, duplicated.record_content()
                ),
            }
        )
        return resolution.model_copy(
            update={
                "reward_ref": duplicated_ref,
                "reward_evidence_refs": duplicated.evidence_refs,
            }
        )

    monkeypatch.setattr(
        service,
        "resolve_evaluation_intent",
        resolve_with_bypass,
    )
    harness = make_harness(
        store=store,
        adapter_registry=registry(CountingProposalAdapter()),
        run=request.run,
        effect_authority=authority,
        evaluation_service=service,
    )
    persisted_schemas: list[str] = []
    real_put = harness._put

    def record_put(schema, content):
        persisted_schemas.append(schema)
        return real_put(schema, content)

    monkeypatch.setattr(harness, "_put", record_put)

    with pytest.raises(
        ValidationError,
        match=(
            "evaluation_result_ref must use schema"
            if bypass == "outcome_schema"
            else "Reward evidence_refs must be unique"
        ),
    ):
        harness.run_step(request)

    assert len(service.calls) == 1
    assert INTENT_RESOLUTION_SCHEMA not in persisted_schemas
    assert harness.resolve_step_result(request.run_id, 0) is None
    acquisition = authority.acquire(
        harness._intent_effect_request(request, service.calls[0]),
        owner_id="terminalization-probe",
        attempt_id="terminalization-probe",
        lease_duration=timedelta(seconds=1),
    )
    assert acquisition.outcome is AcquireOutcome.BUSY
    assert acquisition.terminal is None


def test_candidate_local_failure_does_not_erase_successful_steps(
    tmp_path,
) -> None:
    store = make_store(tmp_path)
    first_adapter = CountingProposalAdapter(status=StepStatus.CONTINUE)
    first_request = proposal_request()
    harness = make_harness(
        store=store,
        adapter_registry=registry(first_adapter),
        run=first_request.run,
        evaluation_service=RecordingEvaluationService(store),
    )
    first, first_ref = harness.run_step(first_request)

    failed_service = RecordingEvaluationService(
        store, outcome=IntentOutcome.FAILED
    )
    second_adapter = CountingProposalAdapter(status=StepStatus.COMPLETE)
    second = make_harness(
        store=store,
        adapter_registry=registry(second_adapter),
        run=first_request.run,
        evaluation_service=failed_service,
    )
    second_request = proposal_request(
        step_index=1,
        prior_step_result_ref=first_ref,
        budget=first.budget,
    )
    second_result, second_ref = second.run_step(second_request)
    assert second_result.status is StepStatus.COMPLETE
    assert second_result.resolved_intents[0].outcome is IntentOutcome.FAILED

    terminal, _ = second.terminalize(
        run=first_request.run,
        step_results=(
            step_result_reference(first),
            step_result_reference(second_result),
        ),
    )
    assert terminal.step_result_refs == (first_ref, second_ref)
    assert len(terminal.proposals) == 1


def test_pre_execution_rejection_is_recorded_without_evidence(
    tmp_path,
) -> None:
    store = make_store(tmp_path)
    adapter = CountingProposalAdapter()
    request = proposal_request()
    harness = make_harness(
        store=store,
        adapter_registry=registry(adapter),
        run=request.run,
        evaluation_service=RecordingEvaluationService(
            store, outcome=IntentOutcome.REJECTED
        ),
    )
    result, _ = harness.run_step(request)
    resolution = result.resolved_intents[0]
    assert resolution.outcome is IntentOutcome.REJECTED
    assert resolution.evaluation_result_ref is None
    assert resolution.reward_evidence_refs == ()
