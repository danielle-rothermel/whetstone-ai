from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event

import pytest
from dr_store import MemoryBackend, ObjectStore, SqliteBackend

from tests.envs.support import (
    execution_policy,
    process_row_job_factory,
)
from whetstone.envs.factory import build_env_experiment
from whetstone.envs.internal_eval import (
    InternalRowJobFactory,
    InternalRowOutcome,
    InternalRowRequest,
    InternalRowResult,
)
from whetstone.evaluation import (
    EngineEvaluation,
    EngineEvaluationService,
    EngineToolEvaluator,
    EvaluationEngine,
    EvaluationEvidence,
    EvaluationOutputComponentTraceStep,
    EvaluationOutputRow,
    EvaluationOutputsRecord,
    EvaluationRequest,
)
from whetstone.evaluation import engine as evaluation_engine_module
from whetstone.evaluation.schema import EvaluationIntentClaim
from whetstone.evaluation_role import EvaluationRole
from whetstone.execution.fanout import ProcessJob
from whetstone.execution.partials import PartialLog
from whetstone.execution.prompt_cache import PromptResultCache
from whetstone.optimization import (
    Candidate,
    EvaluationIntent,
    IntentOutcome,
    Reward,
    ToolCall,
    ToolCapacity,
    ToolCapacityScope,
    ToolConfig,
    ToolDefinition,
    TypedRef,
    candidate_reference,
    tool_capacity_binding,
    tool_config_reference,
    tool_definition_reference,
)
from whetstone.optimization.schema import EvaluationBinding

_DEFAULT_ROW_JOB_FACTORY = process_row_job_factory(
    "tests.envs.process_workers:drive_internal_success"
)


def _experiment(*, repeats: int = 1):
    return build_env_experiment(
        "c18",
        model="openai/test",
        pool_n_per_stratum=2,
        split_sizes=(1, 1, 1),
        repeats=repeats,
    )


def _engine(
    tmp_path,
    *,
    store: ObjectStore,
    row_job_factory: InternalRowJobFactory = _DEFAULT_ROW_JOB_FACTORY,
    repeats: int = 1,
    partial: bool = False,
    cache: bool = False,
) -> EvaluationEngine:
    experiment = _experiment(repeats=repeats)
    return EvaluationEngine(
        store=store,
        experiment=experiment,
        sampling=experiment.eval_configs.internal,
        execution_policy=execution_policy(),
        row_job_factory=row_job_factory,
        partial_log=PartialLog(tmp_path / "partials.jsonl")
        if partial
        else None,
        prompt_cache=PromptResultCache(tmp_path / "cache") if cache else None,
    )


def _binding(
    engine: EvaluationEngine,
    *,
    role: EvaluationRole = EvaluationRole.INTERNAL,
    campaign: str = "evaluation-test",
) -> EvaluationBinding:
    return EvaluationBinding(
        eval_config=engine.eval_config_ref,
        role=role,
        authority_principal=(
            "test-authority" if role is EvaluationRole.OFFICIAL else None
        ),
        campaign=campaign,
    )


def _intent(
    engine: EvaluationEngine,
    *,
    intent_id: str,
    purpose: str,
    candidate: Candidate | None = None,
) -> EvaluationIntent:
    return EvaluationIntent(
        intent_id=intent_id,
        candidate=candidate_reference(
            candidate or engine.experiment.initial_candidate
        ),
        target_eval_config=engine.eval_config_ref,
        evaluation_binding=_binding(engine, campaign=intent_id),
        purpose=purpose,
        run_id="run",
        step_index=0,
        expected_reward_policy_hash=(
            engine.experiment.reward_policy.identity_hash()
        ),
    )


def _blocking_evaluate(
    *,
    result: EngineEvaluation,
    entered: Event,
    release: Event,
    calls: list[EvaluationRequest],
    timeout: float,
) -> Callable[[EvaluationRequest], EngineEvaluation]:
    def blocked(request: EvaluationRequest) -> EngineEvaluation:
        calls.append(request)
        entered.set()
        assert release.wait(timeout=timeout)
        return result

    return blocked


def _fail_unexpected_evaluate(
    _request: EvaluationRequest,
) -> EngineEvaluation:
    raise AssertionError("waiting resolver must not evaluate")


def test_engine_persists_exact_evidence_and_reward(tmp_path) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "engine.sqlite"))
    engine = _engine(tmp_path, store=store)

    result = engine.evaluate(
        EvaluationRequest(
            candidate=engine.experiment.initial_candidate,
            evaluation_binding=_binding(engine),
            purpose="test",
        )
    )

    evidence = result.evidence
    assert store.get(result.evidence_ref.reference) == (
        evidence.record_content()
    )
    assert store.get(evidence.candidate.record_ref.reference)
    assert store.get(
        evidence.evaluation_binding.eval_config.record_ref.reference
    )
    output_record = EvaluationOutputsRecord.model_validate(
        store.get(evidence.outputs_ref.reference)
    )
    assert output_record.record_content() == store.get(
        evidence.outputs_ref.reference
    )
    assert output_record.candidate_id == (
        engine.experiment.initial_candidate.candidate_id
    )
    assert tuple(row.task_identity for row in output_record.outputs) == (
        engine.sampling.task_set.task_identities
    )
    assert store.get(evidence.aggregate_ref.reference)
    assert evidence.reward_ref is not None
    reward = Reward.model_validate(
        store.get(evidence.reward_ref.record_ref.reference)
    )
    assert reward == evidence.reward_ref.record
    assert reward.evidence_refs == (evidence.aggregate_ref,)
    assert evidence.row_accounting.planned == 1
    assert evidence.row_accounting.present == 1
    assert evidence.per_task_counts == (1,)
    assert evidence.evaluation_binding.eval_config == engine.eval_config_ref
    assert evidence.dataset_identity == (
        engine.sampling.task_set.dataset_revision
    )


def test_engine_passes_exact_canonical_row_job_factory(
    tmp_path, monkeypatch
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "factory.sqlite"))
    delegated = process_row_job_factory(
        "tests.envs.process_workers:drive_internal_success"
    )
    submitted: list[InternalRowRequest] = []

    def row_job_factory(request: InternalRowRequest) -> ProcessJob:
        submitted.append(request)
        return delegated(request)

    engine = _engine(
        tmp_path,
        store=store,
        row_job_factory=row_job_factory,
    )
    canonical_run = evaluation_engine_module.run_internal_eval

    def checked_run(*args, **kwargs):
        assert kwargs["row_job_factory"] is row_job_factory
        assert "transport" not in kwargs
        return canonical_run(*args, **kwargs)

    monkeypatch.setattr(
        evaluation_engine_module,
        "run_internal_eval",
        checked_run,
    )

    engine.evaluate(
        EvaluationRequest(
            candidate=engine.experiment.initial_candidate,
            evaluation_binding=_binding(engine),
            purpose="factory-contract",
        )
    )

    assert len(submitted) == 1


def test_engine_rejects_mismatched_process_result_identity(tmp_path) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "identity-mismatch.sqlite"))

    def mismatched(request: InternalRowRequest) -> ProcessJob:
        result = InternalRowResult(
            request_identity=f"mismatched-{request.request_identity}",
            outcome=InternalRowOutcome(score=1.0),
        )
        return ProcessJob(
            entrypoint="tests.envs.process_workers:return_payload",
            payload=result.model_dump(mode="json"),
        )

    engine = _engine(
        tmp_path,
        store=store,
        row_job_factory=mismatched,
    )

    with pytest.raises(
        ValueError,
        match="internal row result does not match its submitted request",
    ):
        engine.evaluate(
            EvaluationRequest(
                candidate=engine.experiment.initial_candidate,
                evaluation_binding=_binding(engine),
                purpose="identity-mismatch",
            )
        )


@pytest.mark.parametrize(
    ("field", "coercible_value"),
    (
        ("concurrency_halved", "false"),
        ("deadline_reached", 0),
    ),
)
def test_evaluation_evidence_rejects_coercible_booleans(
    tmp_path,
    field: str,
    coercible_value: object,
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / f"{field}.sqlite"))
    engine = _engine(tmp_path, store=store)
    evidence = engine.evaluate(
        EvaluationRequest(
            candidate=engine.experiment.initial_candidate,
            evaluation_binding=_binding(engine),
            purpose="test",
        )
    ).evidence
    record = evidence.record_content()
    record[field] = coercible_value

    with pytest.raises(ValueError, match="valid boolean"):
        EvaluationEvidence.model_validate(record)


def test_evaluation_outputs_wire_contract_is_exact() -> None:
    record = EvaluationOutputsRecord(
        candidate_id="candidate-1",
        outputs=(
            EvaluationOutputRow(
                candidate_id="candidate-1",
                instance_id="instance-1",
                task_identity="task-1",
                repeat=0,
                rendered_prompt="Question?",
                output_text="Answer.",
                score=1.0,
                failure_code="",
                component_trace_steps=(
                    EvaluationOutputComponentTraceStep(
                        component_id="predictor",
                        inputs={"question": "Question?"},
                        outputs={"answer": "Answer."},
                    ),
                ),
                finish_reason="stop",
                provider_error=None,
                max_budget=100,
                over_budget=False,
            ),
        ),
    )

    assert record.record_content() == {
        "candidate_id": "candidate-1",
        "outputs": [
            {
                "candidate_id": "candidate-1",
                "instance_id": "instance-1",
                "task_identity": "task-1",
                "repeat": 0,
                "rendered_prompt": "Question?",
                "output_text": "Answer.",
                "score": 1.0,
                "failure_code": "",
                "component_trace_steps": [
                    {
                        "component_id": "predictor",
                        "inputs": {"question": "Question?"},
                        "outputs": {"answer": "Answer."},
                    }
                ],
                "finish_reason": "stop",
                "provider_error": None,
                "max_budget": 100,
                "over_budget": False,
            }
        ],
    }


def test_evaluation_outputs_reject_candidate_mismatch() -> None:
    row = EvaluationOutputRow(
        candidate_id="other",
        instance_id="instance-1",
        task_identity="task-1",
        repeat=0,
        rendered_prompt="Question?",
        output_text="Answer.",
        score=1.0,
        failure_code="",
        component_trace_steps=(),
        finish_reason="stop",
        provider_error=None,
        max_budget=None,
        over_budget=None,
    )

    with pytest.raises(ValueError, match="candidate_id must match"):
        EvaluationOutputsRecord(candidate_id="candidate-1", outputs=(row,))


@pytest.mark.parametrize(
    ("update", "message"),
    (
        ({"repeat": True}, "valid integer"),
        ({"score": float("nan")}, "finite number"),
        ({"unexpected": "drift"}, "Extra inputs are not permitted"),
    ),
)
def test_evaluation_output_row_rejects_wire_schema_drift(
    update, message
) -> None:
    payload = {
        "candidate_id": "candidate-1",
        "instance_id": "instance-1",
        "task_identity": "task-1",
        "repeat": 0,
        "rendered_prompt": "Question?",
        "output_text": "Answer.",
        "score": 1.0,
        "failure_code": "",
        "component_trace_steps": [],
        "finish_reason": "stop",
        "provider_error": None,
        "max_budget": None,
        "over_budget": None,
        **update,
    }

    with pytest.raises(ValueError, match=message):
        EvaluationOutputRow.model_validate(payload)


def test_engine_rejects_output_outside_sampling_plan(
    tmp_path, monkeypatch
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "output-drift.sqlite"))
    engine = _engine(tmp_path, store=store)
    canonical_run = evaluation_engine_module.run_internal_eval

    def drifted_run(*args, **kwargs):
        result = canonical_run(*args, **kwargs)
        assert len(result.outputs) == 1
        return replace(
            result,
            outputs=(
                replace(result.outputs[0], instance_id="unknown-instance"),
            ),
        )

    monkeypatch.setattr(
        evaluation_engine_module,
        "run_internal_eval",
        drifted_run,
    )

    with pytest.raises(ValueError, match="outside the exact sampling plan"):
        engine.evaluate(
            EvaluationRequest(
                candidate=engine.experiment.initial_candidate,
                evaluation_binding=_binding(engine),
                purpose="test",
            )
        )


def test_engine_rejects_output_order_drift(tmp_path, monkeypatch) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "output-order.sqlite"))
    engine = _engine(
        tmp_path,
        store=store,
        repeats=2,
    )
    canonical_run = evaluation_engine_module.run_internal_eval

    def reversed_run(*args, **kwargs):
        result = canonical_run(*args, **kwargs)
        return replace(result, outputs=tuple(reversed(result.outputs)))

    monkeypatch.setattr(
        evaluation_engine_module,
        "run_internal_eval",
        reversed_run,
    )

    with pytest.raises(ValueError, match="must follow sampling"):
        engine.evaluate(
            EvaluationRequest(
                candidate=engine.experiment.initial_candidate,
                evaluation_binding=_binding(engine),
                purpose="test",
            )
        )


def test_invalid_intent_rejects_without_provider_spend(tmp_path) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "reject.sqlite"))
    submitted: list[InternalRowRequest] = []

    def record_submission(request: InternalRowRequest) -> ProcessJob:
        submitted.append(request)
        raise AssertionError("invalid candidate must not create a process job")

    engine = _engine(
        tmp_path,
        store=store,
        row_job_factory=record_submission,
    )
    invalid = Candidate(
        candidate_id="invalid",
        base_ref=engine.experiment.initial_candidate.base_ref,
        payload={"user_prompt_template": "Use {private_gold}."},
    )
    intent = _intent(
        engine,
        intent_id="invalid-intent",
        candidate=invalid,
        purpose="preflight",
    )

    resolution = EngineEvaluationService(
        store=store, engine=engine
    ).resolve_evaluation_intent(intent)

    assert resolution.outcome is IntentOutcome.REJECTED
    assert resolution.evaluation_evidence_refs == ()
    assert submitted == []


def test_resolution_and_prompt_results_replay_after_restart(tmp_path) -> None:
    database = tmp_path / "restart.sqlite"
    store = ObjectStore(SqliteBackend(database))
    delegated = process_row_job_factory(
        "tests.envs.process_workers:drive_internal_success"
    )
    submitted: list[InternalRowRequest] = []

    def record_submission(request: InternalRowRequest) -> ProcessJob:
        submitted.append(request)
        return delegated(request)

    engine = _engine(
        tmp_path,
        store=store,
        row_job_factory=record_submission,
        partial=True,
        cache=True,
    )
    candidate = engine.experiment.initial_candidate
    intent = _intent(
        engine,
        intent_id="restart-intent",
        candidate=candidate,
        purpose="restart",
    )
    first = EngineEvaluationService(
        store=store, engine=engine
    ).resolve_evaluation_intent(intent)
    assert len(submitted) == 1
    assert first.reward_ref is not None
    assert (
        first.evaluation_evidence_refs == first.reward_ref.record.evidence_refs
    )

    fresh_store = ObjectStore(SqliteBackend(database))

    def reject_submission(_request: InternalRowRequest) -> ProcessJob:
        raise AssertionError("durable resolution must replay")

    fresh_engine = _engine(
        tmp_path,
        store=fresh_store,
        row_job_factory=reject_submission,
        partial=True,
        cache=True,
    )
    replay = EngineEvaluationService(
        store=fresh_store, engine=fresh_engine
    ).resolve_evaluation_intent(intent)

    assert replay == first
    assert len(submitted) == 1


def test_two_resolvers_share_one_durable_evaluation(
    tmp_path, monkeypatch
) -> None:
    database = tmp_path / "concurrent.sqlite"
    evaluation_entered = Event()
    waiter_entered = Event()
    release = Event()
    evaluation_calls: list[EvaluationRequest] = []
    first_store = ObjectStore(SqliteBackend(database))
    second_store = ObjectStore(SqliteBackend(database))
    first_engine = _engine(tmp_path, store=first_store)
    second_engine = _engine(tmp_path, store=second_store)
    intent = _intent(
        first_engine,
        intent_id="concurrent-intent",
        purpose="concurrent",
    )
    evaluated = first_engine.evaluate(
        EvaluationRequest(
            candidate=intent.candidate.record,
            evaluation_binding=intent.evaluation_binding,
            purpose=intent.purpose,
        )
    )
    monkeypatch.setattr(
        first_engine,
        "evaluate",
        _blocking_evaluate(
            result=evaluated,
            entered=evaluation_entered,
            release=release,
            calls=evaluation_calls,
            timeout=2,
        ),
    )
    monkeypatch.setattr(
        second_engine,
        "evaluate",
        _fail_unexpected_evaluate,
    )

    def wait_for_winner(_seconds: float) -> None:
        waiter_entered.set()
        assert release.wait(timeout=2)

    first_service = EngineEvaluationService(
        store=first_store, engine=first_engine
    )
    second_service = EngineEvaluationService(
        store=second_store,
        engine=second_engine,
        sleep=wait_for_winner,
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(first_service.resolve_evaluation_intent, intent)
        assert evaluation_entered.wait(timeout=2)
        second = pool.submit(second_service.resolve_evaluation_intent, intent)
        assert waiter_entered.wait(timeout=2)
        assert len(evaluation_calls) == 1
        release.set()
        assert second.result(timeout=10) == first.result(timeout=10)

    assert len(evaluation_calls) == 1


def test_slow_evaluation_renews_claim_on_scripted_tick(
    tmp_path, monkeypatch
) -> None:
    now = [100.0]
    evaluation_entered = Event()
    waiter_entered = Event()
    renewal_wait_entered = Event()
    release_renewal = Event()
    initial_renewal_published = Event()
    scripted_renewal_published = Event()
    release = Event()
    requested_intervals: list[float] = []
    published_claims: list[EvaluationIntentClaim] = []
    evaluation_calls: list[EvaluationRequest] = []

    def wait_for_winner(_seconds: float) -> None:
        waiter_entered.set()
        assert release.wait(timeout=2)

    def scripted_renewal_wait(interval: float, stop: Event) -> bool:
        if not requested_intervals:
            requested_intervals.append(interval)
            renewal_wait_entered.set()
            assert release_renewal.wait(timeout=2)
            return stop.is_set()
        assert stop.wait(timeout=10)
        return True

    def record_renewal(claim: EvaluationIntentClaim) -> None:
        published_claims.append(claim)
        if len(published_claims) == 1:
            initial_renewal_published.set()
        else:
            scripted_renewal_published.set()

    backend = MemoryBackend()
    first_store = ObjectStore(backend)
    second_store = ObjectStore(backend)
    first_engine = _engine(tmp_path, store=first_store)
    second_engine = _engine(tmp_path, store=second_store)
    intent = _intent(
        first_engine,
        intent_id="slow-live-intent",
        purpose="heartbeat",
    )
    evaluated = first_engine.evaluate(
        EvaluationRequest(
            candidate=intent.candidate.record,
            evaluation_binding=intent.evaluation_binding,
            purpose=intent.purpose,
        )
    )
    monkeypatch.setattr(
        first_engine,
        "evaluate",
        _blocking_evaluate(
            result=evaluated,
            entered=evaluation_entered,
            release=release,
            calls=evaluation_calls,
            timeout=2,
        ),
    )
    monkeypatch.setattr(
        second_engine,
        "evaluate",
        _fail_unexpected_evaluate,
    )
    first_service = EngineEvaluationService(
        store=first_store,
        engine=first_engine,
        claim_lease_seconds=3.0,
        clock=lambda: now[0],
        _renewal_wait=scripted_renewal_wait,
        _renewal_published=record_renewal,
    )
    second_service = EngineEvaluationService(
        store=second_store,
        engine=second_engine,
        claim_lease_seconds=3.0,
        clock=lambda: now[0],
        sleep=wait_for_winner,
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(first_service.resolve_evaluation_intent, intent)
        try:
            assert initial_renewal_published.wait(timeout=2)
            assert evaluation_entered.wait(timeout=2)
            assert renewal_wait_entered.wait(timeout=2)
            assert requested_intervals == [1.0]
            initial = published_claims[0]

            now[0] = 102.0
            release_renewal.set()
            assert scripted_renewal_published.wait(timeout=2)
            renewed = published_claims[1]
            assert renewed.event_ordinal == initial.event_ordinal + 1
            assert renewed.heartbeat_ordinal == (initial.heartbeat_ordinal + 1)
            assert renewed.expires_at > initial.expires_at

            now[0] = initial.expires_at + 0.5
            second = pool.submit(
                second_service.resolve_evaluation_intent, intent
            )
            assert waiter_entered.wait(timeout=2)
            assert first_service._latest_claim(intent) == renewed
            assert len(evaluation_calls) == 1
        finally:
            release_renewal.set()
            release.set()
        assert second.result(timeout=10) == first.result(timeout=10)

    assert len(evaluation_calls) == 1


@pytest.mark.sqlite_time_integration
def test_real_sqlite_heartbeat_renews_past_original_expiry(
    tmp_path, monkeypatch
) -> None:
    database = tmp_path / "heartbeat.sqlite"
    evaluation_entered = Event()
    waiter_entered = Event()
    initial_renewal_published = Event()
    renewed_past_original_expiry = Event()
    release = Event()
    published_claims: list[EvaluationIntentClaim] = []
    evaluation_calls: list[EvaluationRequest] = []

    def wait_for_winner(_seconds: float) -> None:
        waiter_entered.set()
        assert release.wait(timeout=10)

    def record_renewal(claim: EvaluationIntentClaim) -> None:
        published_claims.append(claim)
        if len(published_claims) == 1:
            initial_renewal_published.set()
        elif time.time() > published_claims[0].expires_at:
            renewed_past_original_expiry.set()

    first_store = ObjectStore(SqliteBackend(database))
    second_store = ObjectStore(SqliteBackend(database))
    first_engine = _engine(tmp_path, store=first_store)
    second_engine = _engine(tmp_path, store=second_store)
    intent = _intent(
        first_engine,
        intent_id="slow-live-intent",
        purpose="heartbeat",
    )
    evaluated = first_engine.evaluate(
        EvaluationRequest(
            candidate=intent.candidate.record,
            evaluation_binding=intent.evaluation_binding,
            purpose=intent.purpose,
        )
    )
    monkeypatch.setattr(
        first_engine,
        "evaluate",
        _blocking_evaluate(
            result=evaluated,
            entered=evaluation_entered,
            release=release,
            calls=evaluation_calls,
            timeout=10,
        ),
    )
    monkeypatch.setattr(
        second_engine,
        "evaluate",
        _fail_unexpected_evaluate,
    )
    first_service = EngineEvaluationService(
        store=first_store,
        engine=first_engine,
        claim_lease_seconds=0.3,
        _renewal_published=record_renewal,
    )
    second_service = EngineEvaluationService(
        store=second_store,
        engine=second_engine,
        claim_lease_seconds=0.3,
        sleep=wait_for_winner,
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(first_service.resolve_evaluation_intent, intent)
        try:
            assert initial_renewal_published.wait(timeout=10)
            assert evaluation_entered.wait(timeout=10)
            assert renewed_past_original_expiry.wait(timeout=10)
            initial = published_claims[0]
            renewed = published_claims[-1]
            assert renewed.event_ordinal > initial.event_ordinal
            assert renewed.expires_at > initial.expires_at

            second = pool.submit(
                second_service.resolve_evaluation_intent, intent
            )
            assert waiter_entered.wait(timeout=10)
            assert len(evaluation_calls) == 1
        finally:
            release.set()
        assert second.result(timeout=10) == first.result(timeout=10)

    assert len(evaluation_calls) == 1


def test_renewal_wins_same_event_slot_as_stale_takeover(
    tmp_path, monkeypatch
) -> None:
    database = tmp_path / "claim-renewal-race.sqlite"
    now = [100.0]
    renewal_paused = Event()
    stale_takeover_ready = Event()
    renewal_bound = Event()
    evaluation_entered = Event()
    waiter_entered = Event()
    release = Event()
    evaluation_calls: list[EvaluationRequest] = []

    def wait_for_winner(_seconds: float) -> None:
        waiter_entered.set()
        assert release.wait(timeout=2)

    first_store = ObjectStore(SqliteBackend(database))
    second_store = ObjectStore(SqliteBackend(database))
    first_engine = _engine(tmp_path, store=first_store)
    second_engine = _engine(tmp_path, store=second_store)
    intent = _intent(
        first_engine,
        intent_id="renewal-race-intent",
        purpose="renewal-race",
    )
    evaluated = first_engine.evaluate(
        EvaluationRequest(
            candidate=intent.candidate.record,
            evaluation_binding=intent.evaluation_binding,
            purpose=intent.purpose,
        )
    )
    monkeypatch.setattr(
        first_engine,
        "evaluate",
        _blocking_evaluate(
            result=evaluated,
            entered=evaluation_entered,
            release=release,
            calls=evaluation_calls,
            timeout=2,
        ),
    )
    monkeypatch.setattr(
        second_engine,
        "evaluate",
        _fail_unexpected_evaluate,
    )
    first = EngineEvaluationService(
        store=first_store,
        engine=first_engine,
        claim_lease_seconds=1.0,
        clock=lambda: now[0],
    )
    second = EngineEvaluationService(
        store=second_store,
        engine=second_engine,
        claim_lease_seconds=1.0,
        clock=lambda: now[0],
        sleep=wait_for_winner,
    )
    renew_claim = first._renew_claim
    append_claim_event = second._append_claim_event

    def paused_renewal(intent, owned) -> None:
        renewal_paused.set()
        assert stale_takeover_ready.wait(timeout=2)
        renew_claim(intent, owned)
        renewal_bound.set()

    def delayed_takeover(**kwargs):
        prior = kwargs["prior"]
        if prior is not None and kwargs["generation"] == 1:
            stale_takeover_ready.set()
            assert renewal_bound.wait(timeout=2)
        return append_claim_event(**kwargs)

    monkeypatch.setattr(first, "_renew_claim", paused_renewal)
    monkeypatch.setattr(second, "_append_claim_event", delayed_takeover)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_result = pool.submit(first.resolve_evaluation_intent, intent)
        assert renewal_paused.wait(timeout=2)
        now[0] = 102.0
        second_result = pool.submit(second.resolve_evaluation_intent, intent)
        assert stale_takeover_ready.wait(timeout=2)
        assert renewal_bound.wait(timeout=2)
        assert evaluation_entered.wait(timeout=2)
        assert waiter_entered.wait(timeout=2)
        assert len(evaluation_calls) == 1
        release.set()
        assert second_result.result(timeout=10) == first_result.result(
            timeout=10
        )

    assert len(evaluation_calls) == 1


def test_expired_claim_retries_after_resolver_crash(
    tmp_path, monkeypatch
) -> None:
    database = tmp_path / "claim-retry.sqlite"
    now = [100.0]
    delegated = process_row_job_factory(
        "tests.envs.process_workers:drive_internal_success"
    )
    submitted: list[InternalRowRequest] = []
    evaluation_attempts: list[EvaluationRequest] = []

    def record_submission(request: InternalRowRequest) -> ProcessJob:
        submitted.append(request)
        return delegated(request)

    first_store = ObjectStore(SqliteBackend(database))
    first_engine = _engine(
        tmp_path,
        store=first_store,
        row_job_factory=record_submission,
    )

    def crash_once(request: EvaluationRequest) -> EngineEvaluation:
        evaluation_attempts.append(request)
        raise KeyboardInterrupt("simulated resolver crash")

    monkeypatch.setattr(first_engine, "evaluate", crash_once)
    intent = _intent(
        first_engine,
        intent_id="crashed-intent",
        purpose="crash-retry",
    )
    crashed = EngineEvaluationService(
        store=first_store,
        engine=first_engine,
        claim_lease_seconds=1.0,
        clock=lambda: now[0],
        sleep=lambda _seconds: None,
    )

    with pytest.raises(KeyboardInterrupt, match="simulated resolver crash"):
        crashed.resolve_evaluation_intent(intent)

    now[0] = 102.0
    retry_store = ObjectStore(SqliteBackend(database))
    retry_engine = _engine(
        tmp_path,
        store=retry_store,
        row_job_factory=record_submission,
    )
    completed = EngineEvaluationService(
        store=retry_store,
        engine=retry_engine,
        claim_lease_seconds=1.0,
        clock=lambda: now[0],
        sleep=lambda _seconds: None,
    ).resolve_evaluation_intent(intent)

    assert completed.outcome is IntentOutcome.COMPLETED
    assert len(evaluation_attempts) == 1
    assert len(submitted) == 1


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


def test_cache_provenance_avoids_transport_replay(tmp_path) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "cache.sqlite"))
    delegated = process_row_job_factory(
        "tests.envs.process_workers:drive_internal_success"
    )
    submitted: list[InternalRowRequest] = []

    def record_submission(request: InternalRowRequest) -> ProcessJob:
        submitted.append(request)
        return delegated(request)

    engine = _engine(
        tmp_path,
        store=store,
        row_job_factory=record_submission,
        partial=True,
        cache=True,
    )
    base = engine.experiment.initial_candidate
    engine.evaluate(
        EvaluationRequest(
            candidate=base,
            evaluation_binding=_binding(engine, campaign="first"),
            purpose="cache",
        )
    )
    same_prompt = base.model_copy(update={"candidate_id": "same-prompt"})
    result = engine.evaluate(
        EvaluationRequest(
            candidate=same_prompt,
            evaluation_binding=_binding(engine, campaign="second"),
            purpose="cache",
        )
    )

    assert len(submitted) == 2
    assert result.evidence.cache.cache_hit_count == 1
    assert result.evidence.cache.source_call_ids


def test_sampling_repeat_change_changes_exact_eval_identity(tmp_path) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "identity.sqlite"))
    one = _engine(tmp_path, store=store, repeats=1)
    two = _engine(tmp_path, store=store, repeats=2)

    assert (
        one.eval_config_ref.identity_hash != two.eval_config_ref.identity_hash
    )


def test_tool_projection_uses_same_engine_evidence(tmp_path) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "tool.sqlite"))
    engine = _engine(tmp_path, store=store)
    definition = ToolDefinition(
        tool_name="evaluate_candidate",
        input_fields=("base_ref", "model_route", "template"),
        output_fields=("evaluation_evidence_ref", "output_artifact_ref"),
    )
    config = ToolConfig(
        definition=tool_definition_reference(definition),
        endpoint_key="evaluate_candidate",
        eval_config=engine.sampling.eval_config,
        reward_policy_hash=engine.experiment.reward_policy.identity_hash(),
        capacity=ToolCapacity(
            max_accepted_calls=1,
            scope=ToolCapacityScope.GLOBAL,
        ),
        store_namespace_key="tool-projection",
    )
    base = engine.experiment.initial_candidate
    call = ToolCall(
        call_id="tool-call",
        tool_config=tool_config_reference(config),
        capacity_binding=tool_capacity_binding(ToolCapacityScope.GLOBAL),
        args={
            "base_ref": base.base_ref.model_dump(mode="json"),
            "model_route": "openai/test",
            "template": base.payload["user_prompt_template"],
        },
    )

    projected = EngineToolEvaluator(engine).evaluate(call, config)

    assert projected.eval_config_hash == engine.eval_config_ref.identity_hash
    assert len(projected.rollout_refs) == 1
    assert projected.output["evaluation_evidence_ref"] == (
        projected.rollout_refs[0].model_dump(mode="json")
    )
    artifact = TypedRef.model_validate(projected.output["output_artifact_ref"])
    assert store.get(artifact.reference)
