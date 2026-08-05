from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from dr_store import (
    MemoryBackend,
    ObjectNotFoundError,
    ObjectStore,
    SqliteBackend,
)

from tests.envs.support import (
    process_row_job_factory,
)
from tests.evaluation.support import (
    _binding,
    _completed_resolution,
    _engine,
    _intent,
    _load_component_traces,
    _put_typed,
)
from whetstone.coordination.evaluation_claims import (
    EvaluationIntentClaim,
)
from whetstone.coordination.evaluation_service import EngineEvaluationService
from whetstone.core.identity import (
    TerminalFailure,
    TypedRef,
)
from whetstone.evaluation.drivers.internal import (
    InternalRowRequest,
)
from whetstone.evaluation.engine import (
    EngineEvaluation,
    EvaluationRequest,
)
from whetstone.evaluation.schema import (
    EVALUATION_COMPONENT_TRACES_SCHEMA,
    EVALUATION_OUTPUTS_SCHEMA,
    EvaluationFailureEvidence,
)
from whetstone.evaluation.schema_names import (
    EVALUATION_EVIDENCE_SCHEMA,
    EVALUATION_FAILURE_SCHEMA,
)
from whetstone.execution.fanout import ProcessJob
from whetstone.experiment.candidate import (
    candidate_reference,
)
from whetstone.optimization.contracts import (
    INTENT_RESOLUTION_SCHEMA_VERSION,
    EvaluationIntent,
    IntentOutcome,
    IntentResolution,
    ResolutionClass,
    ResolutionDetail,
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


@pytest.mark.parametrize(
    "forgery",
    (
        "missing_object",
        "candidate",
        "binding",
        "graph",
        "task",
        "row_reorder",
        "row_state",
        "labeled_input",
        "step",
        "output",
        "model_copy_dump",
    ),
)
def test_claim_attestation_rejects_forged_component_traces(
    tmp_path,
    forgery: str,
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / f"trace-{forgery}.sqlite"))
    engine = _engine(tmp_path, store=store, repeats=2)
    intent = _intent(
        engine,
        intent_id=f"trace-{forgery}",
        purpose="trace-validation",
    )
    evaluated = engine.evaluate(
        EvaluationRequest(
            candidate=intent.candidate.record,
            evaluation_binding=intent.evaluation_binding,
            purpose=intent.purpose,
        )
    )
    traces = _load_component_traces(store, evaluated.evidence)
    trace_content = traces.record_content()
    if forgery == "candidate":
        trace_content["candidate"] = candidate_reference(
            engine.experiment.ceiling_candidate
        ).model_dump(mode="json")
    elif forgery == "binding":
        trace_content["evaluation_binding"] = _binding(
            engine,
            campaign="forged-trace-binding",
        ).model_dump(mode="json")
    elif forgery == "graph":
        trace_content["graph_hash"] = "f" * 64
    elif forgery == "task":
        trace_content["task_identities"] = ["forged-task"]
        for row in trace_content["rows"]:
            row["task_identity"] = "forged-task"
    elif forgery == "row_reorder":
        trace_content["rows"] = list(reversed(trace_content["rows"]))
    elif forgery == "row_state":
        trace_content["rows"][0]["executed_component_trace"]["row_state"] = (
            "failed"
        )
    elif forgery == "labeled_input":
        trace_content["rows"][0]["executed_component_trace"][
            "executed_component_steps"
        ][0]["inputs"]["prompt"] = "forged prompt"
    elif forgery == "step":
        trace_content["rows"][0]["executed_component_trace"][
            "executed_component_steps"
        ][0]["component_id"] = "forged-step"
    elif forgery == "output":
        trace_content["rows"][0]["executed_component_trace"][
            "executed_component_steps"
        ][0]["outputs"]["generation"] = "forged output"
    elif forgery == "model_copy_dump":
        original_row = traces.rows[0]
        original_trace = original_row.executed_component_trace
        original_step = original_trace.executed_component_steps[0]
        bypassed_step = original_step.model_copy(
            update={"component_id": "copy-bypassed-step"}
        )
        bypassed_trace = original_trace.model_copy(
            update={"executed_component_steps": (bypassed_step,)}
        )
        bypassed_row = original_row.model_copy(
            update={"executed_component_trace": bypassed_trace}
        )
        traces = traces.model_copy(
            update={"rows": (bypassed_row, *traces.rows[1:])}
        )
        trace_content = traces.model_dump(mode="json")
    elif forgery != "missing_object":
        raise AssertionError(f"unhandled forgery {forgery}")

    component_traces_ref = (
        TypedRef(
            schema_name=EVALUATION_COMPONENT_TRACES_SCHEMA,
            content_hash="f" * 64,
        )
        if forgery == "missing_object"
        else _put_typed(
            store,
            EVALUATION_COMPONENT_TRACES_SCHEMA,
            trace_content,
        )
    )
    outputs_content = store.get(evaluated.evidence.outputs_ref.reference)
    assert isinstance(outputs_content, dict)
    outputs_content["component_traces_ref"] = component_traces_ref.model_dump(
        mode="json"
    )
    outputs_ref = _put_typed(
        store,
        EVALUATION_OUTPUTS_SCHEMA,
        outputs_content,
    )
    forged_evidence = evaluated.evidence.model_copy(
        update={
            "component_traces_ref": component_traces_ref,
            "outputs_ref": outputs_ref,
        }
    )
    forged_evidence_ref = _put_typed(
        store,
        EVALUATION_EVIDENCE_SCHEMA,
        forged_evidence.record_content(),
    )
    forged_resolution = _completed_resolution(intent, evaluated).model_copy(
        update={"evaluation_result_ref": forged_evidence_ref}
    )
    service = EngineEvaluationService(store=store, engine=engine)
    service._persist_intent_targets(intent)

    with pytest.raises((ObjectNotFoundError, ValueError)):
        service._validate_result_graph(
            forged_resolution,
            expected_intent=intent,
            require_attestation=False,
        )


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
        first_resolution = first.result(timeout=10)
        assert second.result(timeout=10) == first_resolution

    assert len(evaluation_calls) == 1


def _failing_renewal_wait(_interval: float, _stop: Event) -> bool:
    raise sqlite3.OperationalError("transient store blip")


def test_heartbeat_error_keeps_a_durably_bound_resolution(tmp_path) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "heartbeat-bound.sqlite"))
    engine = _engine(tmp_path, store=store)
    intent = _intent(
        engine,
        intent_id="heartbeat-bound",
        purpose="heartbeat",
    )
    service = EngineEvaluationService(
        store=store,
        engine=engine,
        _renewal_wait=_failing_renewal_wait,
    )

    resolution = service.resolve_evaluation_intent(intent)

    assert resolution.outcome is IntentOutcome.COMPLETED
    bound = store.resolve(service._key(intent))
    assert bound is not None
    assert service._load(bound, expected_intent=intent) == resolution


def test_heartbeat_error_raises_when_nothing_was_bound(
    tmp_path, monkeypatch
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "heartbeat-unbound.sqlite"))
    engine = _engine(tmp_path, store=store)
    intent = _intent(
        engine,
        intent_id="heartbeat-unbound",
        purpose="heartbeat",
    )
    service = EngineEvaluationService(
        store=store,
        engine=engine,
        _renewal_wait=_failing_renewal_wait,
    )

    def bind_without_durability(
        _intent: EvaluationIntent, resolution: IntentResolution
    ) -> IntentResolution:
        return resolution

    monkeypatch.setattr(service, "_bind", bind_without_durability)

    with pytest.raises(RuntimeError, match="lease heartbeat failed"):
        service.resolve_evaluation_intent(intent)

    assert store.resolve(service._key(intent)) is None


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
    resolution_bound = Event()
    release = Event()
    requested_intervals: list[float] = []
    published_claims: list[EvaluationIntentClaim] = []
    evaluation_calls: list[EvaluationRequest] = []

    def wait_for_winner(_seconds: float) -> None:
        waiter_entered.set()
        assert resolution_bound.wait(timeout=10)

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
    bind_resolution = first_service._bind

    def bind_and_publish(intent, resolution):
        bound = bind_resolution(intent, resolution)
        resolution_bound.set()
        return bound

    monkeypatch.setattr(first_service, "_bind", bind_and_publish)

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
        first_resolution = first.result(timeout=10)
        assert second.result(timeout=10) == first_resolution

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
        first_resolution = first.result(timeout=10)
        assert second.result(timeout=10) == first_resolution

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
        first_resolution = first_result.result(timeout=10)
        assert second_result.result(timeout=10) == first_resolution

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


@pytest.mark.parametrize(
    "outcome", (IntentOutcome.COMPLETED, IntentOutcome.FAILED)
)
def test_stale_owner_cannot_attest_after_takeover(
    tmp_path,
    outcome: IntentOutcome,
) -> None:
    now = [100.0]
    backend = MemoryBackend()
    first_store = ObjectStore(backend)
    second_store = ObjectStore(backend)
    first_engine = _engine(tmp_path, store=first_store)
    second_engine = _engine(tmp_path, store=second_store)
    intent = _intent(
        first_engine,
        intent_id=f"stale-attestation-{outcome.value}",
        purpose="claim-fence",
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
    first._persist_intent_targets(intent)
    if outcome is IntentOutcome.COMPLETED:
        evaluated = first_engine.evaluate(
            EvaluationRequest(
                candidate=intent.candidate.record,
                evaluation_binding=intent.evaluation_binding,
                purpose=intent.purpose,
            )
        )
        resolution = _completed_resolution(intent, evaluated)
    else:
        failure = EvaluationFailureEvidence(
            candidate=intent.candidate,
            evaluation_binding=intent.evaluation_binding,
            purpose=intent.purpose,
            exception_type="RuntimeError",
            message="provider unavailable",
        )
        failure_ref = _put_typed(
            first_store,
            EVALUATION_FAILURE_SCHEMA,
            failure.record_content(),
        )
        resolution = IntentResolution(
            schema_version=INTENT_RESOLUTION_SCHEMA_VERSION,
            intent=intent,
            outcome=IntentOutcome.FAILED,
            detail=ResolutionDetail(
                classification=ResolutionClass.INFRASTRUCTURE,
                message=failure.message,
            ),
            evaluation_result_ref=failure_ref,
            resolved_eval_config=intent.target_eval_config,
            terminal_failure=TerminalFailure(
                code="evaluation_RuntimeError",
                message=failure.message,
                details={
                    "evidence_schema": failure_ref.schema_name,
                    "evidence_content_hash": failure_ref.content_hash,
                },
            ),
        )

    stale_claim = first._claim(intent)
    assert stale_claim is not None
    now[0] = 102.0
    winner_claim = second._claim(intent)
    assert winner_claim is not None
    assert winner_claim.generation == stale_claim.generation + 1

    with pytest.raises(RuntimeError, match="not owned"):
        first._publish_result_attestation(
            intent=intent,
            resolution=resolution,
            owned=stale_claim,
        )
    latest = second._latest_claim(intent)
    assert latest is not None
    assert latest.result_attestation_ref is None

    second._publish_result_attestation(
        intent=intent,
        resolution=resolution,
        owned=winner_claim,
    )
    assert second._bind(intent, resolution) == resolution


def test_fresh_resolver_reconciles_terminal_attestation_without_evaluation(
    tmp_path,
    monkeypatch,
) -> None:
    backend = MemoryBackend()
    first_store = ObjectStore(backend)
    fresh_store = ObjectStore(backend)
    first_engine = _engine(tmp_path, store=first_store)
    intent = _intent(
        first_engine,
        intent_id="attestation-reconcile",
        purpose="crash-reconcile",
    )
    evaluated = first_engine.evaluate(
        EvaluationRequest(
            candidate=intent.candidate.record,
            evaluation_binding=intent.evaluation_binding,
            purpose=intent.purpose,
        )
    )
    resolution = _completed_resolution(intent, evaluated)
    first = EngineEvaluationService(store=first_store, engine=first_engine)
    first._persist_intent_targets(intent)
    owned = first._claim(intent)
    assert owned is not None
    first._publish_result_attestation(
        intent=intent,
        resolution=resolution,
        owned=owned,
    )
    assert first_store.resolve(first._key(intent)) is None

    fresh_engine = _engine(tmp_path, store=fresh_store)
    monkeypatch.setattr(fresh_engine, "evaluate", _fail_unexpected_evaluate)
    replay = EngineEvaluationService(
        store=fresh_store,
        engine=fresh_engine,
    ).resolve_evaluation_intent(intent)

    assert replay == resolution
