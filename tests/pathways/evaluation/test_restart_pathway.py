"""Evaluation restart and durable-claim pathway tests."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import cast

import pytest
from dr_serialize import Jsonable
from dr_store import (
    ContentHashMismatchError,
    MemoryBackend,
    ObjectNotFoundError,
    ObjectStore,
    SqliteBackend,
)

from tests.coordination.evaluation_restart_support import (
    assert_restart_rejects_forgery,
    evaluate_intent_bundle,
)
from tests.envs.support import (
    execution_policy,
    process_row_job_factory,
)
from tests.evaluation.support import (
    _bind_without_validation,
    _binding,
    _completed_resolution,
    _engine,
    _intent,
    _load_component_traces,
    _publish_attestation,
    _put_typed,
    _successful_direct_outcome,
)
from whetstone.coordination.evaluation_claims import (
    EvaluationIntentClaim,
)
from whetstone.coordination.evaluation_service import EngineEvaluationService
from whetstone.core.identity import (
    TerminalFailure,
    TypedRef,
)
from whetstone.core.roles import EvaluationRole
from whetstone.envs.code_comp.reward.blended import reward_from_primary_score
from whetstone.evaluation.aggregate import (
    AGGREGATE_SCHEMA,
    RowValue,
    TaskRows,
    unweighted_task_mean,
)
from whetstone.evaluation.drivers.code_comp.direct import (
    DirectRowOutcome,
    DirectRowRequest,
    DirectRowResult,
)
from whetstone.evaluation.engine import (
    EngineEvaluation,
    EvaluationRequest,
)
from whetstone.evaluation.schema import (
    EVALUATION_COMPONENT_TRACES_SCHEMA,
    EVALUATION_OUTPUTS_SCHEMA,
    EvaluationEvidence,
    EvaluationFailureEvidence,
    EvaluationOutputsRecord,
)
from whetstone.evaluation.schema_names import (
    EVALUATION_EVIDENCE_SCHEMA,
    EVALUATION_FAILURE_SCHEMA,
)
from whetstone.evaluation.traces import ExecutedRowState
from whetstone.execution.fanout import ProcessJob
from whetstone.experiment.candidate import (
    Candidate,
    candidate_reference,
)
from whetstone.experiment.reward import (
    reward_reference,
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


def _failing_renewal_wait(_interval: float, _stop: Event) -> bool:
    raise sqlite3.OperationalError("transient store blip")


@pytest.mark.process_integration
def test_evaluator_uses_exact_v2_resolution_wire_and_v3_namespace(
    tmp_path,
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "resolution-wire.sqlite"))
    engine = _engine(tmp_path, store=store)
    intent = _intent(
        engine,
        intent_id="wire-v2",
        purpose="wire-v2",
    )
    service = EngineEvaluationService(store=store, engine=engine)

    resolution = service.resolve_evaluation_intent(intent)
    bound = store.resolve(service._key(intent))
    assert bound is not None
    record = IntentResolution.model_validate(store.get(bound)).model_dump(
        mode="json"
    )

    assert bound.schema == "whetstone.optimization_intent_resolution"
    assert service._key(intent).startswith(
        "whetstone.evaluation_service.v3.intent_resolution:"
    )
    assert service._claim_key(intent, 0).startswith(
        "whetstone.evaluation_service.v3.intent_claim:"
    )
    latest_claim = service._latest_claim(intent)
    assert latest_claim is not None
    assert latest_claim.result_attestation_ref is not None
    attestation = store.get(latest_claim.result_attestation_ref.reference)
    assert isinstance(attestation, dict)
    assert set(attestation) == {"graph_hash", "resolution"}
    assert attestation["graph_hash"] == (
        engine.experiment.generation_graph.graph_hash
    )
    assert attestation["resolution"] == record
    assert record == resolution.model_dump(mode="json")
    assert set(record) == {
        "schema_version",
        "intent",
        "outcome",
        "detail",
        "evaluation_result_ref",
        "reward_evidence_refs",
        "resolved_eval_config",
        "reward_ref",
        "terminal_failure",
    }
    assert record["schema_version"] == 2


@pytest.mark.process_integration
def test_internal_official_failed_and_rejected_resolution_graphs(
    tmp_path,
    monkeypatch,
) -> None:
    internal_store = ObjectStore(
        SqliteBackend(tmp_path / "matrix-internal.sqlite")
    )
    internal_engine = _engine(tmp_path, store=internal_store)
    internal_intent = _intent(
        internal_engine,
        intent_id="matrix-internal",
        purpose="matrix",
    )
    internal = EngineEvaluationService(
        store=internal_store,
        engine=internal_engine,
    ).resolve_evaluation_intent(internal_intent)
    assert internal.outcome is IntentOutcome.COMPLETED
    assert internal.evaluation_result_ref is not None
    assert internal.reward_ref is not None
    assert internal.reward_evidence_refs == (
        internal.reward_ref.record.evidence_refs
    )

    official_store = ObjectStore(
        SqliteBackend(tmp_path / "matrix-official.sqlite")
    )
    official_engine = _engine(
        tmp_path,
        store=official_store,
        role=EvaluationRole.OFFICIAL,
    )
    official_intent = _intent(
        official_engine,
        intent_id="matrix-official",
        purpose="matrix",
        role=EvaluationRole.OFFICIAL,
    )
    official = EngineEvaluationService(
        store=official_store,
        engine=official_engine,
    ).resolve_evaluation_intent(official_intent)
    assert official.outcome is IntentOutcome.COMPLETED
    assert official.evaluation_result_ref is not None
    assert official.reward_ref is None
    assert official.reward_evidence_refs == ()

    failed_store = ObjectStore(
        SqliteBackend(tmp_path / "matrix-failed.sqlite")
    )
    failed_engine = _engine(tmp_path, store=failed_store)
    failed_intent = _intent(
        failed_engine,
        intent_id="matrix-failed",
        purpose="matrix",
    )

    def fail(_request: EvaluationRequest) -> EngineEvaluation:
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(failed_engine, "evaluate", fail)
    failed = EngineEvaluationService(
        store=failed_store,
        engine=failed_engine,
    ).resolve_evaluation_intent(failed_intent)
    assert failed.outcome is IntentOutcome.FAILED
    assert failed.evaluation_result_ref is not None
    assert failed.evaluation_result_ref.schema_name == (
        "whetstone.evaluation_failure"
    )
    assert failed.reward_ref is None
    assert failed.reward_evidence_refs == ()

    rejected_store = ObjectStore(
        SqliteBackend(tmp_path / "matrix-rejected.sqlite")
    )
    rejected_engine = _engine(tmp_path, store=rejected_store)
    invalid = Candidate(
        candidate_id="matrix-invalid",
        base_ref=rejected_engine.experiment.initial_candidate.base_ref,
        payload={"user_prompt_template": "Use {private_gold}."},
    )
    rejected_intent = _intent(
        rejected_engine,
        intent_id="matrix-rejected",
        purpose="matrix",
        candidate=invalid,
    )
    rejected = EngineEvaluationService(
        store=rejected_store,
        engine=rejected_engine,
    ).resolve_evaluation_intent(rejected_intent)
    assert rejected.outcome is IntentOutcome.REJECTED
    assert rejected.evaluation_result_ref is None
    assert rejected.reward_ref is None
    assert rejected.reward_evidence_refs == ()


@pytest.mark.process_integration
def test_prebind_and_restart_reject_coherent_rewritten_output_graph(
    tmp_path,
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "coherent-forgery.sqlite"))
    engine = _engine(tmp_path, store=store)
    intent = _intent(
        engine,
        intent_id="coherent-forgery",
        purpose="graph-validation",
    )
    evaluated = engine.evaluate(
        EvaluationRequest(
            candidate=intent.candidate.record,
            evaluation_binding=intent.evaluation_binding,
            purpose=intent.purpose,
        )
    )
    original_outputs = EvaluationOutputsRecord.model_validate(
        store.get(evaluated.evidence.outputs_ref.reference)
    )
    assert len(original_outputs.outputs) == 1
    original_row = original_outputs.outputs[0]
    rewritten_text = "not the expected answer"
    rewritten_score = 0.0
    rewritten_outputs = original_outputs.model_copy(
        update={
            "outputs": (
                original_row.model_copy(
                    update={
                        "output_text": rewritten_text,
                        "score": rewritten_score,
                        "finish_reason": "length",
                    }
                ),
            )
        }
    )
    rewritten_outputs_ref = _put_typed(
        store,
        EVALUATION_OUTPUTS_SCHEMA,
        rewritten_outputs.record_content(),
    )
    rewritten_aggregate = unweighted_task_mean(
        aggregate_name=evaluated.evidence.aggregate_name,
        graph_hash=engine.experiment.generation_graph.graph_hash,
        evaluation_binding_hash=intent.evaluation_binding.identity_hash(),
        task_rows=(
            TaskRows(
                task_hash=original_row.task_hash,
                rows=(RowValue(value=rewritten_score),),
            ),
        ),
        plan=engine.sampling.evaluation_matrix_plan,
    )
    rewritten_aggregate_ref = _put_typed(
        store,
        AGGREGATE_SCHEMA,
        cast(Jsonable, rewritten_aggregate.record_content()),
    )
    assert rewritten_aggregate_ref == rewritten_aggregate.record_ref()
    rewritten_reward = reward_from_primary_score(
        engine.experiment.reward_policy,
        primary_score=rewritten_aggregate.aggregation_output.value,
        evidence_refs=(rewritten_aggregate_ref,),
    )
    rewritten_reward_ref = reward_reference(rewritten_reward)
    assert (
        _put_typed(
            store,
            rewritten_reward_ref.record_ref.schema_name,
            rewritten_reward.record_content(),
        )
        == rewritten_reward_ref.record_ref
    )
    rewritten_evidence = evaluated.evidence.model_copy(
        update={
            "outputs_ref": rewritten_outputs_ref,
            "aggregate_ref": rewritten_aggregate_ref,
            "aggregate_value": (rewritten_aggregate.aggregation_output.value),
            "aggregate_status": (
                rewritten_aggregate.aggregation_output.status.value
            ),
            "per_task_values": (rewritten_score,),
            "reward_ref": rewritten_reward_ref,
        }
    )
    rewritten_evidence_ref = _put_typed(
        store,
        EVALUATION_EVIDENCE_SCHEMA,
        rewritten_evidence.record_content(),
    )
    rewritten_resolution = _completed_resolution(intent, evaluated).model_copy(
        update={
            "evaluation_result_ref": rewritten_evidence_ref,
            "reward_evidence_refs": (rewritten_aggregate_ref,),
            "reward_ref": rewritten_reward_ref,
        }
    )
    service = EngineEvaluationService(store=store, engine=engine)
    _publish_attestation(
        service=service,
        intent=intent,
        resolution=_completed_resolution(intent, evaluated),
    )

    with pytest.raises(ValueError, match="terminal Evaluation Result"):
        service._bind(intent, rewritten_resolution)
    _bind_without_validation(
        store=store,
        service=service,
        intent=intent,
        resolution=rewritten_resolution,
    )
    with pytest.raises(ValueError, match="terminal Evaluation Result"):
        service.resolve_evaluation_intent(intent)


@pytest.mark.process_integration
def test_prebind_and_restart_reject_rewritten_operational_evidence(
    tmp_path,
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "operational-forgery.sqlite"))
    engine = _engine(tmp_path, store=store)
    intent = _intent(
        engine,
        intent_id="operational-forgery",
        purpose="graph-validation",
    )
    evaluated = engine.evaluate(
        EvaluationRequest(
            candidate=intent.candidate.record,
            evaluation_binding=intent.evaluation_binding,
            purpose=intent.purpose,
        )
    )
    service = EngineEvaluationService(store=store, engine=engine)
    _publish_attestation(
        service=service,
        intent=intent,
        resolution=_completed_resolution(intent, evaluated),
    )
    rewritten_evidence = evaluated.evidence.model_copy(
        update={
            "cache": evaluated.evidence.cache.model_copy(
                update={"partial_row_count": 99}
            ),
            "concurrency_halved": not evaluated.evidence.concurrency_halved,
            "deadline_reached": not evaluated.evidence.deadline_reached,
            "guard_timeouts": evaluated.evidence.guard_timeouts + 1,
        }
    )
    rewritten_evidence_ref = _put_typed(
        store,
        EVALUATION_EVIDENCE_SCHEMA,
        rewritten_evidence.record_content(),
    )
    rewritten_resolution = _completed_resolution(intent, evaluated).model_copy(
        update={"evaluation_result_ref": rewritten_evidence_ref}
    )

    with pytest.raises(ValueError, match="terminal Evaluation Result"):
        service._bind(intent, rewritten_resolution)
    _bind_without_validation(
        store=store,
        service=service,
        intent=intent,
        resolution=rewritten_resolution,
    )
    with pytest.raises(ValueError, match="terminal Evaluation Result"):
        service.resolve_evaluation_intent(intent)


@pytest.mark.process_integration
def test_prebind_and_restart_reject_rewritten_failure_evidence(
    tmp_path,
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "failure-forgery.sqlite"))
    engine = _engine(tmp_path, store=store)
    intent = _intent(
        engine,
        intent_id="failure-forgery",
        purpose="graph-validation",
    )
    service = EngineEvaluationService(store=store, engine=engine)
    service._persist_intent_targets(intent)
    failure = EvaluationFailureEvidence(
        candidate=intent.candidate,
        evaluation_binding=intent.evaluation_binding,
        purpose=intent.purpose,
        exception_type="RuntimeError",
        message="provider unavailable",
    )
    failure_ref = _put_typed(
        store,
        EVALUATION_FAILURE_SCHEMA,
        failure.record_content(),
    )
    terminal = TerminalFailure(
        code="evaluation_RuntimeError",
        message=failure.message,
        details={
            "evidence_schema": failure_ref.schema_name,
            "evidence_content_hash": failure_ref.content_hash,
        },
    )
    canonical = IntentResolution(
        schema_version=INTENT_RESOLUTION_SCHEMA_VERSION,
        intent=intent,
        outcome=IntentOutcome.FAILED,
        detail=ResolutionDetail(
            classification=ResolutionClass.INFRASTRUCTURE,
            message=failure.message,
        ),
        evaluation_result_ref=failure_ref,
        resolved_eval_config=intent.target_eval_config,
        terminal_failure=terminal,
    )
    _publish_attestation(
        service=service,
        intent=intent,
        resolution=canonical,
    )
    wrong_classification = canonical.model_copy(
        update={
            "detail": ResolutionDetail(
                classification=ResolutionClass.UNSCORABLE,
                message=failure.message,
            )
        }
    )
    with pytest.raises(ValueError, match="detail disagrees"):
        service._validate_result_graph(
            wrong_classification,
            expected_intent=intent,
            require_attestation=False,
        )

    rewritten_failure = failure.model_copy(
        update={
            "exception_type": "TimeoutError",
            "message": "forged timeout",
        }
    )
    rewritten_failure_ref = _put_typed(
        store,
        EVALUATION_FAILURE_SCHEMA,
        rewritten_failure.record_content(),
    )
    rewritten_resolution = canonical.model_copy(
        update={
            "detail": ResolutionDetail(
                classification=ResolutionClass.INFRASTRUCTURE,
                message=rewritten_failure.message,
            ),
            "evaluation_result_ref": rewritten_failure_ref,
            "terminal_failure": TerminalFailure(
                code="evaluation_TimeoutError",
                message=rewritten_failure.message,
                details={
                    "evidence_schema": rewritten_failure_ref.schema_name,
                    "evidence_content_hash": (
                        rewritten_failure_ref.content_hash
                    ),
                },
            ),
        }
    )
    with pytest.raises(ValueError, match="terminal Evaluation Result"):
        service._bind(intent, rewritten_resolution)
    _bind_without_validation(
        store=store,
        service=service,
        intent=intent,
        resolution=rewritten_resolution,
    )
    with pytest.raises(ValueError, match="terminal Evaluation Result"):
        service.resolve_evaluation_intent(intent)


@pytest.mark.process_integration
def test_service_accepts_complete_matrix_with_a_failed_row(tmp_path) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "failed-row.sqlite"))

    def one_success_one_failure(request: DirectRowRequest) -> ProcessJob:
        outcome = (
            _successful_direct_outcome(request)
            if request.sample_index == 0
            else DirectRowOutcome(
                submission_score=None,
                output_text=None,
                row_state=ExecutedRowState.FAILED,
                executed_component_steps=(),
                failure_code="provider_unavailable",
                provider_error={"type": "provider_unavailable"},
            )
        )
        return ProcessJob(
            entrypoint="tests.envs.process_workers:return_payload",
            payload=DirectRowResult(
                request_hash=request.request_hash,
                outcome=outcome,
            ).model_dump(mode="json"),
        )

    engine = _engine(
        tmp_path,
        store=store,
        row_job_factory=one_success_one_failure,
        num_samples=2,
        role=EvaluationRole.OFFICIAL,
    )
    intent = _intent(
        engine,
        intent_id="failed-row",
        purpose="failed-row",
        role=EvaluationRole.OFFICIAL,
    )
    service = EngineEvaluationService(store=store, engine=engine)

    resolution = service.resolve_evaluation_intent(intent)

    assert resolution.outcome is IntentOutcome.COMPLETED
    service.validate_resolution_graph(resolution)
    assert resolution.evaluation_result_ref is not None
    evidence = EvaluationEvidence.model_validate(
        store.get(resolution.evaluation_result_ref.reference)
    )
    outputs = EvaluationOutputsRecord.model_validate(
        store.get(evidence.outputs_ref.reference)
    )
    assert tuple((row.failed, row.score) for row in outputs.outputs) == (
        (False, 1.0),
        (True, None),
    )


@pytest.mark.parametrize(
    ("corruption", "expected_error"),
    (
        ("missing", ObjectNotFoundError),
        ("corrupt", ContentHashMismatchError),
    ),
)
@pytest.mark.process_integration
def test_restart_rejects_unresolvable_provider_execution_policy(
    tmp_path,
    corruption: str,
    expected_error: type[Exception],
) -> None:
    database = tmp_path / f"provider-policy-{corruption}.sqlite"
    store = ObjectStore(SqliteBackend(database))
    engine = _engine(tmp_path, store=store)
    intent = _intent(
        engine,
        intent_id=f"provider-policy-{corruption}",
        purpose="provider-policy-restart",
    )
    resolution = EngineEvaluationService(
        store=store,
        engine=engine,
    ).resolve_evaluation_intent(intent)
    assert resolution.outcome is IntentOutcome.COMPLETED
    policy_ref = intent.evaluation_binding.provider_execution_policy_ref
    assert policy_ref is not None

    with sqlite3.connect(database) as connection:
        if corruption == "missing":
            connection.execute(
                "DELETE FROM objects WHERE schema = ? AND content_hash = ?",
                (
                    policy_ref.record_ref.schema_name,
                    policy_ref.record_ref.content_hash,
                ),
            )
        else:
            connection.execute(
                "UPDATE objects SET canonical = ? "
                "WHERE schema = ? AND content_hash = ?",
                (
                    '{"corrupt":true}',
                    policy_ref.record_ref.schema_name,
                    policy_ref.record_ref.content_hash,
                ),
            )

    def reject_submission(_request: DirectRowRequest) -> ProcessJob:
        raise AssertionError("replay must not create a process job")

    restart_store = ObjectStore(SqliteBackend(database))
    restart_engine = _engine(
        tmp_path,
        store=restart_store,
        row_job_factory=reject_submission,
    )
    with pytest.raises(expected_error):
        EngineEvaluationService(
            store=restart_store,
            engine=restart_engine,
        ).resolve_evaluation_intent(intent)


@pytest.mark.parametrize(
    ("corruption", "expected_error"),
    (
        ("missing", ObjectNotFoundError),
        ("corrupt", ContentHashMismatchError),
    ),
)
@pytest.mark.process_integration
def test_restart_rejects_unresolvable_component_trace_artifact(
    tmp_path,
    corruption: str,
    expected_error: type[Exception],
) -> None:
    database = tmp_path / f"component-trace-{corruption}.sqlite"
    store = ObjectStore(SqliteBackend(database))
    engine = _engine(tmp_path, store=store)
    intent = _intent(
        engine,
        intent_id=f"component-trace-{corruption}",
        purpose="component-trace-restart",
    )
    resolution = EngineEvaluationService(
        store=store,
        engine=engine,
    ).resolve_evaluation_intent(intent)
    assert resolution.evaluation_result_ref is not None
    evidence = EvaluationEvidence.model_validate(
        store.get(resolution.evaluation_result_ref.reference)
    )
    trace_ref = evidence.component_traces_ref

    with sqlite3.connect(database) as connection:
        if corruption == "missing":
            connection.execute(
                "DELETE FROM objects WHERE schema = ? AND content_hash = ?",
                (trace_ref.schema_name, trace_ref.content_hash),
            )
        else:
            connection.execute(
                "UPDATE objects SET canonical = ? "
                "WHERE schema = ? AND content_hash = ?",
                (
                    '{"corrupt":true}',
                    trace_ref.schema_name,
                    trace_ref.content_hash,
                ),
            )

    restart_store = ObjectStore(SqliteBackend(database))
    restart_engine = _engine(tmp_path, store=restart_store)
    with pytest.raises(expected_error):
        EngineEvaluationService(
            store=restart_store,
            engine=restart_engine,
        ).resolve_evaluation_intent(intent)


@pytest.mark.process_integration
def test_restart_rejects_result_attested_under_another_provider_policy(
    tmp_path,
) -> None:
    database = tmp_path / "provider-policy-engine-drift.sqlite"
    store = ObjectStore(SqliteBackend(database))
    policy_a = execution_policy(max_attempts=1)
    engine_a = _engine(tmp_path, store=store, provider_policy=policy_a)
    intent = _intent(
        engine_a,
        intent_id="provider-policy-engine-drift",
        purpose="provider-policy-restart",
    )
    resolution = EngineEvaluationService(
        store=store,
        engine=engine_a,
    ).resolve_evaluation_intent(intent)
    assert resolution.outcome is IntentOutcome.COMPLETED

    def reject_submission(_request: DirectRowRequest) -> ProcessJob:
        raise AssertionError("replay must not create a process job")

    restart_store = ObjectStore(SqliteBackend(database))
    engine_b = _engine(
        tmp_path,
        store=restart_store,
        row_job_factory=reject_submission,
        provider_policy=execution_policy(max_attempts=2),
    )
    with pytest.raises(ValueError, match="exact Provider Execution Policy"):
        EngineEvaluationService(
            store=restart_store,
            engine=engine_b,
        ).resolve_evaluation_intent(intent)


@pytest.mark.process_integration
def test_restart_rejects_aggregate_from_another_generation_graph(
    tmp_path,
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "wrong-graph.sqlite"))
    engine = _engine(
        tmp_path,
        store=store,
        role=EvaluationRole.OFFICIAL,
    )
    intent = _intent(
        engine,
        intent_id="wrong-graph",
        purpose="graph-validation",
        role=EvaluationRole.OFFICIAL,
    )
    evaluated = engine.evaluate(
        EvaluationRequest(
            candidate=intent.candidate.record,
            evaluation_binding=intent.evaluation_binding,
            purpose=intent.purpose,
        )
    )
    aggregate_content = store.get(evaluated.evidence.aggregate_ref.reference)
    assert isinstance(aggregate_content, dict)
    aggregate_content["graph_hash"] = "f" * 64
    aggregate_ref = _put_typed(
        store,
        evaluated.evidence.aggregate_ref.schema_name,
        aggregate_content,
    )
    forged_evidence = evaluated.evidence.model_copy(
        update={
            "graph_hash": "f" * 64,
            "graph_config_ref": "f" * 64,
            "aggregate_ref": aggregate_ref,
        }
    )
    forged_evidence_ref = _put_typed(
        store,
        EVALUATION_EVIDENCE_SCHEMA,
        forged_evidence.record_content(),
    )
    resolution = _completed_resolution(intent, evaluated).model_copy(
        update={"evaluation_result_ref": forged_evidence_ref}
    )
    service = EngineEvaluationService(store=store, engine=engine)
    _publish_attestation(
        service=service,
        intent=intent,
        resolution=_completed_resolution(intent, evaluated),
    )
    with pytest.raises(ValueError, match="another generation graph"):
        service._validate_result_graph(
            resolution,
            expected_intent=intent,
            require_attestation=False,
        )
    _bind_without_validation(
        store=store,
        service=service,
        intent=intent,
        resolution=resolution,
    )

    with pytest.raises(ValueError, match="terminal Evaluation Result"):
        service.resolve_evaluation_intent(intent)


@pytest.mark.process_integration
def test_restart_rejects_evidence_resolution_reward_disagreement(
    tmp_path,
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "reward-disagreement.sqlite"))
    engine = _engine(tmp_path, store=store)
    intent = _intent(
        engine,
        intent_id="reward-disagreement",
        purpose="graph-validation",
    )
    evaluated = engine.evaluate(
        EvaluationRequest(
            candidate=intent.candidate.record,
            evaluation_binding=intent.evaluation_binding,
            purpose=intent.purpose,
        )
    )
    other = engine.evaluate(
        EvaluationRequest(
            candidate=intent.candidate.record,
            evaluation_binding=_binding(engine, campaign="other-reward"),
            purpose=intent.purpose,
        )
    )
    assert other.evidence.reward_ref is not None
    forged_resolution = _completed_resolution(intent, evaluated).model_copy(
        update={
            "reward_ref": other.evidence.reward_ref,
            "reward_evidence_refs": (
                other.evidence.reward_ref.record.evidence_refs
            ),
        }
    )
    service = EngineEvaluationService(store=store, engine=engine)
    _publish_attestation(
        service=service,
        intent=intent,
        resolution=_completed_resolution(intent, evaluated),
    )
    with pytest.raises(ValueError, match="disagree on Reward"):
        service._validate_result_graph(
            forged_resolution,
            expected_intent=intent,
            require_attestation=False,
        )
    _bind_without_validation(
        store=store,
        service=service,
        intent=intent,
        resolution=forged_resolution,
    )

    with pytest.raises(ValueError, match="terminal Evaluation Result"):
        service.resolve_evaluation_intent(intent)


@pytest.mark.parametrize(
    "forgery",
    (
        "evidence_binding",
        "evidence_purpose",
        "evidence_dataset",
        "output_binding",
        "output_purpose",
        "output_role",
        "output_split",
        "output_task",
        "output_repeat",
        "output_trace",
        "output_metadata",
        "output_score",
        "output_empty",
        "aggregate_value",
        "missing_output",
    ),
)
@pytest.mark.process_integration
def test_restart_rejects_forged_or_incomplete_result_graphs(
    tmp_path,
    forgery: str,
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / f"forged-{forgery}.sqlite"))
    engine = _engine(tmp_path, store=store)
    intent = _intent(
        engine,
        intent_id=f"forged-{forgery}",
        purpose="graph-validation",
    )
    bundle = evaluate_intent_bundle(engine, intent)
    assert_restart_rejects_forgery(
        store=store,
        engine=engine,
        bundle=bundle,
        forgery=forgery,
    )


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
@pytest.mark.process_integration
def test_claim_attestation_rejects_forged_component_traces(
    tmp_path,
    forgery: str,
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / f"trace-{forgery}.sqlite"))
    engine = _engine(tmp_path, store=store, num_samples=2)
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
        trace_content["task_hashes"] = ["forged-task"]
        for row in trace_content["rows"]:
            row["task_hash"] = "forged-task"
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
        ][0]["outputs"]["provider_generation"] = "forged output"
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


@pytest.mark.process_integration
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


@pytest.mark.process_integration
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


@pytest.mark.process_integration
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


@pytest.mark.process_integration
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


@pytest.mark.process_integration
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


@pytest.mark.process_integration
def test_expired_claim_retries_after_resolver_crash(
    tmp_path, monkeypatch
) -> None:
    database = tmp_path / "claim-retry.sqlite"
    now = [100.0]
    delegated = process_row_job_factory(
        "tests.envs.process_workers:drive_d1_success"
    )
    submitted: list[DirectRowRequest] = []
    evaluation_attempts: list[EvaluationRequest] = []

    def record_submission(request: DirectRowRequest) -> ProcessJob:
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


@pytest.mark.parametrize(
    "outcome",
    (
        IntentOutcome.COMPLETED,
        IntentOutcome.FAILED,
    ),
)
@pytest.mark.process_integration
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


@pytest.mark.process_integration
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
