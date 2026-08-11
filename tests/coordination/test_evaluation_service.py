from __future__ import annotations

import sqlite3
from typing import cast

import pytest
from dr_serialize import Jsonable
from dr_store import (
    ContentHashMismatchError,
    ObjectNotFoundError,
    ObjectStore,
    SqliteBackend,
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
    _successful_internal_outcome,
)
from tests.optimization.support import (
    make_harness,
    make_intent,
    proposal_request,
    proposal_run,
    proposed_candidate,
    python_format_contract,
    registry,
)
from whetstone.coordination.evaluation_service import EngineEvaluationService
from whetstone.core.effects.models import ReplayPolicy
from whetstone.core.identity import (
    TerminalFailure,
    TypedRef,
)
from whetstone.core.roles import EvaluationRole
from whetstone.envs.oracle_operator import env_exact_match_score
from whetstone.envs.registry import env_spec
from whetstone.envs.reward import reward_from_internal_aggregate
from whetstone.evaluation.aggregate import (
    AGGREGATE_SCHEMA,
    RowValue,
    TaskRows,
    unweighted_task_mean,
)
from whetstone.evaluation.drivers.internal import (
    InternalRowOutcome,
    InternalRowRequest,
    InternalRowResult,
)
from whetstone.evaluation.engine import (
    EngineEvaluation,
    EvaluationRequest,
)
from whetstone.evaluation.schema import (
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
)
from whetstone.experiment.reward import (
    reward_reference,
)
from whetstone.optimization.adapters import AdapterOutput
from whetstone.optimization.contracts import (
    INTENT_RESOLUTION_SCHEMA_VERSION,
    BudgetDelta,
    IntentOutcome,
    IntentResolution,
    ResolutionClass,
    ResolutionDetail,
    StepMode,
    StepStatus,
)


def test_service_rejects_provider_policy_mismatch_before_execution(
    tmp_path,
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "service-policy.sqlite"))
    submitted: list[InternalRowRequest] = []

    def reject_submission(request: InternalRowRequest) -> ProcessJob:
        submitted.append(request)
        raise AssertionError("invalid binding must not create a process job")

    engine = _engine(
        tmp_path,
        store=store,
        row_job_factory=reject_submission,
    )
    intent = _intent(
        engine,
        intent_id="service-policy",
        purpose="provider-policy",
    )
    mismatched = intent.model_copy(
        update={
            "evaluation_binding": intent.evaluation_binding.model_copy(
                update={
                    "provider_execution_policy_ref": TypedRef(
                        schema_name="whetstone.provider_execution_policy",
                        content_hash="f" * 64,
                    )
                }
            )
        }
    )

    resolution = EngineEvaluationService(
        store=store,
        engine=engine,
    ).resolve_evaluation_intent(mismatched)

    assert resolution.outcome is IntentOutcome.REJECTED
    assert resolution.detail.classification is ResolutionClass.VALIDATION
    assert resolution.evaluation_result_ref is None
    assert submitted == []


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
    assert resolution.evaluation_result_ref is None
    assert resolution.reward_evidence_refs == ()
    assert submitted == []


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
    assert first.reward_evidence_refs == first.reward_ref.record.evidence_refs
    assert first.evaluation_result_ref is not None
    first_evidence = EvaluationEvidence.model_validate(
        store.get(first.evaluation_result_ref.reference)
    )
    first_outputs = EvaluationOutputsRecord.model_validate(
        store.get(first_evidence.outputs_ref.reference)
    )
    assert first_outputs.component_traces_ref == (
        first_evidence.component_traces_ref
    )
    assert _load_component_traces(store, first_evidence).rows

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


@pytest.mark.parametrize(
    "forgery",
    (
        "candidate",
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
    evaluated = engine.evaluate(
        EvaluationRequest(
            candidate=intent.candidate.record,
            evaluation_binding=intent.evaluation_binding,
            purpose=intent.purpose,
        )
    )
    evidence = evaluated.evidence
    evidence_update: dict[str, object] = {}

    if forgery == "candidate":
        other_candidate = intent.candidate.record.model_copy(
            update={"candidate_id": "candidate-b"}
        )
        other = engine.evaluate(
            EvaluationRequest(
                candidate=other_candidate,
                evaluation_binding=intent.evaluation_binding,
                purpose=intent.purpose,
            )
        )
        evidence_update["outputs_ref"] = other.evidence.outputs_ref
    elif forgery == "evidence_binding":
        evidence_update["evaluation_binding"] = _binding(
            engine,
            campaign="forged-binding",
        )
    elif forgery == "evidence_purpose":
        evidence_update["purpose"] = "forged-purpose"
    elif forgery == "evidence_dataset":
        evidence_update["dataset_hash"] = "forged-dataset"
    elif forgery == "aggregate_value":
        assert evidence.aggregate_value is not None
        evidence_update["aggregate_value"] = evidence.aggregate_value + 1.0
    elif forgery == "missing_output":
        evidence_update["outputs_ref"] = TypedRef(
            schema_name=EVALUATION_OUTPUTS_SCHEMA,
            content_hash="f" * 64,
        )
    else:
        outputs_content = EvaluationOutputsRecord.model_validate(
            store.get(evidence.outputs_ref.reference)
        ).record_content()
        if forgery == "output_binding":
            outputs_content["evaluation_binding"] = _binding(
                engine,
                campaign="forged-output-binding",
            ).model_dump(mode="json")
        elif forgery == "output_purpose":
            outputs_content["purpose"] = "forged-purpose"
        elif forgery == "output_role":
            official_binding = _binding(
                engine,
                role=EvaluationRole.OFFICIAL,
                campaign="forged-output-role",
            )
            outputs_content["evaluation_binding"] = (
                official_binding.model_dump(mode="json")
            )
            outputs_content["evaluation_role"] = "official"
        elif forgery == "output_split":
            outputs_content["split_role"] = "official"
        elif forgery == "output_task":
            outputs_content["task_hashes"] = ["forged-task"]
            outputs_content["outputs"][0]["task_hash"] = "forged-task"
        elif forgery == "output_repeat":
            outputs_content["num_samples"] = 2
        elif forgery == "output_trace":
            outputs_content["outputs"][0]["rendered_prompt"] = "forged prompt"
        elif forgery == "output_metadata":
            outputs_content["outputs"][0].update(
                {
                    "output_text": "forged output",
                    "finish_reason": "length",
                    "provider_error": {"type": "forged"},
                    "failure_code": "forged_failure",
                }
            )
        elif forgery == "output_score":
            outputs_content["outputs"][0]["score"] = 0.0
        elif forgery == "output_empty":
            outputs_content["outputs"] = []
        else:
            raise AssertionError(f"unhandled forgery {forgery}")
        evidence_update["outputs_ref"] = _put_typed(
            store,
            EVALUATION_OUTPUTS_SCHEMA,
            outputs_content,
        )

    forged_evidence = evidence.model_copy(update=evidence_update)
    forged_evidence_ref = _put_typed(
        store,
        EVALUATION_EVIDENCE_SCHEMA,
        forged_evidence.record_content(),
    )
    forged_resolution = _completed_resolution(intent, evaluated).model_copy(
        update={"evaluation_result_ref": forged_evidence_ref}
    )
    service = EngineEvaluationService(store=store, engine=engine)
    _publish_attestation(
        service=service,
        intent=intent,
        resolution=_completed_resolution(intent, evaluated),
    )
    with pytest.raises((ObjectNotFoundError, ValueError)):
        service._bind(intent, forged_resolution)
    _bind_without_validation(
        store=store,
        service=service,
        intent=intent,
        resolution=forged_resolution,
    )

    with pytest.raises((ObjectNotFoundError, ValueError)):
        service.resolve_evaluation_intent(intent)


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
    instance = engine.sampling.tasks[0]
    rewritten_text = "not the expected answer"
    rewritten_score = float(
        env_exact_match_score(
            env=env_spec(engine.experiment.env_name),
            generation=rewritten_text,
            gold=instance.gold,
            evaluation_procedure_config_hash=(
                engine.experiment.generation_graph.procedure_config_hash
            ),
        ).value
    )
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
    rewritten_reward = reward_from_internal_aggregate(
        engine.experiment.reward_policy,
        env_exact_match_value=rewritten_aggregate.aggregation_output.value,
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


def test_service_accepts_complete_matrix_with_a_failed_row(tmp_path) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "failed-row.sqlite"))

    def one_success_one_failure(request: InternalRowRequest) -> ProcessJob:
        outcome = (
            _successful_internal_outcome(request)
            if request.sample_index == 0
            else InternalRowOutcome(
                score=None,
                row_state=ExecutedRowState.FAILED,
                executed_component_steps=(),
                failure_code="provider_unavailable",
                provider_error={"type": "provider_unavailable"},
            )
        )
        return ProcessJob(
            entrypoint="tests.envs.process_workers:return_payload",
            payload=InternalRowResult(
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


def test_concrete_evaluation_service_reaches_harness_boundary(
    tmp_path,
) -> None:
    class EngineProposalAdapter:
        @property
        def key(self) -> str:
            return "proposal-test"

        @property
        def mode(self) -> StepMode:
            return StepMode.PROPOSAL_ONLY

        @property
        def required_replay_policy(self) -> ReplayPolicy:
            return ReplayPolicy.IDEMPOTENT

        def invoke(self, request, handles) -> AdapterOutput:
            assert handles == ()
            base = request.candidates[0]
            template = str(base.payload["user_prompt_template"])
            proposed = proposed_candidate(
                base,
                "harness-evaluation",
                text=f"{template}\n\nBe precise.",
            )
            intent = make_intent(
                proposed,
                run_id=request.run_id,
                step_index=request.step_index,
                binding=_binding(engine, campaign=request.run_id),
                reward_policy=engine.experiment.reward_policy,
            )
            return AdapterOutput(
                proposed_candidates=(proposed,),
                accepted_candidates=(proposed,),
                evaluation_intents=(intent,),
                budget_delta=BudgetDelta(consumed={"generations": 1}),
                proposed_status=StepStatus.COMPLETE,
            )

    store = ObjectStore(SqliteBackend(tmp_path / "harness-service.sqlite"))
    engine = _engine(tmp_path, store=store)
    service = EngineEvaluationService(store=store, engine=engine)
    render_contract = python_format_contract(
        available_fields=("question", "query"),
        required_fields=("question", "query"),
    )
    run = proposal_run(
        reward_policy=engine.experiment.reward_policy,
        template_render_contract=render_contract,
    )
    request = proposal_request(
        run=run,
        candidates=(engine.experiment.initial_candidate,),
    )
    adapter = EngineProposalAdapter()
    assert adapter.required_replay_policy is ReplayPolicy.IDEMPOTENT
    harness = make_harness(
        store=store,
        adapter_registry=registry(adapter),
        run=request.run,
        evaluation_service=service,
        adapter_replay_policy=ReplayPolicy.IDEMPOTENT,
    )

    result, _result_ref = harness.run_step(request)

    assert service.replay_policy is ReplayPolicy.DURABLE_WORKFLOW
    assert len(result.resolved_intents) == 1
    assert result.resolved_intents[0].outcome is IntentOutcome.COMPLETED
    service.validate_resolution_graph(result.resolved_intents[0])


@pytest.mark.parametrize(
    ("corruption", "expected_error"),
    (
        ("missing", ObjectNotFoundError),
        ("corrupt", ContentHashMismatchError),
    ),
)
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

    def reject_submission(_request: InternalRowRequest) -> ProcessJob:
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

    def reject_submission(_request: InternalRowRequest) -> ProcessJob:
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
