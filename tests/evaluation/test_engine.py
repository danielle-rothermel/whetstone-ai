from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
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

from tests.envs.support import (
    execution_policy,
    process_row_job_factory,
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
from whetstone.code_eval.aggregate import (
    ROLLOUT_AGGREGATE_SCHEMA,
    RowValue,
    TaskRows,
    unweighted_task_mean,
)
from whetstone.envs.ed1 import DECODER_TEMPLATE
from whetstone.envs.encdec_rollout import (
    DECODER_NODE_ID,
    ENCODER_NODE_ID,
    build_encdec_rollout_definition,
)
from whetstone.envs.factory import build_env_experiment
from whetstone.envs.internal_eval import (
    ExecutedRowState,
    InternalRowJobFactory,
    InternalRowOutcome,
    InternalRowRequest,
    InternalRowResult,
    _llm_component_step,
)
from whetstone.envs.oracle_operator import env_exact_match_score
from whetstone.envs.registry import env_spec
from whetstone.envs.reward import reward_from_internal_aggregate
from whetstone.envs.rollout_definition import LLM_NODE_ID, render_prompt
from whetstone.evaluation import (
    EngineEvaluation,
    EngineEvaluationService,
    EngineToolEvaluator,
    EvaluationComponentTraces,
    EvaluationComponentTracesRef,
    EvaluationEngine,
    EvaluationEvidence,
    EvaluationEvidenceRef,
    EvaluationFailureEvidence,
    EvaluationFailureEvidenceRef,
    EvaluationOutputRow,
    EvaluationOutputsRecord,
    EvaluationRequest,
)
from whetstone.evaluation import engine as evaluation_engine_module
from whetstone.evaluation.schema import (
    EVALUATION_COMPONENT_TRACES_SCHEMA,
    EVALUATION_COMPONENT_TRACES_SCHEMA_VERSION,
    EVALUATION_EVIDENCE_SCHEMA_VERSION,
    EVALUATION_INTENT_CLAIM_SCHEMA,
    EVALUATION_OUTPUTS_SCHEMA,
    EVALUATION_OUTPUTS_SCHEMA_VERSION,
    EVALUATION_RESULT_ATTESTATION_SCHEMA,
    EvaluationIntentClaim,
    EvaluationResultAttestation,
)
from whetstone.evaluation_role import EvaluationRole
from whetstone.execution.fanout import ProcessJob
from whetstone.execution.partials import PartialLog
from whetstone.execution.prompt_cache import PromptResultCache
from whetstone.optimization import (
    AdapterOutput,
    BudgetDelta,
    Candidate,
    EvaluationIntent,
    IntentOutcome,
    IntentResolution,
    ReplayPolicy,
    ResolutionClass,
    ResolutionDetail,
    Reward,
    StepMode,
    StepStatus,
    TerminalFailure,
    ToolCall,
    ToolCapacity,
    ToolCapacityScope,
    ToolConfig,
    ToolDefinition,
    TypedRef,
    candidate_reference,
    reward_reference,
    tool_capacity_binding,
    tool_config_reference,
    tool_definition_reference,
)
from whetstone.optimization.schema import (
    EVALUATION_BINDING_SCHEMA_VERSION,
    EVALUATION_EVIDENCE_SCHEMA,
    EVALUATION_FAILURE_SCHEMA,
    INTENT_RESOLUTION_SCHEMA,
    INTENT_RESOLUTION_SCHEMA_VERSION,
    EvaluationBinding,
)
from whetstone.provider.policy import ProviderExecutionPolicy

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
    role: EvaluationRole = EvaluationRole.INTERNAL,
    provider_policy: ProviderExecutionPolicy | None = None,
    max_wall_seconds: float | None = None,
) -> EvaluationEngine:
    experiment = _experiment(repeats=repeats)
    sampling = (
        experiment.eval_configs.internal
        if role is EvaluationRole.INTERNAL
        else experiment.eval_configs.official
    )
    return EvaluationEngine(
        store=store,
        experiment=experiment,
        sampling=sampling,
        execution_policy=provider_policy or execution_policy(),
        row_job_factory=row_job_factory,
        max_wall_seconds=max_wall_seconds,
        partial_log=PartialLog(tmp_path / "partials.jsonl")
        if partial
        else None,
        prompt_cache=PromptResultCache(tmp_path / "cache") if cache else None,
    )


def _ed1_graph_engine(*, store: ObjectStore) -> EvaluationEngine:
    base_experiment = _experiment()
    experiment = replace(
        base_experiment,
        rollout_definition=build_encdec_rollout_definition(
            "ed1",
            model="openai/test",
            procedure_config_hash=(
                base_experiment.rollout_definition.procedure_config_hash
            ),
            budget_ratio=None,
        ),
    )
    return EvaluationEngine(
        store=store,
        experiment=experiment,
        sampling=experiment.eval_configs.official,
        execution_policy=execution_policy(),
        row_job_factory=_DEFAULT_ROW_JOB_FACTORY,
    )


def _binding(
    engine: EvaluationEngine,
    *,
    role: EvaluationRole = EvaluationRole.INTERNAL,
    campaign: str = "evaluation-test",
) -> EvaluationBinding:
    return EvaluationBinding(
        schema_version=EVALUATION_BINDING_SCHEMA_VERSION,
        eval_config=engine.eval_config_ref,
        role=role,
        authority_principal=(
            "test-authority" if role is EvaluationRole.OFFICIAL else None
        ),
        campaign=campaign,
        provider_execution_policy_ref=engine.provider_execution_policy_ref,
    )


def _intent(
    engine: EvaluationEngine,
    *,
    intent_id: str,
    purpose: str,
    candidate: Candidate | None = None,
    role: EvaluationRole = EvaluationRole.INTERNAL,
) -> EvaluationIntent:
    return EvaluationIntent(
        intent_id=intent_id,
        candidate=candidate_reference(
            candidate or engine.experiment.initial_candidate
        ),
        target_eval_config=engine.eval_config_ref,
        evaluation_binding=_binding(engine, role=role, campaign=intent_id),
        purpose=purpose,
        run_id="run",
        step_index=0,
        expected_reward_policy_hash=(
            engine.experiment.reward_policy.identity_hash()
            if role is EvaluationRole.INTERNAL
            else None
        ),
    )


def _completed_resolution(
    intent: EvaluationIntent,
    evaluated: EngineEvaluation,
) -> IntentResolution:
    reward_ref = evaluated.evidence.reward_ref
    return IntentResolution(
        schema_version=INTENT_RESOLUTION_SCHEMA_VERSION,
        intent=intent,
        outcome=IntentOutcome.COMPLETED,
        detail=ResolutionDetail(
            classification=ResolutionClass.MEASURED,
            message="candidate evaluated under exact sampling binding",
        ),
        evaluation_result_ref=evaluated.evidence_ref,
        reward_evidence_refs=(
            () if reward_ref is None else reward_ref.record.evidence_refs
        ),
        resolved_eval_config=intent.target_eval_config,
        reward_ref=reward_ref,
    )


def _bind_without_validation(
    *,
    store: ObjectStore,
    service: EngineEvaluationService,
    intent: EvaluationIntent,
    resolution: IntentResolution,
) -> None:
    reference, _ = store.put(
        INTENT_RESOLUTION_SCHEMA,
        resolution.model_dump(mode="json"),
    )
    store.bind(service._key(intent), reference)


def _bind_with_forged_terminal_attestation(
    *,
    store: ObjectStore,
    service: EngineEvaluationService,
    intent: EvaluationIntent,
    resolution: IntentResolution,
) -> None:
    attestation = EvaluationResultAttestation(
        graph_hash=service._engine.experiment.rollout_definition.graph_hash,
        resolution=resolution,
    )
    attestation_ref = _put_typed(
        store,
        EVALUATION_RESULT_ATTESTATION_SCHEMA,
        attestation.record_content(),
    )
    claim = EvaluationIntentClaim(
        intent_ref=service._intent_ref(intent),
        owner_id="forged-restart-fixture",
        event_ordinal=0,
        generation=0,
        heartbeat_ordinal=0,
        expires_at=0.0,
        result_attestation_ref=attestation_ref,
    )
    claim_reference, _ = store.put(
        EVALUATION_INTENT_CLAIM_SCHEMA,
        claim.model_dump(mode="json"),
    )
    store.bind(service._claim_key(intent, 0), claim_reference)
    _bind_without_validation(
        store=store,
        service=service,
        intent=intent,
        resolution=resolution,
    )


def _publish_attestation(
    *,
    service: EngineEvaluationService,
    intent: EvaluationIntent,
    resolution: IntentResolution,
) -> None:
    service._persist_intent_targets(intent)
    owned = service._claim(intent)
    assert owned is not None
    service._publish_result_attestation(
        intent=intent,
        resolution=resolution,
        owned=owned,
    )


def _put_typed(
    store: ObjectStore,
    schema: str,
    content: Jsonable,
) -> TypedRef:
    reference, _ = store.put(schema, content)
    return TypedRef(
        schema_name=reference.schema,
        content_hash=reference.content_hash,
    )


def _load_component_traces(
    store: ObjectStore,
    evidence: EvaluationEvidence,
) -> EvaluationComponentTraces:
    return EvaluationComponentTraces.model_validate_json(
        json.dumps(store.get(evidence.component_traces_ref.reference))
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


def _successful_internal_outcome(
    request: InternalRowRequest,
) -> InternalRowOutcome:
    output_text = request.instance.gold
    prompt = render_prompt(
        env_spec(request.env_name),
        request.candidate,
        request.instance.to_instance(),
    )
    return InternalRowOutcome(
        score=1.0,
        row_state=ExecutedRowState.SUCCESS,
        executed_component_steps=(
            _llm_component_step(
                trace_index=0,
                component_id=LLM_NODE_ID,
                prompt=prompt,
                generation=output_text,
            ),
        ),
        output_text=output_text,
        finish_reason="stop",
    )


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
    component_traces = _load_component_traces(store, evidence)
    assert output_record.record_content() == store.get(
        evidence.outputs_ref.reference
    )
    assert output_record.candidate.record.candidate_id == (
        engine.experiment.initial_candidate.candidate_id
    )
    assert output_record.component_traces_ref == (
        evidence.component_traces_ref
    )
    assert component_traces.candidate == evidence.candidate
    assert component_traces.evaluation_binding == evidence.evaluation_binding
    assert component_traces.graph_hash == evidence.graph_hash
    assert tuple(
        (
            row.instance_id,
            row.task_identity,
            row.repeat,
            row.executed_component_trace.row_state.value,
        )
        for row in component_traces.rows
    ) == tuple(
        (
            row.instance_id,
            row.task_identity,
            row.repeat,
            "success",
        )
        for row in output_record.outputs
    )
    assert (
        component_traces.rows[0]
        .executed_component_trace.executed_component_steps[0]
        .outputs["generation"]
        == output_record.outputs[0].output_text
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


def test_engine_persists_missing_row_state_without_fabricated_steps(
    tmp_path,
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "missing-trace.sqlite"))
    engine = _engine(
        tmp_path,
        store=store,
        role=EvaluationRole.OFFICIAL,
        max_wall_seconds=0.0,
    )

    evaluated = engine.evaluate(
        EvaluationRequest(
            candidate=engine.experiment.initial_candidate,
            evaluation_binding=_binding(engine, role=EvaluationRole.OFFICIAL),
            purpose="missing-trace",
        )
    )

    outputs = EvaluationOutputsRecord.model_validate(
        store.get(evaluated.evidence.outputs_ref.reference)
    )
    trace = (
        _load_component_traces(store, evaluated.evidence)
        .rows[0]
        .executed_component_trace
    )
    assert outputs.outputs[0].missing
    assert trace.row_state is ExecutedRowState.MISSING
    assert trace.executed_component_steps == ()


def test_ed1_trace_persists_encoder_output_and_decoder_failure_prefix(
    tmp_path,
    monkeypatch,
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "ed1-traces.sqlite"))
    engine = _ed1_graph_engine(store=store)
    experiment = engine.experiment
    canonical_run = evaluation_engine_module.run_internal_eval
    run_count = 0
    encoder_text = "compressed description distinct from final code"

    def ed1_traced_run(*args, **kwargs):
        nonlocal run_count
        result = canonical_run(*args, **kwargs)
        original = result.outputs[0]
        instance = engine.sampling.instances[0]
        encoder_prompt = render_prompt(
            env_spec(experiment.env_name),
            kwargs["candidate"],
            instance,
        )
        encode_step = _llm_component_step(
            trace_index=0,
            component_id=ENCODER_NODE_ID,
            prompt=encoder_prompt,
            generation=encoder_text,
        )
        decoder_text = instance.gold
        decode_step = _llm_component_step(
            trace_index=1,
            component_id=DECODER_NODE_ID,
            prompt=DECODER_TEMPLATE.format(encoder_output=encoder_text),
            generation=decoder_text,
        )
        if run_count == 0:
            output = replace(
                original,
                executed_component_steps=(encode_step, decode_step),
                output_text=decoder_text,
            )
            rewritten = replace(result, outputs=(output,))
        else:
            output = replace(
                original,
                row_state=ExecutedRowState.FAILED,
                executed_component_steps=(
                    (encode_step,)
                    if run_count == 1
                    else (encode_step, decode_step)
                ),
                output_text=None if run_count == 1 else decoder_text,
                score=None,
                failure_code=(
                    "decoder_provider_failure"
                    if run_count == 1
                    else "scoring_infrastructure_failure"
                ),
                finish_reason=None,
                provider_error={"type": "provider_unavailable"},
            )
            aggregate = unweighted_task_mean(
                aggregate_name=result.aggregate.name,
                graph_hash=experiment.rollout_definition.graph_hash,
                evaluation_binding_hash=(
                    kwargs["evaluation_binding"].identity_hash()
                ),
                task_rows=(
                    TaskRows(
                        task_identity=(
                            engine.sampling.task_set.task_identities[0]
                        ),
                        rows=(RowValue(failed=True),),
                    ),
                ),
                plan=engine.sampling.evaluation_matrix_plan,
            )
            rewritten = replace(
                result,
                aggregate=aggregate,
                reward=None,
                per_task_scores=(0.0,),
                per_task_counts=(1,),
                outputs=(output,),
            )
        run_count += 1
        return rewritten

    monkeypatch.setattr(
        evaluation_engine_module,
        "run_internal_eval",
        ed1_traced_run,
    )
    service = EngineEvaluationService(store=store, engine=engine)
    successful_intent = _intent(
        engine,
        intent_id="ed1-success",
        purpose="ed1-trace",
        role=EvaluationRole.OFFICIAL,
    )
    failed_intent = _intent(
        engine,
        intent_id="ed1-decoder-failure",
        purpose="ed1-trace",
        role=EvaluationRole.OFFICIAL,
    )
    post_score_failure_intent = _intent(
        engine,
        intent_id="ed1-post-score-failure",
        purpose="ed1-trace",
        role=EvaluationRole.OFFICIAL,
    )

    successful = service.resolve_evaluation_intent(successful_intent)
    failed = service.resolve_evaluation_intent(failed_intent)
    post_score_failure = service.resolve_evaluation_intent(
        post_score_failure_intent
    )

    assert successful.evaluation_result_ref is not None
    successful_evidence = EvaluationEvidence.model_validate(
        store.get(successful.evaluation_result_ref.reference)
    )
    successful_trace = (
        _load_component_traces(store, successful_evidence)
        .rows[0]
        .executed_component_trace
    )
    assert (
        successful_trace.executed_component_steps[0].outputs["generation"]
        == encoder_text
    )
    assert (
        successful_trace.executed_component_steps[1].outputs["generation"]
        != encoder_text
    )

    assert failed.evaluation_result_ref is not None
    failed_evidence = EvaluationEvidence.model_validate(
        store.get(failed.evaluation_result_ref.reference)
    )
    failed_outputs = EvaluationOutputsRecord.model_validate(
        store.get(failed_evidence.outputs_ref.reference)
    )
    failed_trace = (
        _load_component_traces(store, failed_evidence)
        .rows[0]
        .executed_component_trace
    )
    assert failed_trace.row_state is ExecutedRowState.FAILED
    assert tuple(
        step.component_id for step in failed_trace.executed_component_steps
    ) == (ENCODER_NODE_ID,)
    assert (
        failed_trace.executed_component_steps[0].outputs["generation"]
        == encoder_text
    )
    assert failed_outputs.outputs[0].failed
    assert failed_outputs.outputs[0].output_text is None
    assert failed_outputs.outputs[0].score is None

    assert post_score_failure.evaluation_result_ref is not None
    post_score_evidence = EvaluationEvidence.model_validate(
        store.get(post_score_failure.evaluation_result_ref.reference)
    )
    post_score_outputs = EvaluationOutputsRecord.model_validate(
        store.get(post_score_evidence.outputs_ref.reference)
    )
    post_score_trace = (
        _load_component_traces(store, post_score_evidence)
        .rows[0]
        .executed_component_trace
    )
    assert post_score_trace.row_state is ExecutedRowState.FAILED
    assert tuple(
        step.component_id for step in post_score_trace.executed_component_steps
    ) == (ENCODER_NODE_ID, DECODER_NODE_ID)
    assert post_score_outputs.outputs[0].failed
    assert post_score_outputs.outputs[0].output_text == (
        engine.sampling.instances[0].gold
    )
    assert (
        post_score_trace.executed_component_steps[-1].outputs["generation"]
        == post_score_outputs.outputs[0].output_text
    )


@pytest.mark.parametrize(
    ("forgery", "expected_error"),
    (
        ("decoder_prompt", "canonical encoder-output frame"),
        ("failed_terminal_mismatch", "final output"),
        ("failed_prefix_nonnull", "nonterminal component prefix"),
    ),
)
def test_ed1_trace_relationship_forgery_fails_prebind_and_restart(
    tmp_path,
    monkeypatch,
    forgery: str,
    expected_error: str,
) -> None:
    database = tmp_path / f"ed1-{forgery}.sqlite"
    store = ObjectStore(SqliteBackend(database))
    engine = _ed1_graph_engine(store=store)
    canonical_run = evaluation_engine_module.run_internal_eval
    encoder_text = "canonical encoder output"

    def ed1_success(*args, **kwargs):
        result = canonical_run(*args, **kwargs)
        original = result.outputs[0]
        instance = engine.sampling.instances[0]
        encode_step = _llm_component_step(
            trace_index=0,
            component_id=ENCODER_NODE_ID,
            prompt=render_prompt(
                env_spec(engine.experiment.env_name),
                kwargs["candidate"],
                instance,
            ),
            generation=encoder_text,
        )
        decode_step = _llm_component_step(
            trace_index=1,
            component_id=DECODER_NODE_ID,
            prompt=DECODER_TEMPLATE.format(encoder_output=encoder_text),
            generation=instance.gold,
        )
        return replace(
            result,
            outputs=(
                replace(
                    original,
                    executed_component_steps=(encode_step, decode_step),
                    output_text=instance.gold,
                ),
            ),
        )

    monkeypatch.setattr(
        evaluation_engine_module,
        "run_internal_eval",
        ed1_success,
    )
    intent = _intent(
        engine,
        intent_id=f"ed1-{forgery}",
        purpose="ed1-relationship-forgery",
        role=EvaluationRole.OFFICIAL,
    )
    evaluated = engine.evaluate(
        EvaluationRequest(
            candidate=intent.candidate.record,
            evaluation_binding=intent.evaluation_binding,
            purpose=intent.purpose,
        )
    )
    trace_content = _load_component_traces(
        store, evaluated.evidence
    ).record_content()
    trace_payload = trace_content["rows"][0]["executed_component_trace"]
    output_content = store.get(evaluated.evidence.outputs_ref.reference)
    assert isinstance(output_content, dict)
    output_rows = output_content["outputs"]
    assert isinstance(output_rows, list)
    output_row = output_rows[0]
    assert isinstance(output_row, dict)
    if forgery == "decoder_prompt":
        trace_payload["executed_component_steps"][1]["inputs"]["prompt"] = (
            "forged decoder frame"
        )
    else:
        trace_payload["row_state"] = "failed"
        output_row.update(
            {
                "score": None,
                "failed": True,
                "failure_code": "post_execution_failure",
                "provider_error": {"type": "infrastructure"},
            }
        )
        if forgery == "failed_terminal_mismatch":
            output_row["output_text"] = "not the accepted decoder generation"
        else:
            trace_payload["executed_component_steps"] = trace_payload[
                "executed_component_steps"
            ][:1]
            output_row["output_text"] = "fabricated final output"

    trace_ref = _put_typed(
        store,
        EVALUATION_COMPONENT_TRACES_SCHEMA,
        trace_content,
    )
    output_content["component_traces_ref"] = trace_ref.model_dump(mode="json")
    outputs_ref = _put_typed(
        store,
        EVALUATION_OUTPUTS_SCHEMA,
        output_content,
    )
    forged_evidence = evaluated.evidence.model_copy(
        update={
            "component_traces_ref": trace_ref,
            "outputs_ref": outputs_ref,
        }
    )
    evidence_ref = _put_typed(
        store,
        EVALUATION_EVIDENCE_SCHEMA,
        forged_evidence.record_content(),
    )
    forged_resolution = _completed_resolution(intent, evaluated).model_copy(
        update={"evaluation_result_ref": evidence_ref}
    )
    service = EngineEvaluationService(store=store, engine=engine)
    service._persist_intent_targets(intent)

    with pytest.raises(ValueError, match=expected_error):
        service._validate_result_graph(
            forged_resolution,
            expected_intent=intent,
            require_attestation=False,
        )

    _bind_with_forged_terminal_attestation(
        store=store,
        service=service,
        intent=intent,
        resolution=forged_resolution,
    )
    restart_store = ObjectStore(SqliteBackend(database))
    restart_engine = _ed1_graph_engine(store=restart_store)
    with pytest.raises(ValueError, match=expected_error):
        EngineEvaluationService(
            store=restart_store,
            engine=restart_engine,
        ).resolve_evaluation_intent(intent)


@pytest.mark.parametrize(
    ("forgery", "expected_error"),
    (
        ("failed_terminal_mismatch", "final output"),
        ("failed_prefix_nonnull", "nonterminal component prefix"),
    ),
)
def test_one_step_trace_relationship_forgery_fails_prebind_and_restart(
    tmp_path,
    forgery: str,
    expected_error: str,
) -> None:
    database = tmp_path / f"one-step-{forgery}.sqlite"
    store = ObjectStore(SqliteBackend(database))
    engine = _engine(tmp_path, store=store, role=EvaluationRole.OFFICIAL)
    intent = _intent(
        engine,
        intent_id=f"one-step-{forgery}",
        purpose="one-step-relationship-forgery",
        role=EvaluationRole.OFFICIAL,
    )
    evaluated = engine.evaluate(
        EvaluationRequest(
            candidate=intent.candidate.record,
            evaluation_binding=intent.evaluation_binding,
            purpose=intent.purpose,
        )
    )
    trace_content = _load_component_traces(
        store, evaluated.evidence
    ).record_content()
    trace_payload = trace_content["rows"][0]["executed_component_trace"]
    trace_payload["row_state"] = "failed"
    output_content = store.get(evaluated.evidence.outputs_ref.reference)
    assert isinstance(output_content, dict)
    output_rows = output_content["outputs"]
    assert isinstance(output_rows, list)
    output_row = output_rows[0]
    assert isinstance(output_row, dict)
    output_row.update(
        {
            "score": None,
            "failed": True,
            "failure_code": "post_execution_failure",
            "provider_error": {"type": "infrastructure"},
        }
    )
    if forgery == "failed_terminal_mismatch":
        output_row["output_text"] = "not the accepted generation"
    else:
        trace_payload["executed_component_steps"] = []
        output_row["output_text"] = "fabricated final output"

    trace_ref = _put_typed(
        store,
        EVALUATION_COMPONENT_TRACES_SCHEMA,
        trace_content,
    )
    output_content["component_traces_ref"] = trace_ref.model_dump(mode="json")
    outputs_ref = _put_typed(
        store,
        EVALUATION_OUTPUTS_SCHEMA,
        output_content,
    )
    forged_evidence = evaluated.evidence.model_copy(
        update={
            "component_traces_ref": trace_ref,
            "outputs_ref": outputs_ref,
        }
    )
    evidence_ref = _put_typed(
        store,
        EVALUATION_EVIDENCE_SCHEMA,
        forged_evidence.record_content(),
    )
    forged_resolution = _completed_resolution(intent, evaluated).model_copy(
        update={"evaluation_result_ref": evidence_ref}
    )
    service = EngineEvaluationService(store=store, engine=engine)
    service._persist_intent_targets(intent)
    with pytest.raises(ValueError, match=expected_error):
        service._validate_result_graph(
            forged_resolution,
            expected_intent=intent,
            require_attestation=False,
        )

    _bind_with_forged_terminal_attestation(
        store=store,
        service=service,
        intent=intent,
        resolution=forged_resolution,
    )
    restart_store = ObjectStore(SqliteBackend(database))
    restart_engine = _engine(
        tmp_path,
        store=restart_store,
        role=EvaluationRole.OFFICIAL,
    )
    with pytest.raises(ValueError, match=expected_error):
        EngineEvaluationService(
            store=restart_store,
            engine=restart_engine,
        ).resolve_evaluation_intent(intent)


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
            outcome=_successful_internal_outcome(request),
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


def test_engine_rejects_another_provider_execution_policy(tmp_path) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "provider-policy.sqlite"))
    engine = _engine(tmp_path, store=store)
    binding = _binding(engine).model_copy(
        update={
            "provider_execution_policy_ref": TypedRef(
                schema_name="whetstone.provider_execution_policy",
                content_hash="f" * 64,
            )
        }
    )

    with pytest.raises(ValueError, match="exact Provider Execution Policy"):
        engine.evaluate(
            EvaluationRequest(
                candidate=engine.experiment.initial_candidate,
                evaluation_binding=binding,
                purpose="provider-policy",
            )
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


def test_evaluation_outputs_wire_contract_is_exact(tmp_path) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "outputs-wire.sqlite"))
    engine = _engine(tmp_path, store=store)
    candidate = engine.experiment.initial_candidate
    candidate_ref = candidate_reference(candidate)
    binding = _binding(engine)
    component_traces_ref = TypedRef(
        schema_name=EVALUATION_COMPONENT_TRACES_SCHEMA,
        content_hash="a" * 64,
    )
    record = EvaluationOutputsRecord(
        schema_version=EVALUATION_OUTPUTS_SCHEMA_VERSION,
        candidate=candidate_ref,
        evaluation_binding=binding,
        evaluation_role=EvaluationRole.INTERNAL,
        graph_hash=engine.experiment.rollout_definition.graph_hash,
        purpose="wire-contract",
        split_role=engine.sampling.split_role,
        task_identities=("task-1",),
        repeat_count=1,
        component_traces_ref=component_traces_ref,
        outputs=(
            EvaluationOutputRow(
                candidate_id=candidate.candidate_id,
                instance_id="instance-1",
                task_identity="task-1",
                repeat=0,
                rendered_prompt="Question?",
                output_text="Answer.",
                score=1.0,
                failed=False,
                missing=False,
                invalid=False,
                failure_code="",
                finish_reason="stop",
                provider_error=None,
                max_budget=100,
                over_budget=False,
            ),
        ),
    )

    assert record.record_content() == {
        "schema_version": 2,
        "candidate": candidate_ref.model_dump(mode="json"),
        "evaluation_binding": binding.model_dump(mode="json"),
        "evaluation_role": "internal",
        "graph_hash": engine.experiment.rollout_definition.graph_hash,
        "purpose": "wire-contract",
        "split_role": "internal_eval",
        "task_identities": ["task-1"],
        "repeat_count": 1,
        "component_traces_ref": component_traces_ref.model_dump(mode="json"),
        "outputs": [
            {
                "candidate_id": candidate.candidate_id,
                "instance_id": "instance-1",
                "task_identity": "task-1",
                "repeat": 0,
                "rendered_prompt": "Question?",
                "output_text": "Answer.",
                "score": 1.0,
                "failed": False,
                "missing": False,
                "invalid": False,
                "failure_code": "",
                "finish_reason": "stop",
                "provider_error": None,
                "max_budget": 100,
                "over_budget": False,
            }
        ],
    }


def test_component_trace_and_evidence_versions_are_exact(tmp_path) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "trace-wire.sqlite"))
    engine = _engine(tmp_path, store=store)
    evaluated = engine.evaluate(
        EvaluationRequest(
            candidate=engine.experiment.initial_candidate,
            evaluation_binding=_binding(engine),
            purpose="trace-wire",
        )
    )
    evidence = evaluated.evidence
    outputs = EvaluationOutputsRecord.model_validate(
        store.get(evidence.outputs_ref.reference)
    )
    traces = _load_component_traces(store, evidence)
    trace_content = traces.record_content()

    assert EVALUATION_COMPONENT_TRACES_SCHEMA == (
        "whetstone.evaluation_component_traces"
    )
    assert EVALUATION_COMPONENT_TRACES_SCHEMA_VERSION == 1
    assert EVALUATION_OUTPUTS_SCHEMA_VERSION == 2
    assert EVALUATION_EVIDENCE_SCHEMA_VERSION == 2
    assert traces.schema_version == 1
    assert outputs.schema_version == 2
    assert evidence.schema_version == 2
    assert evidence.component_traces_ref.content_hash == (
        "42b08acde3e5aa8a0923e960d5db91fffee150cf2fbc0331d9522747af708942"
    )
    assert evidence.outputs_ref.content_hash == (
        "19ff32c9d9b7ed8d6c70d2d14a725c6887a38f57393f9e8338d3109d3994fc14"
    )
    assert evaluated.evidence_ref.content_hash == (
        "211c24740d389e1a13e8b720cb2e56c8a422c9bf0b8c1827a7f59adc53e82fa9"
    )
    with pytest.raises(ValueError, match="address the exact record"):
        EvaluationComponentTracesRef(
            record=traces,
            record_ref=TypedRef(
                schema_name=EVALUATION_COMPONENT_TRACES_SCHEMA,
                content_hash="f" * 64,
            ),
        )
    assert tuple(trace_content) == (
        "schema_version",
        "candidate",
        "evaluation_binding",
        "evaluation_role",
        "graph_hash",
        "purpose",
        "split_role",
        "task_identities",
        "repeat_count",
        "rows",
    )
    assert tuple(trace_content["rows"][0]) == (
        "instance_id",
        "task_identity",
        "repeat",
        "executed_component_trace",
    )
    assert traces.rows[0].executed_component_trace.model_dump(mode="json") == {
        "row_state": "success",
        "executed_component_steps": [
            traces.rows[0]
            .executed_component_trace.executed_component_steps[0]
            .model_dump(mode="json")
        ],
    }


@pytest.mark.parametrize(
    ("record_name", "wrong_version"),
    (
        ("traces", 0),
        ("outputs", 1),
        ("evidence", 1),
    ),
)
def test_evaluation_artifacts_reject_prior_wire_versions(
    tmp_path,
    record_name: str,
    wrong_version: int,
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / f"{record_name}-v1.sqlite"))
    engine = _engine(tmp_path, store=store)
    evidence = engine.evaluate(
        EvaluationRequest(
            candidate=engine.experiment.initial_candidate,
            evaluation_binding=_binding(engine),
            purpose="hard-cut",
        )
    ).evidence
    if record_name == "traces":
        content = store.get(evidence.component_traces_ref.reference)
        model = EvaluationComponentTraces
    elif record_name == "outputs":
        content = store.get(evidence.outputs_ref.reference)
        model = EvaluationOutputsRecord
    else:
        content = evidence.record_content()
        model = EvaluationEvidence
    assert isinstance(content, dict)
    content["schema_version"] = wrong_version

    with pytest.raises(ValueError, match="schema_version"):
        if model is EvaluationComponentTraces:
            model.model_validate_json(json.dumps(content))
        else:
            model.model_validate(content)


def test_evaluation_outputs_reject_candidate_mismatch(tmp_path) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "outputs-mismatch.sqlite"))
    engine = _engine(tmp_path, store=store)
    row = EvaluationOutputRow(
        candidate_id="other",
        instance_id="instance-1",
        task_identity="task-1",
        repeat=0,
        rendered_prompt="Question?",
        output_text="Answer.",
        score=1.0,
        failed=False,
        missing=False,
        invalid=False,
        failure_code="",
        finish_reason="stop",
        provider_error=None,
        max_budget=None,
        over_budget=None,
    )

    with pytest.raises(ValueError, match="candidate_id must match"):
        EvaluationOutputsRecord(
            schema_version=EVALUATION_OUTPUTS_SCHEMA_VERSION,
            candidate=candidate_reference(engine.experiment.initial_candidate),
            evaluation_binding=_binding(engine),
            evaluation_role=EvaluationRole.INTERNAL,
            graph_hash=engine.experiment.rollout_definition.graph_hash,
            purpose="mismatch",
            split_role=engine.sampling.split_role,
            task_identities=("task-1",),
            repeat_count=1,
            component_traces_ref=TypedRef(
                schema_name=EVALUATION_COMPONENT_TRACES_SCHEMA,
                content_hash="a" * 64,
            ),
            outputs=(row,),
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
        engine.experiment.rollout_definition.graph_hash
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


def test_exact_evaluation_result_refs_reject_forged_hashes(
    tmp_path,
    monkeypatch,
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "exact-ref.sqlite"))
    engine = _engine(tmp_path, store=store)
    evaluated = engine.evaluate(
        EvaluationRequest(
            candidate=engine.experiment.initial_candidate,
            evaluation_binding=_binding(engine),
            purpose="exact-ref",
        )
    )
    forged = TypedRef(
        schema_name=EVALUATION_EVIDENCE_SCHEMA,
        content_hash="f" * 64,
    )

    with pytest.raises(ValueError, match="address the exact record"):
        EvaluationEvidenceRef(
            record=evaluated.evidence,
            record_ref=forged,
        )

    intent = _intent(engine, intent_id="failure-ref", purpose="failure-ref")
    service = EngineEvaluationService(store=store, engine=engine)

    def fail(_request: EvaluationRequest) -> EngineEvaluation:
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(engine, "evaluate", fail)
    failed = service.resolve_evaluation_intent(intent)
    assert failed.evaluation_result_ref is not None
    failure = EvaluationFailureEvidenceRef.model_validate(
        {
            "record": store.get(failed.evaluation_result_ref.reference),
            "record_ref": failed.evaluation_result_ref,
        }
    )
    with pytest.raises(ValueError, match="address the exact record"):
        EvaluationFailureEvidenceRef(
            record=failure.record,
            record_ref=TypedRef(
                schema_name=failed.evaluation_result_ref.schema_name,
                content_hash="e" * 64,
            ),
        )


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
        "failed": False,
        "missing": False,
        "invalid": False,
        "failure_code": "",
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
        evidence_update["dataset_identity"] = "forged-dataset"
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
            outputs_content["task_identities"] = ["forged-task"]
            outputs_content["outputs"][0]["task_identity"] = "forged-task"
        elif forgery == "output_repeat":
            outputs_content["repeat_count"] = 2
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
    instance = engine.sampling.instances[0]
    rewritten_text = "not the expected answer"
    rewritten_score = float(
        env_exact_match_score(
            env=env_spec(engine.experiment.env_name),
            generation=rewritten_text,
            gold=instance.gold,
            evaluation_procedure_config_hash=(
                engine.experiment.rollout_definition.procedure_config_hash
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
        graph_hash=engine.experiment.rollout_definition.graph_hash,
        evaluation_binding_hash=intent.evaluation_binding.identity_hash(),
        task_rows=(
            TaskRows(
                task_identity=original_row.task_identity,
                rows=(RowValue(value=rewritten_score),),
            ),
        ),
        plan=engine.sampling.evaluation_matrix_plan,
    )
    rewritten_aggregate_ref = _put_typed(
        store,
        ROLLOUT_AGGREGATE_SCHEMA,
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
            if request.repeat_index == 0
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
                request_identity=request.request_identity,
                outcome=outcome,
            ).model_dump(mode="json"),
        )

    engine = _engine(
        tmp_path,
        store=store,
        row_job_factory=one_success_one_failure,
        repeats=2,
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
                budget_delta=BudgetDelta(consumed={"rollouts": 1}),
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


def test_restart_rejects_aggregate_from_another_rollout_graph(
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
    with pytest.raises(ValueError, match="another rollout graph"):
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
    cached_outputs = EvaluationOutputsRecord.model_validate(
        store.get(result.evidence.outputs_ref.reference)
    )
    assert cached_outputs.component_traces_ref == (
        result.evidence.component_traces_ref
    )
    assert (
        _load_component_traces(store, result.evidence)
        .rows[0]
        .executed_component_trace.executed_component_steps
    )


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
