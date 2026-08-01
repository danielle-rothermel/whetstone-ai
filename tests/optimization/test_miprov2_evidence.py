from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import Any, cast

import pytest
from dr_code.eval import (
    DefinitionRef,
    EvalConfig,
    RepeatPlan,
    SamplingDefinition,
    TaskSet,
)
from dr_code.eval.identity import SCHEMA_EVAL_CONFIG, identity_hash_for
from dr_serialize import Jsonable
from dr_store import ObjectStore, SqliteBackend

from tests.optimization.support import candidate
from whetstone.envs.internal_eval import (
    ExecutedComponentStep,
    ExecutedComponentTracePayload,
    ExecutedRowState,
)
from whetstone.evaluation.schema import (
    EVALUATION_COMPONENT_TRACES_SCHEMA,
    EVALUATION_OUTPUTS_SCHEMA,
    EvaluationComponentTraceRow,
    EvaluationComponentTraces,
    EvaluationEvidence,
    EvaluationFailureEvidence,
    RowAccounting,
)
from whetstone.evaluation_role import EvaluationRole
from whetstone.optimization.identity import (
    ImmutableJsonObject,
    TerminalFailure,
    TypedRef,
)
from whetstone.optimization.miprov2_bootstrap import BootstrapAttemptPlan
from whetstone.optimization.miprov2_eval_config import (
    Miprov2EvalConfigBinding,
    Miprov2EvalConfigBindingRequest,
    Miprov2EvaluationExecutionPolicy,
    derive_eval_config_reference,
)
from whetstone.optimization.miprov2_evidence import (
    Miprov2EvidenceResolver,
    Miprov2IntentContext,
    persist_miprov2_intent_context,
)
from whetstone.optimization.miprov2_rng import Miprov2DurableBindings
from whetstone.optimization.reward import (
    MissingDataPolicy,
    RewardPolicy,
    RewardTerm,
    apply_reward_policy,
    reward_reference,
)
from whetstone.optimization.schema import (
    EVALUATION_EVIDENCE_SCHEMA,
    EVALUATION_FAILURE_SCHEMA,
    EvaluationBinding,
    EvaluationIntent,
    IntentOutcome,
    IntentResolution,
    ResolutionClass,
    ResolutionDetail,
    candidate_reference,
    eval_config_reference,
)

FULL_A = "a" * 64
FULL_B = "b" * 64
FULL_C = "c" * 64
FULL_D = "d" * 64
TASK = hashlib.sha256(b"miprov2-evidence-task").hexdigest()


def _put(store: ObjectStore, schema: str, record: Jsonable) -> TypedRef:
    ref, _ = store.put(schema, record)
    return TypedRef(schema_name=ref.schema, content_hash=ref.content_hash)


def _source_eval_config():
    definition = DefinitionRef(
        definition_id="miprov2-evidence",
        version="1",
        schema_name="dr_code.eval_definition",
        identity_hash=FULL_A,
    )
    identity = identity_hash_for(
        schema=SCHEMA_EVAL_CONFIG,
        payload={
            "definition_identity": FULL_A,
            "sampling_config": FULL_B,
            "evaluation_procedure_config": FULL_C,
            "aggregation_config": FULL_D,
        },
    )
    return eval_config_reference(
        EvalConfig(
            definition_ref=definition,
            sampling_config_hash=FULL_B,
            evaluation_procedure_config_hash=FULL_C,
            aggregation_config_hash=FULL_D,
            config_identity_hash=identity,
        )
    )


def _bindings() -> Miprov2DurableBindings:
    return Miprov2DurableBindings(
        control_identity_hash=FULL_A,
        prompt_route_identity_hash=FULL_B,
        task_route_identity_hash=FULL_C,
        execution_policy_identity_hash=FULL_D,
        prompt_adapter_identity_hash=FULL_A,
        base_candidate_identity_hash=FULL_B,
        teacher_candidate_identity_hash=FULL_C,
    )


def _fixture(store: ObjectStore, *, reward_policy_hash: str):
    attempt = BootstrapAttemptPlan(
        bindings=_bindings(),
        plan_identity_hash=FULL_D,
        task_index=0,
        task_identity=TASK,
        round_index=0,
        copy_task_model=False,
        rollout_id=None,
        temperature=None,
    )
    policy = Miprov2EvaluationExecutionPolicy(
        num_threads=1,
        max_errors=1,
        provide_traceback=None,
        task_model_identity_hash=FULL_C,
        provider_execution_policy_hash=FULL_D,
    )
    request = Miprov2EvalConfigBindingRequest(
        control_identity_hash=FULL_A,
        source_eval_config=_source_eval_config(),
        purpose="bootstrap",
        effect_identity_hash=attempt.identity_hash(),
        execution_policy=policy,
        task_batch_identities=(TASK,),
    )
    task_set = TaskSet(
        manifest_id="miprov2-evidence-tasks",
        version="1",
        dataset_revision="test",
        task_identities=(TASK,),
    )
    repeat_plan = RepeatPlan(
        plan_id="miprov2-evidence-repeats",
        version="1",
        task_identities=(TASK,),
        repeat_count=1,
    )
    sampling = SamplingDefinition(
        definition_id="miprov2-evidence-sampling",
        version="1",
    ).materialize(
        {
            "task_set_hash": task_set.identity_hash(),
            "repeat_plan_hash": repeat_plan.identity_hash(),
        }
    )
    eval_binding = Miprov2EvalConfigBinding(
        request=request,
        task_set=task_set,
        repeat_plan=repeat_plan,
        sampling_config=sampling,
        eval_config=derive_eval_config_reference(
            request.source_eval_config,
            sampling,
        ),
    )
    exact_binding = EvaluationBinding(
        schema_version=2,
        eval_config=eval_binding.eval_config,
        role=EvaluationRole.INTERNAL,
        campaign="miprov2-evidence",
    )
    candidate_ref = candidate_reference(
        candidate("teacher", text="Encode {query}.")
    )
    intent = EvaluationIntent(
        intent_id="run:miprov2:bootstrap:evidence",
        candidate=candidate_ref,
        target_eval_config=eval_binding.eval_config,
        evaluation_binding=exact_binding,
        purpose="miprov2_bootstrap",
        run_id="run",
        step_index=0,
        expected_reward_policy_hash=reward_policy_hash,
    )
    context = Miprov2IntentContext(
        control_identity_hash=FULL_A,
        run_id="run",
        effect_kind="bootstrap",
        effect_identity_hash=attempt.identity_hash(),
        intent_id=intent.intent_id,
        candidate=candidate_ref,
        task_batch_identities=(TASK,),
        eval_config=eval_binding.eval_config,
        eval_config_binding=eval_binding,
        evaluation_binding=exact_binding,
        execution_policy=policy,
        reward_policy_hash=reward_policy_hash,
        bootstrap_attempt=attempt,
        optimizable_component_id="encode",
        optimizable_trace_index=0,
    )
    persist_miprov2_intent_context(store, context)
    assert context.schema_version == 2
    return intent, context


def _resolution(store: ObjectStore, *, final_text: str = "decoded-final"):
    policy = RewardPolicy(
        policy_name="miprov2-evidence/v1",
        reward_name="score",
        terms=(RewardTerm(name="score", weight=1.0),),
        missing_data=MissingDataPolicy.FAIL,
    )
    intent, _ = _fixture(
        store,
        reward_policy_hash=policy.identity_hash(),
    )
    steps = (
        ExecutedComponentStep(
            trace_index=0,
            component_id="encode",
            input_field_names=("prompt",),
            output_field_names=("generation",),
            inputs=ImmutableJsonObject({"prompt": "encoder prompt"}),
            outputs=ImmutableJsonObject(
                {"generation": "exact encoder output"}
            ),
        ),
        ExecutedComponentStep(
            trace_index=1,
            component_id="decode",
            input_field_names=("prompt",),
            output_field_names=("generation",),
            inputs=ImmutableJsonObject({"prompt": "decoder prompt"}),
            outputs=ImmutableJsonObject({"generation": final_text}),
        ),
    )
    traces = EvaluationComponentTraces(
        schema_version=1,
        candidate=intent.candidate,
        evaluation_binding=intent.evaluation_binding,
        evaluation_role=EvaluationRole.INTERNAL,
        graph_hash=FULL_B,
        purpose=intent.purpose,
        split_role="internal",
        task_identities=(TASK,),
        repeat_count=1,
        rows=(
            EvaluationComponentTraceRow(
                instance_id="instance-1",
                task_identity=TASK,
                repeat=0,
                executed_component_trace=ExecutedComponentTracePayload(
                    row_state=ExecutedRowState.SUCCESS,
                    executed_component_steps=steps,
                ),
            ),
        ),
    )
    traces_ref = _put(
        store,
        EVALUATION_COMPONENT_TRACES_SCHEMA,
        traces.record_content(),
    )
    outputs_ref = _put(
        store, EVALUATION_OUTPUTS_SCHEMA, {"display": final_text}
    )
    aggregate_ref = _put(store, "whetstone.rollout_aggregate", {"score": 0.8})
    reward = apply_reward_policy(
        policy,
        aggregates={"score": 0.8},
        evidence_role=EvaluationRole.INTERNAL,
        evidence_refs=(aggregate_ref,),
    )
    reward_ref = reward_reference(reward)
    _put(store, reward_ref.record_ref.schema_name, reward.record_content())
    evidence = EvaluationEvidence(
        schema_version=2,
        candidate=intent.candidate,
        evaluation_binding=intent.evaluation_binding,
        graph_hash=FULL_B,
        graph_config_ref="graph://miprov2-evidence",
        purpose=intent.purpose,
        dataset_identity="dataset-revision",
        task_identities=(TASK,),
        repeat_count=1,
        per_task_values=(0.8,),
        per_task_counts=(1,),
        row_accounting=RowAccounting(
            planned=1,
            present=1,
            missing=0,
            failed=0,
            invalid=0,
        ),
        component_traces_ref=traces_ref,
        outputs_ref=outputs_ref,
        aggregate_ref=aggregate_ref,
        aggregate_name="score",
        aggregate_value=0.8,
        aggregate_status="ok",
        reward_ref=reward_ref,
    )
    evidence_ref = _put(
        store,
        EVALUATION_EVIDENCE_SCHEMA,
        evidence.record_content(),
    )
    return IntentResolution(
        schema_version=2,
        intent=intent,
        outcome=IntentOutcome.COMPLETED,
        detail=ResolutionDetail(
            classification=ResolutionClass.MEASURED,
            message="measured",
        ),
        evaluation_result_ref=evidence_ref,
        reward_evidence_refs=reward.evidence_refs,
        resolved_eval_config=intent.target_eval_config,
        reward_ref=reward_ref,
    )


def _failed_resolution(store: ObjectStore) -> IntentResolution:
    completed = _resolution(store)
    failure = EvaluationFailureEvidence(
        candidate=completed.intent.candidate,
        evaluation_binding=completed.intent.evaluation_binding,
        purpose=completed.intent.purpose,
        exception_type="ProviderError",
        message="provider failed",
    )
    failure_ref = _put(
        store,
        EVALUATION_FAILURE_SCHEMA,
        failure.record_content(),
    )
    return IntentResolution(
        schema_version=2,
        intent=completed.intent,
        outcome=IntentOutcome.FAILED,
        detail=ResolutionDetail(
            classification=ResolutionClass.PROVIDER,
            message="provider failed",
        ),
        evaluation_result_ref=failure_ref,
        resolved_eval_config=completed.resolved_eval_config,
        terminal_failure=TerminalFailure(
            code="evaluation_failed",
            message="provider failed",
        ),
    )


def _replace_trace(
    store: ObjectStore,
    resolution: IntentResolution,
    trace_record: dict[str, Jsonable],
    *,
    evidence_updates: dict[str, object] | None = None,
) -> IntentResolution:
    trace_ref = _put(
        store,
        EVALUATION_COMPONENT_TRACES_SCHEMA,
        trace_record,
    )
    assert resolution.evaluation_result_ref is not None
    evidence = EvaluationEvidence.model_validate(
        store.get(resolution.evaluation_result_ref.reference)
    ).model_dump(mode="json")
    evidence["component_traces_ref"] = trace_ref.model_dump(mode="json")
    if evidence_updates:
        evidence.update(evidence_updates)
    evidence_ref = _put(store, EVALUATION_EVIDENCE_SCHEMA, evidence)
    payload = resolution.model_dump(mode="json")
    payload["evaluation_result_ref"] = evidence_ref.model_dump(mode="json")
    return IntentResolution.model_validate(payload)


def _trace_record(
    store: ObjectStore, resolution: IntentResolution
) -> tuple[TypedRef, dict[str, Any]]:
    assert resolution.evaluation_result_ref is not None
    evidence = EvaluationEvidence.model_validate(
        store.get(resolution.evaluation_result_ref.reference)
    )
    trace_ref = evidence.component_traces_ref
    trace = deepcopy(store.get(trace_ref.reference))
    if not isinstance(trace, dict):
        raise AssertionError("stored component traces must be a JSON object")
    return trace_ref, cast(dict[str, Any], trace)


def test_ed1_bootstrap_uses_exact_encoder_step_not_decoder_or_display(
    tmp_path,
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "miprov2-evidence.sqlite"))
    resolution = _resolution(store, final_text="different decoded display")

    result = Miprov2EvidenceResolver(store).resolve_bootstrap(resolution)

    assert len(result.trace_steps) == 1
    selected = result.trace_steps[0]
    assert selected.component_id == "encode"
    assert selected.inputs == {"prompt": "encoder prompt"}
    assert selected.outputs == {"generation": "exact encoder output"}
    assert "different decoded display" not in str(selected.model_dump())
    assert result.source_output_identity == (
        "f9e2592008aebdf3fa64f8cdfc62a296136f5abe67a2f163b51d91fac26478e0"
    )


def test_empty_trace_cannot_synthesize_demo_from_display_output(
    tmp_path,
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "empty-trace.sqlite"))
    resolution = _resolution(store, final_text="tempting display fallback")
    _, record = _trace_record(store, resolution)
    record["rows"][0]["executed_component_trace"][
        "executed_component_steps"
    ] = []
    exact = _replace_trace(store, resolution, record)

    with pytest.raises(ValueError, match="exactly once"):
        Miprov2EvidenceResolver(store).resolve_bootstrap(exact)


def test_missing_exact_trace_artifact_fails_without_scan_or_redrive(
    tmp_path,
    monkeypatch,
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "missing-trace.sqlite"))
    resolution = _resolution(store)
    trace_ref, _ = _trace_record(store, resolution)
    canonical_get = store.get

    def missing(reference):
        if reference == trace_ref.reference:
            raise KeyError("exact trace artifact missing")
        return canonical_get(reference)

    monkeypatch.setattr(store, "get", missing)

    with pytest.raises(KeyError, match="exact trace artifact missing"):
        Miprov2EvidenceResolver(store).resolve_bootstrap(resolution)


def test_foreign_trace_candidate_fails_exact_evidence_match(tmp_path) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "foreign-trace.sqlite"))
    resolution = _resolution(store)
    _, record = _trace_record(store, resolution)
    record["candidate"] = candidate_reference(
        candidate("foreign", text="Foreign {query}.")
    ).model_dump(mode="json")
    exact = _replace_trace(store, resolution, record)

    with pytest.raises(ValueError, match="conflict"):
        Miprov2EvidenceResolver(store).resolve_bootstrap(exact)


def test_foreign_intent_binding_fails_persisted_context_match(
    tmp_path,
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "foreign-binding.sqlite"))
    resolution = _resolution(store)
    payload = resolution.model_dump(mode="json")
    payload["intent"]["evaluation_binding"]["campaign"] = "foreign"
    foreign = IntentResolution.model_validate(payload)

    with pytest.raises(ValueError, match="persisted context"):
        Miprov2EvidenceResolver(store).resolve_bootstrap(foreign)


def test_foreign_trace_split_fails_exact_graph_context(tmp_path) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "foreign-split.sqlite"))
    resolution = _resolution(store)
    _, record = _trace_record(store, resolution)
    record["split_role"] = "validation"
    exact = _replace_trace(store, resolution, record)

    with pytest.raises(ValueError, match="conflict"):
        Miprov2EvidenceResolver(store).resolve_bootstrap(exact)


def test_reordered_trace_indexes_fail_strict_trace_validation(
    tmp_path,
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "reordered-trace.sqlite"))
    resolution = _resolution(store)
    _, record = _trace_record(store, resolution)
    steps = record["rows"][0]["executed_component_trace"][
        "executed_component_steps"
    ]
    steps[0]["trace_index"] = 1
    steps[1]["trace_index"] = 0
    exact = _replace_trace(store, resolution, record)

    with pytest.raises(ValueError, match="contiguous"):
        Miprov2EvidenceResolver(store).resolve_bootstrap(exact)


def test_tampered_trace_bytes_fail_content_reference(
    tmp_path, monkeypatch
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "tampered-trace.sqlite"))
    resolution = _resolution(store)
    trace_ref, record = _trace_record(store, resolution)
    record["rows"][0]["executed_component_trace"]["executed_component_steps"][
        0
    ]["outputs"]["generation"] = "tampered"
    canonical_get = store.get

    def tampered(reference):
        if reference == trace_ref.reference:
            return record
        return canonical_get(reference)

    monkeypatch.setattr(store, "get", tampered)

    with pytest.raises(ValueError, match="address"):
        Miprov2EvidenceResolver(store).resolve_bootstrap(resolution)


def test_decoder_failed_prefix_is_audit_only_not_a_demo(tmp_path) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "decoder-failed.sqlite"))
    resolution = _resolution(store)
    _, record = _trace_record(store, resolution)
    payload = record["rows"][0]["executed_component_trace"]
    payload["row_state"] = "failed"
    payload["executed_component_steps"] = payload["executed_component_steps"][
        :1
    ]
    exact = _replace_trace(
        store,
        resolution,
        record,
        evidence_updates={
            "row_accounting": {
                "planned": 1,
                "present": 0,
                "missing": 0,
                "failed": 1,
                "invalid": 0,
            }
        },
    )

    with pytest.raises(ValueError, match="successful row"):
        Miprov2EvidenceResolver(store).resolve_bootstrap(exact)


def test_bootstrap_failure_uses_exact_failure_ref_without_reward(
    tmp_path,
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "bootstrap-failure.sqlite"))
    resolution = _failed_resolution(store)

    result = Miprov2EvidenceResolver(store).resolve_bootstrap_failure(
        resolution
    )

    assert result.metric_present is False
    assert result.score is None
    assert result.trace_steps == ()
    assert resolution.evaluation_result_ref is not None
    assert result.source_rollout_identity == (
        resolution.evaluation_result_ref.content_hash
    )


def test_bootstrap_rejection_cannot_masquerade_as_executed_failure(
    tmp_path,
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "bootstrap-rejected.sqlite"))
    completed = _resolution(store)
    rejected = IntentResolution(
        schema_version=2,
        intent=completed.intent,
        outcome=IntentOutcome.REJECTED,
        detail=ResolutionDetail(
            classification=ResolutionClass.VALIDATION,
            message="rejected before execution",
        ),
        resolved_eval_config=completed.resolved_eval_config,
    )

    with pytest.raises(ValueError, match="FAILED outcome"):
        Miprov2EvidenceResolver(store).resolve_bootstrap_failure(rejected)


def test_restart_resolves_the_same_exact_trace_and_step_refs(tmp_path) -> None:
    database = tmp_path / "restart.sqlite"
    first_store = ObjectStore(SqliteBackend(database))
    resolution = _resolution(first_store)
    first = Miprov2EvidenceResolver(first_store).resolve_bootstrap(resolution)

    restarted = Miprov2EvidenceResolver(
        ObjectStore(SqliteBackend(database))
    ).resolve_bootstrap(
        IntentResolution.model_validate_json(resolution.model_dump_json())
    )

    assert restarted == first
