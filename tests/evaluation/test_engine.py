from __future__ import annotations

import json
from dataclasses import replace

import pytest
from dr_store import (
    ObjectStore,
    SqliteBackend,
)

from tests.envs.support import (
    execution_policy,
    process_row_job_factory,
)
from tests.evaluation.support import (
    _DEFAULT_ROW_JOB_FACTORY,
    _binding,
    _engine,
    _experiment,
    _intent,
    _load_component_traces,
    _successful_internal_outcome,
    _uncached_experiment,
)
from whetstone.coordination.evaluation_service import EngineEvaluationService
from whetstone.core.identity import (
    TypedRef,
)
from whetstone.core.roles import EvaluationRole
from whetstone.envs.code_comp.constants import (
    CODE_COMP_ENV_NAME,
    DECODER_TEMPLATE,
)
from whetstone.envs.code_comp.generation_graph.encdec import (
    DECODER_NODE_ID,
    ENCODER_NODE_ID,
    build_encdec_generation_graph,
    build_encoder_provider_call_config,
)
from whetstone.envs.generation_graph import render_prompt
from whetstone.envs.registry import env_spec
from whetstone.evaluation import engine as evaluation_engine_module
from whetstone.evaluation.aggregate import (
    RowValue,
    TaskRows,
    unweighted_task_mean,
)
from whetstone.evaluation.drivers.internal import (
    InternalRowRequest,
    InternalRowResult,
    _llm_component_step,
)
from whetstone.evaluation.engine import (
    EngineEvaluation,
    EvaluationEngine,
    EvaluationRequest,
)
from whetstone.evaluation.schema import (
    EVALUATION_COMPONENT_TRACES_SCHEMA,
    EVALUATION_COMPONENT_TRACES_SCHEMA_VERSION,
    EVALUATION_EVIDENCE_SCHEMA_VERSION,
    EVALUATION_OUTPUTS_SCHEMA_VERSION,
    EvaluationComponentTraces,
    EvaluationComponentTracesRef,
    EvaluationEvidence,
    EvaluationEvidenceRef,
    EvaluationFailureEvidenceRef,
    EvaluationOutputRow,
    EvaluationOutputsRecord,
)
from whetstone.evaluation.schema_names import (
    EVALUATION_EVIDENCE_SCHEMA,
)
from whetstone.evaluation.traces import ExecutedRowState
from whetstone.execution.fanout import ProcessJob
from whetstone.execution.partials import PartialLog
from whetstone.experiment.candidate import (
    candidate_reference,
)
from whetstone.experiment.reward import (
    Reward,
)


def _ed1_graph_engine(*, store: ObjectStore) -> EvaluationEngine:
    base_experiment = _experiment()
    experiment = replace(
        base_experiment,
        generation_graph=build_encdec_generation_graph(
            CODE_COMP_ENV_NAME,
            provider_call_config=build_encoder_provider_call_config(
                "openai/test"
            ),
            procedure_config_hash=(
                base_experiment.generation_graph.procedure_config_hash
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


def test_uncached_experiment_uses_real_production_constructor() -> None:
    experiment = _uncached_experiment()

    assert experiment.env_name == "c18"
    assert len(experiment.eval_configs.internal.tasks) == 1
    assert len(experiment.eval_configs.official.tasks) == 1


def test_engine_run_delegates_encdec_to_code_comp_dispatch(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.envs.support import synthetic_code_comp_tasks
    from whetstone.envs.code_comp import (
        CodeCompMode,
        build_code_comp_experiment,
    )

    class _StopEval(Exception):
        pass

    experiment = build_code_comp_experiment(
        CodeCompMode.ENCDEC,
        tasks=synthetic_code_comp_tasks(1),
        internal_n=1,
        official_n=1,
    )
    store = ObjectStore(SqliteBackend(tmp_path / "code-comp-dispatch.sqlite"))
    engine = EvaluationEngine(
        store=store,
        experiment=experiment,
        sampling=experiment.eval_configs.internal,
        execution_policy=execution_policy(),
        row_job_factory=_DEFAULT_ROW_JOB_FACTORY,
    )
    sentinel: dict[str, object] = {"called": False}

    def fake_run_code_comp_eval(*args, **kwargs):
        sentinel["called"] = True
        raise _StopEval()

    monkeypatch.setattr(
        evaluation_engine_module,
        "run_code_comp_eval",
        fake_run_code_comp_eval,
    )
    request = EvaluationRequest(
        candidate=experiment.initial_candidate,
        evaluation_binding=_binding(engine),
        purpose="code-comp-dispatch-test",
    )
    with pytest.raises(_StopEval):
        engine._run(request)
    assert sentinel["called"] is True


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
            row.task_id,
            row.task_hash,
            row.sample_index,
            row.executed_component_trace.row_state.value,
        )
        for row in component_traces.rows
    ) == tuple(
        (
            row.task_id,
            row.task_hash,
            row.sample_index,
            "success",
        )
        for row in output_record.outputs
    )
    assert (
        component_traces.rows[0]
        .executed_component_trace.executed_component_steps[0]
        .outputs["provider_generation"]
        == output_record.outputs[0].output_text
    )
    assert tuple(row.task_hash for row in output_record.outputs) == (
        engine.sampling.task_set.task_hashes
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
    assert evidence.dataset_hash == (engine.sampling.task_set.dataset_revision)


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
        instance = engine.sampling.tasks[0]
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
                graph_hash=experiment.generation_graph.graph_hash,
                evaluation_binding_hash=(
                    kwargs["evaluation_binding"].identity_hash()
                ),
                task_rows=(
                    TaskRows(
                        task_hash=(engine.sampling.task_set.task_hashes[0]),
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
        successful_trace.executed_component_steps[0].outputs[
            "provider_generation"
        ]
        == encoder_text
    )
    assert (
        successful_trace.executed_component_steps[1].outputs[
            "provider_generation"
        ]
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
        failed_trace.executed_component_steps[0].outputs["provider_generation"]
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
        engine.sampling.tasks[0].gold
    )
    assert (
        post_score_trace.executed_component_steps[-1].outputs[
            "provider_generation"
        ]
        == post_score_outputs.outputs[0].output_text
    )


@pytest.mark.process_integration
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
            request_hash=f"mismatched-{request.request_hash}",
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
        graph_hash=engine.experiment.generation_graph.graph_hash,
        purpose="wire-contract",
        split_role=engine.sampling.split_role,
        task_hashes=("task-1",),
        num_samples=1,
        component_traces_ref=component_traces_ref,
        outputs=(
            EvaluationOutputRow(
                candidate_id=candidate.candidate_id,
                task_id="instance-1",
                task_hash="task-1",
                task_index=0,
                sample_index=0,
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
                code_submission_result=None,
            ),
        ),
    )

    assert record.record_content() == {
        "schema_version": 4,
        "candidate": candidate_ref.model_dump(mode="json"),
        "evaluation_binding": binding.model_dump(mode="json"),
        "evaluation_role": "internal",
        "graph_hash": engine.experiment.generation_graph.graph_hash,
        "purpose": "wire-contract",
        "split_role": "internal_eval",
        "task_hashes": ["task-1"],
        "num_samples": 1,
        "component_traces_ref": component_traces_ref.model_dump(mode="json"),
        "outputs": [
            {
                "candidate_id": candidate.candidate_id,
                "task_id": "instance-1",
                "task_hash": "task-1",
                "task_index": 0,
                "sample_index": 0,
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
                "code_submission_result": None,
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
    assert EVALUATION_COMPONENT_TRACES_SCHEMA_VERSION == 2
    assert EVALUATION_OUTPUTS_SCHEMA_VERSION == 4
    assert EVALUATION_EVIDENCE_SCHEMA_VERSION == 3
    assert traces.schema_version == 2
    assert outputs.schema_version == 4
    assert evidence.schema_version == 3
    assert evidence.component_traces_ref.content_hash == (
        "bf842459a3a5e772bc6889862c16948524c60ef7a8ac514ba14fd3da374e16ea"
    )
    assert evidence.outputs_ref.content_hash == (
        "fc277176e9bd7c168732e73577fa7a78a10076673c6980a5fcd4a57a3aecc7c6"
    )
    assert evaluated.evidence_ref.content_hash == (
        "8080365649c85f8c5cd2a266abeb4d2ee0314cf0b08e7d3e33c928ad7bc4d1f7"
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
        "task_hashes",
        "num_samples",
        "rows",
    )
    assert tuple(trace_content["rows"][0]) == (
        "task_id",
        "task_hash",
        "task_index",
        "sample_index",
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
        ("traces", 1),
        ("outputs", 2),
        ("evidence", 2),
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
        task_id="instance-1",
        task_hash="task-1",
        task_index=0,
        sample_index=0,
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
            graph_hash=engine.experiment.generation_graph.graph_hash,
            purpose="mismatch",
            split_role=engine.sampling.split_role,
            task_hashes=("task-1",),
            num_samples=1,
            component_traces_ref=TypedRef(
                schema_name=EVALUATION_COMPONENT_TRACES_SCHEMA,
                content_hash="a" * 64,
            ),
            outputs=(row,),
        )


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
        ({"sample_index": True}, "valid integer"),
        ({"score": float("nan")}, "finite number"),
        ({"unexpected": "drift"}, "Extra inputs are not permitted"),
    ),
)
def test_evaluation_output_row_rejects_wire_schema_drift(
    update, message
) -> None:
    payload = {
        "candidate_id": "candidate-1",
        "task_id": "instance-1",
        "task_hash": "task-1",
        "sample_index": 0,
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
            outputs=(replace(result.outputs[0], sample_index=99),),
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
        num_samples=2,
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


@pytest.mark.process_integration
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


def test_cache_evidence_excludes_another_bindings_partial_rows(
    tmp_path,
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "binding-scope.sqlite"))
    engine = _engine(tmp_path, store=store, partial=True)
    partial_log = PartialLog(tmp_path / "partials.jsonl")
    candidate = engine.experiment.initial_candidate

    first = engine.evaluate(
        EvaluationRequest(
            candidate=candidate,
            evaluation_binding=_binding(engine, campaign="first-binding"),
            purpose="binding-scope",
        )
    )
    second = engine.evaluate(
        EvaluationRequest(
            candidate=candidate,
            evaluation_binding=_binding(engine, campaign="second-binding"),
            purpose="binding-scope",
        )
    )

    rows = partial_log.load()
    assert {row.unit for row in rows} == {candidate.candidate_id}
    assert len({row.request_hash for row in rows}) == len(rows)
    assert len(rows) == first.evidence.cache.partial_row_count + (
        second.evidence.cache.partial_row_count
    )
    assert second.evidence.cache.partial_row_count == (
        first.evidence.cache.partial_row_count
    )


def test_sampling_repeat_change_changes_exact_eval_identity(tmp_path) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "identity.sqlite"))
    one = _engine(tmp_path, store=store, num_samples=1)
    two = _engine(tmp_path, store=store, num_samples=2)

    assert one.eval_config_ref.config_hash != two.eval_config_ref.config_hash
