from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from dr_dspy.graph import (
    BindingRef,
    FieldRole,
    FieldSpec,
    GraphRunStatus,
    GraphSpec,
    NodeConfig,
    NodeSpec,
    graph_digest,
)
from dr_dspy.harness.flow import stable_prediction_id_from_dimensions
from dr_dspy.humaneval.parsed_tests import HumanEvalTestCaseKind
from dr_dspy.humaneval.scoring import GeneratedCodeOutcome
from dr_dspy.humaneval.task import EvaluationCaseStatus, EvaluationCaseSummary
from dr_dspy.lm.boundary import (
    EndpointKind,
    ProviderKind,
    openai_responses_config,
)
from dr_dspy.records import (
    BatchSubmitItemEnqueueStatus,
    BatchSubmitItemInsertStatus,
    BatchSubmitItemRecord,
    BatchSubmitOperationRecord,
    BatchSubmitOperationStatus,
    DimensionsPayload,
    ExtractedCodePayload,
    FailureMetadataPayload,
    GenerationRunRecord,
    GenerationRunStatus,
    GenerationRunSummaryPayload,
    GenerationTerminalErrorPayload,
    GraphSnapshotPayload,
    MetricsPayload,
    NodeAttemptRecord,
    NodeAttemptStatus,
    NodeOutputPayload,
    PerTestResultPayload,
    PredictionProjectionRecord,
    PredictionSpecRecord,
    ProviderConfigRef,
    PythonLeakageMetricsPayload,
    ScoreAttemptRecord,
    ScoreAttemptStatus,
    TaskInputsPayload,
    TaskSnapshotPayload,
    TextMetricsPayload,
    dimensions_digest,
    fair_order_key,
    stable_prediction_id,
)

NOW = datetime(2026, 6, 29, 12, 0, tzinfo=UTC)


def test_generation_run_status_matches_graph_run_status_values() -> None:
    assert {status.value for status in GenerationRunStatus} == {
        status.value for status in GraphRunStatus
    }


def _node(
    node_id: str,
    *,
    bindings: dict[str, str] | None = None,
    output_field: str = "output",
) -> NodeSpec:
    input_bindings = {
        name: BindingRef.model_validate(ref)
        for name, ref in (bindings or {}).items()
    }
    fields = [
        FieldSpec(name=name, role=FieldRole.INPUT)
        for name in input_bindings
    ]
    fields.append(FieldSpec(name=output_field, role=FieldRole.OUTPUT))
    return NodeSpec(
        id=node_id,
        config=NodeConfig(
            fields=tuple(fields),
            input_bindings=input_bindings,
            output_field=output_field,
        ),
    )


def _direct_graph() -> GraphSpec:
    return GraphSpec(
        nodes=(_node("direct", bindings={"prompt": "task.prompt"}),),
        terminal_node_id="direct",
    )


def _provider() -> ProviderConfigRef:
    return ProviderConfigRef(
        provider_kind=ProviderKind.OPENAI,
        endpoint_kind=EndpointKind.RESPONSES,
        model="gpt-test",
        throttle_key="openai:responses:gpt-test",
        parameters={"temperature": 0.2},
    )


def _dimensions(**values: Any) -> DimensionsPayload:
    return DimensionsPayload(values={"temperature": 0.2, **values})


def _prediction_spec(
    *,
    graph: GraphSpec | None = None,
    dimensions: DimensionsPayload | None = None,
) -> PredictionSpecRecord:
    graph = graph or _direct_graph()
    dimensions = dimensions or _dimensions()
    graph_id = graph_digest(graph)
    dimensions_id = dimensions_digest(dimensions)
    provider = _provider()
    prediction_id = stable_prediction_id(
        experiment_name="exp",
        task_id="HumanEval/0",
        graph_digest=graph_id,
        dimensions_digest=dimensions_id,
        repetition_seed=7,
        provider_kind=provider.provider_kind.value,
        endpoint_kind=provider.endpoint_kind.value,
        model=provider.model,
        throttle_key=provider.throttle_key,
    )
    return PredictionSpecRecord(
        prediction_id=prediction_id,
        experiment_name="exp",
        task_id="HumanEval/0",
        repetition_seed=7,
        graph=GraphSnapshotPayload(
            graph=graph,
            graph_digest=graph_id,
            layout="direct",
        ),
        dimensions=dimensions,
        dimensions_digest=dimensions_id,
        task=TaskSnapshotPayload(
            task_id="HumanEval/0",
            inputs=TaskInputsPayload(values={"prompt": "write add"}),
        ),
        provider_configs=(provider,),
        provider_axis=provider,
        fair_order_seed="seed",
        fair_order_key=fair_order_key(
            experiment_seed="seed",
            prediction_id=prediction_id,
            provider=provider.provider_kind.value,
            endpoint_kind=provider.endpoint_kind.value,
            model=provider.model,
            throttle_key=provider.throttle_key,
            graph_layout="direct",
            task_id="HumanEval/0",
            repetition_seed=7,
            config_axis=dimensions_id,
        ),
        created_at=NOW,
    )


def _failure() -> FailureMetadataPayload:
    return FailureMetadataPayload(
        error_type="builtins.RuntimeError",
        message="boom",
        metadata={"node": "direct"},
    )


def test_prediction_spec_rejects_extra_fields_and_dumps_json() -> None:
    spec = _prediction_spec()

    dumped = spec.model_dump(mode="json")

    assert dumped["prediction_id"] == spec.prediction_id
    assert dumped["provider_configs"][0]["provider_kind"] == "openai"
    assert dumped["graph"]["graph"]["terminal_node_id"] == "direct"
    with pytest.raises(ValidationError):
        PredictionSpecRecord.model_validate({**dumped, "extra": "nope"})


def test_graph_snapshot_validates_digest() -> None:
    graph = _direct_graph()

    with pytest.raises(ValidationError, match="graph_digest"):
        GraphSnapshotPayload(
            graph=graph,
            graph_digest="wrong",
            layout="direct",
        )


def test_prediction_spec_validates_dimensions_digest() -> None:
    dumped = _prediction_spec().model_dump(mode="json")
    dumped["dimensions_digest"] = "wrong"

    with pytest.raises(ValidationError, match="dimensions_digest"):
        PredictionSpecRecord.model_validate(dumped)


def test_prediction_spec_validates_stable_prediction_id() -> None:
    dumped = _prediction_spec().model_dump(mode="json")
    dumped["prediction_id"] = "wrong"

    with pytest.raises(ValidationError, match="prediction_id"):
        PredictionSpecRecord.model_validate(dumped)


def test_prediction_spec_validates_fair_order_key() -> None:
    dumped = _prediction_spec().model_dump(mode="json")
    dumped["fair_order_key"] = "wrong"

    with pytest.raises(ValidationError, match="fair_order_key"):
        PredictionSpecRecord.model_validate(dumped)


def test_prediction_spec_validates_task_bindings() -> None:
    dumped = _prediction_spec().model_dump(mode="json")
    dumped["graph"]["graph"]["nodes"][0]["config"]["input_bindings"][
        "prompt"
    ] = "task.promt"
    graph = GraphSpec.model_validate(dumped["graph"]["graph"])
    dumped["graph"]["graph_digest"] = graph_digest(graph)

    with pytest.raises(
        ValidationError,
        match="task binding field\\(s\\) 'promt' not in allowed task fields",
    ):
        PredictionSpecRecord.model_validate(dumped)


def test_provider_config_ref_converts_from_runtime_provider_config() -> None:
    runtime_config = openai_responses_config(model="gpt-test")

    ref = ProviderConfigRef.from_config(
        runtime_config,
        config_id="decoder",
        parameters={"temperature": 0.2},
    )

    assert ref.provider_kind is ProviderKind.OPENAI
    assert ref.endpoint_kind is EndpointKind.RESPONSES
    assert ref.model == "gpt-test"
    assert ref.config_id == "decoder"
    assert ref.throttle_key == "openai:responses:gpt-test"
    assert ref.parameters == {"temperature": 0.2}


def test_prediction_spec_requires_provider_axis_member() -> None:
    provider = _provider()
    other_provider = ProviderConfigRef(
        provider_kind=ProviderKind.OPENROUTER,
        endpoint_kind=EndpointKind.CHAT_COMPLETIONS,
        model="other",
        throttle_key="openrouter:chat_completions:other",
    )
    spec = _prediction_spec().model_copy(
        update={
            "provider_configs": (provider,),
            "provider_axis": other_provider,
        }
    )

    with pytest.raises(ValidationError, match="provider_axis"):
        PredictionSpecRecord.model_validate(spec.model_dump(mode="json"))


def test_stable_prediction_id_changes_with_graph_or_dimensions() -> None:
    base = _prediction_spec()
    changed_dimensions = _prediction_spec(
        dimensions=_dimensions(temperature=0.9)
    )
    changed_graph = _prediction_spec(
        graph=GraphSpec(
            nodes=(
                _node("encoder", bindings={"prompt": "task.prompt"}),
                _node("decoder", bindings={"text": "encoder"}),
            ),
            terminal_node_id="decoder",
        )
    )

    assert base.prediction_id != changed_dimensions.prediction_id
    assert base.prediction_id != changed_graph.prediction_id
    assert base.dimensions_digest != changed_dimensions.dimensions_digest
    assert base.graph.graph_digest != changed_graph.graph.graph_digest


def test_stable_prediction_id_changes_with_provider_axis() -> None:
    base = _prediction_spec()
    other_provider = ProviderConfigRef(
        provider_kind=ProviderKind.OPENROUTER,
        endpoint_kind=EndpointKind.CHAT_COMPLETIONS,
        model="other-model",
        throttle_key="openrouter:chat_completions:other-model",
    )
    other_prediction_id = stable_prediction_id(
        experiment_name=base.experiment_name,
        task_id=base.task_id,
        graph_digest=base.graph.graph_digest,
        dimensions_digest=base.dimensions_digest,
        repetition_seed=base.repetition_seed,
        provider_kind=other_provider.provider_kind.value,
        endpoint_kind=other_provider.endpoint_kind.value,
        model=other_provider.model,
        throttle_key=other_provider.throttle_key,
    )
    changed_provider = base.model_copy(
        update={
            "prediction_id": other_prediction_id,
            "provider_configs": (other_provider,),
            "provider_axis": other_provider,
            "fair_order_key": fair_order_key(
                experiment_seed=base.fair_order_seed,
                prediction_id=other_prediction_id,
                provider=other_provider.provider_kind.value,
                endpoint_kind=other_provider.endpoint_kind.value,
                model=other_provider.model,
                throttle_key=other_provider.throttle_key,
                graph_layout=base.graph.layout,
                task_id=base.task_id,
                repetition_seed=base.repetition_seed,
                config_axis=base.dimensions_digest,
            ),
        }
    )

    PredictionSpecRecord.model_validate(changed_provider.model_dump(mode="json"))
    assert base.prediction_id != changed_provider.prediction_id


def test_v1_prediction_ids_are_not_v0_compatible() -> None:
    v1 = _prediction_spec()
    v0 = stable_prediction_id_from_dimensions(
        experiment_name=v1.experiment_name,
        task_id=v1.task_id,
        dimensions=v1.dimensions.values,
        repetition_seed=v1.repetition_seed,
        digest_length=24,
    )

    assert v1.prediction_id != v0


def test_fair_order_key_mixes_endpoint_throttle_and_config_axes() -> None:
    base = fair_order_key(
        experiment_seed="seed",
        prediction_id="prediction",
        provider="openai",
        endpoint_kind="responses",
        model="model",
        throttle_key="openai:responses:model",
        graph_layout="direct",
        task_id="HumanEval/0",
        repetition_seed=1,
        config_axis="temperature=0.2",
    )
    changed_endpoint = fair_order_key(
        experiment_seed="seed",
        prediction_id="prediction",
        provider="openai",
        endpoint_kind="chat_completions",
        model="model",
        throttle_key="openai:chat_completions:model",
        graph_layout="direct",
        task_id="HumanEval/0",
        repetition_seed=1,
        config_axis="temperature=0.2",
    )
    changed_config = fair_order_key(
        experiment_seed="seed",
        prediction_id="prediction",
        provider="openai",
        endpoint_kind="responses",
        model="model",
        throttle_key="openai:responses:model",
        graph_layout="direct",
        task_id="HumanEval/0",
        repetition_seed=1,
        config_axis="temperature=0.9",
    )

    assert base != changed_endpoint
    assert base != changed_config


def test_node_attempt_statuses_are_only_invoked_terminal_outcomes() -> None:
    with pytest.raises(ValidationError):
        NodeAttemptRecord.model_validate(
            {
                "node_attempt_id": "node-attempt-1",
                "generation_run_id": "run-1",
                "prediction_id": "prediction-1",
                "node_id": "decoder",
                "attempt_index": 0,
                "status": "blocked",
                "started_at": NOW,
                "completed_at": NOW,
            }
        )


def test_generation_run_can_store_terminal_blocked_result() -> None:
    record = GenerationRunRecord(
        generation_run_id="run-1",
        prediction_id="prediction-1",
        attempt_index=0,
        status=GenerationRunStatus.BLOCKED,
        terminal_node_id="decoder",
        summary=GenerationRunSummaryPayload(
            execution_order=("encoder", "decoder"),
            terminal_node_id="decoder",
            terminal_error=GenerationTerminalErrorPayload(
                node_id="decoder",
                status=GenerationRunStatus.BLOCKED,
                blocked_by=("encoder",),
            ),
        ),
        started_at=NOW,
        completed_at=NOW,
    )

    assert record.status is GenerationRunStatus.BLOCKED
    assert record.summary.terminal_error is not None
    assert record.summary.terminal_error.blocked_by == ("encoder",)


def test_partial_generation_run_rejects_terminal_error() -> None:
    with pytest.raises(ValidationError, match="terminal_error"):
        GenerationRunRecord(
            generation_run_id="run-1",
            prediction_id="prediction-1",
            attempt_index=0,
            status=GenerationRunStatus.PARTIAL,
            terminal_node_id="direct",
            summary=GenerationRunSummaryPayload(
                execution_order=("direct", "bad"),
                terminal_node_id="direct",
                terminal_output="ok",
                terminal_error=GenerationTerminalErrorPayload(
                    node_id="bad",
                    status=GenerationRunStatus.ERROR,
                ),
            ),
            started_at=NOW,
            completed_at=NOW,
        )


def test_partial_generation_run_accepts_terminal_output() -> None:
    record = GenerationRunRecord(
        generation_run_id="run-1",
        prediction_id="prediction-1",
        attempt_index=0,
        status=GenerationRunStatus.PARTIAL,
        terminal_node_id="direct",
        terminal_output_node_id="direct",
        summary=GenerationRunSummaryPayload(
            execution_order=("direct", "bad"),
            terminal_node_id="direct",
            terminal_output="ok",
        ),
        started_at=NOW,
        completed_at=NOW,
    )

    assert record.status is GenerationRunStatus.PARTIAL
    assert record.summary.terminal_error is None
    assert record.summary.terminal_output == "ok"


def test_successful_node_attempt_requires_output() -> None:
    with pytest.raises(ValidationError, match="require output"):
        NodeAttemptRecord(
            node_attempt_id="node-attempt-1",
            generation_run_id="run-1",
            prediction_id="prediction-1",
            node_id="direct",
            attempt_index=0,
            status=NodeAttemptStatus.SUCCESS,
            started_at=NOW,
            completed_at=NOW,
        )

    record = NodeAttemptRecord(
        node_attempt_id="node-attempt-1",
        generation_run_id="run-1",
        prediction_id="prediction-1",
        node_id="direct",
        attempt_index=0,
        status=NodeAttemptStatus.SUCCESS,
        provider_config=_provider(),
        output=NodeOutputPayload(values={"output": "def add(): pass"}),
        started_at=NOW,
        completed_at=NOW,
    )

    assert record.model_dump(mode="json")["output"]["values"] == {
        "output": "def add(): pass"
    }


def test_node_attempt_success_and_error_payloads_are_exclusive() -> None:
    success = {
        "node_attempt_id": "node-attempt-1",
        "generation_run_id": "run-1",
        "prediction_id": "prediction-1",
        "node_id": "direct",
        "attempt_index": 0,
        "status": "success",
        "output": {"values": {"output": "ok"}},
        "failure": _failure().model_dump(mode="json"),
        "started_at": NOW,
        "completed_at": NOW,
    }
    error = {
        **success,
        "status": "error",
        "failure": _failure().model_dump(mode="json"),
    }

    with pytest.raises(ValidationError, match="cannot have failure"):
        NodeAttemptRecord.model_validate(success)
    with pytest.raises(ValidationError, match="cannot have output"):
        NodeAttemptRecord.model_validate(error)


def test_score_attempt_success_and_error_shapes() -> None:
    success = ScoreAttemptRecord(
        score_attempt_id="score-1",
        prediction_id="prediction-1",
        generation_run_id="run-1",
        attempt_index=0,
        scoring_profile_id="humaneval",
        scoring_profile_version="v1",
        parser_profile_id="best-effort",
        parser_version="v1",
        status=ScoreAttemptStatus.SUCCESS,
        generated_code_outcome=GeneratedCodeOutcome.PASSED,
        score=1.0,
        metrics=MetricsPayload(
            profile_id="humaneval",
            profile_version="v1",
            text=TextMetricsPayload(
                character_count=12,
                byte_count=12,
                line_count=1,
                nonempty_line_count=1,
                word_count=2,
                average_word_length=5.5,
            ),
            python_leakage=PythonLeakageMetricsPayload(
                keyword_count=1,
                code_marker_count=1,
                fenced_code_block_count=0,
                code_like_line_count=1,
                operator_count=0,
            ),
            custom={"passed": True},
        ),
        started_at=NOW,
        completed_at=NOW,
    )

    assert success.model_dump(mode="json")["status"] == "success"
    assert success.generated_code_outcome is GeneratedCodeOutcome.PASSED
    with pytest.raises(ValidationError, match="require failure"):
        ScoreAttemptRecord(
            score_attempt_id="score-2",
            prediction_id="prediction-1",
            generation_run_id="run-1",
            attempt_index=0,
            scoring_profile_id="humaneval",
            scoring_profile_version="v1",
            parser_profile_id="best-effort",
            parser_version="v1",
            status=ScoreAttemptStatus.ERROR,
            started_at=NOW,
            completed_at=NOW,
        )


def test_score_attempt_success_and_error_payloads_are_exclusive() -> None:
    success = {
        "score_attempt_id": "score-1",
        "prediction_id": "prediction-1",
        "generation_run_id": "run-1",
        "attempt_index": 0,
        "scoring_profile_id": "humaneval",
        "scoring_profile_version": "v1",
        "parser_profile_id": "best-effort",
        "parser_version": "v1",
        "status": "success",
        "score": 1.0,
        "failure": _failure().model_dump(mode="json"),
        "started_at": NOW,
        "completed_at": NOW,
    }
    error = {
        **success,
        "status": "error",
        "failure": _failure().model_dump(mode="json"),
    }

    with pytest.raises(ValidationError, match="cannot have failure"):
        ScoreAttemptRecord.model_validate(success)
    with pytest.raises(ValidationError, match="cannot have score"):
        ScoreAttemptRecord.model_validate(error)


def test_score_attempt_error_allows_partial_diagnostics() -> None:
    attempt = ScoreAttemptRecord(
        score_attempt_id="score-error",
        prediction_id="prediction-1",
        generation_run_id="run-1",
        attempt_index=0,
        scoring_profile_id="humaneval",
        scoring_profile_version="v1",
        parser_profile_id="best-effort",
        parser_version="v1",
        status=ScoreAttemptStatus.ERROR,
        generated_code_outcome=GeneratedCodeOutcome.EXTRACTION_FAILED,
        extracted_code=ExtractedCodePayload(
            raw_generation="def broken(",
            parser_profile_id="best-effort",
            parser_version="v1",
        ),
        metrics=MetricsPayload(
            profile_id="humaneval",
            profile_version="v1",
            text=TextMetricsPayload(
                character_count=11,
                byte_count=11,
                line_count=1,
                nonempty_line_count=1,
                word_count=2,
                average_word_length=5.5,
            ),
        ),
        failure=_failure(),
        started_at=NOW,
        completed_at=NOW,
    )

    assert attempt.generated_code_outcome is (
        GeneratedCodeOutcome.EXTRACTION_FAILED
    )
    assert attempt.metrics is not None
    assert attempt.extracted_code is not None


def test_score_attempt_allows_distinct_metrics_profile() -> None:
    attempt = ScoreAttemptRecord(
        score_attempt_id="score-1",
        prediction_id="prediction-1",
        generation_run_id="run-1",
        attempt_index=0,
        scoring_profile_id="humaneval",
        scoring_profile_version="v1",
        parser_profile_id="best-effort",
        parser_version="v1",
        status=ScoreAttemptStatus.SUCCESS,
        score=1.0,
        metrics=MetricsPayload(
            profile_id="humaneval-metrics",
            profile_version="v1",
        ),
        started_at=NOW,
        completed_at=NOW,
    )

    assert attempt.metrics is not None
    assert attempt.metrics.profile_id == "humaneval-metrics"


def test_score_attempt_rejects_mismatched_parser_profile() -> None:
    with pytest.raises(
        ValidationError,
        match="extracted_code parser_profile_id",
    ):
        ScoreAttemptRecord(
            score_attempt_id="score-1",
            prediction_id="prediction-1",
            generation_run_id="run-1",
            attempt_index=0,
            scoring_profile_id="humaneval",
            scoring_profile_version="v1",
            parser_profile_id="best-effort",
            parser_version="v1",
            status=ScoreAttemptStatus.ERROR,
            extracted_code=ExtractedCodePayload(
                raw_generation="def broken(",
                parser_profile_id="other",
                parser_version="v1",
            ),
            failure=_failure(),
            started_at=NOW,
            completed_at=NOW,
        )


def test_score_attempt_attempt_index_must_be_non_negative() -> None:
    with pytest.raises(ValidationError, match="attempt_index"):
        ScoreAttemptRecord(
            score_attempt_id="score-1",
            prediction_id="prediction-1",
            generation_run_id="run-1",
            attempt_index=-1,
            scoring_profile_id="humaneval",
            scoring_profile_version="v1",
            parser_profile_id="best-effort",
            parser_version="v1",
            status=ScoreAttemptStatus.SUCCESS,
            score=1.0,
            started_at=NOW,
            completed_at=NOW,
        )


def test_per_test_result_aligns_with_humaneval_case_summary() -> None:
    summary = EvaluationCaseSummary(
        task_id="HumanEval/0",
        case_id="case_0",
        function_name="add",
        status=EvaluationCaseStatus.PASSED,
        message="",
        test_type=HumanEvalTestCaseKind.INPUT_RESULT,
        input_repr="[1, 2]",
        expected_output_repr="3",
        actual_output_repr="3",
    )

    payload = PerTestResultPayload.from_evaluation_case(summary)

    assert payload.model_dump(mode="json") == {
        "task_id": "HumanEval/0",
        "test_id": "case_0",
        "function_name": "add",
        "status": "passed",
        "message": "",
        "test_type": "input_result",
        "input_repr": "[1, 2]",
        "expected_output_repr": "3",
        "actual_output_repr": "3",
    }


def test_projection_and_batch_records_validate_json_contracts() -> None:
    projection = PredictionProjectionRecord(
        prediction_id="prediction-1",
        generation_run_id="run-1",
        score_attempt_id="score-1",
        projection_profile_id="analysis",
        projection_version="v1",
        selected_at=NOW,
        selection_reason="latest validated score",
    )
    operation = BatchSubmitOperationRecord(
        operation_key="op-1",
        experiment_name="exp",
        status=BatchSubmitOperationStatus.PARTIAL,
        requested_count=2,
        inserted_count=1,
        failed_count=1,
        created_at=NOW,
        completed_at=NOW,
    )
    item = BatchSubmitItemRecord(
        batch_submit_item_id="item-1",
        operation_key="op-1",
        item_index=1,
        prediction_id="prediction-1",
        fair_order_key="abc",
        insert_status=BatchSubmitItemInsertStatus.INSERTED,
        enqueue_status=BatchSubmitItemEnqueueStatus.FAILED,
        failure=_failure(),
        created_at=NOW,
    )

    assert projection.model_dump(mode="json")["selected_at"].startswith(
        "2026-06-29"
    )
    assert operation.model_dump(mode="json")["status"] == "partial"
    assert item.model_dump(mode="json")["failure"]["message"] == "boom"


def test_projection_requires_selected_generation_or_score() -> None:
    with pytest.raises(ValidationError, match="projection requires"):
        PredictionProjectionRecord(
            prediction_id="prediction-1",
            projection_profile_id="analysis",
            projection_version="v1",
            selected_at=NOW,
        )
