from __future__ import annotations

from datetime import UTC, datetime

import pytest
from dr_code.humaneval.scoring import GeneratedCodeOutcome
from dr_graph import (
    BindingRef,
    FieldRole,
    FieldSpec,
    GraphRunStatus,
    GraphSpec,
    NodeConfig,
    NodeError,
    NodeOutput,
    NodeSpec,
    graph_digest,
)
from dr_providers import EndpointKind, ProviderKind
from pydantic import ValidationError

from whetstone.db import io
from whetstone.records import (
    DEFAULT_SCORE_DATASET_NAME,
    DEFAULT_SCORE_DATASET_SPLIT,
    DimensionsPayload,
    ExperimentRecord,
    GenerationRunRecord,
    GenerationRunStatus,
    GenerationRunSummaryPayload,
    GraphSnapshotPayload,
    MetricsPayload,
    NodeAttemptRecord,
    NodeAttemptStatus,
    NodeOutputPayload,
    PredictionProjectionRecord,
    PredictionSpecRecord,
    ProviderConfigRef,
    ResponseMetadataPayload,
    ScoreAttemptRecord,
    ScoreAttemptStatus,
    TaskInputsPayload,
    TaskSnapshotPayload,
    TextMetricsPayload,
    UsageCostPayload,
    dimensions_digest,
    fair_order_key,
    stable_prediction_id,
)
from whetstone.records import models as records_models

NOW = datetime(2026, 6, 29, 12, 0, tzinfo=UTC)


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
        op="llm_call",
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


def test_prediction_spec_row_uses_explicit_provider_axis() -> None:
    graph = _direct_graph()
    graph_id = graph_digest(graph)
    dimensions = DimensionsPayload(values={"budget_ratio": 0.5})
    dimensions_id = dimensions_digest(dimensions)
    encoder = ProviderConfigRef(
        provider_kind=ProviderKind.OPENROUTER,
        endpoint_kind=EndpointKind.CHAT_COMPLETIONS,
        model="encoder-model",
        throttle_key="openrouter:chat_completions:encoder-model",
    )
    decoder = ProviderConfigRef(
        provider_kind=ProviderKind.OPENAI,
        endpoint_kind=EndpointKind.RESPONSES,
        model="decoder-model",
        throttle_key="openai:responses:decoder-model",
    )
    prediction_id = stable_prediction_id(
        experiment_name="exp",
        task_id="HumanEval/0",
        graph_digest=graph_id,
        dimensions_digest=dimensions_id,
        repetition_seed=0,
        provider_kind=decoder.provider_kind.value,
        endpoint_kind=decoder.endpoint_kind.value,
        model=decoder.model,
        throttle_key=decoder.throttle_key,
    )
    fair_key = fair_order_key(
        experiment_seed="seed",
        prediction_id=prediction_id,
        provider=decoder.provider_kind.value,
        endpoint_kind=decoder.endpoint_kind.value,
        model=decoder.model,
        throttle_key=decoder.throttle_key,
        graph_layout="encdec",
        task_id="HumanEval/0",
        repetition_seed=0,
        config_axis=dimensions_id,
    )
    record = PredictionSpecRecord(
        prediction_id=prediction_id,
        experiment_name="exp",
        task_id="HumanEval/0",
        repetition_seed=0,
        graph=GraphSnapshotPayload(
            graph=graph,
            graph_digest=graph_id,
            layout="encdec",
        ),
        dimensions=dimensions,
        dimensions_digest=dimensions_id,
        task=TaskSnapshotPayload(
            task_id="HumanEval/0",
            inputs=TaskInputsPayload(values={"prompt": "write add"}),
        ),
        provider_configs=(encoder, decoder),
        provider_axis=decoder,
        fair_order_seed="seed",
        fair_order_key=fair_key,
        created_at=NOW,
    )

    row = io.prediction_spec_row(record)

    assert row["provider_kind"] == "openai"
    assert row["endpoint_kind"] == "responses"
    assert row["model"] == "decoder-model"
    assert row["throttle_key"] == "openai:responses:decoder-model"
    assert row["provider_axis_config_id"] is None
    assert row["fair_order_seed"] == "seed"
    assert row["fair_order_key"] == fair_key
    assert row["provider_configs"] == [
        encoder.model_dump(mode="json"),
        decoder.model_dump(mode="json"),
    ]


def test_node_attempt_row_keeps_provider_snapshot_and_index_columns() -> None:
    provider = ProviderConfigRef(
        provider_kind=ProviderKind.OPENAI,
        endpoint_kind=EndpointKind.RESPONSES,
        model="decoder-model",
        throttle_key="openai:responses:decoder-model",
    )
    record = NodeAttemptRecord(
        node_attempt_id="node-attempt-1",
        generation_run_id="run-1",
        prediction_id="prediction-1",
        node_id="decoder",
        attempt_index=0,
        status=NodeAttemptStatus.SUCCESS,
        provider_config=provider,
        output=io.node_output_payload_from_graph_output(
            NodeOutput(values={"code": "def add(): pass"})
        ),
        started_at=NOW,
        completed_at=NOW,
    )

    row = io.node_attempt_row(record)

    assert row["provider_kind"] == "openai"
    assert row["endpoint_kind"] == "responses"
    assert row["model"] == "decoder-model"
    assert row["config_id"] is None
    assert row["provider_config"] == provider.model_dump(mode="json")
    assert row["output"] == {
        "values": {"code": "def add(): pass"},
        "metadata": {},
    }


def test_prediction_spec_row_rejects_provider_index_drift() -> None:
    graph = _direct_graph()
    graph_id = graph_digest(graph)
    dimensions = DimensionsPayload(values={"budget_ratio": 0.5})
    dimensions_id = dimensions_digest(dimensions)
    provider = ProviderConfigRef(
        provider_kind=ProviderKind.OPENAI,
        endpoint_kind=EndpointKind.RESPONSES,
        model="decoder-model",
        throttle_key="openai:responses:decoder-model",
    )
    prediction_id = stable_prediction_id(
        experiment_name="exp",
        task_id="HumanEval/0",
        graph_digest=graph_id,
        dimensions_digest=dimensions_id,
        repetition_seed=0,
        provider_kind=provider.provider_kind.value,
        endpoint_kind=provider.endpoint_kind.value,
        model=provider.model,
        throttle_key=provider.throttle_key,
    )
    record = PredictionSpecRecord(
        prediction_id=prediction_id,
        experiment_name="exp",
        task_id="HumanEval/0",
        repetition_seed=0,
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
            repetition_seed=0,
            config_axis=dimensions_id,
        ),
        created_at=NOW,
    )

    row = io.prediction_spec_row(record)
    row["provider_kind"] = "openrouter"

    with pytest.raises(ValueError, match="provider_configs snapshot"):
        io._validate_prediction_spec_provider_row(row)


def test_node_attempt_row_rejects_provider_index_drift() -> None:
    provider = ProviderConfigRef(
        provider_kind=ProviderKind.OPENAI,
        endpoint_kind=EndpointKind.RESPONSES,
        model="decoder-model",
        throttle_key="openai:responses:decoder-model",
    )
    record = NodeAttemptRecord(
        node_attempt_id="node-attempt-1",
        generation_run_id="run-1",
        prediction_id="prediction-1",
        node_id="decoder",
        attempt_index=0,
        status=NodeAttemptStatus.SUCCESS,
        provider_config=provider,
        output=io.node_output_payload_from_graph_output(
            NodeOutput(values={"code": "def add(): pass"})
        ),
        started_at=NOW,
        completed_at=NOW,
    )

    row = io.node_attempt_row(record)
    row["model"] = "other-model"

    with pytest.raises(ValueError, match="provider_config snapshot"):
        io._validate_node_attempt_provider_row(row)


def test_failure_payload_from_node_error_is_persistable() -> None:
    error = NodeError(
        error_type="builtins.RuntimeError",
        message="boom",
        failure_class="permanent",
        metadata={"node": "decoder"},
    )

    payload = io.failure_payload_from_node_error(error)

    assert payload.model_dump(mode="json") == {
        "failure_class": "permanent",
        "error_type": "builtins.RuntimeError",
        "underlying_exception_type": None,
        "message": "boom",
        "metadata": {"node": "decoder"},
    }


def test_failure_payload_promotes_underlying_exception_type() -> None:
    error = NodeError(
        error_type="whetstone.eval_failures.TransientFailureError",
        message="provider failed",
        failure_class="transient",
        metadata={
            "underlying_exception_type": "openai.AuthenticationError",
            "status_code": 401,
        },
    )

    payload = io.failure_payload_from_node_error(error)

    assert payload.underlying_exception_type == "openai.AuthenticationError"
    assert payload.metadata == {"status_code": 401}


def test_score_attempt_row_includes_generated_code_outcome() -> None:
    record = ScoreAttemptRecord(
        score_attempt_id="score-1",
        prediction_id="prediction-1",
        generation_run_id="run-1",
        attempt_index=0,
        scoring_profile_id="humaneval",
        scoring_profile_version="v1",
        parser_profile_id="best-effort",
        parser_version="v1",
        dataset_name=DEFAULT_SCORE_DATASET_NAME,
        dataset_split=DEFAULT_SCORE_DATASET_SPLIT,
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
            ),
            custom={"passed": True},
        ),
        started_at=NOW,
        completed_at=NOW,
    )

    row = io.score_attempt_row(record)

    assert row["attempt_index"] == 0
    assert row["generated_code_outcome"] == "passed"
    assert row["metrics"] == {
        "profile_id": "humaneval",
        "profile_version": "v1",
        "task_tests": None,
        "text": {
            "character_count": 12,
            "byte_count": 12,
            "line_count": 1,
            "nonempty_line_count": 1,
            "word_count": 2,
            "average_word_length": None,
            "punctuation_count": None,
            "symbol_count": None,
        },
        "python_leakage": None,
        "ast": None,
        "compression": {},
        "stages": [],
        "custom": {"passed": True},
    }


@pytest.mark.parametrize(
    ("graph_status", "generation_status"),
    [
        (GraphRunStatus.SUCCESS, GenerationRunStatus.SUCCESS),
        (GraphRunStatus.ERROR, GenerationRunStatus.ERROR),
        (GraphRunStatus.BLOCKED, GenerationRunStatus.BLOCKED),
        (GraphRunStatus.PARTIAL, GenerationRunStatus.PARTIAL),
    ],
)
def test_generation_status_from_graph_status_maps_terminal_values(
    graph_status: GraphRunStatus,
    generation_status: GenerationRunStatus,
) -> None:
    assert io.generation_status_from_graph_status(graph_status) is (
        generation_status
    )


def test_prediction_spec_row_round_trips_through_record_from_row() -> None:
    graph = _direct_graph()
    graph_id = graph_digest(graph)
    dimensions = DimensionsPayload(values={"budget_ratio": 0.5})
    dimensions_id = dimensions_digest(dimensions)
    provider = ProviderConfigRef(
        provider_kind=ProviderKind.OPENAI,
        endpoint_kind=EndpointKind.RESPONSES,
        model="decoder-model",
        throttle_key="openai:responses:decoder-model",
    )
    prediction_id = stable_prediction_id(
        experiment_name="exp",
        task_id="HumanEval/0",
        graph_digest=graph_id,
        dimensions_digest=dimensions_id,
        repetition_seed=0,
        provider_kind=provider.provider_kind.value,
        endpoint_kind=provider.endpoint_kind.value,
        model=provider.model,
        throttle_key=provider.throttle_key,
    )
    record = PredictionSpecRecord(
        prediction_id=prediction_id,
        experiment_name="exp",
        task_id="HumanEval/0",
        repetition_seed=0,
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
            repetition_seed=0,
            config_axis=dimensions_id,
        ),
        created_at=NOW,
    )

    round_tripped = io.prediction_spec_record_from_row(
        io.prediction_spec_row(record)
    )

    assert round_tripped == record


def test_node_attempt_row_round_trips_through_record_from_row() -> None:
    provider = ProviderConfigRef(
        provider_kind=ProviderKind.OPENAI,
        endpoint_kind=EndpointKind.RESPONSES,
        model="decoder-model",
        throttle_key="openai:responses:decoder-model",
    )
    record = NodeAttemptRecord(
        node_attempt_id="node-attempt-1",
        generation_run_id="run-1",
        prediction_id="prediction-1",
        node_id="decoder",
        attempt_index=0,
        status=NodeAttemptStatus.SUCCESS,
        provider_config=provider,
        output=io.node_output_payload_from_graph_output(
            NodeOutput(values={"code": "def add(): pass"})
        ),
        started_at=NOW,
        completed_at=NOW,
    )

    round_tripped = io.node_attempt_record_from_row(
        io.node_attempt_row(record)
    )

    assert round_tripped == record


def test_score_attempt_row_round_trips_through_record_from_row() -> None:
    record = ScoreAttemptRecord(
        score_attempt_id="score-1",
        prediction_id="prediction-1",
        generation_run_id="run-1",
        attempt_index=0,
        scoring_profile_id="humaneval",
        scoring_profile_version="v1",
        parser_profile_id="best-effort",
        parser_version="v1",
        dataset_name=DEFAULT_SCORE_DATASET_NAME,
        dataset_split=DEFAULT_SCORE_DATASET_SPLIT,
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
            ),
            custom={"passed": True},
        ),
        started_at=NOW,
        completed_at=NOW,
    )

    round_tripped = io.score_attempt_record_from_row(
        io.score_attempt_row(record)
    )

    assert round_tripped == record


def test_generation_run_row_round_trips_through_record_from_row() -> None:
    record = GenerationRunRecord(
        generation_run_id="run-1",
        prediction_id="prediction-1",
        attempt_index=0,
        status=GenerationRunStatus.SUCCESS,
        terminal_node_id="direct",
        terminal_output_node_id="direct",
        summary=GenerationRunSummaryPayload(
            execution_order=("direct",),
            terminal_node_id="direct",
            terminal_output="ok",
        ),
        started_at=NOW,
        completed_at=NOW,
    )

    round_tripped = io.generation_run_record_from_row(
        io.generation_run_row(record)
    )

    assert round_tripped == record


def test_projection_rows_round_trip() -> None:
    projection = PredictionProjectionRecord(
        prediction_id="prediction-1",
        generation_run_id="run-1",
        score_attempt_id="score-1",
        projection_profile_id="analysis",
        projection_version="v1",
        selected_at=NOW,
        selection_reason="latest validated score",
    )
    experiment = ExperimentRecord(
        experiment_name="exp",
        description="round trip",
        config_metadata={"seed": "seed"},
        created_at=NOW,
    )

    assert io.prediction_projection_record_from_row(
        io.prediction_projection_row(projection)
    ) == projection
    assert io.experiment_record_from_row(io.experiment_row(experiment)) == (
        experiment
    )


def test_prediction_spec_row_rejects_oversized_jsonb_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(records_models, "TASK_INPUTS_MAX_BYTES", 1024)
    oversized_prompt = "x" * (records_models.TASK_INPUTS_MAX_BYTES + 1)
    graph = _direct_graph()
    graph_id = graph_digest(graph)
    dimensions = DimensionsPayload(values={"budget_ratio": 0.5})
    dimensions_id = dimensions_digest(dimensions)
    provider = ProviderConfigRef(
        provider_kind=ProviderKind.OPENAI,
        endpoint_kind=EndpointKind.RESPONSES,
        model="decoder-model",
        throttle_key="openai:responses:decoder-model",
    )
    prediction_id = stable_prediction_id(
        experiment_name="exp",
        task_id="HumanEval/0",
        graph_digest=graph_id,
        dimensions_digest=dimensions_id,
        repetition_seed=0,
        provider_kind=provider.provider_kind.value,
        endpoint_kind=provider.endpoint_kind.value,
        model=provider.model,
        throttle_key=provider.throttle_key,
    )
    oversized_prompt = "x" * (records_models.TASK_INPUTS_MAX_BYTES + 1)
    with pytest.raises(ValidationError, match="task inputs"):
        PredictionSpecRecord(
            prediction_id=prediction_id,
            experiment_name="exp",
            task_id="HumanEval/0",
            repetition_seed=0,
            graph=GraphSnapshotPayload(
                graph=graph,
                graph_digest=graph_id,
                layout="direct",
            ),
            dimensions=dimensions,
            dimensions_digest=dimensions_id,
            task=TaskSnapshotPayload(
                task_id="HumanEval/0",
                inputs=TaskInputsPayload(values={"prompt": oversized_prompt}),
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
                repetition_seed=0,
                config_axis=dimensions_id,
            ),
            created_at=NOW,
        )


def test_node_attempt_row_rejects_oversized_usage_metadata_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(records_models, "PROVIDER_TELEMETRY_MAX_BYTES", 1024)
    provider = ProviderConfigRef(
        provider_kind=ProviderKind.OPENAI,
        endpoint_kind=EndpointKind.RESPONSES,
        model="decoder-model",
        throttle_key="openai:responses:decoder-model",
    )
    telemetry_limit = records_models.PROVIDER_TELEMETRY_MAX_BYTES + 1
    oversized_metadata = {"blob": "x" * telemetry_limit}
    with pytest.raises(ValidationError, match="usage metadata"):
        NodeAttemptRecord(
            node_attempt_id="node-attempt-1",
            generation_run_id="run-1",
            prediction_id="prediction-1",
            node_id="decoder",
            attempt_index=0,
            status=NodeAttemptStatus.SUCCESS,
            provider_config=provider,
            output=NodeOutputPayload(values={"output": "ok"}),
            usage_cost=UsageCostPayload(usage_metadata=oversized_metadata),
            started_at=NOW,
            completed_at=NOW,
        )


def test_node_attempt_row_rejects_oversized_response_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(records_models, "PROVIDER_TELEMETRY_MAX_BYTES", 1024)
    provider = ProviderConfigRef(
        provider_kind=ProviderKind.OPENAI,
        endpoint_kind=EndpointKind.RESPONSES,
        model="decoder-model",
        throttle_key="openai:responses:decoder-model",
    )
    telemetry_limit = records_models.PROVIDER_TELEMETRY_MAX_BYTES + 1
    oversized_metadata = {"blob": "x" * telemetry_limit}
    with pytest.raises(ValidationError, match="response metadata"):
        NodeAttemptRecord(
            node_attempt_id="node-attempt-1",
            generation_run_id="run-1",
            prediction_id="prediction-1",
            node_id="decoder",
            attempt_index=0,
            status=NodeAttemptStatus.SUCCESS,
            provider_config=provider,
            output=NodeOutputPayload(values={"output": "ok"}),
            response_metadata=ResponseMetadataPayload(
                response_metadata=oversized_metadata,
            ),
            started_at=NOW,
            completed_at=NOW,
        )


def test_node_attempt_row_rejects_oversized_node_output_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(records_models, "NODE_OUTPUT_MAX_BYTES", 1024)
    provider = ProviderConfigRef(
        provider_kind=ProviderKind.OPENAI,
        endpoint_kind=EndpointKind.RESPONSES,
        model="decoder-model",
        throttle_key="openai:responses:decoder-model",
    )
    oversized_output = "x" * (records_models.NODE_OUTPUT_MAX_BYTES + 1)
    with pytest.raises(ValidationError, match="node output"):
        NodeAttemptRecord(
            node_attempt_id="node-attempt-1",
            generation_run_id="run-1",
            prediction_id="prediction-1",
            node_id="decoder",
            attempt_index=0,
            status=NodeAttemptStatus.SUCCESS,
            provider_config=provider,
            output=NodeOutputPayload(values={"output": oversized_output}),
            started_at=NOW,
            completed_at=NOW,
        )
