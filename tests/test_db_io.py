from __future__ import annotations

from datetime import UTC, datetime

from dr_dspy.db import io
from dr_dspy.graph import (
    BindingRef,
    FieldRole,
    FieldSpec,
    GraphSpec,
    NodeConfig,
    NodeError,
    NodeOutput,
    NodeSpec,
    graph_digest,
)
from dr_dspy.humaneval.scoring import GeneratedCodeOutcome
from dr_dspy.lm.boundary import EndpointKind, ProviderKind
from dr_dspy.records import (
    DimensionsPayload,
    GraphSnapshotPayload,
    MetricsPayload,
    NodeAttemptRecord,
    NodeAttemptStatus,
    PredictionSpecRecord,
    ProviderConfigRef,
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


def test_prediction_spec_row_uses_explicit_provider_axis() -> None:
    graph = _direct_graph()
    graph_id = graph_digest(graph)
    dimensions = DimensionsPayload(values={"budget_ratio": 0.5})
    dimensions_id = dimensions_digest(dimensions)
    prediction_id = stable_prediction_id(
        experiment_name="exp",
        task_id="HumanEval/0",
        graph_digest=graph_id,
        dimensions_digest=dimensions_id,
        repetition_seed=0,
    )
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
    assert row["provider_config"] == provider.model_dump(mode="json")
    assert row["output"] == {
        "values": {"code": "def add(): pass"},
        "metadata": {},
    }


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
        "message": "boom",
        "metadata": {"node": "decoder"},
    }


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
        "custom": {"passed": True},
    }
