from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from dr_dspy.graph import (
    BindingRef,
    FieldRole,
    FieldSpec,
    GraphSpec,
    NodeConfig,
    NodeSpec,
    graph_digest,
)
from dr_dspy.lm.boundary import EndpointKind, ProviderKind
from dr_dspy.records import (
    DimensionsPayload,
    GraphSnapshotPayload,
    NodeAttemptRecord,
    NodeAttemptStatus,
    NodeOutputPayload,
    PredictionSpecRecord,
    ProviderConfigRef,
    TaskInputsPayload,
    TaskSnapshotPayload,
    dimensions_digest,
    fair_order_key,
    stable_prediction_id,
)
from dr_dspy.records.providers import find_provider_config_ref

NOW = datetime(2026, 6, 29, 12, 0, tzinfo=UTC)


def _node(node_id: str) -> NodeSpec:
    return NodeSpec(
        id=node_id,
        config=NodeConfig(
            fields=(
                FieldSpec(name="prompt", role=FieldRole.INPUT),
                FieldSpec(name="output", role=FieldRole.OUTPUT),
            ),
            input_bindings={
                "prompt": BindingRef.model_validate("task.prompt"),
            },
            output_field="output",
        ),
    )


def _shared_provider(*, config_id: str | None) -> ProviderConfigRef:
    return ProviderConfigRef(
        provider_kind=ProviderKind.OPENAI,
        endpoint_kind=EndpointKind.RESPONSES,
        model="shared-model",
        config_id=config_id,
        throttle_key="openai:responses:shared-model",
    )


def test_prediction_spec_requires_config_id_for_ambiguous_configs() -> None:
    encoder = _shared_provider(config_id=None)
    decoder = _shared_provider(config_id="decoder")
    graph = GraphSpec(nodes=(_node("direct"),), terminal_node_id="direct")
    graph_id = graph_digest(graph)
    dimensions = DimensionsPayload(values={"temperature": 0.2})
    dimensions_id = dimensions_digest(dimensions)
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

    with pytest.raises(ValidationError, match="config_id is required"):
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
                inputs=TaskInputsPayload(values={"prompt": "write add"}),
            ),
            provider_configs=(encoder, decoder),
            provider_axis=decoder,
            fair_order_seed="seed",
            fair_order_key=fair_order_key(
                experiment_seed="seed",
                prediction_id=prediction_id,
                provider=decoder.provider_kind.value,
                endpoint_kind=decoder.endpoint_kind.value,
                model=decoder.model,
                throttle_key=decoder.throttle_key,
                graph_layout="direct",
                task_id="HumanEval/0",
                repetition_seed=0,
                config_axis=dimensions_id,
            ),
            created_at=NOW,
        )


def test_find_provider_config_ref_disambiguates_by_config_id() -> None:
    encoder = _shared_provider(config_id="encoder")
    decoder = _shared_provider(config_id="decoder")

    match = find_provider_config_ref(
        (encoder, decoder),
        provider_kind=decoder.provider_kind.value,
        endpoint_kind=decoder.endpoint_kind.value,
        model=decoder.model,
        throttle_key=decoder.throttle_key,
        config_id="decoder",
    )

    assert match.config_id == "decoder"


def test_successful_node_attempt_requires_provider_config() -> None:
    with pytest.raises(ValidationError, match="require provider_config"):
        NodeAttemptRecord(
            node_attempt_id="node-attempt-1",
            generation_run_id="run-1",
            prediction_id="prediction-1",
            node_id="direct",
            attempt_index=0,
            status=NodeAttemptStatus.SUCCESS,
            output=NodeOutputPayload(values={"output": "ok"}),
            started_at=NOW,
            completed_at=NOW,
        )
