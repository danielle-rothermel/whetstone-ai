"""Shared prediction-graph fixtures for unit and integration tests."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from dr_dspy.graph import (
    BindingRef,
    FieldRole,
    FieldSpec,
    GraphSpec,
    NodeConfig,
    NodeOutput,
    NodeSpec,
    graph_digest,
)
from dr_dspy.lm.boundary import EndpointKind, ProviderKind
from dr_dspy.platform.node_execution import NodeStepResult, NodeStepStatus
from dr_dspy.records import (
    DimensionsPayload,
    GraphSnapshotPayload,
    PredictionSpecRecord,
    ProviderConfigRef,
    TaskInputsPayload,
    TaskSnapshotPayload,
    dimensions_digest,
    fair_order_key,
    stable_prediction_id,
)

NOW = datetime(2026, 6, 29, 12, 0, tzinfo=UTC)


def direct_node(
    node_id: str = "direct",
    *,
    bindings: dict[str, str] | None = None,
    output_field: str = "output",
    user_prompt_template: str = "{prompt}",
    system_prompt: str | None = None,
    provider_config_id: str | None = None,
    parameters: dict[str, Any] | None = None,
) -> NodeSpec:
    input_bindings = {
        name: BindingRef.model_validate(ref)
        for name, ref in (bindings or {"prompt": "task.prompt"}).items()
    }
    fields = [
        FieldSpec(name=name, role=FieldRole.INPUT)
        for name in input_bindings
    ]
    fields.append(FieldSpec(name=output_field, role=FieldRole.OUTPUT))
    metadata: dict[str, Any] = {"user_prompt_template": user_prompt_template}
    if system_prompt is not None:
        metadata["system_prompt"] = system_prompt
    if provider_config_id is not None:
        metadata["provider_config_id"] = provider_config_id
    return NodeSpec(
        id=node_id,
        config=NodeConfig(
            fields=tuple(fields),
            input_bindings=input_bindings,
            output_field=output_field,
            parameters=parameters or {},
            metadata=metadata,
        ),
    )


def encoder_node() -> NodeSpec:
    return direct_node(
        "encoder",
        bindings={"prompt": "task.prompt"},
        output_field="description",
        user_prompt_template="Describe {prompt}",
        provider_config_id="encoder",
    )


def decoder_node() -> NodeSpec:
    return direct_node(
        "decoder",
        bindings={"description": "encoder.description"},
        output_field="code",
        user_prompt_template="Write code from {description}",
        provider_config_id="decoder",
    )


def encdec_graph() -> GraphSpec:
    return GraphSpec(
        nodes=(decoder_node(), encoder_node()),
        terminal_node_id="decoder",
    )


def encdec_spec(
    *,
    task_inputs: dict[str, Any] | None = None,
    experiment_name: str = "exp",
) -> PredictionSpecRecord:
    return prediction_spec(
        encdec_graph(),
        layout="encdec",
        providers=(
            provider_ref(config_id="encoder", model="encoder-model"),
            provider_ref(config_id="decoder", model="decoder-model"),
        ),
        task_inputs=task_inputs,
        experiment_name=experiment_name,
    )


def provider_ref(
    *,
    config_id: str | None = "main",
    model: str = "gpt-test",
) -> ProviderConfigRef:
    return ProviderConfigRef(
        provider_kind=ProviderKind.OPENAI,
        endpoint_kind=EndpointKind.RESPONSES,
        model=model,
        config_id=config_id,
        throttle_key=f"openai:responses:{model}",
        parameters={"temperature": 0.2},
    )


def prediction_spec(
    graph: GraphSpec,
    *,
    providers: tuple[ProviderConfigRef, ...] | None = None,
    provider_axis: ProviderConfigRef | None = None,
    layout: str = "direct",
    task_inputs: dict[str, Any] | None = None,
    experiment_name: str = "exp",
    task_id: str = "HumanEval/0",
    repetition_seed: int = 0,
    fair_order_seed: str = "seed",
    created_at: datetime | None = None,
) -> PredictionSpecRecord:
    providers = providers or (provider_ref(),)
    provider_axis = provider_axis or providers[0]
    graph_id = graph_digest(graph)
    dimensions = DimensionsPayload(values={"temperature": 0.2})
    dimensions_id = dimensions_digest(dimensions)
    prediction_id = stable_prediction_id(
        experiment_name=experiment_name,
        task_id=task_id,
        graph_digest=graph_id,
        dimensions_digest=dimensions_id,
        repetition_seed=repetition_seed,
        provider_kind=provider_axis.provider_kind.value,
        endpoint_kind=provider_axis.endpoint_kind.value,
        model=provider_axis.model,
        throttle_key=provider_axis.throttle_key,
    )
    return PredictionSpecRecord(
        prediction_id=prediction_id,
        experiment_name=experiment_name,
        task_id=task_id,
        repetition_seed=repetition_seed,
        graph=GraphSnapshotPayload(
            graph=graph,
            graph_digest=graph_id,
            layout=layout,
        ),
        dimensions=dimensions,
        dimensions_digest=dimensions_id,
        task=TaskSnapshotPayload(
            task_id=task_id,
            inputs=TaskInputsPayload(
                values=task_inputs or {"prompt": "write add"}
            ),
        ),
        provider_configs=providers,
        provider_axis=provider_axis,
        fair_order_seed=fair_order_seed,
        fair_order_key=fair_order_key(
            experiment_seed=fair_order_seed,
            prediction_id=prediction_id,
            provider=provider_axis.provider_kind.value,
            endpoint_kind=provider_axis.endpoint_kind.value,
            model=provider_axis.model,
            throttle_key=provider_axis.throttle_key,
            graph_layout=layout,
            task_id=task_id,
            repetition_seed=repetition_seed,
            config_axis=dimensions_id,
        ),
        created_at=created_at or NOW,
    )


def step_success(
    node: NodeSpec,
    value: str,
    *,
    provider: ProviderConfigRef | None = None,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
) -> NodeStepResult:
    started = started_at or NOW
    completed = completed_at or NOW
    return NodeStepResult.success(
        node_id=node.id,
        provider_config=provider or provider_ref(),
        output=NodeOutput(values={node.config.output_field: value}),
        usage_metadata={"total_tokens": 3},
        provider_cost=0.01,
        response_metadata={"id": f"response-{node.id}"},
        started_at=started,
        completed_at=completed,
    )


def step_error(
    node: NodeSpec,
    message: str,
    *,
    provider: ProviderConfigRef | None = None,
) -> NodeStepResult:
    from dr_dspy.eval_failures import FailureClass
    from dr_dspy.records import FailureMetadataPayload

    return NodeStepResult(
        node_id=node.id,
        status=NodeStepStatus.ERROR,
        provider_config=provider or provider_ref(),
        failure=FailureMetadataPayload(
            failure_class=FailureClass.PERMANENT,
            error_type="PermanentFailureError",
            message=message,
            metadata={"node_id": node.id},
        ),
        started_at=NOW,
        completed_at=NOW,
    )
