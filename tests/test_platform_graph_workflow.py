from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from dr_dspy.eval_failures import FailureClass, PermanentFailureError
from dr_dspy.graph import (
    BindingRef,
    FieldRole,
    FieldSpec,
    GraphRunStatus,
    GraphSpec,
    NodeConfig,
    NodeOutput,
    NodeSpec,
    graph_digest,
)
from dr_dspy.lm.boundary import (
    EndpointKind,
    ProviderConfig,
    ProviderKind,
)
from dr_dspy.platform.graph_workflow import execute_prediction_graph
from dr_dspy.platform.node_execution import (
    NodeStepResult,
    NodeStepStatus,
    execute_lm_node,
    provider_config_ref_for_node,
)
from dr_dspy.records import (
    DimensionsPayload,
    FailureMetadataPayload,
    GenerationRunStatus,
    GraphSnapshotPayload,
    PredictionSpecRecord,
    ProviderConfigRef,
    TaskInputsPayload,
    TaskSnapshotPayload,
    dimensions_digest,
    fair_order_key,
    stable_generation_run_id,
    stable_node_attempt_id,
    stable_prediction_id,
)

NOW = datetime(2026, 6, 29, 12, 0, tzinfo=UTC)
LATER = NOW + timedelta(seconds=5)


def _node(
    node_id: str,
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
        for name, ref in (bindings or {}).items()
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


def _provider(
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


def _spec(
    graph: GraphSpec,
    *,
    providers: tuple[ProviderConfigRef, ...] | None = None,
    provider_axis: ProviderConfigRef | None = None,
    layout: str = "direct",
    task_inputs: dict[str, Any] | None = None,
) -> PredictionSpecRecord:
    providers = providers or (_provider(),)
    provider_axis = provider_axis or providers[0]
    graph_id = graph_digest(graph)
    dimensions = DimensionsPayload(values={"temperature": 0.2})
    dimensions_id = dimensions_digest(dimensions)
    prediction_id = stable_prediction_id(
        experiment_name="exp",
        task_id="HumanEval/0",
        graph_digest=graph_id,
        dimensions_digest=dimensions_id,
        repetition_seed=0,
    )
    return PredictionSpecRecord(
        prediction_id=prediction_id,
        experiment_name="exp",
        task_id="HumanEval/0",
        repetition_seed=0,
        graph=GraphSnapshotPayload(
            graph=graph,
            graph_digest=graph_id,
            layout=layout,
        ),
        dimensions=dimensions,
        dimensions_digest=dimensions_id,
        task=TaskSnapshotPayload(
            task_id="HumanEval/0",
            inputs=TaskInputsPayload(
                values=task_inputs or {"prompt": "write add"}
            ),
        ),
        provider_configs=providers,
        provider_axis=provider_axis,
        fair_order_seed="seed",
        fair_order_key=fair_order_key(
            experiment_seed="seed",
            prediction_id=prediction_id,
            provider=provider_axis.provider_kind.value,
            endpoint_kind=provider_axis.endpoint_kind.value,
            model=provider_axis.model,
            throttle_key=provider_axis.throttle_key,
            graph_layout=layout,
            task_id="HumanEval/0",
            repetition_seed=0,
            config_axis=dimensions_id,
        ),
        created_at=NOW,
    )


def _step_success(
    node: NodeSpec,
    value: str,
    *,
    provider: ProviderConfigRef | None = None,
) -> NodeStepResult:
    return NodeStepResult.success(
        node_id=node.id,
        provider_config=provider or _provider(),
        output=NodeOutput(values={node.config.output_field: value}),
        usage_metadata={"total_tokens": 3},
        provider_cost=0.01,
        response_metadata={"id": f"response-{node.id}"},
        started_at=NOW,
        completed_at=NOW,
    )


def _step_error(node: NodeSpec, message: str) -> NodeStepResult:
    return NodeStepResult(
        node_id=node.id,
        status=NodeStepStatus.ERROR,
        provider_config=_provider(),
        failure=FailureMetadataPayload(
            failure_class=FailureClass.PERMANENT,
            error_type="builtins.RuntimeError",
            message=message,
            metadata={"node_id": node.id},
        ),
        started_at=NOW,
        completed_at=NOW,
    )


def test_direct_graph_success_persists_generation_and_node_attempt() -> None:
    graph = GraphSpec(
        nodes=(_node("direct", bindings={"prompt": "task.prompt"}),),
        terminal_node_id="direct",
    )
    spec = _spec(graph)
    generation_run_id = stable_generation_run_id(
        prediction_id=spec.prediction_id,
        attempt_index=0,
    )

    execution = execute_prediction_graph(
        spec=spec,
        attempt_index=0,
        generation_run_id=generation_run_id,
        started_at=NOW,
        completed_at=LATER,
        run_node_step=lambda step_spec, node, inputs: _step_success(
            node,
            f"code for {inputs['prompt']}",
        ),
    )

    assert execution.generation_run.status is GenerationRunStatus.SUCCESS
    assert execution.generation_run.summary.terminal_output == (
        "code for write add"
    )
    assert len(execution.node_attempts) == 1
    attempt = execution.node_attempts[0]
    assert attempt.node_id == "direct"
    assert attempt.output is not None
    assert attempt.output.values == {"output": "code for write add"}
    assert attempt.usage_cost.provider_cost == 0.01
    assert attempt.response_metadata.response_metadata == {
        "id": "response-direct"
    }


def test_encdec_graph_success_persists_encoder_and_decoder_attempts() -> None:
    encoder = _node(
        "encoder",
        bindings={"prompt": "task.prompt"},
        output_field="description",
        user_prompt_template="Describe {prompt}",
    )
    decoder = _node(
        "decoder",
        bindings={"description": "encoder.description"},
        output_field="code",
        user_prompt_template="Write code from {description}",
    )
    graph = GraphSpec(nodes=(decoder, encoder), terminal_node_id="decoder")
    spec = _spec(graph, layout="encdec")

    def run_node_step(
        step_spec: PredictionSpecRecord,
        node: NodeSpec,
        inputs: Mapping[str, Any],
    ) -> NodeStepResult:
        if node.id == "encoder":
            return _step_success(node, "plain description")
        return _step_success(
            node,
            f"def f(): return {inputs['description']!r}",
        )

    execution = execute_prediction_graph(
        spec=spec,
        attempt_index=0,
        generation_run_id="run-1",
        started_at=NOW,
        completed_at=LATER,
        run_node_step=run_node_step,
    )

    assert execution.generation_run.status is GenerationRunStatus.SUCCESS
    assert [attempt.node_id for attempt in execution.node_attempts] == [
        "encoder",
        "decoder",
    ]
    assert execution.generation_run.summary.terminal_output == (
        "def f(): return 'plain description'"
    )


def test_failed_upstream_node_blocks_downstream_without_node_attempt() -> None:
    encoder = _node("encoder", bindings={"prompt": "task.prompt"})
    decoder = _node("decoder", bindings={"text": "encoder"})
    graph = GraphSpec(nodes=(decoder, encoder), terminal_node_id="decoder")
    spec = _spec(graph, layout="encdec")

    execution = execute_prediction_graph(
        spec=spec,
        attempt_index=0,
        generation_run_id="run-1",
        started_at=NOW,
        completed_at=LATER,
        run_node_step=lambda step_spec, node, inputs: _step_error(
            node,
            "provider failed",
        ),
    )

    assert execution.graph_result.status is GraphRunStatus.BLOCKED
    assert execution.generation_run.status is GenerationRunStatus.BLOCKED
    assert [attempt.node_id for attempt in execution.node_attempts] == [
        "encoder"
    ]
    assert execution.node_attempts[0].failure is not None
    assert execution.generation_run.summary.terminal_error is not None
    assert execution.generation_run.summary.terminal_error.blocked_by == (
        "encoder",
    )


def test_independent_branch_failure_records_partial_generation() -> None:
    graph = GraphSpec(
        nodes=(_node("terminal"), _node("bad")),
        terminal_node_id="terminal",
    )
    spec = _spec(graph, task_inputs={})

    def run_node_step(
        step_spec: PredictionSpecRecord,
        node: NodeSpec,
        inputs: Mapping[str, Any],
    ) -> NodeStepResult:
        if node.id == "bad":
            return _step_error(node, "boom")
        return _step_success(node, "ok")

    execution = execute_prediction_graph(
        spec=spec,
        attempt_index=0,
        generation_run_id="run-1",
        started_at=NOW,
        completed_at=LATER,
        run_node_step=run_node_step,
    )

    assert execution.graph_result.status is GraphRunStatus.PARTIAL
    assert execution.generation_run.status is GenerationRunStatus.PARTIAL
    assert [attempt.node_id for attempt in execution.node_attempts] == [
        "bad",
        "terminal",
    ]


def test_lm_node_executor_sends_exact_messages_and_metadata() -> None:
    node = _node(
        "direct",
        bindings={"prompt": "task.prompt"},
        output_field="code",
        user_prompt_template="Solve: {prompt}",
        system_prompt="Write Python.",
        parameters={"token_limit": 20},
    )
    graph = GraphSpec(nodes=(node,), terminal_node_id="direct")
    spec = _spec(graph)
    captured: dict[str, Any] = {}

    def client_factory(config: ProviderConfig) -> object:
        captured["config"] = config
        return object()

    def provider_caller(client: Any, request: Any) -> dict[str, Any]:
        captured["request"] = request
        return {
            "id": "resp-1",
            "model": "gpt-test",
            "status": "completed",
            "output_text": "def add(a, b): return a + b",
            "usage": {
                "input_tokens": 1,
                "output_tokens": 2,
                "total_tokens": 3,
                "cost": 0.02,
            },
        }

    result = execute_lm_node(
        spec=spec,
        node=node,
        node_inputs={"prompt": "write add"},
        client_factory=client_factory,
        provider_caller=provider_caller,
    )

    assert result.status is NodeStepStatus.SUCCESS
    request = captured["request"]
    assert request.kwargs == {
        "model": "gpt-test",
        "instructions": "Write Python.",
        "input": [{"role": "user", "content": "Solve: write add"}],
        "temperature": 0.2,
        "max_output_tokens": 20,
    }
    assert result.output is not None
    assert result.output.values == {"code": "def add(a, b): return a + b"}
    assert result.usage_cost.usage_metadata["total_tokens"] == 3
    assert result.usage_cost.provider_cost == 0.02
    assert result.response_metadata.response_metadata["id"] == "resp-1"


def test_multiple_provider_configs_require_node_provider_config_id() -> None:
    node = _node("direct", bindings={"prompt": "task.prompt"})
    graph = GraphSpec(nodes=(node,), terminal_node_id="direct")
    spec = _spec(
        graph,
        providers=(
            _provider(config_id="encoder", model="encoder-model"),
            _provider(config_id="decoder", model="decoder-model"),
        ),
    )

    with pytest.raises(PermanentFailureError, match="provider_config_id"):
        provider_config_ref_for_node(spec=spec, node=node)


def test_deterministic_generation_and_node_attempt_ids() -> None:
    generation_run_id = stable_generation_run_id(
        prediction_id="prediction-1",
        attempt_index=2,
    )

    assert generation_run_id == stable_generation_run_id(
        prediction_id="prediction-1",
        attempt_index=2,
    )
    assert generation_run_id != stable_generation_run_id(
        prediction_id="prediction-1",
        attempt_index=3,
    )
    assert stable_node_attempt_id(
        generation_run_id=generation_run_id,
        node_id="decoder",
        attempt_index=0,
    ) == stable_node_attempt_id(
        generation_run_id=generation_run_id,
        node_id="decoder",
        attempt_index=0,
    )
