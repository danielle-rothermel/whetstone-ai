"""Shared HumanEval scoring fixtures for unit and integration tests."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

from dr_dspy.graph import (
    BindingRef,
    FieldRole,
    FieldSpec,
    GraphSpec,
    NodeConfig,
    NodeSpec,
    graph_digest,
)
from dr_dspy.humaneval.task import HumanEvalTask
from dr_dspy.lm.boundary import EndpointKind, ProviderKind
from dr_dspy.records import (
    DimensionsPayload,
    GenerationRunRecord,
    GenerationRunStatus,
    GenerationRunSummaryPayload,
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
    stable_generation_run_id,
    stable_prediction_id,
)

NOW = datetime(2026, 6, 29, 12, 0, tzinfo=UTC)
LATER = NOW + timedelta(seconds=1)


def scoring_task(*, test: str | None = None) -> HumanEvalTask:
    return HumanEvalTask(
        task_id="HumanEval/fixture",
        prompt="def add_one(x):\n",
        canonical_solution="    return x + 1\n",
        entry_point="add_one",
        test=test
        or (
            "def check(candidate):\n"
            "    inputs = [(1,), (2,)]\n"
            "    results = [2, 3]\n"
            "    for inp, expected in zip(inputs, results):\n"
            "        assertion(candidate(*inp), expected)\n"
        ),
    )


def scoring_node(
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


def scoring_graph(layout: str = "direct") -> GraphSpec:
    if layout == "encdec":
        return GraphSpec(
            nodes=(
                scoring_node(
                    "encoder",
                    bindings={"prompt": "task.prompt"},
                    output_field="description",
                ),
                scoring_node(
                    "decoder",
                    bindings={"description": "encoder.description"},
                    output_field="code",
                ),
            ),
            terminal_node_id="decoder",
        )
    return GraphSpec(
        nodes=(scoring_node("direct", bindings={"prompt": "task.prompt"}),),
        terminal_node_id="direct",
    )


def scoring_provider() -> ProviderConfigRef:
    return ProviderConfigRef(
        provider_kind=ProviderKind.OPENAI,
        endpoint_kind=EndpointKind.RESPONSES,
        model="gpt-test",
        throttle_key="openai:responses:gpt-test",
    )


def scoring_prediction_spec(layout: str = "direct") -> PredictionSpecRecord:
    graph = scoring_graph(layout)
    graph_id = graph_digest(graph)
    dimensions = DimensionsPayload(values={"temperature": 0.2})
    dimensions_id = dimensions_digest(dimensions)
    provider = scoring_provider()
    prediction_id = stable_prediction_id(
        experiment_name="exp",
        task_id="HumanEval/fixture",
        graph_digest=graph_id,
        dimensions_digest=dimensions_id,
        repetition_seed=0,
        provider_kind=provider.provider_kind.value,
        endpoint_kind=provider.endpoint_kind.value,
        model=provider.model,
        throttle_key=provider.throttle_key,
    )
    return PredictionSpecRecord(
        prediction_id=prediction_id,
        experiment_name="exp",
        task_id="HumanEval/fixture",
        repetition_seed=0,
        graph=GraphSnapshotPayload(
            graph=graph,
            graph_digest=graph_id,
            layout=layout,
        ),
        dimensions=dimensions,
        dimensions_digest=dimensions_id,
        task=TaskSnapshotPayload(
            task_id="HumanEval/fixture",
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
            graph_layout=layout,
            task_id="HumanEval/fixture",
            repetition_seed=0,
            config_axis=dimensions_id,
        ),
        created_at=NOW,
    )


def successful_generation_run(
    spec: PredictionSpecRecord,
    raw_generation: Any,
    *,
    generation_run_id: str | None = None,
) -> GenerationRunRecord:
    resolved_generation_run_id = generation_run_id or stable_generation_run_id(
        prediction_id=spec.prediction_id,
        attempt_index=0,
    )
    return GenerationRunRecord(
        generation_run_id=resolved_generation_run_id,
        prediction_id=spec.prediction_id,
        attempt_index=0,
        status=GenerationRunStatus.SUCCESS,
        terminal_node_id=spec.graph.graph.terminal_node_id,
        terminal_output_node_id=spec.graph.graph.terminal_node_id,
        summary=GenerationRunSummaryPayload(
            execution_order=tuple(node.id for node in spec.graph.graph.nodes),
            terminal_node_id=spec.graph.graph.terminal_node_id,
            terminal_output=raw_generation,
        ),
        started_at=NOW,
        completed_at=LATER,
    )


def unique_generation_run_id() -> str:
    return f"integration-score-{uuid.uuid4().hex}"


def seeded_scoring_target(
    *,
    raw_generation: str = "def add_one(x):\n    return x + 1\n",
) -> tuple[PredictionSpecRecord, GenerationRunRecord]:
    spec = scoring_prediction_spec()
    run = successful_generation_run(
        spec,
        raw_generation,
        generation_run_id=unique_generation_run_id(),
    )
    return spec, run


def scoring_node_attempt(
    spec: PredictionSpecRecord,
    *,
    node_id: str,
    values: Mapping[str, Any],
) -> NodeAttemptRecord:
    return NodeAttemptRecord(
        node_attempt_id=f"node-{node_id}",
        generation_run_id=stable_generation_run_id(
            prediction_id=spec.prediction_id,
            attempt_index=0,
        ),
        prediction_id=spec.prediction_id,
        node_id=node_id,
        attempt_index=0,
        status=NodeAttemptStatus.SUCCESS,
        provider_config=spec.provider_axis,
        output=NodeOutputPayload(values=dict(values)),
        started_at=NOW,
        completed_at=LATER,
    )
