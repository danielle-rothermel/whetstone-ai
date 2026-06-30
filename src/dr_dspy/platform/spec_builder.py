"""Production helpers for building validated v1 prediction specs."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    model_validator,
)

from dr_dspy.graph import (
    BindingRef,
    FieldRole,
    FieldSpec,
    GraphSpec,
    NodeConfig,
    NodeSpec,
    graph_digest,
)
from dr_dspy.humaneval.sampling import (
    SampledHumanEvalTask,
    sample_human_eval_tasks,
    sample_human_eval_tasks_from_rows,
)
from dr_dspy.humaneval.task import HumanEvalTask
from dr_dspy.lm.boundary import EndpointKind, ProviderKind
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

DEFAULT_FAIR_ORDER_SEED = "seed"
DEFAULT_DIMENSIONS_AXES: tuple[dict[str, Any], ...] = ({"temperature": 0.2},)
DEFAULT_REPETITION_SEEDS: tuple[int, ...] = (0,)


class GraphLayout(StrEnum):
    DIRECT = "direct"
    ENCDEC = "encdec"


class DatasetSpecConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: StrictStr
    split: StrictStr
    sample_seed: StrictInt = 0
    sample_count: StrictInt = 1

    @model_validator(mode="after")
    def validate_sample_count(self) -> DatasetSpecConfig:
        if self.sample_count < 1:
            raise ValueError("sample_count must be positive")
        return self


class ProviderSpecConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: StrictStr
    config_id: StrictStr | None = None
    provider_kind: ProviderKind = ProviderKind.OPENAI
    endpoint_kind: EndpointKind = EndpointKind.RESPONSES
    parameters: dict[StrictStr, Any] = Field(default_factory=dict)


class ExperimentSpecConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_name: StrictStr
    graph_layout: GraphLayout
    dataset: DatasetSpecConfig
    fair_order_seed: StrictStr = DEFAULT_FAIR_ORDER_SEED
    repetition_seeds: tuple[StrictInt, ...] = DEFAULT_REPETITION_SEEDS
    dimensions_axes: tuple[dict[StrictStr, Any], ...] = DEFAULT_DIMENSIONS_AXES
    providers: tuple[ProviderSpecConfig, ...]

    @model_validator(mode="after")
    def validate_layout_providers(self) -> ExperimentSpecConfig:
        if not self.repetition_seeds:
            raise ValueError("repetition_seeds must not be empty")
        if not self.dimensions_axes:
            raise ValueError("dimensions_axes must not be empty")
        if self.graph_layout is GraphLayout.DIRECT:
            if len(self.providers) != 1:
                raise ValueError(
                    "direct graph_layout requires exactly one provider"
                )
        elif len(self.providers) != 2:
            raise ValueError(
                "encdec graph_layout requires exactly two providers"
            )
        else:
            config_ids = {
                provider.config_id for provider in self.providers
            }
            if config_ids != {"encoder", "decoder"}:
                raise ValueError(
                    "encdec providers must use config_id encoder and decoder"
                )
        return self


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


def direct_graph() -> GraphSpec:
    return GraphSpec(
        nodes=(direct_node("direct", output_field="output"),),
        terminal_node_id="direct",
    )


def encdec_graph() -> GraphSpec:
    return GraphSpec(
        nodes=(decoder_node(), encoder_node()),
        terminal_node_id="decoder",
    )


def provider_ref(
    *,
    config_id: str | None = "main",
    model: str = "gpt-test",
    provider_kind: ProviderKind = ProviderKind.OPENAI,
    endpoint_kind: EndpointKind = EndpointKind.RESPONSES,
    parameters: dict[str, Any] | None = None,
) -> ProviderConfigRef:
    return ProviderConfigRef(
        provider_kind=provider_kind,
        endpoint_kind=endpoint_kind,
        model=model,
        config_id=config_id,
        throttle_key=f"{provider_kind.value}:{endpoint_kind.value}:{model}",
        parameters=dict(parameters or {"temperature": 0.2}),
    )


def provider_ref_from_config(config: ProviderSpecConfig) -> ProviderConfigRef:
    return provider_ref(
        config_id=config.config_id,
        model=config.model,
        provider_kind=config.provider_kind,
        endpoint_kind=config.endpoint_kind,
        parameters=config.parameters,
    )


def task_snapshot_from_humaneval(task: HumanEvalTask) -> TaskSnapshotPayload:
    return TaskSnapshotPayload(
        task_id=task.task_id,
        inputs=TaskInputsPayload(
            values={
                "prompt": task.prompt,
                "test": task.test,
                "entry_point": task.entry_point,
            }
        ),
        metadata={
            "canonical_solution": task.canonical_solution,
            "ground_truth_code": task.ground_truth_code,
        },
    )


def prediction_spec(
    graph: GraphSpec,
    *,
    providers: tuple[ProviderConfigRef, ...] | None = None,
    provider_axis: ProviderConfigRef | None = None,
    layout: str = "direct",
    task: TaskSnapshotPayload | None = None,
    task_id: str = "HumanEval/0",
    dimensions: DimensionsPayload | None = None,
    experiment_name: str = "exp",
    repetition_seed: int = 0,
    fair_order_seed: str = DEFAULT_FAIR_ORDER_SEED,
    created_at: datetime | None = None,
) -> PredictionSpecRecord:
    providers = providers or (provider_ref(),)
    provider_axis = provider_axis or providers[0]
    graph_id = graph_digest(graph)
    dimensions = dimensions or DimensionsPayload(values={"temperature": 0.2})
    dimensions_id = dimensions_digest(dimensions)
    task_snapshot = task or TaskSnapshotPayload(
        task_id=task_id,
        inputs=TaskInputsPayload(values={"prompt": "write add"}),
    )
    if task_snapshot.task_id != task_id:
        raise ValueError("task snapshot task_id must match task_id")
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
        task=task_snapshot,
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
        created_at=created_at or datetime.now(UTC),
    )


def encdec_spec(
    *,
    task: TaskSnapshotPayload | None = None,
    task_id: str = "HumanEval/0",
    experiment_name: str = "exp",
) -> PredictionSpecRecord:
    return prediction_spec(
        encdec_graph(),
        layout=GraphLayout.ENCDEC.value,
        providers=(
            provider_ref(config_id="encoder", model="encoder-model"),
            provider_ref(config_id="decoder", model="decoder-model"),
        ),
        task=task,
        task_id=task_id,
        experiment_name=experiment_name,
    )


def graph_for_layout(layout: GraphLayout) -> GraphSpec:
    if layout is GraphLayout.DIRECT:
        return direct_graph()
    return encdec_graph()


def providers_for_config(
    config: ExperimentSpecConfig,
) -> tuple[ProviderConfigRef, ...]:
    return tuple(
        provider_ref_from_config(provider) for provider in config.providers
    )


def sample_tasks_for_config(
    config: ExperimentSpecConfig,
    *,
    rows: Sequence[dict[str, Any]] | None = None,
) -> tuple[SampledHumanEvalTask, ...]:
    if rows is None:
        return tuple(
            sample_human_eval_tasks(
                seed=config.dataset.sample_seed,
                sample_count=config.dataset.sample_count,
                dataset_name=config.dataset.name,
                dataset_split=config.dataset.split,
            )
        )
    return tuple(
        sample_human_eval_tasks_from_rows(
            rows,
            seed=config.dataset.sample_seed,
            sample_count=config.dataset.sample_count,
        )
    )


def iter_experiment_specs(
    config: ExperimentSpecConfig,
    *,
    rows: Sequence[dict[str, Any]] | None = None,
) -> Iterator[PredictionSpecRecord]:
    graph = graph_for_layout(config.graph_layout)
    layout = config.graph_layout.value
    providers = providers_for_config(config)
    provider_axis = providers[0]
    sampled_tasks = sample_tasks_for_config(config, rows=rows)
    for sampled in sampled_tasks:
        task_snapshot = task_snapshot_from_humaneval(sampled.task)
        for repetition_seed in config.repetition_seeds:
            for axis_values in config.dimensions_axes:
                dimensions = DimensionsPayload(values=dict(axis_values))
                yield prediction_spec(
                    graph,
                    providers=providers,
                    provider_axis=provider_axis,
                    layout=layout,
                    task=task_snapshot,
                    task_id=sampled.task.task_id,
                    dimensions=dimensions,
                    experiment_name=config.experiment_name,
                    repetition_seed=repetition_seed,
                    fair_order_seed=config.fair_order_seed,
                )


def load_experiment_spec_config(path: Path) -> ExperimentSpecConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ExperimentSpecConfig.model_validate(payload)


def write_prediction_specs_jsonl(
    specs: Sequence[PredictionSpecRecord],
    output: Path | None,
) -> Literal["stdout"] | Path:
    lines = [spec.model_dump_json() for spec in specs]
    payload = "\n".join(lines)
    if payload:
        payload += "\n"
    if output is None:
        print(payload, end="")
        return "stdout"
    output.write_text(payload, encoding="utf-8")
    return output
