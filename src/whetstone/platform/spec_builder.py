"""Production helpers for building validated v1 prediction specs."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from dr_code.humaneval import (
    HumanEvalTask,
    SampledHumanEvalTask,
    sample_human_eval_tasks_from_rows,
)
from dr_graph import (
    BindingRef,
    FieldRole,
    FieldSpec,
    GraphSpec,
    NodeConfig,
    NodeSpec,
    graph_digest,
)
from dr_providers import EndpointKind, ProviderKind
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    model_validator,
)

from whetstone.node_ops import LLM_CALL_OP
from whetstone.platform.dataset_snapshot import (
    HumanEvalSnapshot,
    load_humaneval_snapshot,
)
from whetstone.records import (
    DatasetSnapshotIdentityPayload,
    DimensionsPayload,
    GraphSnapshotPayload,
    PredictionSpecRecord,
    ProviderConfigRef,
    TaskInputsPayload,
    TaskSnapshotPayload,
    dimensions_digest,
    stable_prediction_id,
)

DEFAULT_DIMENSIONS_AXES: tuple[dict[str, Any], ...] = ({"temperature": 0.2},)
DEFAULT_REPETITION_SEEDS: tuple[int, ...] = (0,)
DEFAULT_HUMANEVAL_INSTRUCTIONS_START = (
    "Provide a concise description of the following code."
)
DEFAULT_MIN_ENCODER_CHAR_BUDGET = 50
DEFAULT_CONFIGS_ROOT = Path(__file__).resolve().parents[3] / "configs"

HUMANEVAL_ENCODER_USER_PROMPT_TEMPLATE = (
    "{instructions_start}\n"
    "Use at most {budget} characters.\n"
    "\n"
    "```python\n"
    "{gt_code}\n"
    "```\n"
    "{instructions_end}"
)
HUMANEVAL_DECODER_USER_PROMPT_TEMPLATE = (
    "Write functional code in Python according to the following description.\n"
    "Output only the final answer, without any descriptions or surrounding\n"
    "characters.\n"
    "\n"
    "{encoded_desc}"
)

EncDecShape = Literal["legacy", "humaneval"]


class GraphLayout(StrEnum):
    DIRECT = "direct"
    ENCDEC = "encdec"


class DatasetSpecConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: StrictStr
    split: StrictStr
    snapshot_path: StrictStr
    sample_seed: StrictInt = 0
    sample_count: StrictInt = 1

    @model_validator(mode="after")
    def validate_sample_count(self) -> DatasetSpecConfig:
        if self.sample_count < 1:
            raise ValueError("sample_count must be positive")
        if not self.snapshot_path:
            raise ValueError("snapshot_path must not be empty")
        return self


class ProviderSpecConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: StrictStr
    config_id: StrictStr | None = None
    provider_kind: ProviderKind = ProviderKind.OPENAI
    endpoint_kind: EndpointKind = EndpointKind.RESPONSES
    parameters: dict[StrictStr, Any] = Field(default_factory=dict)


class HumanevalEncDecConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instructions_start: StrictStr = DEFAULT_HUMANEVAL_INSTRUCTIONS_START
    instructions_end: StrictStr = ""
    encoder_system_prompt: StrictStr = ""
    min_encoder_char_budget: StrictInt = DEFAULT_MIN_ENCODER_CHAR_BUDGET

    @model_validator(mode="after")
    def validate_min_encoder_char_budget(self) -> HumanevalEncDecConfig:
        if self.min_encoder_char_budget < 1:
            raise ValueError("min_encoder_char_budget must be positive")
        return self


def _validate_encdec_provider_config_ids(
    providers: tuple[ProviderSpecConfig, ...],
) -> None:
    if len(providers) != 2:
        raise ValueError("encdec providers require exactly two entries")
    config_ids = {provider.config_id for provider in providers}
    if config_ids != {"encoder", "decoder"}:
        raise ValueError(
            "encdec providers must use config_id encoder and decoder"
        )


def _validate_humaneval_experiment_axes(
    *,
    graph_layout: GraphLayout,
    encdec_shape: EncDecShape,
    dimensions_axes: tuple[dict[StrictStr, Any], ...],
) -> None:
    if encdec_shape != "humaneval":
        return
    if graph_layout is not GraphLayout.ENCDEC:
        raise ValueError("humaneval encdec_shape requires encdec graph_layout")
    for axis in dimensions_axes:
        if "compression_target" not in axis:
            raise ValueError(
                "humaneval dimensions_axes require compression_target"
            )


class ModelConfigFragment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: StrictStr
    providers: tuple[ProviderSpecConfig, ...]

    @model_validator(mode="after")
    def validate_encdec_providers(self) -> ModelConfigFragment:
        _validate_encdec_provider_config_ids(self.providers)
        return self


class SplitConfigFragment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: StrictStr
    dataset: DatasetSpecConfig


class ComposableExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_name: StrictStr
    graph_layout: GraphLayout
    split: StrictStr
    model_configs: tuple[StrictStr, ...]
    repetition_seeds: tuple[StrictInt, ...] = DEFAULT_REPETITION_SEEDS
    dimensions_axes: tuple[dict[StrictStr, Any], ...] = DEFAULT_DIMENSIONS_AXES
    encdec_shape: EncDecShape = "legacy"
    humaneval_encdec: HumanevalEncDecConfig | None = None

    @model_validator(mode="after")
    def validate_composable_experiment(self) -> ComposableExperimentConfig:
        if not self.model_configs:
            raise ValueError("model_configs must not be empty")
        if not self.repetition_seeds:
            raise ValueError("repetition_seeds must not be empty")
        if not self.dimensions_axes:
            raise ValueError("dimensions_axes must not be empty")
        if self.graph_layout is GraphLayout.DIRECT:
            raise ValueError(
                "composable experiments currently require encdec graph_layout"
            )
        _validate_humaneval_experiment_axes(
            graph_layout=self.graph_layout,
            encdec_shape=self.encdec_shape,
            dimensions_axes=self.dimensions_axes,
        )
        return self


class ExperimentSpecConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_name: StrictStr
    graph_layout: GraphLayout
    dataset: DatasetSpecConfig
    repetition_seeds: tuple[StrictInt, ...] = DEFAULT_REPETITION_SEEDS
    dimensions_axes: tuple[dict[StrictStr, Any], ...] = DEFAULT_DIMENSIONS_AXES
    providers: tuple[ProviderSpecConfig, ...]
    encdec_shape: EncDecShape = "legacy"
    humaneval_encdec: HumanevalEncDecConfig | None = None

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
            _validate_encdec_provider_config_ids(self.providers)
        _validate_humaneval_experiment_axes(
            graph_layout=self.graph_layout,
            encdec_shape=self.encdec_shape,
            dimensions_axes=self.dimensions_axes,
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
    resolved_parameters: dict[str, Any] = dict(parameters or {})
    resolved_parameters["user_prompt_template"] = user_prompt_template
    if system_prompt is not None:
        resolved_parameters["system_prompt"] = system_prompt
    if provider_config_id is not None:
        resolved_parameters["provider_config_id"] = provider_config_id
    return NodeSpec(
        id=node_id,
        op=LLM_CALL_OP,
        config=NodeConfig(
            fields=tuple(fields),
            input_bindings=input_bindings,
            output_field=output_field,
            parameters=resolved_parameters,
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


def humaneval_encoder_node(
    *,
    humaneval_encdec: HumanevalEncDecConfig | None = None,
) -> NodeSpec:
    cfg = humaneval_encdec or HumanevalEncDecConfig()
    system_prompt = cfg.encoder_system_prompt or None
    return direct_node(
        "encoder",
        bindings={
            "instructions_start": "task.instructions_start",
            "budget": "task.budget",
            "gt_code": "task.gt_code",
            "instructions_end": "task.instructions_end",
        },
        output_field="description",
        user_prompt_template=HUMANEVAL_ENCODER_USER_PROMPT_TEMPLATE,
        system_prompt=system_prompt,
        provider_config_id="encoder",
    )


def humaneval_decoder_node() -> NodeSpec:
    return direct_node(
        "decoder",
        bindings={"encoded_desc": "encoder.description"},
        output_field="code",
        user_prompt_template=HUMANEVAL_DECODER_USER_PROMPT_TEMPLATE,
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


def humaneval_encdec_graph(
    *,
    humaneval_encdec: HumanevalEncDecConfig | None = None,
) -> GraphSpec:
    return GraphSpec(
        nodes=(
            humaneval_decoder_node(),
            humaneval_encoder_node(humaneval_encdec=humaneval_encdec),
        ),
        terminal_node_id="decoder",
    )


def encoder_char_budget(
    *,
    compression_target: float,
    gt_code: str,
    min_budget: int,
) -> int:
    return max(min_budget, round(compression_target * len(gt_code)))


def humaneval_gt_code(task: HumanEvalTask) -> str:
    cleaned = task.ground_truth_code_without_comments
    if cleaned is not None:
        return cleaned
    return task.ground_truth_code


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


def task_snapshot_from_humaneval(
    task: HumanEvalTask,
    *,
    snapshot_identity: DatasetSnapshotIdentityPayload | None = None,
) -> TaskSnapshotPayload:
    metadata: dict[str, Any] = {
        "canonical_solution": task.canonical_solution,
        "ground_truth_code": task.ground_truth_code,
    }
    source = None
    if snapshot_identity is not None:
        source = f"sha256:{snapshot_identity.sha256}"
        metadata["dataset_snapshot"] = snapshot_identity.model_dump(
            mode="json"
        )
    return TaskSnapshotPayload(
        task_id=task.task_id,
        inputs=TaskInputsPayload(
            values={
                "prompt": task.prompt,
                "test": task.test,
                "entry_point": task.entry_point,
            }
        ),
        source=source,
        metadata=metadata,
    )


def humaneval_encdec_task_snapshot(
    task: HumanEvalTask,
    *,
    compression_target: float,
    humaneval_encdec: HumanevalEncDecConfig,
    snapshot_identity: DatasetSnapshotIdentityPayload | None = None,
) -> TaskSnapshotPayload:
    gt_code = humaneval_gt_code(task)
    budget = encoder_char_budget(
        compression_target=compression_target,
        gt_code=gt_code,
        min_budget=humaneval_encdec.min_encoder_char_budget,
    )
    base = task_snapshot_from_humaneval(
        task,
        snapshot_identity=snapshot_identity,
    )
    inputs = dict(base.inputs.values)
    inputs.update(
        {
            "gt_code": gt_code,
            "budget": budget,
            "instructions_start": humaneval_encdec.instructions_start,
            "instructions_end": humaneval_encdec.instructions_end,
        }
    )
    return TaskSnapshotPayload(
        task_id=base.task_id,
        inputs=TaskInputsPayload(values=inputs),
        metadata=base.metadata,
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


def graph_for_layout(
    layout: GraphLayout,
    *,
    encdec_shape: EncDecShape = "legacy",
    humaneval_encdec: HumanevalEncDecConfig | None = None,
) -> GraphSpec:
    if layout is GraphLayout.DIRECT:
        return direct_graph()
    if encdec_shape == "humaneval":
        return humaneval_encdec_graph(humaneval_encdec=humaneval_encdec)
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
    snapshot: HumanEvalSnapshot | None = None,
) -> tuple[SampledHumanEvalTask, ...]:
    if snapshot is None:
        snapshot = load_humaneval_snapshot(
            dataset_name=config.dataset.name,
            dataset_split=config.dataset.split,
            snapshot_path=config.dataset.snapshot_path,
        )
    else:
        _validate_injected_snapshot(config, snapshot)
    return tuple(
        sample_human_eval_tasks_from_rows(
            snapshot.rows,
            seed=config.dataset.sample_seed,
            sample_count=config.dataset.sample_count,
        )
    )


def iter_experiment_specs(
    config: ExperimentSpecConfig,
    *,
    snapshot: HumanEvalSnapshot | None = None,
) -> Iterator[PredictionSpecRecord]:
    if snapshot is None:
        snapshot = load_humaneval_snapshot(
            dataset_name=config.dataset.name,
            dataset_split=config.dataset.split,
            snapshot_path=config.dataset.snapshot_path,
        )
    else:
        _validate_injected_snapshot(config, snapshot)
    snapshot_identity = snapshot.identity
    humaneval_cfg = config.humaneval_encdec or HumanevalEncDecConfig()
    graph = graph_for_layout(
        config.graph_layout,
        encdec_shape=config.encdec_shape,
        humaneval_encdec=humaneval_cfg,
    )
    layout = config.graph_layout.value
    providers = providers_for_config(config)
    provider_axis = providers[0]
    sampled_tasks = sample_tasks_for_config(
        config,
        snapshot=snapshot,
    )
    for sampled in sampled_tasks:
        for repetition_seed in config.repetition_seeds:
            for axis_values in config.dimensions_axes:
                dimensions = DimensionsPayload(values=dict(axis_values))
                if config.encdec_shape == "humaneval":
                    compression_target = float(
                        axis_values["compression_target"]
                    )
                    task_snapshot = humaneval_encdec_task_snapshot(
                        sampled.task,
                        compression_target=compression_target,
                        humaneval_encdec=humaneval_cfg,
                        snapshot_identity=snapshot_identity,
                    )
                else:
                    task_snapshot = task_snapshot_from_humaneval(
                        sampled.task,
                        snapshot_identity=snapshot_identity,
                    )
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
                )


def load_experiment_spec_config(path: Path) -> ExperimentSpecConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ExperimentSpecConfig.model_validate(payload)


def resolve_config_path(configs_root: Path, ref: str) -> Path:
    root = configs_root.resolve()
    resolved = (root / ref).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(
            f"config path {ref!r} escapes configs root {root}"
        )
    return resolved


def load_json_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config file must contain a JSON object: {path}")
    return payload


def load_model_config_fragment(path: Path) -> ModelConfigFragment:
    return ModelConfigFragment.model_validate(load_json_config(path))


def load_split_config_fragment(path: Path) -> SplitConfigFragment:
    return SplitConfigFragment.model_validate(load_json_config(path))


def load_composable_experiment_config(
    path: Path,
) -> ComposableExperimentConfig:
    return ComposableExperimentConfig.model_validate(load_json_config(path))


def expand_composable_experiment(
    config: ComposableExperimentConfig,
    *,
    configs_root: Path,
) -> Iterator[ExperimentSpecConfig]:
    split = load_split_config_fragment(
        resolve_config_path(configs_root, config.split)
    )
    for model_ref in config.model_configs:
        model = load_model_config_fragment(
            resolve_config_path(configs_root, model_ref)
        )
        yield ExperimentSpecConfig(
            experiment_name=config.experiment_name,
            graph_layout=config.graph_layout,
            dataset=split.dataset,
            repetition_seeds=config.repetition_seeds,
            dimensions_axes=config.dimensions_axes,
            providers=model.providers,
            encdec_shape=config.encdec_shape,
            humaneval_encdec=config.humaneval_encdec,
        )


def load_experiment_configs(
    path: Path,
    *,
    configs_root: Path,
) -> Iterator[ExperimentSpecConfig]:
    payload = load_json_config(path)
    if "model_configs" in payload:
        composable = ComposableExperimentConfig.model_validate(payload)
        yield from expand_composable_experiment(
            composable,
            configs_root=configs_root,
        )
        return
    yield ExperimentSpecConfig.model_validate(payload)


def iter_experiment_specs_from_file(
    path: Path,
    *,
    configs_root: Path = DEFAULT_CONFIGS_ROOT,
    snapshot: HumanEvalSnapshot | None = None,
) -> Iterator[PredictionSpecRecord]:
    for config in load_experiment_configs(path, configs_root=configs_root):
        yield from iter_experiment_specs(
            config,
            snapshot=snapshot,
        )


def _validate_injected_snapshot(
    config: ExperimentSpecConfig,
    snapshot: HumanEvalSnapshot,
) -> None:
    snapshot.validate_content_coupling()
    if snapshot.identity.header.dataset_id != config.dataset.name:
        raise ValueError(
            "injected snapshot must match the configured dataset"
        )


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
