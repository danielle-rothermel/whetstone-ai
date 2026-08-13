from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from dr_graph import GraphConfig, graph_hash
from dr_providers import (
    ControlConstraints,
    PROVIDER_CALL_CONFIG_SCHEMA,
    ProviderCallConfig,
    ProviderCallDefinition,
    ProviderKind,
    Protocol,
    TokenLimitParameter,
)
from pydantic import BaseModel, ConfigDict, StrictStr

from whetstone.core.identity import typed_ref_for_record
from whetstone.eval import (
    EvalProcedureDefinition,
    MetricExtractionDefinition,
    MetricQuestionBinding,
    PreprocessingDefinition,
    SCHEMA_EVAL_PROCEDURE_CONFIG,
)
from whetstone.eval.aggregate import CompletenessPolicy, aggregation_definition
from whetstone.experiment.candidate import (
    Candidate,
    TemplateRenderContract,
    TemplateRenderKind,
    candidate_reference,
)
from whetstone.experiment.env import Experiment
from whetstone.experiment.graph.rollout_template import build_single_llm_eval_graph
from whetstone.experiment.reward import RewardPolicy, RewardTerm
from whetstone.experiment.sampling import (
    INTERNAL_EVAL,
    OFFICIAL,
    EvalConfigs,
    derive_eval_split,
)

TOY_NAMESPACE = "whetstone.toy"
TOY_DATASET_REVISION = "toy/v1"
TOY_MUTATION_FIELD = "user_prompt_template"
TOY_ROOT_BASE_SCHEMA = "whetstone.toy.root_candidate"
DEFAULT_TOY_TEMPLATE = "Reply briefly to: {prompt}"


@dataclass(frozen=True, slots=True)
class ToyTask:
    task_id: str
    prompt_inputs: dict[str, str]
    gold: str = ""

    @property
    def task_hash(self) -> str:
        payload = {
            "task_id": self.task_id,
            "prompt_inputs": self.prompt_inputs,
            "gold": self.gold,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()


class ToyPromptFrame(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    frame_id: StrictStr = "toy-prompt/v1"
    template: StrictStr = DEFAULT_TOY_TEMPLATE
    mutation_field: StrictStr = TOY_MUTATION_FIELD


@dataclass(frozen=True, slots=True)
class _ToyRolloutGraph:
    graph_config: GraphConfig
    provider_call_config: ProviderCallConfig
    procedure_hash: str
    graph_hash_value: str

    @property
    def graph_hash(self) -> str:
        return self.graph_hash_value

    @property
    def procedure_config_hash(self) -> str:
        return self.procedure_hash


def _task_hash(task: ToyTask) -> str:
    return task.task_hash


def _reference_procedure() -> tuple[object, str]:
    preprocessing = PreprocessingDefinition(
        definition_id=f"{TOY_NAMESPACE}.preprocessing",
        version="1",
        steps=(),
    ).materialize()
    metric_extraction = MetricExtractionDefinition(
        definition_id=f"{TOY_NAMESPACE}.metric_extraction",
        version="1",
        questions=(
            MetricQuestionBinding(metric="score", on="submission"),
        ),
    ).materialize(resolved_operators=(("score", "1"),))
    procedure = EvalProcedureDefinition(
        definition_id=f"{TOY_NAMESPACE}.evaluation_procedure",
        version="1",
    ).materialize(
        preprocessing=preprocessing,
        metric_extraction=metric_extraction,
        assignment={"zero_denominator": "not_applicable"},
    )
    return procedure, procedure.config_hash


def _reference_provider_call_config() -> ProviderCallConfig:
    definition = ProviderCallDefinition(
        definition_id=f"{TOY_NAMESPACE}.provider/v1",
        route={
            "provider": ProviderKind.OPENAI,
            "protocol": Protocol.CHAT_COMPLETIONS,
            "model": "fake-model",
        },
        constraints=ControlConstraints(
            token_limit_parameter=TokenLimitParameter.MAX_TOKENS
        ),
    )
    return ProviderCallConfig(definition=definition, controls={}, extensions={})


def _reference_rollout_graph() -> _ToyRolloutGraph:
    provider_call_config = _reference_provider_call_config()
    procedure, procedure_hash = _reference_procedure()
    provider_ref_hash = str(
        typed_ref_for_record(
            PROVIDER_CALL_CONFIG_SCHEMA,
            provider_call_config.model_dump(mode="json"),
        ).content_hash
    )
    graph_config = build_single_llm_eval_graph(
        provider_call_config_hash=provider_ref_hash,
        evaluation_procedure_config_schema=SCHEMA_EVAL_PROCEDURE_CONFIG,
        evaluation_procedure_config_hash=procedure_hash,
    )
    return _ToyRolloutGraph(
        graph_config=graph_config,
        provider_call_config=provider_call_config,
        procedure_hash=procedure_hash,
        graph_hash_value=graph_hash(graph_config),
    )


def _reference_aggregation():
    return aggregation_definition(f"{TOY_NAMESPACE}.aggregation").materialize(
        {
            "reduction": "mean",
            "missing_data": "propagate",
        }
    )


def _toy_candidate(
    *,
    candidate_id: str,
    template: str,
) -> Candidate:
    root_ref = typed_ref_for_record(TOY_ROOT_BASE_SCHEMA, {"kind": "root"})
    candidate = Candidate(
        candidate_id=candidate_id,
        base_ref=root_ref,
        payload={TOY_MUTATION_FIELD: template},
    )
    return candidate_reference(candidate).record


def build_toy_experiment(
    *,
    internal_tasks: tuple[ToyTask, ...] | None = None,
    official_tasks: tuple[ToyTask, ...] | None = None,
    num_seeds: int = 1,
    initial_template: str = DEFAULT_TOY_TEMPLATE,
    ceiling_template: str | None = None,
) -> Experiment:
    """Build an in-memory toy Experiment wired to the single-node eval graph."""
    if num_seeds < 1:
        raise ValueError("num_seeds must be at least 1")
    resolved_internal = internal_tasks or (
        ToyTask(task_id="task-a", prompt_inputs={"prompt": "hello A"}, gold="A"),
        ToyTask(task_id="task-b", prompt_inputs={"prompt": "hello B"}, gold="B"),
    )
    resolved_official = official_tasks or (
        ToyTask(task_id="task-c", prompt_inputs={"prompt": "hello C"}, gold="C"),
    )
    if not resolved_internal or not resolved_official:
        raise ValueError("toy experiment requires non-empty internal and official tasks")

    rollout_graph = _reference_rollout_graph()
    procedure, procedure_hash = _reference_procedure()
    aggregation = _reference_aggregation()
    internal = derive_eval_split(
        namespace=TOY_NAMESPACE,
        dataset_revision=TOY_DATASET_REVISION,
        split_role=INTERNAL_EVAL,
        tasks=resolved_internal,
        task_hash_of=_task_hash,
        procedure=procedure,
        aggregation=aggregation,
        num_seeds=num_seeds,
    )
    official = derive_eval_split(
        namespace=TOY_NAMESPACE,
        dataset_revision=TOY_DATASET_REVISION,
        split_role=OFFICIAL,
        tasks=resolved_official,
        task_hash_of=_task_hash,
        procedure=procedure,
        aggregation=aggregation,
        num_seeds=num_seeds,
    )
    eval_configs = EvalConfigs(
        env_name=TOY_NAMESPACE,
        procedure_config_hash=procedure_hash,
        internal=internal,
        official=official,
        held_out_task_hashes=(),
    )
    render_contract = TemplateRenderContract(
        kind=TemplateRenderKind.PYTHON_FORMAT_V1,
        available_fields=("prompt",),
        required_fields=("prompt",),
    )
    render_contract.validate_template(initial_template)
    ceiling = ceiling_template or initial_template
    render_contract.validate_template(ceiling)
    reward_policy = RewardPolicy(
        policy_name="toy-reward",
        terms=(RewardTerm(name="score", weight=1.0),),
    )
    return Experiment(
        env_name=TOY_NAMESPACE,
        rollout_graph=rollout_graph,
        initial_candidate=_toy_candidate(
            candidate_id="toy-initial",
            template=initial_template,
        ),
        ceiling_candidate=_toy_candidate(
            candidate_id="toy-ceiling",
            template=ceiling,
        ),
        eval_configs=eval_configs,
        reward_policy=reward_policy,
        completeness_policy=CompletenessPolicy(),
    )


def toy_template_render_contract() -> TemplateRenderContract:
    return TemplateRenderContract(
        kind=TemplateRenderKind.PYTHON_FORMAT_V1,
        available_fields=("prompt",),
        required_fields=("prompt",),
    )


__all__ = [
    "DEFAULT_TOY_TEMPLATE",
    "TOY_MUTATION_FIELD",
    "TOY_NAMESPACE",
    "ToyPromptFrame",
    "ToyTask",
    "build_toy_experiment",
    "toy_template_render_contract",
]
