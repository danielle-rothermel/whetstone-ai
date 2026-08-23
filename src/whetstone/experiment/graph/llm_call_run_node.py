from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from dr_graph import NodeConfig, NodeOutput, RunNode
from dr_providers import PROVIDER_CALL_CONFIG_SCHEMA, ProviderCallConfig

from whetstone.core.identity import (
    IdentityRef,
    assert_materialized_ref_matches,
    identity_ref_from_config_variable,
)
from whetstone.eval.drivers.graph_execution import (
    METADATA_PROMPT_KEY,
    METADATA_SUBMISSION_RESULT_KEY,
    GenerationNodeError,
    single_node_input,
)
from whetstone.eval.eval_procedure import (
    EvalProcedureRunner,
    accepts_seed_index,
)
from whetstone.eval.protocol import EvalTaskView
from whetstone.experiment.graph.nodes import (
    EVAL_NODE_TYPE,
    EVAL_OUTPUT_FIELD,
    LLM_CALL_NODE_TYPE,
    PROVIDER_CALL_CONFIG_VARIABLE,
    PROVIDER_GENERATION_OUTPUT_FIELD,
    eval_node_procedure_hash,
)
from whetstone.provider.llm_call import (
    LlmCallContext,
    build_provider_request,
    call_execution_metadata,
    execute_llm_call,
    provider_result_text,
)

__all__ = [
    "EvalRunNodeDeps",
    "LlmCallRunNodeDeps",
    "build_eval_run_node",
    "build_llm_call_run_node",
]

ProviderCallConfigResolver = Callable[[IdentityRef], ProviderCallConfig]


@dataclass(frozen=True, slots=True)
class LlmCallRunNodeDeps:
    context: LlmCallContext
    resolve_provider_call_config: ProviderCallConfigResolver
    graph_hash: str
    rng_seed: int
    task_id: str
    seed_index: int = 0
    drive_ordinal: int = 0
    phase: str = ""
    unit: str = ""
    split_role: str | None = None
    request_identity_sink: list[str] | None = None


@dataclass(frozen=True, slots=True)
class EvalRunNodeDeps:
    runner: EvalProcedureRunner
    task: EvalTaskView
    #: Which repeat of ``task`` this row is. The ``EvalProcedureRunner``
    #: protocol does not take it -- a real eval procedure scores one
    #: generation and must not depend on which repeat produced it. It is
    #: carried here so a runner that opts in (``SeedAwareEvalProcedureRunner``,
    #: the toy scorer used to make repeat-mean assertions non-vacuous) can
    #: read it. Runners that do not opt in are called exactly as before.
    seed_index: int = 0


def build_llm_call_run_node(deps: LlmCallRunNodeDeps) -> RunNode:
    def run_llm_node(
        node: NodeConfig, node_inputs: Mapping[str, Any]
    ) -> NodeOutput:
        if node.node_type != LLM_CALL_NODE_TYPE:
            raise TypeError(
                f"expected node type {LLM_CALL_NODE_TYPE!r}, "
                f"got {node.node_type!r}"
            )
        prompt = single_node_input(node, node_inputs)
        variable = node.variables[PROVIDER_CALL_CONFIG_VARIABLE]
        if not isinstance(variable, Mapping):
            raise ValueError("provider call config reference is malformed")
        config_ref = identity_ref_from_config_variable(variable)
        provider_config = deps.resolve_provider_call_config(config_ref)
        assert_materialized_ref_matches(
            record=provider_config,
            ref=config_ref,
            schema=PROVIDER_CALL_CONFIG_SCHEMA,
        )
        request = build_provider_request(
            provider_config=provider_config,
            rng_seed=deps.rng_seed,
            prompt=prompt,
            prompt_adapter=deps.context.prompt_adapter,
        )
        logical_call_id = (
            f"graph:{deps.graph_hash}:{node.node_id}:"
            f"{request.config.identity_hash}"
        )
        try:
            execution = execute_llm_call(
                context=deps.context,
                request=request,
                logical_call_id=logical_call_id,
                task_id=deps.task_id,
                seed_index=deps.seed_index,
                drive_ordinal=deps.drive_ordinal,
                phase=deps.phase,
                unit=deps.unit,
                split_role=deps.split_role,
                request_identity_sink=deps.request_identity_sink,
            )
            text = provider_result_text(execution.result)
        except (GenerationNodeError, ValueError) as exc:
            metadata = getattr(exc, "metadata", {})
            raise GenerationNodeError(str(exc), metadata=metadata) from exc
        output_field = node.output_field or PROVIDER_GENERATION_OUTPUT_FIELD
        metadata = call_execution_metadata(execution)
        metadata[METADATA_PROMPT_KEY] = prompt
        return NodeOutput(
            values={output_field: text},
            metadata=metadata,
        )

    return run_llm_node


def build_eval_run_node(deps: EvalRunNodeDeps) -> RunNode:
    def run_eval_node(
        node: NodeConfig, node_inputs: Mapping[str, Any]
    ) -> NodeOutput:
        if node.node_type != EVAL_NODE_TYPE:
            raise TypeError(
                f"expected node type {EVAL_NODE_TYPE!r}, got {node.node_type!r}"
            )
        procedure_hash = eval_node_procedure_hash(node.variables)
        runner = deps.runner
        if accepts_seed_index(runner):
            score, submission_result, extra_metadata = runner.run_eval_node(
                node_id=node.node_id,
                node_inputs=node_inputs,
                evaluation_procedure_config_hash=procedure_hash,
                task=deps.task,
                seed_index=deps.seed_index,
            )
        else:
            score, submission_result, extra_metadata = runner.run_eval_node(
                node_id=node.node_id,
                node_inputs=node_inputs,
                evaluation_procedure_config_hash=procedure_hash,
                task=deps.task,
            )
        metadata = dict(extra_metadata)
        if submission_result is not None:
            metadata[METADATA_SUBMISSION_RESULT_KEY] = submission_result
        output_field = node.output_field or EVAL_OUTPUT_FIELD
        return NodeOutput(
            values={output_field: score},
            metadata=metadata,
        )

    return run_eval_node
