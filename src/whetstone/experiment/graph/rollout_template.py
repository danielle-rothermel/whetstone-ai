from __future__ import annotations

from dr_graph import GraphConfig, GraphDefinition
from dr_providers import PROVIDER_CALL_CONFIG_SCHEMA

from whetstone.experiment.graph.nodes import (
    eval_node_definition,
    eval_variable_assignment,
    llm_call_node_definition,
    llm_call_variable_assignment,
)

LLM_NODE_ID = "generate"
EVAL_NODE_ID = "evaluate"

PROMPT_EXTERNAL_INPUT = "task.prompt"

__all__ = [
    "EVAL_NODE_ID",
    "LLM_NODE_ID",
    "PROMPT_EXTERNAL_INPUT",
    "PROVIDER_CALL_CONFIG_SCHEMA",
    "build_single_llm_eval_graph",
    "single_llm_eval_graph_definition",
]


def single_llm_eval_graph_definition() -> GraphDefinition:
    """Minimal LLM Call -> terminal Eval graph definition."""
    llm = llm_call_node_definition(
        LLM_NODE_ID,
        prompt_source=PROMPT_EXTERNAL_INPUT,
    )
    eval_node = eval_node_definition(
        EVAL_NODE_ID,
        upstream_sources={"provider_generation": LLM_NODE_ID},
    )
    return GraphDefinition(nodes=(llm, eval_node), terminal_node_id=EVAL_NODE_ID)


def build_single_llm_eval_graph(
    *,
    provider_call_config_schema: str = PROVIDER_CALL_CONFIG_SCHEMA,
    provider_call_config_hash: str,
    evaluation_procedure_config_schema: str,
    evaluation_procedure_config_hash: str,
) -> GraphConfig:
    """Materialize a two-node rollout graph with caller-supplied config hashes."""
    definition = single_llm_eval_graph_definition()
    assignments = {
        LLM_NODE_ID: llm_call_variable_assignment(
            provider_call_config_schema=provider_call_config_schema,
            provider_call_config_hash=provider_call_config_hash,
        ),
        EVAL_NODE_ID: eval_variable_assignment(
            evaluation_procedure_config_schema=(
                evaluation_procedure_config_schema
            ),
            evaluation_procedure_config_hash=(
                evaluation_procedure_config_hash
            ),
        ),
    }
    return definition.materialize(assignments)
