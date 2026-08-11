from __future__ import annotations

from dataclasses import dataclass

from dr_graph import GraphConfig, GraphDefinition, graph_hash
from dr_providers import ProviderCallConfig, openrouter_chat_config

from whetstone.envs.rollout_definition import (
    EVAL_NODE_ID,
    LLM_NODE_ID,
    PROMPT_EXTERNAL_INPUT,
    PROVIDER_CALL_CONFIG_SCHEMA,
)
from whetstone.experiment.graph.nodes import (
    eval_node_definition,
    eval_variable_assignment,
    llm_call_node_definition,
    llm_call_variable_assignment,
)

DIRECT_ENV_NAME = "d1"
D1_ENV_NAME = DIRECT_ENV_NAME

D1_PROCEDURE_CONFIG_SCHEMA = "whetstone.d1_code_eval_procedure"

D1_RENAMED_ARM = "renamed"

D1_INPUT_ARMS: tuple[str, ...] = (
    "original",
    "docstring",
    "signature",
    "name",
    D1_RENAMED_ARM,
)

D1_DEFAULT_RENAME_TOKEN = "target_fxn"

D1_WRAPPER_FRAME = "{body}\n{input_arm}"


def render_d1_frame(body: str, *, input_arm: str) -> str:
    """Compose the immutable d1 wrapper frame around a mutable strategy body.

    ``body`` is the Mutation-Surface payload (the strategy sentence ONLY);
    ``input_arm`` is the frozen input-arm text. A body carrying a
    ``{placeholder}`` would raise here -- but intake validation rejects such
    bodies first (the frame owns every placeholder).
    """
    return D1_WRAPPER_FRAME.format(body=body, input_arm=input_arm)


@dataclass(frozen=True, slots=True)
class D1RolloutDefinition:
    """The d1 direct Rollout Definition graph + the config references it binds.

    A single LLM Call Node -> terminal Eval Node (the SAME two-node shape the
    QA envs use), with the code-eval Evaluation Procedure on the Eval Node. The
    FROZEN ``input_arm`` folds into ``graph_hash`` (a distinct arm is a
    distinct graph variant), so a d1 cell on ``renamed`` is identity-distinct
    from one on ``original``.
    """

    env_name: str
    definition: GraphDefinition
    provider_call_config: ProviderCallConfig
    procedure_config_hash: str
    input_arm: str
    graph_config: GraphConfig

    @property
    def graph_hash(self) -> str:
        """The native dr-graph Graph Config Identity Hash."""
        return graph_hash(self.graph_config)


def d1_graph_definition() -> GraphDefinition:
    """The d1 direct LLM Call -> terminal Eval Graph Definition.

    The SAME two-node shape as the QA graph, but the LLM Call Node DECLARES the
    input-arm control Variable (reusing the ``character_budget_rule`` slot to
    carry the FROZEN input-arm token) so a distinct input arm yields a distinct
    ``graph_hash`` -- the arm is an output-affecting knob that MUST fold into
    graph identity, exactly as ed1 folds its budget ratio.
    """
    llm = llm_call_node_definition(
        LLM_NODE_ID,
        prompt_source=PROMPT_EXTERNAL_INPUT,
        declares_character_budget=True,
    )
    ev = eval_node_definition(
        EVAL_NODE_ID,
        upstream_sources={"generation": LLM_NODE_ID},
    )
    return GraphDefinition(nodes=(llm, ev), terminal_node_id=EVAL_NODE_ID)


def d1_arm_token(input_arm: str, rename_token: str) -> str:
    """The identity-bearing control token for one (arm, rename token) pair.

    ``rename_token`` folds in ONLY for the ``renamed`` arm -- it is the text
    substituted for every canonical name in that arm, so two ``renamed`` cells
    with different tokens are different experiments. The other arms never read
    the token, so folding it there would churn their identities for a value
    they ignore.
    """
    if input_arm == D1_RENAMED_ARM:
        return f"d1_input_arm:{input_arm}|rename={rename_token}"
    return f"d1_input_arm:{input_arm}"


def build_d1_graph_config(
    *,
    provider_call_config_hash: str,
    evaluation_procedure_config_hash: str,
    input_arm: str,
    rename_token: str = D1_DEFAULT_RENAME_TOKEN,
) -> GraphConfig:
    """Materialize the d1 Graph Config binding the route, procedure, and arm.

    The LLM Call Node carries the Provider Call Config reference AND the FROZEN
    input-arm control token (in the declared budget-variable slot); the Eval
    Node carries the code-eval Procedure reference. A distinct arm -- or, on
    the ``renamed`` arm, a distinct ``rename_token`` -- yields a distinct
    ``graph_hash`` (identity-folded by construction).
    """
    definition = d1_graph_definition()
    assignments = {
        LLM_NODE_ID: llm_call_variable_assignment(
            provider_call_config_schema=PROVIDER_CALL_CONFIG_SCHEMA,
            provider_call_config_hash=provider_call_config_hash,
            character_budget_rule=d1_arm_token(input_arm, rename_token),
        ),
        EVAL_NODE_ID: eval_variable_assignment(
            evaluation_procedure_config_schema=D1_PROCEDURE_CONFIG_SCHEMA,
            evaluation_procedure_config_hash=(
                evaluation_procedure_config_hash
            ),
        ),
    }
    return definition.materialize(assignments)


def build_d1_rollout_definition(
    *,
    model: str,
    procedure_config_hash: str,
    input_arm: str,
    rename_token: str = D1_DEFAULT_RENAME_TOKEN,
) -> D1RolloutDefinition:
    """Build the d1 direct Rollout Definition for one (model, input arm)."""
    provider_call_config = openrouter_chat_config(model=model)
    graph_config = build_d1_graph_config(
        provider_call_config_hash=provider_call_config.identity_hash,
        evaluation_procedure_config_hash=procedure_config_hash,
        input_arm=input_arm,
        rename_token=rename_token,
    )
    return D1RolloutDefinition(
        env_name=D1_ENV_NAME,
        definition=d1_graph_definition(),
        provider_call_config=provider_call_config,
        procedure_config_hash=procedure_config_hash,
        input_arm=input_arm,
        graph_config=graph_config,
    )


__all__ = [
    "D1_DEFAULT_RENAME_TOKEN",
    "D1_ENV_NAME",
    "D1_INPUT_ARMS",
    "D1_PROCEDURE_CONFIG_SCHEMA",
    "D1_RENAMED_ARM",
    "D1_WRAPPER_FRAME",
    "D1RolloutDefinition",
    "build_d1_graph_config",
    "build_d1_rollout_definition",
    "d1_arm_token",
    "d1_graph_definition",
    "render_d1_frame",
]
