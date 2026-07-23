"""The Encoder -> Decoder -> Eval three-node Rollout Definition (enc-dec).

The enc-dec HumanEval compression experiment's Rollout Definition role graph,
per
``design/vocab_and_defs.html`` ("Its variable-bearing shape is Encoder LLM Call
Node -> Decoder LLM Call Node -> Eval Node, with the Eval Node as the unique
terminal Node"):

    Encoder LLM Call Node -> Decoder LLM Call Node -> Eval Node (terminal)

* The **encoder** renders the Mutation-Surface encoder ``user_prompt_template``
  against the task's ``INPUT_CODE`` and a per-task character budget
  ``MAX_BUDGET = round(budget_ratio * chars(INPUT_CODE))``. It declares the
  Character Budget Variable (``CharacterBudgetRule(ratio=budget_ratio)``), so
  the
  ratio folds into ``graph_hash`` (the concrete budget bound is a runtime Graph
  External Input, never in identity).
* The **decoder** conditions ONLY on the encoder's description (its
  ``prompt_source`` is the encoder Node's Generation output) and reconstructs
  code.
* The **Eval Node** (terminal) consumes the DECODER Generation, runs the
  HumanEval test suite (Binary Test Pass Score on the decoder output) and the
  zstd-19 compression scoring (Compression Ratio on the ENCODER output vs the
  ground-truth code). The **same Model Route plays both** encoder and decoder.

This composes the existing closed Node primitives (``graph/nodes.py``) and the
Character Budget binding (``graph/character_budget.py``) -- no new graph
capability, a second LLM node between encoder and eval plus the budget
Variable.
The encoder ``user_prompt_template`` is the Mutation Surface optimizers mutate.
"""

from __future__ import annotations

from dataclasses import dataclass

from dr_graph import GraphConfig, GraphDefinition, graph_hash
from dr_providers import ProviderCallConfig, openrouter_chat_config

from whetstone.graph.character_budget import CharacterBudgetRule
from whetstone.graph.nodes import (
    eval_node_definition,
    eval_variable_assignment,
    llm_call_node_definition,
    llm_call_variable_assignment,
)

#: Node ids for the three-node graph.
ENCODER_NODE_ID = "encode"
DECODER_NODE_ID = "decode"
EVAL_NODE_ID = "evaluate"

#: The single Graph External Input the ENCODER prompt binds to: the rendered
#: encoder prompt (the encoder template filled with INPUT_CODE + the budget).
ENCODER_PROMPT_EXTERNAL_INPUT = "task.encoder_prompt"

#: The decoder Node's declared upstream input field (the encoder's Generation).
_DECODER_INPUT_FIELD = "description"

#: The Eval Node's declared upstream input field (the decoder's Generation).
_EVAL_INPUT_FIELD = "submission"

#: The Provider Call Config schema name (the LLM Call Nodes' static Variable
#: typed reference), matching the QA graph's schema so the identity domain is
#: shared.
PROVIDER_CALL_CONFIG_SCHEMA = "dr_providers.provider_call_config"

#: The Evaluation Procedure Config schema for the enc-dec code eval procedure.
ENCDEC_PROCEDURE_CONFIG_SCHEMA = "whetstone.encdec_code_eval_procedure"


@dataclass(frozen=True, slots=True)
class EncDecRolloutDefinition:
    """The enc-dec Rollout Definition graph + the config references it binds.

    ``definition`` is the native three-node :class:`GraphDefinition`;
    ``provider_call_config`` is the shared encoder/decoder route (its Identity
    Hash is BOTH LLM nodes' static Variable). ``budget_ratio`` is the
    identity-bearing Character Budget ratio (a distinct ratio is a distinct
    ``graph_hash``). ``procedure_config_hash`` is the code-eval Evaluation
    Procedure Config identity the Eval Node carries.
    """

    env_name: str
    definition: GraphDefinition
    provider_call_config: ProviderCallConfig
    procedure_config_hash: str
    budget_ratio: float
    graph_config: GraphConfig

    @property
    def graph_hash(self) -> str:
        """The native dr-graph Graph Config Identity Hash."""
        return graph_hash(self.graph_config)

    @property
    def budget_rule(self) -> CharacterBudgetRule:
        """The Character Budget derivation rule for this graph."""
        return CharacterBudgetRule(ratio=self.budget_ratio)


def build_encoder_provider_call_config(model: str) -> ProviderCallConfig:
    """The native OpenRouter Provider Call Config for the enc/dec task model.

    A minimal chat Config over ``model`` -- the SAME route plays both encoder
    and decoder, so its Identity Hash is both LLM nodes' Provider Call Config
    static Variable.
    """
    return openrouter_chat_config(model=model)


def encdec_graph_definition() -> GraphDefinition:
    """The Encoder -> Decoder -> terminal Eval three-node Graph Definition.

    The encoder declares the Character Budget Variable; the decoder's prompt is
    the encoder's Generation output; the Eval Node consumes the decoder's
    Generation and is the unique terminal Node.
    """
    encoder = llm_call_node_definition(
        ENCODER_NODE_ID,
        prompt_source=ENCODER_PROMPT_EXTERNAL_INPUT,
        declares_character_budget=True,
    )
    decoder = llm_call_node_definition(
        DECODER_NODE_ID,
        # The decoder conditions ONLY on the encoder's description output.
        prompt_source=ENCODER_NODE_ID,
    )
    ev = eval_node_definition(
        EVAL_NODE_ID,
        upstream_sources={_EVAL_INPUT_FIELD: DECODER_NODE_ID},
    )
    return GraphDefinition(
        nodes=(encoder, decoder, ev), terminal_node_id=EVAL_NODE_ID
    )


def build_encdec_graph_config(
    *,
    provider_call_config_hash: str,
    evaluation_procedure_config_hash: str,
    budget_ratio: float,
) -> GraphConfig:
    """Materialize the enc-dec Graph Config binding both routes + the budget.

    BOTH LLM Call Nodes carry the SAME Provider Call Config reference (encoder
    ==
    decoder route); the ENCODER additionally carries the Character Budget
    ``ratio`` Variable, so a distinct ``budget_ratio`` yields a distinct
    ``graph_hash``. The Eval Node carries the code-eval Procedure reference.
    """
    definition = encdec_graph_definition()
    budget_rule = CharacterBudgetRule(ratio=budget_ratio)
    assignments = {
        ENCODER_NODE_ID: llm_call_variable_assignment(
            provider_call_config_schema=PROVIDER_CALL_CONFIG_SCHEMA,
            provider_call_config_hash=provider_call_config_hash,
            character_budget_rule=budget_rule.identity_value(),
        ),
        DECODER_NODE_ID: llm_call_variable_assignment(
            provider_call_config_schema=PROVIDER_CALL_CONFIG_SCHEMA,
            provider_call_config_hash=provider_call_config_hash,
        ),
        EVAL_NODE_ID: eval_variable_assignment(
            evaluation_procedure_config_schema=(
                ENCDEC_PROCEDURE_CONFIG_SCHEMA
            ),
            evaluation_procedure_config_hash=(
                evaluation_procedure_config_hash
            ),
        ),
    }
    return definition.materialize(assignments)


def build_encdec_rollout_definition(
    env_name: str,
    *,
    model: str,
    procedure_config_hash: str,
    budget_ratio: float,
) -> EncDecRolloutDefinition:
    """Build the enc-dec Rollout Definition graph for one (model, ratio).

    Wires the shared encoder/decoder Provider Call Config (``model``) across
    both LLM nodes, the Character Budget ``ratio`` onto the encoder, and the
    code-eval Evaluation Procedure Config onto the terminal Eval Node.
    """
    provider_call_config = build_encoder_provider_call_config(model)
    graph_config = build_encdec_graph_config(
        provider_call_config_hash=provider_call_config.identity_hash,
        evaluation_procedure_config_hash=procedure_config_hash,
        budget_ratio=budget_ratio,
    )
    return EncDecRolloutDefinition(
        env_name=env_name,
        definition=encdec_graph_definition(),
        provider_call_config=provider_call_config,
        procedure_config_hash=procedure_config_hash,
        budget_ratio=budget_ratio,
        graph_config=graph_config,
    )


__all__ = [
    "DECODER_NODE_ID",
    "ENCDEC_PROCEDURE_CONFIG_SCHEMA",
    "ENCODER_NODE_ID",
    "ENCODER_PROMPT_EXTERNAL_INPUT",
    "EVAL_NODE_ID",
    "EncDecRolloutDefinition",
    "build_encdec_graph_config",
    "build_encdec_rollout_definition",
    "build_encoder_provider_call_config",
    "encdec_graph_definition",
]
