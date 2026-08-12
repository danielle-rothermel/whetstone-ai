from __future__ import annotations

from dataclasses import dataclass

from dr_graph import GraphConfig, GraphDefinition, graph_hash
from dr_providers import ProviderCallConfig, openrouter_chat_config

from whetstone.experiment.graph.character_budget import CharacterBudgetRule
from whetstone.experiment.graph.nodes import (
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

#: The decoder Node's upstream input field (the encoder's provider generation).
_DECODER_INPUT_FIELD = "description"

#: The Eval Node's upstream input field (the decoder's provider generation).
_EVAL_INPUT_FIELD = "submission"

#: The Provider Call Config schema name (the LLM Call Nodes' static Variable
#: typed reference), matching the QA graph's schema so the identity domain is
#: shared.
PROVIDER_CALL_CONFIG_SCHEMA = "dr_providers.provider_call_config"

#: The Evaluation Procedure Config schema for the enc-dec code eval procedure.
ENCDEC_PROCEDURE_CONFIG_SCHEMA = "whetstone.code_comp.encdec_procedure"


@dataclass(frozen=True, slots=True)
class EncDecGenerationGraph:
    """The enc-dec Generation Graph graph + the config references it binds.

    ``definition`` is the native three-node :class:`GraphDefinition`;
    ``encoder_call_config`` and ``decoder_call_config`` are the exact provider
    routes for each LLM node (they may differ). ``budget_ratio`` is the
    identity-bearing Character Budget ratio (a distinct ratio is a distinct
    ``graph_hash``). ``procedure_config_hash`` is the code-eval Evaluation
    Procedure Config identity the Eval Node carries.
    """

    env_name: str
    definition: GraphDefinition
    encoder_call_config: ProviderCallConfig
    decoder_call_config: ProviderCallConfig
    procedure_config_hash: str
    #: The Character Budget ratio, or ``None`` for the no-budget frame, which
    #: has no budget clause or MAX_BUDGET. A ``None`` budget yields a
    #: DISTINCT ``graph_hash`` from any ratio (identity-folded).
    budget_ratio: float | None
    graph_config: GraphConfig

    @property
    def provider_call_config(self) -> ProviderCallConfig:
        """Backward-compatible alias for the encoder route."""
        return self.encoder_call_config

    @property
    def graph_hash(self) -> str:
        """The native dr-graph Graph Config Identity Hash."""
        return graph_hash(self.graph_config)

    @property
    def budget_rule(self) -> CharacterBudgetRule | None:
        """The Character Budget derivation rule, or ``None`` (no-budget)."""
        if self.budget_ratio is None:
            return None
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
    the encoder's provider-generation output; the Eval Node consumes the
    decoder's
    ProviderGeneration and is the unique terminal Node.
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


#: The encoder Character Budget identity token for the no-budget frame. The
#: distinct sentinel prevents collision with a ratio's graph hash.
_NO_BUDGET_IDENTITY = "no_budget"


def build_encdec_graph_config(
    *,
    encoder_call_config_hash: str,
    decoder_call_config_hash: str | None = None,
    evaluation_procedure_config_hash: str,
    budget_ratio: float | None,
) -> GraphConfig:
    """Materialize the enc-dec Graph Config binding both routes + the budget.

    Encoder and decoder LLM Call Nodes may carry distinct Provider Call Config
    references; when ``decoder_call_config_hash`` is omitted the encoder hash
    is reused. The ENCODER additionally carries the Character Budget
    ``ratio`` Variable, so a distinct ``budget_ratio`` yields a distinct
    ``graph_hash``. ``budget_ratio=None`` binds the NO-BUDGET sentinel (a
    distinct graph). The Eval Node carries the code-eval Procedure reference.
    """
    definition = encdec_graph_definition()
    decoder_hash = decoder_call_config_hash or encoder_call_config_hash
    budget_label = (
        _NO_BUDGET_IDENTITY
        if budget_ratio is None
        else CharacterBudgetRule(ratio=budget_ratio).identity_value()
    )
    assignments = {
        ENCODER_NODE_ID: llm_call_variable_assignment(
            provider_call_config_schema=PROVIDER_CALL_CONFIG_SCHEMA,
            provider_call_config_hash=encoder_call_config_hash,
            character_budget_rule=budget_label,
        ),
        DECODER_NODE_ID: llm_call_variable_assignment(
            provider_call_config_schema=PROVIDER_CALL_CONFIG_SCHEMA,
            provider_call_config_hash=decoder_hash,
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


def build_encdec_generation_graph(
    env_name: str,
    *,
    provider_call_config: ProviderCallConfig | None = None,
    encoder_call_config: ProviderCallConfig | None = None,
    decoder_call_config: ProviderCallConfig | None = None,
    procedure_config_hash: str,
    budget_ratio: float | None,
) -> EncDecGenerationGraph:
    """Build the enc-dec Generation Graph for one or two provider routes.

    Wires encoder and decoder Provider Call Configs onto their LLM nodes, the
    Character Budget ``ratio`` onto the encoder, and the code-eval Evaluation
    Procedure Config onto the terminal Eval Node. Provider lane, protocol,
    model, and generation controls therefore fold into graph identity through
    the exact Configs.
    """
    if provider_call_config is not None:
        if encoder_call_config is not None or decoder_call_config is not None:
            raise ValueError(
                "provider_call_config is mutually exclusive with "
                "encoder_call_config and decoder_call_config"
            )
        encoder_call_config = provider_call_config
    if encoder_call_config is None:
        raise ValueError(
            "encoder_call_config or provider_call_config required"
        )
    decoder = decoder_call_config or encoder_call_config
    graph_config = build_encdec_graph_config(
        encoder_call_config_hash=encoder_call_config.identity_hash,
        decoder_call_config_hash=(
            None
            if decoder.identity_hash == encoder_call_config.identity_hash
            else decoder.identity_hash
        ),
        evaluation_procedure_config_hash=procedure_config_hash,
        budget_ratio=budget_ratio,
    )
    return EncDecGenerationGraph(
        env_name=env_name,
        definition=encdec_graph_definition(),
        encoder_call_config=encoder_call_config,
        decoder_call_config=decoder,
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
    "EncDecGenerationGraph",
    "build_encdec_generation_graph",
    "build_encdec_graph_config",
    "build_encoder_provider_call_config",
    "encdec_graph_definition",
]
