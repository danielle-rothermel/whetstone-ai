"""Closed, versioned Whetstone Node Definitions.

Whetstone owns the *semantics* of its two Node kinds; the concrete native
representation is dr-graph's ``NodeDefinition`` / ``NodeConfig``, and all
Node fields participate in ``graph_hash``. Whetstone introduces no separate
Node type registry, no Node hash, and no parallel config model.

Two closed Node Definitions exist:

``whetstone.llm-call/v1``
    Primary output is a Generation. It references a native dr-providers
    Provider Call Config by a typed reference plus its Identity Hash, carried
    as a *static Variable assignment* (never a Node Input Source). The full
    provider result is retained as provenance at runtime, not in identity.

``whetstone.eval/v1``
    Consumes declared upstream Node Outputs from the current Graph Run via
    Node Input Sources. Its Evaluation Procedure Config is a typed reference
    plus Identity Hash carried as a *static Variable assignment* — never a
    Node Input Source and never a context-bound runtime input.

Because the Provider Call Config reference and the Evaluation Procedure
Config reference are Node ``variables`` (not ``input_sources``), they are in
the Graph Config identity payload and changing either changes ``graph_hash``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dr_graph import (
    FieldRole,
    NodeDefinition,
    NodeFieldSpec,
    as_node_input_source_ref,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

# Closed Node Definition versioned type identifiers. These are the only two
# Node kinds Whetstone defines; the strings are stable identity-bearing
# ``node_type`` values that appear in the Graph Config identity payload.
LLM_CALL_NODE_TYPE = "whetstone.llm-call/v1"
EVAL_NODE_TYPE = "whetstone.eval/v1"

# The one required static Variable name on each Node kind. Both are typed
# references *plus Identity Hash* to a native config owned by another repo.
PROVIDER_CALL_CONFIG_VARIABLE = "provider_call_config_ref"
EVALUATION_PROCEDURE_CONFIG_VARIABLE = "evaluation_procedure_config_ref"

# Character Budget derivation rule / ratio: a Graph Definition Variable that
# lives in Graph Config identity (see ``character_budget``).
CHARACTER_BUDGET_VARIABLE = "character_budget_rule"

# Primary declared outputs.
GENERATION_OUTPUT_FIELD = "generation"
EVAL_OUTPUT_FIELD = "evaluation"


def _typed_config_ref(
    *, schema_name: str, identity_hash: str
) -> dict[str, str]:
    """A typed config reference plus Identity Hash, JSON-safe for identity.

    This is the static Variable value that names a native config owned by
    another repo (a dr-providers Provider Call Config or a dr-code Evaluation
    Procedure Config). It carries the owning schema name and the full
    64-char Identity Hash so changing the referenced config changes
    ``graph_hash``.
    """
    return {"schema_name": schema_name, "identity_hash": identity_hash}


def llm_call_node_definition(
    node_id: str,
    *,
    prompt_source: str,
    declares_character_budget: bool = False,
    output_field: str = GENERATION_OUTPUT_FIELD,
) -> NodeDefinition:
    """Build the closed ``whetstone.llm-call/v1`` Node Definition.

    ``prompt_source`` is the Node Input Source ref for the prompt input
    (a Graph External Input such as ``task.prompt`` or an upstream Node
    Output). The Provider Call Config is a *static Variable*, not an input
    source. When ``declares_character_budget`` is set, the Character Budget
    derivation rule/ratio is also a declared static Variable in Graph Config
    identity.
    """
    variable_names = {PROVIDER_CALL_CONFIG_VARIABLE}
    if declares_character_budget:
        variable_names = variable_names | {CHARACTER_BUDGET_VARIABLE}
    return NodeDefinition(
        node_id=node_id,
        node_type=LLM_CALL_NODE_TYPE,
        fields=(
            NodeFieldSpec(name="prompt", role=FieldRole.INPUT),
            NodeFieldSpec(name=output_field, role=FieldRole.OUTPUT),
        ),
        input_sources={
            "prompt": as_node_input_source_ref(prompt_source),
        },
        output_field=output_field,
        variable_names=frozenset(variable_names),
    )


def llm_call_variable_assignment(
    *,
    provider_call_config_schema: str,
    provider_call_config_hash: str,
    character_budget_rule: Any | None = None,
) -> dict[str, Any]:
    """Static Variable assignment for one ``whetstone.llm-call/v1`` Node."""
    values: dict[str, Any] = {
        PROVIDER_CALL_CONFIG_VARIABLE: _typed_config_ref(
            schema_name=provider_call_config_schema,
            identity_hash=provider_call_config_hash,
        ),
    }
    if character_budget_rule is not None:
        values[CHARACTER_BUDGET_VARIABLE] = character_budget_rule
    return values


def eval_node_definition(
    node_id: str,
    *,
    upstream_sources: Mapping[str, str],
    output_field: str = EVAL_OUTPUT_FIELD,
) -> NodeDefinition:
    """Build the closed ``whetstone.eval/v1`` Node Definition.

    ``upstream_sources`` maps each declared eval input field to exactly one
    upstream Node Output (or a Graph External Input) via a Node Input Source
    ref. The Evaluation Procedure Config is a *static Variable* — never an
    input source.
    """
    if not upstream_sources:
        raise ValueError(
            "eval node must declare at least one upstream Node Output source"
        )
    fields = (
        *(
            NodeFieldSpec(name=name, role=FieldRole.INPUT)
            for name in upstream_sources
        ),
        NodeFieldSpec(name=output_field, role=FieldRole.OUTPUT),
    )
    return NodeDefinition(
        node_id=node_id,
        node_type=EVAL_NODE_TYPE,
        fields=fields,
        input_sources={
            name: as_node_input_source_ref(ref)
            for name, ref in upstream_sources.items()
        },
        output_field=output_field,
        variable_names=frozenset({EVALUATION_PROCEDURE_CONFIG_VARIABLE}),
    )


def eval_variable_assignment(
    *,
    evaluation_procedure_config_schema: str,
    evaluation_procedure_config_hash: str,
) -> dict[str, Any]:
    """The static Variable assignment for one ``whetstone.eval/v1`` Node."""
    return {
        EVALUATION_PROCEDURE_CONFIG_VARIABLE: _typed_config_ref(
            schema_name=evaluation_procedure_config_schema,
            identity_hash=evaluation_procedure_config_hash,
        ),
    }


def eval_node_procedure_hash(node_variables: Mapping[str, Any]) -> str:
    """Extract the Evaluation Procedure Config Identity Hash from an Eval
    Node's static Variable assignment.

    Raises ``KeyError`` if the Node does not carry the procedure reference
    Variable, and ``ValueError`` if it is malformed.
    """
    ref = node_variables[EVALUATION_PROCEDURE_CONFIG_VARIABLE]
    if not isinstance(ref, dict) or "identity_hash" not in ref:
        raise ValueError(
            "evaluation procedure config reference is malformed: "
            f"{ref!r}"
        )
    return str(ref["identity_hash"])


__all__ = [
    "CHARACTER_BUDGET_VARIABLE",
    "EVALUATION_PROCEDURE_CONFIG_VARIABLE",
    "EVAL_NODE_TYPE",
    "EVAL_OUTPUT_FIELD",
    "GENERATION_OUTPUT_FIELD",
    "LLM_CALL_NODE_TYPE",
    "PROVIDER_CALL_CONFIG_VARIABLE",
    "eval_node_definition",
    "eval_node_procedure_hash",
    "eval_variable_assignment",
    "llm_call_node_definition",
    "llm_call_variable_assignment",
]
