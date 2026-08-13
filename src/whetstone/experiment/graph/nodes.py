from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dr_graph import (
    FieldRole,
    NodeDefinition,
    NodeFieldSpec,
    as_node_input_source_ref,
)

from whetstone.core.identity import require_full_hash

LLM_CALL_NODE_TYPE = "whetstone.llm-call/v1"
EVAL_NODE_TYPE = "whetstone.eval/v1"


PROVIDER_CALL_CONFIG_VARIABLE = "provider_call_config_ref"
EVAL_PROCEDURE_CONFIG_VARIABLE = "evaluation_procedure_config_ref"


PROVIDER_GENERATION_OUTPUT_FIELD = "provider_generation"
EVAL_OUTPUT_FIELD = "evaluation"


def _require_config_hash(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(
            f"{field} must be a full 64-char lowercase SHA-256 hash, "
            f"got {value!r}"
        )
    return require_full_hash(value, field=field)


def _typed_config_ref(
    *, schema_name: str, identity_hash: str
) -> dict[str, str]:
    if not isinstance(schema_name, str) or not schema_name.strip():
        raise ValueError("config reference schema_name must be non-empty")
    return {
        "schema_name": schema_name,
        "identity_hash": _require_config_hash(
            identity_hash, field="config reference identity_hash"
        ),
    }


def llm_call_node_definition(
    node_id: str,
    *,
    prompt_source: str,
    output_field: str = PROVIDER_GENERATION_OUTPUT_FIELD,
) -> NodeDefinition:
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
        variable_names=frozenset({PROVIDER_CALL_CONFIG_VARIABLE}),
    )


def llm_call_variable_assignment(
    *,
    provider_call_config_schema: str,
    provider_call_config_hash: str,
) -> dict[str, Any]:
    return {
        PROVIDER_CALL_CONFIG_VARIABLE: _typed_config_ref(
            schema_name=provider_call_config_schema,
            identity_hash=provider_call_config_hash,
        ),
    }


def eval_node_definition(
    node_id: str,
    *,
    upstream_sources: Mapping[str, str],
    output_field: str = EVAL_OUTPUT_FIELD,
) -> NodeDefinition:
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
        variable_names=frozenset({EVAL_PROCEDURE_CONFIG_VARIABLE}),
    )


def eval_variable_assignment(
    *,
    evaluation_procedure_config_schema: str,
    evaluation_procedure_config_hash: str,
) -> dict[str, Any]:
    return {
        EVAL_PROCEDURE_CONFIG_VARIABLE: _typed_config_ref(
            schema_name=evaluation_procedure_config_schema,
            identity_hash=evaluation_procedure_config_hash,
        ),
    }


def eval_node_procedure_hash(node_variables: Mapping[str, Any]) -> str:
    ref = node_variables[EVAL_PROCEDURE_CONFIG_VARIABLE]
    if not isinstance(ref, Mapping):
        raise ValueError(
            f"evaluation procedure config reference is malformed: {ref!r}"
        )
    schema_name = ref.get("schema_name")
    if not isinstance(schema_name, str) or not schema_name.strip():
        raise ValueError(
            "evaluation procedure config reference schema_name must be "
            "non-empty"
        )
    return _require_config_hash(
        ref.get("identity_hash"),
        field="evaluation procedure config reference identity_hash",
    )


__all__ = [
    "EVAL_PROCEDURE_CONFIG_VARIABLE",
    "EVAL_NODE_TYPE",
    "EVAL_OUTPUT_FIELD",
    "LLM_CALL_NODE_TYPE",
    "PROVIDER_CALL_CONFIG_VARIABLE",
    "PROVIDER_GENERATION_OUTPUT_FIELD",
    "eval_node_definition",
    "eval_node_procedure_hash",
    "eval_variable_assignment",
    "llm_call_node_definition",
    "llm_call_variable_assignment",
]
