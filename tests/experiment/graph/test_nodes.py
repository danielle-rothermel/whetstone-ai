"""Closed, versioned Node Definition contracts."""

from __future__ import annotations

import pytest
from dr_graph import graph_hash

from tests.experiment.graph.support import build_graph_config, fake_hash
from whetstone.experiment.graph.nodes import (
    EVAL_NODE_TYPE,
    EVALUATION_PROCEDURE_CONFIG_VARIABLE,
    GENERATION_OUTPUT_FIELD,
    LLM_CALL_NODE_TYPE,
    PROVIDER_CALL_CONFIG_VARIABLE,
    eval_node_definition,
    eval_node_procedure_hash,
    eval_variable_assignment,
    llm_call_node_definition,
    llm_call_variable_assignment,
)


def test_llm_call_node_uses_closed_versioned_type() -> None:
    node = llm_call_node_definition("generate", prompt_source="task.prompt")

    assert LLM_CALL_NODE_TYPE == "whetstone.llm-call/v1"
    assert node.model_dump(mode="json") == {
        "node_id": "generate",
        "node_type": LLM_CALL_NODE_TYPE,
        "fields": [
            {
                "name": "prompt",
                "role": "input",
                "type_name": "str",
                "description": None,
            },
            {
                "name": GENERATION_OUTPUT_FIELD,
                "role": "output",
                "type_name": "str",
                "description": None,
            },
        ],
        "input_sources": {"prompt": "task.prompt"},
        "output_field": GENERATION_OUTPUT_FIELD,
        "variable_names": [PROVIDER_CALL_CONFIG_VARIABLE],
    }


def test_eval_node_uses_closed_versioned_type() -> None:
    node = eval_node_definition(
        "evaluate", upstream_sources={"candidate": "generate"}
    )
    assert EVAL_NODE_TYPE == "whetstone.eval/v1"
    assert node.model_dump(mode="json") == {
        "node_id": "evaluate",
        "node_type": EVAL_NODE_TYPE,
        "fields": [
            {
                "name": "candidate",
                "role": "input",
                "type_name": "str",
                "description": None,
            },
            {
                "name": "evaluation",
                "role": "output",
                "type_name": "str",
                "description": None,
            },
        ],
        "input_sources": {"candidate": "generate"},
        "output_field": "evaluation",
        "variable_names": [EVALUATION_PROCEDURE_CONFIG_VARIABLE],
    }


def test_eval_node_requires_an_upstream_source() -> None:
    with pytest.raises(ValueError, match="upstream Node Output"):
        eval_node_definition("evaluate", upstream_sources={})


def test_provider_call_config_reference_is_in_graph_hash() -> None:
    proc = fake_hash("c")
    base = build_graph_config(
        provider_call_config_hash=fake_hash("a"),
        evaluation_procedure_config_hash=proc,
    )
    changed = build_graph_config(
        provider_call_config_hash=fake_hash("b"),
        evaluation_procedure_config_hash=proc,
    )
    # Changing the Provider Call Config reference changes graph_hash.
    assert graph_hash(base) != graph_hash(changed)


def test_procedure_reference_change_changes_graph_hash() -> None:
    pcc = fake_hash("a")
    base = build_graph_config(
        provider_call_config_hash=pcc,
        evaluation_procedure_config_hash=fake_hash("c"),
    )
    changed = build_graph_config(
        provider_call_config_hash=pcc,
        evaluation_procedure_config_hash=fake_hash("d"),
    )
    assert graph_hash(base) != graph_hash(changed)


def test_eval_variable_assignment_carries_typed_ref_and_hash() -> None:
    assignment = eval_variable_assignment(
        evaluation_procedure_config_schema="dr_code.evaluation_procedure.config",
        evaluation_procedure_config_hash=fake_hash("e"),
    )
    ref = assignment[EVALUATION_PROCEDURE_CONFIG_VARIABLE]
    assert ref["schema_name"] == "dr_code.evaluation_procedure.config"
    assert ref["identity_hash"] == fake_hash("e")


@pytest.mark.parametrize(
    "schema_name",
    ["", "   "],
)
def test_typed_config_references_require_nonempty_schema(
    schema_name: str,
) -> None:
    with pytest.raises(ValueError, match="schema_name must be non-empty"):
        llm_call_variable_assignment(
            provider_call_config_schema=schema_name,
            provider_call_config_hash=fake_hash("a"),
        )


@pytest.mark.parametrize(
    "identity_hash",
    ["short", "g" * 64, "A" * 64],
)
def test_typed_config_references_require_canonical_hash(
    identity_hash: str,
) -> None:
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        eval_variable_assignment(
            evaluation_procedure_config_schema=(
                "dr_code.evaluation_procedure.config"
            ),
            evaluation_procedure_config_hash=identity_hash,
        )


def test_eval_procedure_hash_validates_deserialized_reference() -> None:
    identity_hash = fake_hash("e")
    variables = {
        EVALUATION_PROCEDURE_CONFIG_VARIABLE: {
            "schema_name": "dr_code.evaluation_procedure.config",
            "identity_hash": identity_hash,
        }
    }
    assert eval_node_procedure_hash(variables) == identity_hash


def test_eval_procedure_hash_requires_reference_variable() -> None:
    with pytest.raises(KeyError, match=EVALUATION_PROCEDURE_CONFIG_VARIABLE):
        eval_node_procedure_hash({})


@pytest.mark.parametrize(
    "reference",
    [
        None,
        {},
        {
            "schema_name": "",
            "identity_hash": "e" * 64,
        },
        {
            "schema_name": "dr_code.evaluation_procedure.config",
            "identity_hash": None,
        },
        {
            "schema_name": "dr_code.evaluation_procedure.config",
            "identity_hash": "E" * 64,
        },
    ],
)
def test_eval_procedure_hash_rejects_malformed_reference(reference) -> None:
    with pytest.raises(ValueError):
        eval_node_procedure_hash(
            {EVALUATION_PROCEDURE_CONFIG_VARIABLE: reference}
        )
