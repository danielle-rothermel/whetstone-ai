"""Closed, versioned Node Definitions (deliverable 3)."""

from __future__ import annotations

import pytest
from dr_graph import graph_hash

from tests.graph.support import build_graph_config, fake_hash
from whetstone.graph.nodes import (
    EVAL_NODE_TYPE,
    EVALUATION_PROCEDURE_CONFIG_VARIABLE,
    GENERATION_OUTPUT_FIELD,
    LLM_CALL_NODE_TYPE,
    PROVIDER_CALL_CONFIG_VARIABLE,
    eval_node_definition,
    eval_variable_assignment,
    llm_call_node_definition,
)


def test_llm_call_node_uses_closed_versioned_type() -> None:
    node = llm_call_node_definition("generate", prompt_source="task.prompt")
    assert node.node_type == LLM_CALL_NODE_TYPE == "whetstone.llm-call/v1"
    assert node.output_field == GENERATION_OUTPUT_FIELD
    # Provider Call Config is a declared static Variable, not an input source.
    assert PROVIDER_CALL_CONFIG_VARIABLE in node.variable_names
    assert "prompt" in node.input_sources
    assert PROVIDER_CALL_CONFIG_VARIABLE not in node.input_sources


def test_eval_node_uses_closed_versioned_type() -> None:
    node = eval_node_definition(
        "evaluate", upstream_sources={"candidate": "generate"}
    )
    assert node.node_type == EVAL_NODE_TYPE == "whetstone.eval/v1"
    # Evaluation Procedure Config is a static Variable, never a Node Input
    # Source.
    assert EVALUATION_PROCEDURE_CONFIG_VARIABLE in node.variable_names
    assert EVALUATION_PROCEDURE_CONFIG_VARIABLE not in node.input_sources
    # It consumes a declared upstream Node Output.
    assert node.input_sources["candidate"].dependency_node_id == "generate"


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
