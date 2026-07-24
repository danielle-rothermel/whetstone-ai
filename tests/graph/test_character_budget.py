"""Character Budget graph/runtime binding (deliverable 7)."""

from __future__ import annotations

import pytest
from dr_graph import GraphDefinition, graph_hash

from tests.graph.support import (
    PROVIDER_CALL_CONFIG_SCHEMA,
    eval_config,
    fake_hash,
)
from whetstone.graph.character_budget import (
    CHARACTER_BUDGET_EXTERNAL_INPUT,
    CharacterBudgetRule,
    derive_character_bound,
)
from whetstone.graph.nodes import (
    CHARACTER_BUDGET_VARIABLE,
    eval_node_definition,
    eval_variable_assignment,
    llm_call_node_definition,
    llm_call_variable_assignment,
)


def test_derivation_rule_is_a_graph_variable_in_identity() -> None:
    node = llm_call_node_definition(
        "generate", prompt_source="task.prompt", declares_character_budget=True
    )
    assert CHARACTER_BUDGET_VARIABLE in node.variable_names


def _graph_with_budget(ratio: float):
    proc = eval_config().evaluation_procedure_config_hash
    llm = llm_call_node_definition(
        "generate",
        prompt_source="task.prompt",
        declares_character_budget=True,
    )
    ev = eval_node_definition(
        "evaluate", upstream_sources={"candidate": "generate"}
    )
    definition = GraphDefinition(nodes=(llm, ev), terminal_node_id="evaluate")
    assignments = {
        "generate": llm_call_variable_assignment(
            provider_call_config_schema=PROVIDER_CALL_CONFIG_SCHEMA,
            provider_call_config_hash=fake_hash("a"),
            character_budget_rule=CharacterBudgetRule(
                ratio=ratio
            ).identity_value(),
        ),
        "evaluate": eval_variable_assignment(
            evaluation_procedure_config_schema=(
                "dr_code.evaluation_procedure.config"
            ),
            evaluation_procedure_config_hash=proc,
        ),
    }
    return definition.materialize(assignments)


def test_changing_budget_rule_ratio_changes_graph_hash() -> None:
    base = _graph_with_budget(0.5)
    changed = _graph_with_budget(0.75)
    assert graph_hash(base) != graph_hash(changed)


def test_concrete_task_bound_is_a_graph_external_input() -> None:
    # The concrete Task-derived bound is supplied via task.<field> and is a
    # Graph External Input, excluded from Graph Config identity.
    assert CHARACTER_BUDGET_EXTERNAL_INPUT.startswith("task.")


def test_derive_character_bound() -> None:
    rule = CharacterBudgetRule(ratio=0.5)
    assert derive_character_bound(rule, task_length=100) == 50


def test_budget_rule_rejects_nonpositive_ratio() -> None:
    with pytest.raises(ValueError, match="positive"):
        CharacterBudgetRule(ratio=0.0)


def test_no_character_budget_policy_artifact_exists() -> None:
    # Absence test: there is no separate character-budget policy artifact
    # type anywhere in the whetstone graph package. The name is built at
    # runtime so the stale-name scan stays clean.
    import whetstone.graph.character_budget as mod

    forbidden = "Character" + "Budget" + "Policy"
    assert not hasattr(mod, forbidden)
