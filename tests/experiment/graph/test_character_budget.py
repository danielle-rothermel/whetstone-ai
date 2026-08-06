from __future__ import annotations

import pytest
from dr_graph import GraphDefinition, graph_hash

from tests.experiment.graph.support import (
    PROVIDER_CALL_CONFIG_SCHEMA,
    eval_config,
    fake_hash,
)
from whetstone.experiment.graph.character_budget import (
    CharacterBudgetRule,
    derive_character_bound,
)
from whetstone.experiment.graph.nodes import (
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
                "whetstone.evaluation_procedure.config"
            ),
            evaluation_procedure_config_hash=proc,
        ),
    }
    return definition.materialize(assignments)


def test_changing_budget_rule_ratio_changes_graph_hash() -> None:
    base = _graph_with_budget(0.5)
    changed = _graph_with_budget(0.75)
    assert graph_hash(base) != graph_hash(changed)


def test_derive_character_bound() -> None:
    rule = CharacterBudgetRule(ratio=0.5)
    assert derive_character_bound(rule, task_length=100) == 50


def test_derivation_uses_python_round_semantics() -> None:
    rule = CharacterBudgetRule(ratio=0.5)
    assert derive_character_bound(rule, task_length=3) == 2


def test_budget_rule_rejects_nonpositive_ratio() -> None:
    with pytest.raises(ValueError, match="positive"):
        CharacterBudgetRule(ratio=0.0)


@pytest.mark.parametrize(
    "ratio",
    [float("nan"), float("inf"), float("-inf")],
)
def test_budget_rule_rejects_nonfinite_ratio(ratio: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        CharacterBudgetRule(ratio=ratio)


def test_huge_finite_derived_bound_has_stable_error() -> None:
    rule = CharacterBudgetRule(ratio=1e308)
    with pytest.raises(ValueError, match="derived character bound"):
        derive_character_bound(rule, task_length=2)


@pytest.mark.parametrize(
    ("ratio", "task_length", "expected"),
    [
        (0.3, 10, 3),
        (1 / 3, 3, 1),
        (2 / 3, 3, 2),
    ],
)
def test_derivation_preserves_float_then_round_semantics(
    ratio: float,
    task_length: int,
    expected: int,
) -> None:
    assert (
        derive_character_bound(
            CharacterBudgetRule(ratio=ratio),
            task_length=task_length,
        )
        == expected
    )


def test_zero_length_task_has_zero_bound() -> None:
    assert (
        derive_character_bound(CharacterBudgetRule(ratio=1e308), task_length=0)
        == 0
    )
