"""Golden wire keys for the persisted run cost report.

``RunCostReport`` is serialized into ``OptimResult.cost``, which is a
content-addressed record, so its exact key spelling is a persisted format.
The literals below are written out by hand on purpose: deriving them from the
models would make this test agree with any silent drift instead of catching
it. Changing a key set is a persisted-format change and must bump
``COST_REPORT_SCHEMA_VERSION`` here and in ``whetstone.optim.cost``.
"""

from __future__ import annotations

from whetstone.optim.cost import (
    COST_REPORT_SCHEMA,
    COST_REPORT_SCHEMA_VERSION,
    CostRole,
    ProposerCallUsage,
    RoleCost,
    RunCostReport,
)

GOLDEN_COST_REPORT_KEYS = frozenset(
    {
        "schema_version",
        "task_model",
        "proposer",
    }
)

GOLDEN_ROLE_COST_KEYS = frozenset(
    {
        "calls",
        "input_tokens",
        "output_tokens",
        "priced_calls",
        "unpriced_calls",
        "usd",
    }
)

GOLDEN_PROPOSER_CALL_USAGE_KEYS = frozenset(
    {
        "prompt_tokens",
        "completion_tokens",
        "usd",
    }
)


def test_persisted_schema_name_and_version_are_pinned() -> None:
    assert COST_REPORT_SCHEMA == "whetstone.optim_run_cost"
    assert COST_REPORT_SCHEMA_VERSION == 1


def test_cost_role_values_are_pinned() -> None:
    """Role names are persisted map keys, not display strings."""
    assert CostRole.TASK_MODEL.value == "task_model"
    assert CostRole.PROPOSER.value == "proposer"


def test_run_cost_report_wire_keys_are_pinned() -> None:
    content = RunCostReport().record_content()
    assert set(content) == GOLDEN_COST_REPORT_KEYS
    assert set(content["task_model"]) == GOLDEN_ROLE_COST_KEYS
    assert set(content["proposer"]) == GOLDEN_ROLE_COST_KEYS


def test_role_cost_wire_keys_are_pinned() -> None:
    assert set(RoleCost().model_dump(mode="json")) == GOLDEN_ROLE_COST_KEYS


def test_proposer_call_usage_wire_keys_are_pinned() -> None:
    assert (
        set(ProposerCallUsage().model_dump(mode="json"))
        == GOLDEN_PROPOSER_CALL_USAGE_KEYS
    )


def test_default_report_carries_its_schema_version() -> None:
    """The version travels with the payload, not only with the code."""
    assert RunCostReport().record_content()["schema_version"] == 1
