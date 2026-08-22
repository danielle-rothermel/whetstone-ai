"""Golden pins for the persisted evaluation-evidence wire format.

`EvalEvidence` is content-addressed and stored, so its schema name, version,
and exact field spelling are a persisted-format contract. These literals are
written out by hand on purpose: deriving them from the model would make the
test agree with any drift it was meant to catch.
"""

from __future__ import annotations

from whetstone.core.roles import EvalRole
from whetstone.eval.schema import (
    EVAL_EVIDENCE_SCHEMA_VERSION,
    EVAL_OUTPUTS_SCHEMA,
    EVAL_OUTPUTS_SCHEMA_VERSION,
    EVAL_TRACES_SCHEMA,
    EVAL_TRACES_SCHEMA_VERSION,
    EvalEvidence,
    EvalOutputRow,
)
from whetstone.eval.schema_names import EVAL_EVIDENCE_SCHEMA

EXPECTED_EVIDENCE_FIELDS = (
    "schema_version",
    "candidate",
    "eval_config_ref",
    "eval_role",
    "provider_execution_policy_ref",
    "graph_hash",
    "graph_config_ref",
    "metadata",
    "dataset_hash",
    "task_hashes",
    "num_seeds",
    "per_task_values",
    "per_task_counts",
    "row_accounting",
    "traces_ref",
    "outputs_ref",
    "aggregate_ref",
    "aggregate_name",
    "aggregate_value",
    "aggregate_status",
    "reward_ref",
    "cache",
    "deadline_reached",
)


def test_persisted_schema_names_and_versions_are_pinned() -> None:
    assert EVAL_EVIDENCE_SCHEMA == "whetstone.eval_evidence"
    assert EVAL_EVIDENCE_SCHEMA_VERSION == 5
    assert EVAL_OUTPUTS_SCHEMA == "whetstone.eval_outputs"
    assert EVAL_OUTPUTS_SCHEMA_VERSION == 5
    assert EVAL_TRACES_SCHEMA == "whetstone.eval_component_traces"
    assert EVAL_TRACES_SCHEMA_VERSION == 2


def test_eval_evidence_wire_fields_are_pinned() -> None:
    assert tuple(EvalEvidence.model_fields) == EXPECTED_EVIDENCE_FIELDS


def test_eval_evidence_declares_the_current_schema_version() -> None:
    """The literal in the model and the exported constant cannot drift apart."""
    annotation = EvalEvidence.model_fields["schema_version"].annotation
    assert annotation.__args__ == (EVAL_EVIDENCE_SCHEMA_VERSION,)


def test_retired_scheduler_evidence_fields_are_gone() -> None:
    """The worker pool cannot produce these, so they must not be persisted."""
    assert "concurrency_halved" not in EvalEvidence.model_fields
    assert "guard_timeouts" not in EvalEvidence.model_fields


#: The persisted spellings of every evaluation role, written by hand. These
#: strings land inside stored `EvalEvidence` records, so renaming a member or
#: changing a value silently orphans every record that carries the old
#: spelling.
EXPECTED_EVAL_ROLE_LITERALS = (
    ("INTERNAL", "internal"),
    ("OFFICIAL", "official"),
    ("HELD_OUT", "held_out"),
)


def test_eval_role_wire_literals_are_pinned() -> None:
    assert tuple(
        (member.name, member.value) for member in EvalRole
    ) == EXPECTED_EVAL_ROLE_LITERALS


def test_eval_role_values_are_unique() -> None:
    """`@verify(UNIQUE)` on the enum; asserted here so the decorator cannot
    be dropped without a test noticing."""
    values = [member.value for member in EvalRole]
    assert len(set(values)) == len(values)

EXPECTED_OUTPUT_ROW_FIELDS = (
    "candidate_id",
    "task_id",
    "task_hash",
    "task_index",
    "seed_index",
    "rendered_prompt",
    "output_text",
    "score",
    "failed",
    "missing",
    "invalid",
    "failure_code",
    "finish_reason",
    "provider_error",
    "max_budget",
    "over_budget",
    "submission_result",
    "prompt_tokens",
    "completion_tokens",
    "provider_cost",
)


def test_eval_output_row_wire_fields_are_pinned() -> None:
    """Output rows carry the task-model usage run cost is derived from."""
    assert tuple(EvalOutputRow.model_fields) == EXPECTED_OUTPUT_ROW_FIELDS
