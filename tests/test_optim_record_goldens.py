"""Golden wire keys for the content-addressed optimization records.

These records are content-addressed, so their exact serialized key sets are
their persisted identity. The literals below are written out by hand on
purpose: deriving them from the models would make this test agree with any
silent drift instead of catching it. Changing a key set is a persisted-format
change and must bump the matching schema version here and in
``whetstone.optim.contracts``.
"""

from __future__ import annotations

from whetstone.optim.contracts import (
    OPTIM_RESULT_SCHEMA,
    OPTIM_RESULT_SCHEMA_VERSION,
    OPTIM_RUN_SCHEMA,
    OPTIM_RUN_SCHEMA_VERSION,
    STEP_REQUEST_SCHEMA,
    STEP_REQUEST_SCHEMA_VERSION,
    STEP_RESULT_SCHEMA,
    STEP_RESULT_SCHEMA_VERSION,
    OptimResult,
    OptimRun,
    OptimStepRequest,
    OptimStepResult,
    OutputContract,
    SearchEvidence,
)

GOLDEN_SCHEMA_NAMES = {
    "optim_run": "whetstone.optim_run",
    "step_request": "whetstone.optim_step_request",
    "step_result": "whetstone.optim_step_result",
    "optim_result": "whetstone.optim_result",
}

GOLDEN_SCHEMA_VERSIONS = {
    "optim_run": 1,
    "step_request": 2,
    "step_result": 2,
    "optim_result": 2,
}

GOLDEN_OUTPUT_CONTRACT_KEYS = frozenset(
    {
        "returned_proposal_count",
        "terminal_proposal_count",
        "require_distinct_bases",
    }
)

GOLDEN_STEP_REQUEST_KEYS = frozenset(
    {
        "run",
        "step_id",
        "kind",
        "kind_label",
        "step_index",
        "prior_step_result_ref",
        "prior_state_ref",
        "prior_history_ref",
        "candidates",
        "pools",
        "hyperparameters",
        "budget",
        "step_output_contract",
    }
)

GOLDEN_SEARCH_EVIDENCE_KEYS = frozenset(
    {
        "eval_request_id",
        "candidate",
        "outcome",
        "eval_result_ref",
        "reward_ref",
        "reward_evidence_refs",
    }
)

GOLDEN_STEP_RESULT_KEYS = frozenset(
    {
        "request",
        "proposed_candidates",
        "accepted_candidates",
        "resolved_intents",
        "search_evidence",
        "tool_evidence",
        "state_ref",
        "history_ref",
        "budget_delta",
        "budget",
        "status",
        "terminal_failure",
        "seed_retained",
        "provenance_note",
        "provenance_ordinal",
    }
)

GOLDEN_OPTIM_RESULT_KEYS = frozenset(
    {
        "run",
        "proposals",
        "step_results",
        "cost",
        "terminal_failure",
        "seed_retained",
        "provenance_note",
        "provenance_ordinal",
    }
)

GOLDEN_OPTIM_RUN_KEYS = frozenset(
    {
        "run_id",
        "optimizer_config",
        "adapter_key",
        "mode",
        "terminal_output_contract",
        "template_render_contract",
        "mutation_field",
        "reward_policy",
        "tool_configs",
    }
)


def test_persisted_schema_names_are_pinned() -> None:
    assert OPTIM_RUN_SCHEMA == GOLDEN_SCHEMA_NAMES["optim_run"]
    assert STEP_REQUEST_SCHEMA == GOLDEN_SCHEMA_NAMES["step_request"]
    assert STEP_RESULT_SCHEMA == GOLDEN_SCHEMA_NAMES["step_result"]
    assert OPTIM_RESULT_SCHEMA == GOLDEN_SCHEMA_NAMES["optim_result"]


def test_persisted_schema_versions_are_pinned() -> None:
    assert OPTIM_RUN_SCHEMA_VERSION == GOLDEN_SCHEMA_VERSIONS["optim_run"]
    assert (
        STEP_REQUEST_SCHEMA_VERSION
        == GOLDEN_SCHEMA_VERSIONS["step_request"]
    )
    assert STEP_RESULT_SCHEMA_VERSION == GOLDEN_SCHEMA_VERSIONS["step_result"]
    assert (
        OPTIM_RESULT_SCHEMA_VERSION
        == GOLDEN_SCHEMA_VERSIONS["optim_result"]
    )


def test_output_contract_wire_keys_are_pinned() -> None:
    assert (
        set(OutputContract(returned_proposal_count=1).model_dump(mode="json"))
        == GOLDEN_OUTPUT_CONTRACT_KEYS
    )


def test_optim_run_wire_keys_are_pinned() -> None:
    assert set(OptimRun.model_fields) == GOLDEN_OPTIM_RUN_KEYS


def test_step_request_wire_keys_are_pinned() -> None:
    assert set(OptimStepRequest.model_fields) == GOLDEN_STEP_REQUEST_KEYS


def test_step_result_wire_keys_are_pinned() -> None:
    assert set(OptimStepResult.model_fields) == GOLDEN_STEP_RESULT_KEYS


def test_search_evidence_wire_keys_are_pinned() -> None:
    assert set(SearchEvidence.model_fields) == GOLDEN_SEARCH_EVIDENCE_KEYS


def test_optim_result_wire_keys_are_pinned() -> None:
    assert set(OptimResult.model_fields) == GOLDEN_OPTIM_RESULT_KEYS


def test_seed_retained_serializes_on_the_wire() -> None:
    """The no-improvement signal is a persisted field, not a derived view."""
    dumped = OutputContract(
        returned_proposal_count=0,
        terminal_proposal_count=1,
    ).model_dump(mode="json")
    assert dumped["terminal_proposal_count"] == 1
    assert dumped["returned_proposal_count"] == 0
    assert "seed_retained" in OptimStepResult.model_fields
    assert "seed_retained" in OptimResult.model_fields
