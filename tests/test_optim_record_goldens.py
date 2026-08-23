"""Golden wire keys for the content-addressed optimization records.

These records are content-addressed, so their exact serialized key sets are
their persisted identity. The literals below are written out by hand on
purpose: deriving them from the models would make this test agree with any
silent drift instead of catching it. Changing a key set is a persisted-format
change and must bump the matching schema version here and in
``whetstone.optim.contracts``.
"""

from __future__ import annotations

from whetstone.core.identity import IdentityRef, TypedRef
from whetstone.experiment.candidate import candidate_reference
from whetstone.optim.contracts import (
    OPTIM_RESULT_SCHEMA,
    OPTIM_RESULT_SCHEMA_VERSION,
    OPTIM_RUN_SCHEMA,
    OPTIM_RUN_SCHEMA_VERSION,
    STEP_REQUEST_SCHEMA,
    STEP_REQUEST_SCHEMA_VERSION,
    STEP_RESULT_SCHEMA,
    STEP_RESULT_SCHEMA_VERSION,
    SUPERSEDED_FAILURE_CODES_KEY,
    IntentOutcome,
    OptimResult,
    OptimRun,
    OptimStepRequest,
    OptimStepResult,
    OutputContract,
    SearchEvidence,
    StepKind,
    StepMode,
    StepStatus,
    optimization_run_reference,
    step_request_reference,
    step_result_reference,
)
from whetstone.testing.toy.experiment import (
    TOY_MUTATION_FIELD,
    build_toy_experiment,
    toy_template_render_contract,
)

GOLDEN_SCHEMA_NAMES = {
    "optim_run": "whetstone.optim_run",
    "step_request": "whetstone.optim_step_request",
    "step_result": "whetstone.optim_step_result",
    "optim_result": "whetstone.optim_result",
}

GOLDEN_SCHEMA_VERSIONS = {
    "optim_run": 3,
    "step_request": 3,
    "step_result": 4,
    "optim_result": 3,
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
        "optim_run_id",
        "optim_step_index",
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
        "proposer_usage",
        "tool_evidence",
        "state_ref",
        "history_ref",
        "budget_delta",
        "budget",
        "status",
        "terminal_failure",
        "seed_retained",
        "retained_candidate_ref",
        "provenance_note",
        "provenance_ordinal",
    }
)

#: A Step failure's ``details`` key naming the nested failure codes the
#: Step supersedes. It is not a model field -- it lives inside a
#: persisted ``TerminalFailure.details`` payload -- so the key set tests
#: above cannot catch it drifting. Written out by hand for the same
#: reason they are.
GOLDEN_SUPERSEDED_FAILURE_CODES_KEY = "superseded_failure_codes"

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
        "initial_candidate_ref",
        "mutation_field",
        "reward_policy",
        "tool_configs",
    }
)


# --- minimal valid instances ----------------------------------------------
#
# The goldens below compare against ``model_dump(mode="json")``, not
# ``model_fields``: the serialized key set is what gets content-addressed, so
# an alias, an ``exclude``, or a custom serializer must fail here.


def _typed_ref(schema_name: str) -> TypedRef:
    return TypedRef(schema_name=schema_name, content_hash="a" * 64)


def _optim_run() -> OptimRun:
    experiment = build_toy_experiment(num_seeds=1)
    return OptimRun(
        run_id="golden-run",
        optimizer_config=IdentityRef(
            record_ref=_typed_ref("whetstone.optim_control"),
            record_hash="b" * 64,
        ),
        adapter_key="gepa",
        mode=StepMode.PROPOSAL_ONLY,
        terminal_output_contract=OutputContract(returned_proposal_count=0),
        template_render_contract=toy_template_render_contract(),
        initial_candidate_ref=candidate_reference(
            experiment.initial_candidate
        ),
        mutation_field=TOY_MUTATION_FIELD,
        reward_policy=experiment.reward_policy,
    )


def _step_request() -> OptimStepRequest:
    experiment = build_toy_experiment(num_seeds=1)
    return OptimStepRequest(
        run=optimization_run_reference(_optim_run()),
        step_id="golden-run:0",
        kind=StepKind.PROPOSAL,
        step_index=0,
        candidates=(experiment.initial_candidate,),
        step_output_contract=OutputContract(
            returned_proposal_count=0,
            terminal_proposal_count=0,
        ),
    )


def _step_result() -> OptimStepResult:
    experiment = build_toy_experiment(num_seeds=1)
    return OptimStepResult(
        request=step_request_reference(_step_request()),
        status=StepStatus.COMPLETE,
        seed_retained=True,
        retained_candidate_ref=candidate_reference(
            experiment.initial_candidate
        ),
    )


def _search_evidence() -> SearchEvidence:
    experiment = build_toy_experiment(num_seeds=1)
    return SearchEvidence(
        eval_request_id="golden-eval",
        optim_run_id="golden-run",
        optim_step_index=0,
        candidate=candidate_reference(experiment.initial_candidate),
        outcome=IntentOutcome.REJECTED,
    )


def _optim_result() -> OptimResult:
    return OptimResult(
        run=optimization_run_reference(_optim_run()),
        proposals=(),
        step_results=(step_result_reference(_step_result()),),
        seed_retained=True,
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
    assert (
        set(_optim_run().model_dump(mode="json")) == GOLDEN_OPTIM_RUN_KEYS
    )


def test_step_request_wire_keys_are_pinned() -> None:
    assert (
        set(_step_request().model_dump(mode="json"))
        == GOLDEN_STEP_REQUEST_KEYS
    )


def test_step_result_wire_keys_are_pinned() -> None:
    assert (
        set(_step_result().model_dump(mode="json"))
        == GOLDEN_STEP_RESULT_KEYS
    )


def test_search_evidence_wire_keys_are_pinned() -> None:
    assert (
        set(_search_evidence().model_dump(mode="json"))
        == GOLDEN_SEARCH_EVIDENCE_KEYS
    )


def test_optim_result_wire_keys_are_pinned() -> None:
    assert (
        set(_optim_result().model_dump(mode="json"))
        == GOLDEN_OPTIM_RESULT_KEYS
    )


def test_terminal_proposal_count_serializes_on_the_wire() -> None:
    """The split continuing/terminal cardinality is a persisted field."""
    dumped = OutputContract(
        returned_proposal_count=0,
        terminal_proposal_count=1,
    ).model_dump(mode="json")
    assert dumped["terminal_proposal_count"] == 1
    assert dumped["returned_proposal_count"] == 0


def test_seed_retained_serializes_on_the_wire() -> None:
    """The no-improvement signal is a persisted field, not a derived view."""
    assert _step_result().model_dump(mode="json")["seed_retained"] is True
    assert _optim_result().model_dump(mode="json")["seed_retained"] is True


def test_superseded_failure_codes_key_is_pinned() -> None:
    """The supersession key is persisted inside a Step failure.

    A superseding Step Result records the nested codes it stands for
    under this key, and the Step Result validator reads it back to check
    that set. Renaming it would silently turn every stored supersession
    into an unexplained disagreement with its own evidence, so the
    literal is pinned here rather than derived.
    """
    assert (
        SUPERSEDED_FAILURE_CODES_KEY == GOLDEN_SUPERSEDED_FAILURE_CODES_KEY
    )


def test_the_gepa_detailed_result_schema_version_is_pinned() -> None:
    """A persisted-format string has one owner and one pinned literal."""
    from whetstone.optim.gepa.control import GEPA_RESULT_SCHEMA_VERSION

    assert GEPA_RESULT_SCHEMA_VERSION == "whetstone.gepa_detailed_result/v2"
