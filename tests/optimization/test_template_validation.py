"""One hard-cut proposer-draft and mutation validation path."""

from inspect import signature

import pytest

from whetstone.experiment.candidate import (
    Candidate,
    TemplateRenderContract,
    TemplateRenderKind,
    candidate_reference,
)
from whetstone.experiment.reward import (
    RewardPolicy,
    RewardTerm,
)
from whetstone.optimization.contracts import (
    OptimizationRun,
    OutputContract,
    StepMode,
)
from whetstone.optimization.proposal.mutation import (
    DiffCheckError,
    ProposalValidationError,
    candidate_from_draft,
    diff_check,
)
from whetstone.optimization.proposal.proposer import ProposalDraft

from .support import base_ref, candidate, optimizer_config_ref


def python_format_contract(
    *,
    available_fields: tuple[str, ...] = ("query",),
    required_fields: tuple[str, ...] = (),
) -> TemplateRenderContract:
    return TemplateRenderContract(
        kind=TemplateRenderKind.PYTHON_FORMAT_V1,
        available_fields=available_fields,
        required_fields=required_fields,
    )


def proposal_run(
    contract: TemplateRenderContract | None = None,
) -> OptimizationRun:
    return OptimizationRun(
        run_id="template-validation",
        optimizer_config=optimizer_config_ref("proposal"),
        adapter_key="proposal-test",
        mode=StepMode.PROPOSAL_ONLY,
        terminal_output_contract=OutputContract(returned_proposal_count=1),
        template_render_contract=contract or python_format_contract(),
        reward_policy=RewardPolicy(
            policy_name="template-validation/v1",
            terms=(RewardTerm(name="score", weight=1.0),),
        ),
    )


def test_successful_draft_becomes_the_only_surface_mutation() -> None:
    base = candidate(text="old")
    proposed = candidate_from_draft(
        base=base,
        candidate_id="P1",
        draft=ProposalDraft(template="Use {query} carefully"),
        run=proposal_run(),
    )
    assert proposed.payload["user_prompt_template"] == "Use {query} carefully"
    assert proposed.payload["fixed"] == base.payload["fixed"]


def test_failed_draft_never_falls_back_to_base_template() -> None:
    base = candidate(text="old")
    with pytest.raises(ProposalValidationError, match="timeout"):
        candidate_from_draft(
            base=base,
            candidate_id="P1",
            draft=ProposalDraft.failure(detail="timeout"),
            run=proposal_run(),
        )


def test_unrenderable_placeholders_fail_before_candidate_creation() -> None:
    with pytest.raises(ProposalValidationError, match="question"):
        candidate_from_draft(
            base=candidate(),
            candidate_id="P1",
            draft=ProposalDraft(template="{question}"),
            run=proposal_run(),
        )


@pytest.mark.parametrize(
    "template",
    (
        "{}",
        "{0}",
        "{query.missing}",
        "{query[missing]}",
        "{query!s}",
        "{query!r}",
        "{query!a}",
        "{query!z}",
        "{query:}",
        "{query:>10}",
        "{query:*^10.5s}",
        "{value:{width}}",
        "{query",
        "query}",
        "{value:{width}",
        "{value:{width!s}",
        "{value:{width!!s}}",
    ),
)
def test_python_format_rejects_hostile_or_ambiguous_syntax(
    template: str,
) -> None:
    with pytest.raises(ProposalValidationError):
        candidate_from_draft(
            base=candidate(),
            candidate_id="P1",
            draft=ProposalDraft(template=template),
            run=proposal_run(
                python_format_contract(
                    available_fields=("query", "value", "width")
                )
            ),
        )


def test_python_format_escaped_braces_are_literal_and_renderable() -> None:
    template = "literal {{query.missing}} and {{value[missing]}}"
    contract = python_format_contract(available_fields=())
    assert contract.placeholder_fields(template) == ()
    assert contract.render(template, {}) == (
        "literal {query.missing} and {value[missing]}"
    )
    proposed = candidate_from_draft(
        base=candidate(),
        candidate_id="P1",
        draft=ProposalDraft(template=template),
        run=proposal_run(contract),
    )
    assert proposed.payload["user_prompt_template"] == template


def test_accepted_template_renders_actual_string_placeholder_values() -> None:
    template = "literal {{query}} | {query} | {answer}"
    contract = python_format_contract(available_fields=("query", "answer"))
    proposed = candidate_from_draft(
        base=candidate(),
        candidate_id="P1",
        draft=ProposalDraft(template=template),
        run=proposal_run(contract),
    )

    rendered_template = proposed.payload["user_prompt_template"]
    assert isinstance(rendered_template, str)
    assert (
        contract.render(
            rendered_template, {"query": "abcdef", "answer": "yes"}
        )
        == "literal {query} | abcdef | yes"
    )


@pytest.mark.parametrize(
    ("base_value", "proposed_value"),
    ((True, 1), (1, 1.0)),
)
def test_diff_check_uses_strict_json_scalar_equality(
    base_value: object,
    proposed_value: object,
) -> None:
    base = Candidate(
        candidate_id="A",
        base_ref=base_ref(),
        payload={"user_prompt_template": "old", "fixed": base_value},
    )
    proposed = Candidate(
        candidate_id="P1",
        base_ref=candidate_reference(base).record_ref,
        payload={"user_prompt_template": "new", "fixed": proposed_value},
    )
    with pytest.raises(DiffCheckError, match="non-surface"):
        diff_check(base=base, proposed=proposed)


def test_diff_check_uses_strict_json_nested_equality() -> None:
    base = Candidate(
        candidate_id="A",
        base_ref=base_ref(),
        payload={
            "user_prompt_template": "old",
            "fixed": {"nested": [{"value": True}]},
        },
    )
    proposed = Candidate(
        candidate_id="P1",
        base_ref=candidate_reference(base).record_ref,
        payload={
            "user_prompt_template": "new",
            "fixed": {"nested": [{"value": 1}]},
        },
    )
    with pytest.raises(DiffCheckError, match="non-surface"):
        diff_check(base=base, proposed=proposed)


def test_diff_check_rejects_an_unchanged_mutation_field() -> None:
    base = candidate(text="same")
    proposed = Candidate(
        candidate_id="P1",
        base_ref=candidate_reference(base).record_ref,
        payload=base.model_dump(mode="json")["payload"],
    )
    with pytest.raises(DiffCheckError, match="must differ"):
        diff_check(base=base, proposed=proposed)


def test_diff_check_has_no_configurable_mutation_field() -> None:
    assert tuple(signature(diff_check).parameters) == ("base", "proposed")


def test_diff_check_preserves_protected_array_order_and_multiplicity() -> None:
    base = Candidate(
        candidate_id="A",
        base_ref=base_ref(),
        payload={
            "user_prompt_template": "old",
            "fixed": ["a", "a", "b"],
        },
    )
    proposed = Candidate(
        candidate_id="P1",
        base_ref=candidate_reference(base).record_ref,
        payload={
            "user_prompt_template": "new",
            "fixed": ["a", "b", "a"],
        },
    )
    with pytest.raises(DiffCheckError, match="non-surface"):
        diff_check(base=base, proposed=proposed)


def test_multiple_candidates_may_share_a_base() -> None:
    base = candidate()
    proposed = tuple(
        candidate_from_draft(
            base=base,
            candidate_id=f"P{index}",
            draft=ProposalDraft(template=template),
            run=proposal_run(),
        )
        for index, template in enumerate(
            ("First {query}", "Second {query}"),
            start=1,
        )
    )
    assert tuple(item.base_ref for item in proposed) == (
        candidate_reference(base).record_ref,
        candidate_reference(base).record_ref,
    )


def test_draft_clones_nested_payload_before_mutating() -> None:
    base = Candidate(
        candidate_id="A",
        base_ref=base_ref(),
        payload={
            "user_prompt_template": "old",
            "nested": {"items": [{"value": 1}, {"value": 2}]},
        },
    )

    proposed = candidate_from_draft(
        base=base,
        candidate_id="P1",
        draft=ProposalDraft(template="new {query}"),
        run=proposal_run(),
    )

    assert proposed.payload["nested"] == base.payload["nested"]
    assert proposed.payload.to_json()["nested"] == {
        "items": [{"value": 1}, {"value": 2}]
    }


def test_draft_accepts_exact_render_contract_authority() -> None:
    proposed = candidate_from_draft(
        base=candidate(),
        candidate_id="P1",
        draft=ProposalDraft(template="{query}"),
        run=python_format_contract(),
    )

    assert proposed.payload["user_prompt_template"] == "{query}"
