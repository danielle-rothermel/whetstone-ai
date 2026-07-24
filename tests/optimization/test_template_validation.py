"""One hard-cut proposer-draft and mutation validation path."""

import pytest

from whetstone.optimization import (
    POSITIONAL_FIELD_TOKEN,
    ProposalDraft,
    ProposalValidationError,
    candidate_from_draft,
    invalid_template_placeholders,
)

from .support import candidate


def test_successful_draft_becomes_the_only_surface_mutation() -> None:
    base = candidate(text="old")
    proposed = candidate_from_draft(
        base=base,
        candidate_id="P1",
        draft=ProposalDraft(template="Use {query} carefully"),
        valid_template_keys={"query"},
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
            valid_template_keys={"query"},
        )


def test_unrenderable_placeholders_fail_before_candidate_creation() -> None:
    with pytest.raises(ProposalValidationError, match="question"):
        candidate_from_draft(
            base=candidate(),
            candidate_id="P1",
            draft=ProposalDraft(template="{question}"),
            valid_template_keys={"query"},
        )
    assert invalid_template_placeholders("{} {query}", {"query"}) == (
        POSITIONAL_FIELD_TOKEN,
    )


def test_malformed_placeholders_are_typed_validation_failures() -> None:
    with pytest.raises(ProposalValidationError, match="malformed"):
        candidate_from_draft(
            base=candidate(),
            candidate_id="P1",
            draft=ProposalDraft(template="{query"),
            valid_template_keys={"query"},
        )


def test_required_placeholder_removal_is_rejected() -> None:
    with pytest.raises(ProposalValidationError, match="removes"):
        candidate_from_draft(
            base=candidate(),
            candidate_id="P1",
            draft=ProposalDraft(template="No template variables"),
            valid_template_keys={"query"},
            required_template_keys={"query"},
        )
