from __future__ import annotations

import string
from typing import TYPE_CHECKING

from dr_serialize import canonical_json

if TYPE_CHECKING:
    from whetstone.experiment.candidate import (
        Candidate,
    )
    from whetstone.optimization.contracts import (
        OptimizationRun,
        OptimizationRunRef,
    )
    from whetstone.optimization.proposal.proposer import ProposalDraft

__all__ = [
    "MUTATION_FIELD",
    "DiffCheckError",
    "ProposalValidationError",
    "candidate_from_draft",
    "diff_check",
    "template_placeholder_fields",
    "validate_candidate_template",
]

# The single allowed mutation field across every optimizing run here.
MUTATION_FIELD = "user_prompt_template"


def template_placeholder_fields(template: str) -> tuple[str, ...]:
    """Return every placeholder occurrence in one template, in order.

    Parses the pinned brace syntax without importing the candidate runtime, so
    callers validating a bare replacement template need no render contract.
    Raises :class:`ValueError` when the braces are malformed.
    """

    fields: list[str] = []
    for _literal, field_name, _spec, _conversion in string.Formatter().parse(
        template
    ):
        if field_name is None:
            continue
        if field_name == "":
            fields.append("<positional>")
            continue
        head = field_name.replace("[", ".").split(".", 1)[0]
        fields.append("<positional>" if head.isdigit() else head)
    return tuple(fields)


class DiffCheckError(ValueError):
    """A proposed candidate failed the Mutation-Surface diff check."""


class ProposalValidationError(DiffCheckError):
    """A typed proposer draft cannot become a valid candidate."""


def _validated_optimization_run(
    run: OptimizationRun | OptimizationRunRef,
) -> OptimizationRun:
    """Revalidate an exact run authority at its mutation boundary."""

    from whetstone.optimization.contracts import (
        OptimizationRun,
        OptimizationRunRef,
    )

    if type(run) is OptimizationRun:
        return OptimizationRun.model_validate(run.model_dump(mode="python"))
    if type(run) is OptimizationRunRef:
        return OptimizationRunRef.model_validate(
            run.model_dump(mode="python")
        ).record
    raise TypeError("run must be an exact OptimizationRun or RunRef")


def candidate_from_draft(
    *,
    base: Candidate,
    candidate_id: str,
    draft: ProposalDraft,
    run: OptimizationRun | OptimizationRunRef,
) -> Candidate:
    """The sole draft-to-candidate validation path.

    Failed drafts remain failures. Successful drafts must use only renderable
    placeholders and then pass the same mutation-surface diff check as every
    other proposal. There is no base-template fallback.
    """
    exact_run = _validated_optimization_run(run)
    if draft.failed:
        failure = draft.terminal_failure
        if failure is None:
            raise AssertionError(
                "failed proposal draft lacks terminal failure"
            )
        raise ProposalValidationError(failure.message)
    from whetstone.experiment.candidate import (
        Candidate,
        candidate_reference,
    )

    try:
        exact_run.template_render_contract.validate_template(draft.template)
    except ValueError as error:
        raise ProposalValidationError(
            f"proposal template violates its render contract: {error}"
        ) from error
    payload = base.payload.to_json()
    payload[MUTATION_FIELD] = draft.template
    proposed = Candidate(
        candidate_id=candidate_id,
        base_ref=candidate_reference(base).record_ref,
        payload=payload,
    )
    diff_check(base=base, proposed=proposed)
    return proposed


def validate_candidate_template(
    *,
    candidate: Candidate,
    run: OptimizationRun | OptimizationRunRef,
) -> None:
    """Validate one candidate template under an exact run's authority."""
    exact_run = _validated_optimization_run(run)
    exact_run.template_render_contract.validate_template(
        candidate.payload.get(MUTATION_FIELD)
    )


def diff_check(
    *,
    base: Candidate,
    proposed: Candidate,
) -> None:
    """Validate a proposal against its base under the Mutation Surface.

    Raises :class:`DiffCheckError` unless the proposal:

    * binds the exact same base (``base_ref`` byte-matches),
    * supplies a non-empty ``MUTATION_FIELD`` value, and
    * canonical-JSON-matches the base on **every** other payload key (no
      added, dropped, or altered non-surface field).
    """
    from whetstone.experiment.candidate import candidate_reference

    expected_base_ref = candidate_reference(base).record_ref
    if proposed.base_ref != expected_base_ref:
        raise DiffCheckError(
            f"proposal binds base {proposed.base_ref!r}, not the exact "
            f"request candidate {expected_base_ref!r}"
        )
    value = proposed.payload.get(MUTATION_FIELD)
    if type(value) is not str or value == "":
        raise DiffCheckError(
            f"proposal must supply a non-empty {MUTATION_FIELD!r} template"
        )
    base_payload = base.model_dump(mode="json")["payload"]
    proposed_payload = proposed.model_dump(mode="json")["payload"]
    if MUTATION_FIELD not in base_payload:
        raise DiffCheckError(
            f"base candidate must supply the {MUTATION_FIELD!r} mutation field"
        )
    base_value = base_payload[MUTATION_FIELD]
    if type(base_value) is not str:
        raise DiffCheckError(
            f"base candidate {MUTATION_FIELD!r} mutation field must be "
            "a string"
        )
    if value == base_value:
        raise DiffCheckError(
            f"proposal {MUTATION_FIELD!r} mutation must differ from its base"
        )
    base_others = {
        k: v for k, v in base_payload.items() if k != MUTATION_FIELD
    }
    prop_others = {
        k: v for k, v in proposed_payload.items() if k != MUTATION_FIELD
    }
    if canonical_json(prop_others) != canonical_json(base_others):
        raise DiffCheckError(
            "proposal changes a field outside the Mutation Surface "
            f"({MUTATION_FIELD!r} only): non-surface payload diverged "
            "from base"
        )
