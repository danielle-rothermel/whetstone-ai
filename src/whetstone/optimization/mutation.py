"""The Mutation Surface + Diff Check for proposal validation.

Every optimizing run in this system shares one Mutation Surface: the encoder
``user_prompt_template`` field only. A proposal is valid iff it changes only
that field relative to its named base candidate and canonical-JSON-matches the
base everywhere else (the "diff check", ``concrete-changes.html`` / the run
docs' shared harness expectation #3). This module makes that check a small,
reusable, testable function so both COPRO and MIPROv2 reject the same way.

The check is applied by the adapter *before* it emits a candidate: an invalid
draft (empty template, a payload that touches a non-surface field, or a
mutated base binding) is rejected and never becomes a proposal or an Evaluation
Intent. Rejection is data, not an exception path the harness has to unwind: the
adapter records the rejected draft as provenance and either retries within its
attempt cap or fails the Step per its cardinality rule.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dr_serialize import canonical_json

if TYPE_CHECKING:
    from whetstone.optimization.proposer import ProposalDraft
    from whetstone.optimization.schema import (
        Candidate,
        OptimizationRun,
        OptimizationRunRef,
        TemplateRenderContract,
    )

__all__ = [
    "MUTATION_FIELD",
    "DiffCheckError",
    "ProposalValidationError",
    "candidate_from_draft",
    "diff_check",
    "validate_candidate_template",
]

# The single allowed mutation field across every optimizing run here.
MUTATION_FIELD = "user_prompt_template"


class DiffCheckError(ValueError):
    """A proposed candidate failed the Mutation-Surface diff check."""


class ProposalValidationError(DiffCheckError):
    """A typed proposer draft cannot become a valid candidate."""


def candidate_from_draft(
    *,
    base: Candidate,
    candidate_id: str,
    draft: ProposalDraft,
    run: OptimizationRun | OptimizationRunRef | TemplateRenderContract,
) -> Candidate:
    """The sole draft-to-candidate validation path.

    Failed drafts remain failures. Successful drafts must use only renderable
    placeholders and then pass the same mutation-surface diff check as every
    other proposal. There is no base-template fallback.
    """
    if draft.failed:
        failure = draft.terminal_failure
        if failure is None:
            raise AssertionError(
                "failed proposal draft lacks terminal failure"
            )
        raise ProposalValidationError(failure.message)
    from whetstone.optimization.schema import (
        Candidate,
        OptimizationRun,
        OptimizationRunRef,
        TemplateRenderContract,
        candidate_reference,
    )

    if type(run) is OptimizationRun:
        contract = run.template_render_contract
    elif type(run) is OptimizationRunRef:
        contract = run.record.template_render_contract
    elif type(run) is TemplateRenderContract:
        contract = run
    else:
        raise TypeError(
            "run must be an exact OptimizationRun, RunRef, or render contract"
        )
    try:
        contract.validate_template(draft.template)
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
    from whetstone.optimization.schema import (
        OptimizationRun,
        OptimizationRunRef,
    )

    if type(run) is OptimizationRun:
        exact_run = run
    elif type(run) is OptimizationRunRef:
        exact_run = run.record
    else:
        raise TypeError("run must be an exact OptimizationRun or RunRef")
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
    from whetstone.optimization.schema import candidate_reference

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
