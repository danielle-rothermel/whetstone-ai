"""The Mutation Surface + Diff Check for proposal validation.

Every optimizing run in this system shares one Mutation Surface: the encoder
``user_prompt_template`` field only. A proposal is valid iff it changes only
that field relative to its named base candidate and byte-matches the base
everywhere else (the "diff check", ``concrete-changes.html`` / the run docs'
shared harness expectation #3). This module makes that check a small,
reusable, testable function so both COPRO and MIPROv2 reject the same way.

The check is applied by the adapter *before* it emits a candidate: an invalid
draft (empty template, a payload that touches a non-surface field, or a
mutated base binding) is rejected and never becomes a proposal or an Evaluation
Intent. Rejection is data, not an exception path the harness has to unwind: the
adapter records the rejected draft as provenance and either retries within its
attempt cap or fails the Step per its cardinality rule.
"""

from __future__ import annotations

import string
from collections import Counter
from collections.abc import Iterable

from whetstone.optimization.proposer import ProposalDraft
from whetstone.optimization.schema import Candidate

__all__ = [
    "MUTATION_FIELD",
    "POSITIONAL_FIELD_TOKEN",
    "DiffCheckError",
    "ProposalValidationError",
    "candidate_from_draft",
    "diff_check",
    "invalid_template_placeholders",
    "template_placeholder_fields",
]

# The single allowed mutation field across every optimizing run here.
MUTATION_FIELD = "user_prompt_template"
POSITIONAL_FIELD_TOKEN = "<positional>"


def template_placeholder_fields(template: str) -> tuple[str, ...]:
    fields: list[str] = []
    for _literal, field_name, _spec, _conv in string.Formatter().parse(
        template
    ):
        if field_name is None:
            continue
        if field_name == "":
            fields.append(POSITIONAL_FIELD_TOKEN)
            continue
        head = field_name.replace("[", ".").split(".", 1)[0]
        fields.append(POSITIONAL_FIELD_TOKEN if head.isdigit() else head)
    return tuple(fields)


def invalid_template_placeholders(
    template: str, valid_keys: Iterable[str]
) -> tuple[str, ...]:
    allowed = set(valid_keys)
    offending: list[str] = []
    seen: set[str] = set()
    for field_name in template_placeholder_fields(template):
        if field_name in allowed or field_name in seen:
            continue
        seen.add(field_name)
        offending.append(field_name)
    return tuple(offending)


class DiffCheckError(ValueError):
    """A proposed candidate failed the Mutation-Surface diff check."""


class ProposalValidationError(DiffCheckError):
    """A typed proposer draft cannot become a valid candidate."""


def candidate_from_draft(
    *,
    base: Candidate,
    candidate_id: str,
    draft: ProposalDraft,
    valid_template_keys: Iterable[str],
    required_template_keys: Iterable[str] = (),
) -> Candidate:
    """The sole draft-to-candidate validation path.

    Failed drafts remain failures. Successful drafts must use only renderable
    placeholders and then pass the same mutation-surface diff check as every
    other proposal. There is no base-template fallback.
    """
    if draft.failed:
        raise ProposalValidationError(
            draft.failure_detail or "proposer failed without detail"
        )
    try:
        invalid = invalid_template_placeholders(
            draft.template, valid_template_keys
        )
    except ValueError as exc:
        raise ProposalValidationError(
            f"proposal template has malformed placeholders: {exc}"
        ) from exc
    if invalid:
        raise ProposalValidationError(
            "proposal template contains unavailable placeholders: "
            + ", ".join(invalid)
        )
    proposed_fields = Counter(template_placeholder_fields(draft.template))
    required_fields = Counter(required_template_keys)
    missing = tuple(
        field
        for field, count in required_fields.items()
        if proposed_fields[field] < count
    )
    if missing:
        raise ProposalValidationError(
            "proposal template removes required placeholders: "
            + ", ".join(missing)
        )
    proposed = Candidate(
        candidate_id=candidate_id,
        base_ref=base.base_ref,
        payload={**base.payload, MUTATION_FIELD: draft.template},
    )
    diff_check(base=base, proposed=proposed)
    return proposed


def diff_check(
    *,
    base: Candidate,
    proposed: Candidate,
    mutation_field: str = MUTATION_FIELD,
) -> None:
    """Validate a proposal against its base under the Mutation Surface.

    Raises :class:`DiffCheckError` unless the proposal:

    * binds the exact same base (``base_ref`` byte-matches),
    * supplies a non-empty ``mutation_field`` value, and
    * byte-matches the base on **every** other payload key (no added, dropped,
      or altered non-surface field).
    """
    if proposed.base_ref != base.base_ref:
        raise DiffCheckError(
            f"proposal binds base {proposed.base_ref!r}, not its named base "
            f"{base.base_ref!r}"
        )
    value = proposed.payload.get(mutation_field)
    if not isinstance(value, str) or value == "":
        raise DiffCheckError(
            f"proposal must supply a non-empty {mutation_field!r} template"
        )
    base_others = {
        k: v for k, v in base.payload.items() if k != mutation_field
    }
    prop_others = {
        k: v for k, v in proposed.payload.items() if k != mutation_field
    }
    if prop_others != base_others:
        raise DiffCheckError(
            "proposal changes a field outside the Mutation Surface "
            f"({mutation_field!r} only): non-surface payload diverged "
            "from base"
        )
