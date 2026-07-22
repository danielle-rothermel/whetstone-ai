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

from whetstone.optimization.schema import Candidate

__all__ = [
    "MUTATION_FIELD",
    "DiffCheckError",
    "diff_check",
]

# The single allowed mutation field across every optimizing run here.
MUTATION_FIELD = "user_prompt_template"


class DiffCheckError(ValueError):
    """A proposed candidate failed the Mutation-Surface diff check."""


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
