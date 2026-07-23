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
from collections.abc import Iterable

from whetstone.optimization.schema import Candidate

__all__ = [
    "MUTATION_FIELD",
    "POSITIONAL_FIELD_TOKEN",
    "DiffCheckError",
    "diff_check",
    "invalid_template_placeholders",
    "template_placeholder_fields",
]

# The single allowed mutation field across every optimizing run here.
MUTATION_FIELD = "user_prompt_template"

#: The reported token for a positional (``{}``) or index (``{0}``) format
#: field. A render restricted to keyword ``prompt_inputs`` can never fill a
#: positional field, so any positional field is an invalid placeholder; it is
#: surfaced under this stable token so a rejection reason names it readably.
POSITIONAL_FIELD_TOKEN = "<positional>"


def template_placeholder_fields(template: str) -> tuple[str, ...]:
    """The ordered ``str.format`` placeholder fields a template references.

    Parses ``template`` with :func:`string.Formatter().parse` -- the exact
    engine ``str.format`` uses at render time -- so the fields returned are the
    ones a render would try to fill. Escaped ``{{``/``}}`` literals carry no
    field and are skipped. A positional (``{}``) or index (``{0}``) field is
    reported as :data:`POSITIONAL_FIELD_TOKEN`, since a keyword-only render
    over ``prompt_inputs`` can never fill it. Only the top-level field NAME is
    reported (an attribute/index suffix like ``{grid[0]}`` -> ``grid``),
    because that is the key the render's ``**prompt_inputs`` lookup uses.
    """
    fields: list[str] = []
    for _literal, field_name, _spec, _conv in string.Formatter().parse(
        template
    ):
        if field_name is None:
            continue  # a literal run (or an escaped brace), no field
        if field_name == "":
            fields.append(POSITIONAL_FIELD_TOKEN)
            continue
        # Strip an attribute/index suffix: ``grid[0]``/``grid.x`` -> ``grid``.
        head = field_name.replace("[", ".").split(".", 1)[0]
        if head.isdigit():
            fields.append(POSITIONAL_FIELD_TOKEN)  # an index field {0}
        else:
            fields.append(head)
    return tuple(fields)


def invalid_template_placeholders(
    template: str, valid_keys: Iterable[str]
) -> tuple[str, ...]:
    """The offending placeholder fields a template references but cannot fill.

    ``valid_keys`` are the render's known keyword inputs (the env's public
    ``prompt_inputs`` keys, plus the fields the env's own probe templates use).
    Returns the ordered, de-duplicated field names that are NOT valid: an
    unknown named field, or a positional/index field (reported as
    :data:`POSITIONAL_FIELD_TOKEN`). Empty tuple means every placeholder is
    fillable -- the template renders without a ``KeyError``.
    """
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
