from __future__ import annotations

import string
from typing import TYPE_CHECKING

from dr_serialize import canonical_json

if TYPE_CHECKING:
    from whetstone.experiment.candidate import (
        Candidate,
    )
    from whetstone.optim.contracts import (
        OptimRun,
        OptimRunRef,
    )
    from whetstone.optim.proposal.proposer import ProposalDraft

__all__ = [
    "DiffCheckError",
    "ProposalValidationError",
    "candidate_from_draft",
    "diff_check",
    "resolve_mutation_field",
    "template_placeholder_fields",
    "validate_candidate_template",
]


def template_placeholder_fields(template: str) -> tuple[str, ...]:

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
    pass


class ProposalValidationError(DiffCheckError):
    pass


def _validated_optimization_run(
    run: OptimRun | OptimRunRef,
) -> OptimRun:

    from whetstone.optim.contracts import (
        OptimRun,
        OptimRunRef,
    )

    if type(run) is OptimRun:
        return OptimRun.model_validate(run.model_dump(mode="python"))
    if type(run) is OptimRunRef:
        return OptimRunRef.model_validate(
            run.model_dump(mode="python")
        ).record
    raise TypeError("run must be an exact OptimRun or RunRef")


def resolve_mutation_field(
    *,
    run: OptimRun | OptimRunRef | None = None,
    mutation_field: str | None = None,
) -> str:
    if mutation_field is not None:
        if type(mutation_field) is not str or not mutation_field.strip():
            raise ValueError("mutation_field must be a non-empty string")
        return mutation_field
    if run is None:
        raise TypeError("resolve_mutation_field requires run or mutation_field")
    return _validated_optimization_run(run).mutation_field


def candidate_from_draft(
    *,
    base: Candidate,
    candidate_id: str,
    draft: ProposalDraft,
    run: OptimRun | OptimRunRef,
) -> Candidate:
    exact_run = _validated_optimization_run(run)
    field = exact_run.mutation_field
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
    payload[field] = draft.template
    proposed = Candidate(
        candidate_id=candidate_id,
        base_ref=candidate_reference(base).record_ref,
        payload=payload,
    )
    diff_check(base=base, proposed=proposed, run=exact_run)
    return proposed


def validate_candidate_template(
    *,
    candidate: Candidate,
    run: OptimRun | OptimRunRef,
) -> None:
    exact_run = _validated_optimization_run(run)
    field = exact_run.mutation_field
    exact_run.template_render_contract.validate_template(
        candidate.payload.get(field)
    )


def diff_check(
    *,
    base: Candidate,
    proposed: Candidate,
    run: OptimRun | OptimRunRef,
) -> None:
    from whetstone.experiment.candidate import candidate_reference

    field = resolve_mutation_field(run=run)
    expected_base_ref = candidate_reference(base).record_ref
    if proposed.base_ref != expected_base_ref:
        raise DiffCheckError(
            f"proposal binds base {proposed.base_ref!r}, not the exact "
            f"request candidate {expected_base_ref!r}"
        )
    value = proposed.payload.get(field)
    if type(value) is not str or value == "":
        raise DiffCheckError(
            f"proposal must supply a non-empty {field!r} template"
        )
    base_payload = base.model_dump(mode="json")["payload"]
    proposed_payload = proposed.model_dump(mode="json")["payload"]
    if field not in base_payload:
        raise DiffCheckError(
            f"base candidate must supply the {field!r} mutation field"
        )
    base_value = base_payload[field]
    if type(base_value) is not str:
        raise DiffCheckError(
            f"base candidate {field!r} mutation field must be a string"
        )
    if value == base_value:
        raise DiffCheckError(
            f"proposal {field!r} mutation must differ from its base"
        )
    base_others = {k: v for k, v in base_payload.items() if k != field}
    prop_others = {k: v for k, v in proposed_payload.items() if k != field}
    if canonical_json(prop_others) != canonical_json(base_others):
        raise DiffCheckError(
            "proposal changes a field outside the Mutation Surface "
            f"({field!r} only): non-surface payload diverged from base"
        )
