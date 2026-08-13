from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from whetstone.experiment.candidate import (
    Candidate,
    CandidateRef,
    TemplateRenderContract,
    TemplateRenderKind,
)
from whetstone.optim.contracts import (
    OptimRun,
    OptimRunRef,
)
from whetstone.optim.proposal.mutation import (
    _validated_optimization_run,
    candidate_from_draft,
    diff_check,
)
from whetstone.optim.proposal.proposer import ProposalDraft


def _format_literal_text(
    value: str,
    *,
    template_render_contract: TemplateRenderContract,
    context: str,
) -> str:

    if template_render_contract.kind is TemplateRenderKind.PYTHON_FORMAT_V1:
        return value.replace("{", "{{").replace("}", "}}")
    if template_render_contract.kind is TemplateRenderKind.LITERAL_REPLACE_V1:
        field = template_render_contract.available_fields[0]
        token = f"{{{field}}}"
        if token in value:
            raise ValueError(
                "literal_replace/v1 cannot losslessly compose MIPROv2 "
                f"{context} containing active token {token!r}"
            )
    return value


def _format_literal_json(
    value: object,
    *,
    template_render_contract: TemplateRenderContract,
    context: str,
) -> str:

    return _format_literal_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")),
        template_render_contract=template_render_contract,
        context=context,
    )


def compose_user_prompt_template(
    components: Sequence[Mapping[str, Any]],
    *,
    template_render_contract: TemplateRenderContract,
) -> str:

    if len(components) != 1:
        raise ValueError(
            "MIPROv2 rendering requires exactly one optimizable component"
        )
    sections: list[str] = []
    for ordinal, component in enumerate(components):
        component_id = component.get("component_id")
        instruction = component.get("instruction")
        if type(component_id) is not str or not component_id:
            raise ValueError("MIPROv2 component_id must be non-empty")
        if type(instruction) is not str or not instruction:
            raise ValueError("MIPROv2 instruction must be non-empty")
        metadata = {
            "component_id": component_id,
            "demo_identity_hash": component.get("demo_identity_hash"),
            "demo_index": component.get("demo_index"),
            "instruction_identity_hash": component.get(
                "instruction_identity_hash"
            ),
            "instruction_index": component.get("instruction_index"),
            "ordinal": ordinal,
        }
        demonstrations = component.get("demo_set")
        rendered_component_id = _format_literal_text(
            component_id,
            template_render_contract=template_render_contract,
            context="component id",
        )
        sections.extend(
            (
                f"## Component {ordinal + 1}: {rendered_component_id}",
                "### Metadata",
                _format_literal_json(
                    metadata,
                    template_render_contract=template_render_contract,
                    context="metadata",
                ),
                "### Instruction",
                instruction,
                "### Demonstrations",
                (
                    "[]"
                    if demonstrations is None
                    else _format_literal_json(
                        demonstrations,
                        template_render_contract=template_render_contract,
                        context="demonstrations",
                    )
                ),
            )
        )
    return "\n".join(sections)


def candidate_from_components(
    *,
    base: CandidateRef,
    candidate_id: str,
    components: Sequence[Mapping[str, Any]],
    run: OptimRun | OptimRunRef,
) -> Candidate:

    template_render_contract = _validated_optimization_run(
        run
    ).template_render_contract
    candidate = candidate_from_draft(
        base=base.record,
        candidate_id=candidate_id,
        draft=ProposalDraft(
            template=compose_user_prompt_template(
                components,
                template_render_contract=template_render_contract,
            )
        ),
        run=run,
    )
    if candidate.base_ref != base.record_ref:
        raise AssertionError("canonical candidate did not bind the exact base")
    diff_check(base=base.record, proposed=candidate, run=run)
    return candidate


__all__ = ["candidate_from_components", "compose_user_prompt_template"]
