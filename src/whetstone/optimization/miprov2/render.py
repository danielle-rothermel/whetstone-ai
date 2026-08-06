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
from whetstone.optimization.proposal.mutation import (
    candidate_from_draft,
    diff_check,
)
from whetstone.optimization.proposal.proposer import ProposalDraft


def _format_literal_json(value: object, *, escape_braces: bool) -> str:
    """Encode JSON that renders back to itself under the run's contract.

    Only ``python_format/v1`` consumes ``{{``/``}}`` as brace escapes. Under
    the literal contracts those pairs survive rendering verbatim, so doubling
    them there would deliver malformed JSON to the task model.
    """

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    if not escape_braces:
        return encoded
    return encoded.replace("{", "{{").replace("}", "}}")


def compose_user_prompt_template(
    components: Sequence[Mapping[str, Any]],
    *,
    template_render_contract: TemplateRenderContract,
) -> str:
    """Compose ordered instructions, metadata, and demonstrations exactly."""

    escape_braces = (
        template_render_contract.kind is TemplateRenderKind.PYTHON_FORMAT_V1
    )
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
        sections.extend(
            (
                f"## Component {ordinal + 1}: {component_id}",
                "### Metadata",
                _format_literal_json(metadata, escape_braces=escape_braces),
                "### Instruction",
                instruction,
                "### Demonstrations",
                (
                    "[]"
                    if demonstrations is None
                    else _format_literal_json(
                        demonstrations, escape_braces=escape_braces
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
    template_render_contract: TemplateRenderContract,
) -> Candidate:
    """Create one exact candidate through the canonical mutation boundary."""

    candidate = candidate_from_draft(
        base=base.record,
        candidate_id=candidate_id,
        draft=ProposalDraft(
            template=compose_user_prompt_template(
                components,
                template_render_contract=template_render_contract,
            )
        ),
        run=template_render_contract,
    )
    if candidate.base_ref != base.record_ref:
        raise AssertionError("canonical candidate did not bind the exact base")
    diff_check(base=base.record, proposed=candidate)
    return candidate


__all__ = ["candidate_from_components", "compose_user_prompt_template"]
