"""Canonical MIPROv2 projection onto the sole candidate mutation field."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from whetstone.optimization.mutation import candidate_from_draft, diff_check
from whetstone.optimization.proposer import ProposalDraft
from whetstone.optimization.schema import (
    Candidate,
    CandidateRef,
    TemplateRenderContract,
)


def _format_literal_json(value: object) -> str:
    """Encode JSON as literal text inside a Python-format template."""

    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"))
        .replace("{", "{{")
        .replace("}", "}}")
    )


def compose_user_prompt_template(
    components: Sequence[Mapping[str, Any]],
) -> str:
    """Compose ordered instructions, metadata, and demonstrations exactly."""

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
                _format_literal_json(metadata),
                "### Instruction",
                instruction,
                "### Demonstrations",
                (
                    "[]"
                    if demonstrations is None
                    else _format_literal_json(demonstrations)
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
        draft=ProposalDraft(template=compose_user_prompt_template(components)),
        run=template_render_contract,
    )
    if candidate.base_ref != base.record_ref:
        raise AssertionError("canonical candidate did not bind the exact base")
    diff_check(base=base.record, proposed=candidate)
    return candidate


__all__ = ["candidate_from_components", "compose_user_prompt_template"]
