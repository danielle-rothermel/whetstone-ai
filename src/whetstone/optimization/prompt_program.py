"""Optimizer-agnostic, versioned structured prompt-program rendering."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictStr,
    model_validator,
)

from whetstone.optimization.identity import (
    compute_identity_hash,
    reject_non_json,
)
from whetstone.optimization.schema import Candidate

PROMPT_PROGRAM_PAYLOAD_FIELD = "structured_prompt_program"
PROMPT_PROGRAM_SCHEMA = "whetstone.structured_prompt_program"
PROMPT_PROGRAM_SCHEMA_VERSION = 1
PROMPT_PROGRAM_RENDERER_VERSION = "whetstone_plain_examples/v1"


class PromptProgramExample(BaseModel):
    """One structured example, independent of instruction text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_example(self) -> PromptProgramExample:
        if not self.inputs and not self.outputs:
            raise ValueError("prompt-program example cannot be empty")
        reject_non_json(self.inputs, field="prompt-program example inputs")
        reject_non_json(self.outputs, field="prompt-program example outputs")
        return self


class PromptProgramComponent(BaseModel):
    """Examples bound to one native candidate instruction field."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    component_id: StrictStr
    candidate_field: StrictStr
    examples: tuple[PromptProgramExample, ...] = ()

    @model_validator(mode="after")
    def _validate_component(self) -> PromptProgramComponent:
        if not self.component_id or not self.candidate_field:
            raise ValueError("prompt-program component fields are required")
        return self


class PromptProgram(BaseModel):
    """A versioned structured program consumed by an execution renderer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    renderer_version: Literal["whetstone_plain_examples/v1"] = (
        PROMPT_PROGRAM_RENDERER_VERSION
    )
    components: tuple[PromptProgramComponent, ...]

    @model_validator(mode="after")
    def _validate_program(self) -> PromptProgram:
        if not self.components:
            raise ValueError("structured prompt program needs components")
        ids = tuple(component.component_id for component in self.components)
        fields = tuple(
            component.candidate_field for component in self.components
        )
        if len(ids) != len(set(ids)) or len(fields) != len(set(fields)):
            raise ValueError("prompt-program components must be unique")
        return self

    def identity_hash(self) -> str:
        return compute_identity_hash(
            schema=PROMPT_PROGRAM_SCHEMA,
            schema_version=PROMPT_PROGRAM_SCHEMA_VERSION,
            payload=self.model_dump(mode="json"),
        )


def prompt_program(candidate: Candidate) -> PromptProgram | None:
    raw = candidate.payload.get(PROMPT_PROGRAM_PAYLOAD_FIELD)
    if raw is None:
        return None
    return PromptProgram.model_validate(raw)


def prompt_program_instruction_template(
    candidate: Candidate,
    *,
    candidate_field: str,
) -> str:
    """Return the exact native instruction selected for a render surface."""

    program = prompt_program(candidate)
    if program is not None:
        matches = tuple(
            component
            for component in program.components
            if component.candidate_field == candidate_field
        )
        if len(matches) != 1:
            raise ValueError(
                "render surface must match exactly one prompt-program "
                "component"
            )
    template = candidate.payload.get(candidate_field)
    if type(template) is not str or not template:
        raise ValueError("candidate instruction must be a non-empty string")
    return template


def render_prompt_program(
    candidate: Candidate,
    *,
    candidate_field: str,
    render_instruction: Callable[[str], str],
) -> str:
    """Render examples and the task instruction without mutating either."""

    template = prompt_program_instruction_template(
        candidate,
        candidate_field=candidate_field,
    )
    rendered_instruction = render_instruction(template)
    program = prompt_program(candidate)
    if program is None:
        return rendered_instruction
    component = next(
        item
        for item in program.components
        if item.candidate_field == candidate_field
    )
    if not component.examples:
        return rendered_instruction
    rendered_examples = "\n\n".join(
        _render_example(index, example)
        for index, example in enumerate(component.examples, start=1)
    )
    return (
        "Use the following worked examples as demonstrations.\n\n"
        f"{rendered_examples}\n\n"
        "Now complete the current task.\n\n"
        f"{rendered_instruction}"
    )


def _render_example(index: int, example: PromptProgramExample) -> str:
    return "\n".join(
        (
            f"### Example {index}",
            "Inputs:",
            *_render_fields(example.inputs),
            "Outputs:",
            *_render_fields(example.outputs),
        )
    )


def _render_fields(values: dict[str, Any]) -> tuple[str, ...]:
    if not values:
        return ("(none)",)
    return tuple(
        f"{name}: {_render_value(value)}" for name, value in values.items()
    )


def _render_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


__all__ = [
    "PROMPT_PROGRAM_PAYLOAD_FIELD",
    "PROMPT_PROGRAM_RENDERER_VERSION",
    "PROMPT_PROGRAM_SCHEMA",
    "PROMPT_PROGRAM_SCHEMA_VERSION",
    "PromptProgram",
    "PromptProgramComponent",
    "PromptProgramExample",
    "prompt_program",
    "prompt_program_instruction_template",
    "render_prompt_program",
]
