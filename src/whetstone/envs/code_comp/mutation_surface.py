from __future__ import annotations

from whetstone.envs.code_comp.constants import (
    ED1_INVALID_BODY,
    ENCODER_FRAME,
    ENCODER_FRAME_NO_BUDGET,
)
from whetstone.experiment.candidate import (
    TemplateRenderContract,
    TemplateRenderKind,
)

_ED1_BODY_RENDER_CONTRACT = TemplateRenderContract(
    kind=TemplateRenderKind.PYTHON_FORMAT_V1,
    available_fields=(),
)


class InstructionBodyError(ValueError):
    """A mutable ED1/D1 body violated the environment-owned frame contract."""

    code = ED1_INVALID_BODY

    def __init__(self, offending: tuple[str, ...]) -> None:
        self.offending = offending
        super().__init__(
            f"{self.code}: body contains forbidden tokens {list(offending)}"
        )


def render_encoder_frame(
    body: str, *, input_code: str, max_budget: int | None
) -> str:
    """Compose the immutable encoder frame around a mutable body."""
    if max_budget is None:
        return ENCODER_FRAME_NO_BUDGET.format(body=body, input_code=input_code)
    return ENCODER_FRAME.format(
        body=body, input_code=input_code, max_budget=max_budget
    )


def instruction_body_rejection(body: str) -> tuple[str, ...]:
    """Return offending tokens for an invalid encoder body, else empty."""
    offending: list[str] = []
    seen: set[str] = set()
    try:
        placeholder_fields = _ED1_BODY_RENDER_CONTRACT.placeholder_fields(body)
    except ValueError:
        return ("{",) if "{" in body else ("}",)
    for field_name in placeholder_fields:
        token = "{" + field_name + "}"
        if token not in seen:
            seen.add(token)
            offending.append(token)
    if "```" in body and "```" not in seen:
        offending.append("```")
    return tuple(offending)


def validate_instruction_body(body: str) -> None:
    """Reject invalid body text before any provider call can be made."""
    offending = instruction_body_rejection(body)
    if offending:
        raise InstructionBodyError(offending)


__all__ = [
    "InstructionBodyError",
    "instruction_body_rejection",
    "render_encoder_frame",
    "validate_instruction_body",
]
