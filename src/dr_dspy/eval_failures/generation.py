from __future__ import annotations

from dr_dspy.eval_failures.exceptions import EmptyGenerationError

__all__ = [
    "require_generation_text",
    "validate_direct_generation",
    "validate_encdec_generation",
]


def require_generation_text(text: str | None, *, output_field: str) -> str:
    """Shared path for generation outputs before they become result fields."""
    if text is None or not text.strip():
        raise EmptyGenerationError(
            f"empty generation for output field {output_field!r}",
            metadata={"output_field": output_field},
        )
    return text


def validate_encdec_generation(*, description: str, code: str) -> None:
    require_generation_text(description, output_field="description")
    require_generation_text(code, output_field="code")


def validate_direct_generation(*, code: str) -> None:
    require_generation_text(code, output_field="code")
