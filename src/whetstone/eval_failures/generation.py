from __future__ import annotations

from whetstone.eval_failures.exceptions import EmptyGenerationError

__all__ = [
    "require_generation_text",
]


def require_generation_text(text: str | None, *, output_field: str) -> str:
    """Shared path for generation outputs before they become result fields."""
    if text is None or not text.strip():
        raise EmptyGenerationError(
            f"empty generation for output field {output_field!r}",
            metadata={"output_field": output_field},
        )
    return text
