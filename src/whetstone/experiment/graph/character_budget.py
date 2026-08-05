"""Character Budget graph/runtime binding.

The Character Budget is a character-count bound applied during a Rollout.
Its binding is deliberately split so that identity stays clean:

* The output-affecting *derivation rule or ratio* is a Graph Definition
  Variable. Its assignment belongs to Graph Config identity, so changing it
  changes ``graph_hash``. It is carried as an LLM Call Node static Variable
  (see ``nodes.CHARACTER_BUDGET_VARIABLE``).
* The *concrete Task-derived bound* (an integer character count computed
  from a Task at runtime) is used by the environment when rendering the
  encoder prompt. The rendered prompt is the Graph External Input; the bound
  itself is not part of Graph Config / Rollout Variant identity.

Whetstone owns this experiment binding directly. There is deliberately no
separate character-budget policy artifact — no dedicated type, schema,
config, or identity. Both forms also stay separate from the compression byte
denominator, which is a dr-code concern.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, StrictStr, model_validator


class CharacterBudgetRule(BaseModel):
    """The output-affecting derivation rule / ratio for the Character Budget.

    This is a plain identity-bearing value assigned as a Graph Definition
    Variable (an LLM Call Node static Variable); it is NOT a standalone
    policy artifact. ``ratio`` derives the bound from a Task-provided base
    length; ``kind`` distinguishes derivation strategies if more are added.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: StrictStr = "ratio_of_task_length"
    ratio: float

    @model_validator(mode="after")
    def _validate(self) -> CharacterBudgetRule:
        if not self.kind:
            raise ValueError("character budget rule kind must be non-empty")
        if not math.isfinite(self.ratio) or self.ratio <= 0:
            raise ValueError(
                "character budget ratio must be finite and positive"
            )
        return self

    def identity_value(self) -> dict[str, object]:
        """The JSON-safe identity-bearing form assigned as a Node Variable."""
        return {"kind": self.kind, "ratio": self.ratio}


def derive_character_bound(
    rule: CharacterBudgetRule, *, task_length: int
) -> int:
    """Derive the concrete character-count bound from a Task-provided length.

    The environment uses the result to render the encoder prompt at runtime;
    the result is never entered into Graph Config identity.
    """
    if task_length < 0:
        raise ValueError("task_length must be non-negative")
    try:
        scaled = rule.ratio * task_length
    except OverflowError as error:
        raise ValueError(
            "derived character bound exceeds the supported finite range"
        ) from error
    if not math.isfinite(scaled):
        raise ValueError(
            "derived character bound exceeds the supported finite range"
        )
    return round(scaled)


__all__ = [
    "CharacterBudgetRule",
    "derive_character_bound",
]
