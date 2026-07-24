"""Character Budget graph/runtime binding.

The Character Budget is a character-count bound applied during a Rollout.
Its binding is deliberately split so that identity stays clean:

* The output-affecting *derivation rule or ratio* is a Graph Definition
  Variable. Its assignment belongs to Graph Config identity, so changing it
  changes ``graph_hash``. It is carried as an LLM Call Node static Variable
  (see ``nodes.CHARACTER_BUDGET_VARIABLE``).
* The *concrete Task-derived bound* (an integer character count computed
  from a Task at runtime) is a Graph External Input, supplied through the
  ``task.<field>`` namespace and excluded from Graph Config / Rollout
  Variant identity.

Whetstone owns this experiment binding directly. There is deliberately no
separate character-budget policy artifact — no dedicated type, schema,
config, or identity. Both forms also stay separate from the compression byte
denominator, which is a dr-code concern.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, StrictStr, model_validator

# The Graph External Input field through which a concrete Task-derived
# character budget bound is supplied at runtime. It is NOT in identity.
CHARACTER_BUDGET_EXTERNAL_INPUT = "task.character_budget"


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
        if not (self.ratio > 0):
            raise ValueError("character budget ratio must be positive")
        return self

    def identity_value(self) -> dict[str, object]:
        """The JSON-safe identity-bearing form assigned as a Node Variable."""
        return {"kind": self.kind, "ratio": self.ratio}


def derive_character_bound(
    rule: CharacterBudgetRule, *, task_length: int
) -> int:
    """Derive the concrete character-count bound from a Task-provided length.

    The result is a Graph External Input value (runtime), never entered into
    Graph Config identity.
    """
    if task_length < 0:
        raise ValueError("task_length must be non-negative")
    return int(rule.ratio * task_length)


__all__ = [
    "CHARACTER_BUDGET_EXTERNAL_INPUT",
    "CharacterBudgetRule",
    "derive_character_bound",
]
