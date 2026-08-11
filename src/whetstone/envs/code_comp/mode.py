from __future__ import annotations

from enum import UNIQUE, StrEnum, verify


@verify(UNIQUE)
class CodeCompMode(StrEnum):
    """HumanEval code-compression experiment modes."""

    DIRECT = "direct"
    ENCDEC = "encdec"
    ENCDEC_MUTANT = "encdec_mutant"


def code_comp_identity_prefix(mode: CodeCompMode) -> str:
    """Return the identity namespace prefix for one code_comp mode."""
    return f"whetstone.code_comp.{mode.value}"


__all__ = ["CodeCompMode", "code_comp_identity_prefix"]
