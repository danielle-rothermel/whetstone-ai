from enum import UNIQUE, StrEnum, verify


@verify(UNIQUE)
class EvalRole(StrEnum):
    INTERNAL = "internal"
    OFFICIAL = "official"


__all__ = ["EvalRole"]
