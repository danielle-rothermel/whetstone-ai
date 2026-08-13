from enum import UNIQUE, StrEnum, verify


@verify(UNIQUE)
class EvaluationRole(StrEnum):
    INTERNAL = "internal"
    OFFICIAL = "official"


__all__ = ["EvaluationRole"]
