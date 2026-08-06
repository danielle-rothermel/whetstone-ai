from enum import UNIQUE, StrEnum, verify


@verify(UNIQUE)
class EvaluationRole(StrEnum):
    """Closed role of an evaluation.

    Persisted payloads must spell role values explicitly; never derive a
    payload shape by iterating over this enum.
    """

    INTERNAL = "internal"
    OFFICIAL = "official"


__all__ = ["EvaluationRole"]
