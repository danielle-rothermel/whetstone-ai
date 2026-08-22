from enum import UNIQUE, StrEnum, verify


@verify(UNIQUE)
class EvalRole(StrEnum):
    """The evidence role a stored evaluation belongs to.

    These values are persisted inside ``EvalEvidence`` (schema
    ``whetstone.eval_evidence``), so the spellings are a wire contract and
    are pinned by a golden literal test.
    """

    INTERNAL = "internal"
    OFFICIAL = "official"
    HELD_OUT = "held_out"


__all__ = ["EvalRole"]
