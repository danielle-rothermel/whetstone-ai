from __future__ import annotations


class CandidateEvaluationFailure(RuntimeError):
    """An internal-path candidate could not be scored into a Reward.

    Raised when an internal aggregate is missing/incomplete under a FAIL
    missing-data policy. Optimizers handle this as a typed candidate failure.
    """


__all__ = [
    "CandidateEvaluationFailure",
]
