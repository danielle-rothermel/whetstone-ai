from __future__ import annotations

from typing import Protocol, runtime_checkable

from whetstone.core.identity import compute_identity_hash

DEFAULT_SUBMISSION_RESULT_FIELD = "submission_result"
DEFAULT_GEPA_SUBMISSION_PROJECTOR_IDENTITY_HASH = compute_identity_hash(
    schema="whetstone.gepa.submission_projector",
    schema_version=1,
    payload={
        "projector": "default",
        "submission_result_field": DEFAULT_SUBMISSION_RESULT_FIELD,
    },
)


@runtime_checkable
class GepaSubmissionProjector(Protocol):
    def submission_result_field(self) -> str: ...

    def prediction_failed(
        self, *, row_failed: bool, submission: object
    ) -> bool: ...

    def feedback_text(self, *, score: float, submission: object) -> str: ...

    def test_results(self, *, submission: object) -> dict[str, object] | None: ...

    def projector_identity_hash(self) -> str: ...


def _parse_score_record(submission: object) -> dict[str, object] | None:
    if not isinstance(submission, dict):
        return None
    score = submission.get("score")
    if not isinstance(score, dict):
        return None
    return score


class DefaultGepaSubmissionProjector:
    def __init__(
        self,
        *,
        submission_result_field: str = DEFAULT_SUBMISSION_RESULT_FIELD,
    ) -> None:
        if not submission_result_field:
            raise ValueError("submission_result_field must be non-empty")
        self._submission_result_field = submission_result_field

    def submission_result_field(self) -> str:
        return self._submission_result_field

    def projector_identity_hash(self) -> str:
        if self._submission_result_field == DEFAULT_SUBMISSION_RESULT_FIELD:
            return DEFAULT_GEPA_SUBMISSION_PROJECTOR_IDENTITY_HASH
        return compute_identity_hash(
            schema="whetstone.gepa.submission_projector",
            schema_version=1,
            payload={
                "projector": "default",
                "submission_result_field": self._submission_result_field,
            },
        )

    def prediction_failed(
        self, *, row_failed: bool, submission: object
    ) -> bool:
        """Whether reflection should treat this prediction as a failure.

        ``row_failed`` is the caller's canonical row verdict -- a row is
        failed exactly when it carries no score. It deliberately replaces the
        old ``failure_code`` test: a scored row may carry a code as
        explanation without having failed, and a blank generation is exactly
        that. A blank output must reach reflection as a completed, failing
        prediction, because the blank *is* the signal reflection should learn
        from.
        """
        if row_failed:
            return True
        score = _parse_score_record(submission)
        if score is None:
            return False
        if (
            score.get("passed") is False
            and score.get("infrastructure_unknown") is not True
        ):
            return True
        return False

    def feedback_text(self, *, score: float, submission: object) -> str:
        base = f"This trajectory got a score of {score}."
        if not isinstance(submission, dict):
            return base
        score_record = _parse_score_record(submission)
        if score_record is not None and score_record.get("passed") is True:
            return base
        outcome = submission.get("outcome")
        if not isinstance(outcome, str) or not outcome:
            if score_record is not None:
                nested = score_record.get("outcome")
                outcome = nested if isinstance(nested, str) else None
        if isinstance(outcome, str) and outcome:
            return f"{base} Outcome: {outcome}."
        return base

    def test_results(self, *, submission: object) -> dict[str, object] | None:
        if not isinstance(submission, dict):
            return None
        test_results = submission.get("test_results")
        if test_results is not None:
            return {"test_results": test_results}
        return None


__all__ = [
    "DEFAULT_GEPA_SUBMISSION_PROJECTOR_IDENTITY_HASH",
    "DEFAULT_SUBMISSION_RESULT_FIELD",
    "DefaultGepaSubmissionProjector",
    "GepaSubmissionProjector",
]
