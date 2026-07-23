"""The narrow dr-code HumanEval scoring seam for ed1 correctness.

This is the ONLY module that touches dr-code's HumanEval *execution/scoring*
surface, so the flip to ``impl/02-preprocessing-integration`` has one place to
re-check. It wraps dr-code's high-level
:func:`dr_code.humaneval.score_humaneval_submission` (preprocessing extracts
code from the raw decoder submission, then the task's test suite runs in the
injectable subprocess runner) into a tiny :class:`CodeScore` the ed1 drive
reads: ``passed`` (Binary Test Pass Score 1/0) and ``infrastructure_unknown``
(the harness could not deliver a definitive verdict -> the rollout fails, never
scores 0, per the design invariant).

dr-code APIs called (the flip-risk surface -- confirmed against the INSTALLED
``impl/01-eval-kernel`` build, whose signature differs from older docs):

* ``dr_code.humaneval.score_humaneval_submission(*, raw_submission, task,
  timeout_seconds, run_in_subprocess=...)`` -- raw submission -> typed score.
  The default ``preprocessing_runner`` + ``run_in_subprocess``
  (``run_python_subprocess``) run LOCALLY (a plain subprocess, no container),
  so the pilot + tests need no Docker.
* ``dr_code.humaneval.HumanEvalTask`` (the test-suite-bearing task)
* ``dr_code.humaneval.SubmissionOutcome`` / ``CompletedScore``
* ``dr_code.humaneval.DEFAULT_HUMANEVAL_TIMEOUT_SECONDS``
* ``dr_code.humaneval.subprocess_runner.SubprocessRunner`` /
  ``SubprocessCompletedProcess`` / ``run_python_subprocess`` (the local
  runner).
"""

from __future__ import annotations

from dataclasses import dataclass

from dr_code.humaneval import (
    DEFAULT_HUMANEVAL_TIMEOUT_SECONDS,
    CompletedScore,
    HumanEvalTask,
    SubmissionOutcome,
    score_humaneval_submission,
)
from dr_code.humaneval.subprocess_runner import (
    SubprocessRunner,
    run_python_subprocess,
)

#: The outcomes that are INFRASTRUCTURE-UNKNOWN (no definitive pass/fail): the
#: rollout fails, never scores 0. Everything else is a definitive scorable
#: outcome (``passed`` -> 1; any other definitive outcome -> 0).
_INFRASTRUCTURE_UNKNOWN_OUTCOMES = frozenset(
    {
        SubmissionOutcome.HARNESS_FAILURE,
        SubmissionOutcome.EVALUATION_INCOMPLETE,
        SubmissionOutcome.TIMED_OUT,
    }
)


@dataclass(frozen=True, slots=True)
class CodeScore:
    """The ed1 correctness outcome for one decoder submission.

    ``passed`` is the Binary Test Pass Score (the submission passes all tests).
    ``infrastructure_unknown`` is True when dr-code could not deliver a
    definitive verdict (harness failure / evaluation incomplete / timeout) --
    the
    rollout must fail, never score 0. ``outcome`` retains the dr-code label.
    """

    passed: bool
    infrastructure_unknown: bool
    outcome: str


def score_ed1_submission(
    *,
    raw_submission: str,
    task: HumanEvalTask,
    run_in_subprocess: SubprocessRunner = run_python_subprocess,
    timeout_seconds: float = DEFAULT_HUMANEVAL_TIMEOUT_SECONDS,
) -> CodeScore:
    """Score one decoder submission -> :class:`CodeScore`.

    Delegates to dr-code's ``score_humaneval_submission`` (preprocessing +
    subprocess test run) and projects its typed outcome onto the ed1
    correctness
    invariant: ``PASSED`` -> passed; a HARNESS_FAILURE / EVALUATION_INCOMPLETE
    /
    TIMED_OUT -> infrastructure unknown (the rollout fails); every other
    definitive outcome (tests failed, no candidates, ...) -> definitive fail
    (score 0). The subprocess runner is injectable; its default runs locally
    (no container), so tests + the pilot need no Docker.
    """
    result = score_humaneval_submission(
        raw_submission=raw_submission,
        task=task,
        timeout_seconds=timeout_seconds,
        run_in_subprocess=run_in_subprocess,
    )
    outcome = result.outcome if isinstance(result, CompletedScore) else None
    if outcome is None or outcome in _INFRASTRUCTURE_UNKNOWN_OUTCOMES:
        return CodeScore(
            passed=False,
            infrastructure_unknown=True,
            outcome=str(outcome) if outcome is not None else "no_score",
        )
    return CodeScore(
        passed=outcome is SubmissionOutcome.PASSED,
        infrastructure_unknown=False,
        outcome=str(outcome),
    )


__all__ = [
    "CodeScore",
    "score_ed1_submission",
]
