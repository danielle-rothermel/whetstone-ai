"""The dr-code HumanEval scoring boundary for ed1 correctness.

This module binds ed1 to one explicit dr-code parser profile and the canonical
bounded Python execution API. It projects dr-code's typed score into the small
:class:`CodeScore` contract consumed by the ed1 evaluation drive:
``passed`` is the typed boolean projection of dr-code's SubmissionOutcome and
``infrastructure_unknown`` marks an execution result that cannot be treated as
a definitive zero.
"""

from __future__ import annotations

from dataclasses import dataclass

from dr_code.execution import PythonSubprocessRunner, run_python_subprocess
from dr_code.humaneval import (
    BEST_EFFORT_HUMANEVAL_PARSER_PROFILE,
    DEFAULT_HUMANEVAL_TIMEOUT_SECONDS,
    CodeParserProfile,
    HarnessFailure,
    HumanEvalTask,
    SubmissionOutcome,
    score_humaneval_submission,
)

#: The parser contract for decoder submissions. The profile id and version are
#: also folded into the ed1 evaluation procedure identity.
ED1_PARSER_PROFILE = BEST_EFFORT_HUMANEVAL_PARSER_PROFILE

#: The outcomes that are INFRASTRUCTURE-UNKNOWN (no definitive pass/fail): the
#: rollout fails, never scores 0. A :class:`HarnessFailure` is handled
#: separately because it is not a completed submission outcome.
_INFRASTRUCTURE_UNKNOWN_OUTCOMES = frozenset(
    {
        SubmissionOutcome.EVALUATION_INCOMPLETE,
        SubmissionOutcome.TIMED_OUT,
    }
)


@dataclass(frozen=True, slots=True)
class CodeScore:
    """The ed1 correctness outcome for one decoder submission.

    ``passed`` is the natural typed boolean projection of dr-code's
    :class:`SubmissionOutcome`: it is true exactly for ``PASSED``.
    ``infrastructure_unknown`` is true when dr-code could not deliver a
    definitive verdict (harness failure / evaluation incomplete / timeout);
    the rollout must fail, never score 0. ``outcome`` retains the dr-code
    label.

    ED1M extends the projection with ``fidelity_to_mutant``, the fractional
    reward-bearing row value (fraction of inputs matching the mutant).
    ``attractor_pull`` is the reported contamination measurement (fraction of
    discriminating inputs that snapped to canonical), never a reward. Both are
    ``None`` for ED1/D1, whose HumanEval Submission Score is ``float(passed)``.
    """

    passed: bool
    infrastructure_unknown: bool
    outcome: str
    fidelity_to_mutant: float | None = None
    attractor_pull: float | None = None

    @property
    def row_value(self) -> float:
        """The environment's primary row metric value.

        ED1M returns fractional ``fidelity_to_mutant``. ED1/D1 return the
        HumanEval Submission Score, the 0.0/1.0 numeric projection of
        ``passed``.
        """
        return (
            self.fidelity_to_mutant
            if self.fidelity_to_mutant is not None
            else float(self.passed)
        )


def score_ed1_submission(
    *,
    raw_submission: str,
    task: HumanEvalTask,
    parser_profile: CodeParserProfile = ED1_PARSER_PROFILE,
    run_in_subprocess: PythonSubprocessRunner = run_python_subprocess,
    timeout_seconds: float = DEFAULT_HUMANEVAL_TIMEOUT_SECONDS,
) -> CodeScore:
    """Score one decoder submission -> :class:`CodeScore`.

    Delegates to dr-code's ``score_humaneval_submission`` (preprocessing +
    subprocess test run) and projects its typed outcome onto the ed1
    correctness
    invariant: ``PASSED`` -> passed; a typed harness failure,
    ``EVALUATION_INCOMPLETE``, or ``TIMED_OUT`` -> infrastructure unknown (the
    rollout fails); every other completed outcome (tests failed, extraction
    failed, no top-level functions, ...) -> definitive fail (score 0).
    """
    result = score_humaneval_submission(
        raw_submission=raw_submission,
        task=task,
        parser_profile=parser_profile,
        timeout_seconds=timeout_seconds,
        run_in_subprocess=run_in_subprocess,
    )
    if isinstance(result, HarnessFailure):
        return CodeScore(
            passed=False,
            infrastructure_unknown=True,
            outcome=result.kind,
        )
    outcome = result.outcome
    if outcome in _INFRASTRUCTURE_UNKNOWN_OUTCOMES:
        return CodeScore(
            passed=False,
            infrastructure_unknown=True,
            outcome=str(outcome),
        )
    return CodeScore(
        passed=outcome is SubmissionOutcome.PASSED,
        infrastructure_unknown=False,
        outcome=str(outcome),
    )


__all__ = [
    "ED1_PARSER_PROFILE",
    "CodeScore",
    "score_ed1_submission",
]
