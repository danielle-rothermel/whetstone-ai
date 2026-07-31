"""The dr-code HumanEval scoring boundary for ed1 correctness.

This module binds ed1 to one explicit dr-code parser profile and the canonical
bounded Python execution API. It projects dr-code's typed score into the small
:class:`CodeScore` contract consumed by the ed1 evaluation drive:
``passed`` is the Binary Test Pass Score and ``infrastructure_unknown`` marks
an execution result that cannot be treated as a definitive zero.
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

    ``passed`` is the Binary Test Pass Score (the submission passes all tests).
    ``infrastructure_unknown`` is True when dr-code could not deliver a
    definitive verdict (harness failure / evaluation incomplete / timeout) --
    the
    rollout must fail, never score 0. ``outcome`` retains the dr-code label.

    ed1m (behavioral-mutant) extension: ``fidelity`` is the FRACTIONAL
    reward-bearing per-row score (fraction of inputs matching the mutant), used
    IN PLACE of the binary ``passed`` when present; ``attractor_pull`` is the
    REPORTED contamination measurement (fraction of discriminating inputs that
    snapped to canonical), never a reward. Both ``None`` for the QA/ed1 binary
    scorer, where ``passed`` is the sole score.
    """

    passed: bool
    infrastructure_unknown: bool
    outcome: str
    fidelity: float | None = None
    attractor_pull: float | None = None

    @property
    def row_value(self) -> float:
        """The per-row reward-bearing score: fractional fidelity, else 0/1."""
        return (
            self.fidelity if self.fidelity is not None else float(self.passed)
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
