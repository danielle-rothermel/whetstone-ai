"""Candidate Correctness: the all-candidates any-passing policy.

Whetstone owns the correctness policy; dr-code owns the typed candidates and
the compile-valid Code Artifact. This module evaluates a *set* of candidates
against a Task's full test suite and produces one of a small closed set of
**distinguishable** outcomes.

The policy has two load-bearing rules from the design invariants:

1. **All candidates, any passing.** Correctness succeeds (Binary Test Pass
   Score 1) when *any* fully evaluated Code Artifact passes the Task's full
   suite. It fails with score 0 only when *all* fully evaluated Code Artifacts
   **definitively** fail. Every candidate is examined; the first pass short of
   an infrastructure-unknown does not need to short-circuit the examination,
   but a single pass is sufficient for success.

2. **Infrastructure-unknown is never zero.** An infrastructure-unknown outcome
   (the sandbox/runner could not deliver a definitive pass/fail verdict) MUST
   fail the *rollout* rather than collapse to Binary Test Pass Score 0. A
   correctness result carrying infrastructure uncertainty has **no** score:
   the caller terminates the rollout as failed.

The following causes remain **distinguishable** (never collapsed onto one
another), matching the design's distinguishability invariant:

* ``NO_INPUT`` — no Submission Text reached the pipeline.
* ``NO_TRACE`` — a preprocessing trace was never produced.
* ``MISSING_TRACE_KEY`` — the trace exists but the candidate-set key is absent.
* ``PREPROCESSING_FAILURE`` — the native dr-code ``Absent`` role: a causal
  failure while preprocessing a *present* input.
* ``EMPTY_CANDIDATE_SET`` — preprocessing ran and zero candidates survived
  (a valid dr-code outcome, **not** a Preprocessing Failure).
* ``COMPILE_FAILURE`` — candidates exist but none compiled into a Code
  Artifact.
* ``DEFINITIVE_TEST_FAILURE`` — every compiled Code Artifact ran the suite to
  a definitive verdict and all failed.
* ``INFRASTRUCTURE_FAILURE`` — at least one candidate's verdict is
  infrastructure-unknown; the rollout must fail.

The first five reuse dr-code's :class:`~dr_code.eval.AbsenceMode` values so the
distinction is the *same* one the kernel draws; the last three are Whetstone
correctness-policy states over the compiled artifacts.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from dr_code.eval import AbsenceMode, CodeArtifact, CodeCandidateSet


class CandidateVerdict(StrEnum):
    """The verdict of running the full test suite against one Code Artifact.

    ``INFRASTRUCTURE_UNKNOWN`` is deliberately distinct from ``FAILED``: the
    runner could not obtain a definitive pass/fail (sandbox unavailable, timed
    out at the infrastructure layer, forged/unreadable output, ...). It must
    never be read as a definitive failure.
    """

    PASSED = "passed"
    FAILED = "failed"
    INFRASTRUCTURE_UNKNOWN = "infrastructure_unknown"


class CorrectnessOutcome(StrEnum):
    """The closed, distinguishable outcomes of Candidate Correctness.

    The first five mirror dr-code's five absence causes; the remaining three
    are Whetstone correctness-policy states. Every value is distinguishable —
    none is collapsed onto another.
    """

    PASSED = "passed"
    DEFINITIVE_TEST_FAILURE = "definitive_test_failure"
    COMPILE_FAILURE = "compile_failure"
    EMPTY_CANDIDATE_SET = "empty_candidate_set"
    PREPROCESSING_FAILURE = "preprocessing_failure"
    NO_INPUT = "no_input"
    NO_TRACE = "no_trace"
    MISSING_TRACE_KEY = "missing_trace_key"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"


#: Outcomes that are definitive scorable results (Binary Test Pass Score
#: is defined). Infrastructure failure is intentionally excluded: it has no
#: score and fails the rollout.
_SCORABLE = frozenset(
    {
        CorrectnessOutcome.PASSED,
        CorrectnessOutcome.DEFINITIVE_TEST_FAILURE,
        CorrectnessOutcome.COMPILE_FAILURE,
        CorrectnessOutcome.EMPTY_CANDIDATE_SET,
        CorrectnessOutcome.PREPROCESSING_FAILURE,
        CorrectnessOutcome.NO_INPUT,
        CorrectnessOutcome.NO_TRACE,
        CorrectnessOutcome.MISSING_TRACE_KEY,
    }
)

# The five dr-code absence causes crosswalk to their correctness outcomes,
# proving the distinction is the same one the kernel draws.
_ABSENCE_TO_OUTCOME: dict[AbsenceMode, CorrectnessOutcome] = {
    AbsenceMode.NO_INPUT: CorrectnessOutcome.NO_INPUT,
    AbsenceMode.NO_TRACE: CorrectnessOutcome.NO_TRACE,
    AbsenceMode.MISSING_TRACE_KEY: CorrectnessOutcome.MISSING_TRACE_KEY,
    AbsenceMode.PREPROCESSING_FAILURE: (
        CorrectnessOutcome.PREPROCESSING_FAILURE
    ),
    AbsenceMode.EMPTY_CANDIDATE_SET: CorrectnessOutcome.EMPTY_CANDIDATE_SET,
}


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    """The verdict of evaluating one compiled candidate against the suite."""

    position: int
    verdict: CandidateVerdict
    #: Free-form detail retained for provenance (e.g. failing assertion,
    #: infrastructure error class). Never interpreted by the policy.
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class CorrectnessResult:
    """The outcome of Candidate Correctness over one candidate set.

    Carries the closed :class:`CorrectnessOutcome`, the per-candidate
    evaluations that were run (empty for the absence/compile causes), and,
    for the absence causes, the dr-code :class:`AbsenceMode` it crosswalks
    from (proving the same kernel distinction).
    """

    outcome: CorrectnessOutcome
    evaluations: tuple[CandidateEvaluation, ...] = ()
    absence_mode: AbsenceMode | None = None

    def __post_init__(self) -> None:
        absence_outcomes = set(_ABSENCE_TO_OUTCOME.values())
        if self.outcome in absence_outcomes:
            if self.absence_mode is None:
                raise ValueError(
                    f"{self.outcome} must carry its dr-code absence_mode"
                )
            if _ABSENCE_TO_OUTCOME[self.absence_mode] != self.outcome:
                raise ValueError(
                    "absence_mode does not crosswalk to outcome: "
                    f"{self.absence_mode} -> {self.outcome}"
                )
        elif self.absence_mode is not None:
            raise ValueError(f"{self.outcome} must not carry an absence_mode")

    @property
    def is_scorable(self) -> bool:
        """Whether a Binary Test Pass Score is defined for this outcome.

        ``False`` exactly for ``INFRASTRUCTURE_FAILURE`` — the rollout fails
        and no correctness score is produced.
        """

        return self.outcome in _SCORABLE

    @property
    def is_infrastructure_failure(self) -> bool:
        return self.outcome is CorrectnessOutcome.INFRASTRUCTURE_FAILURE

    @property
    def passed(self) -> bool:
        return self.outcome is CorrectnessOutcome.PASSED


# A runner maps one compiled Code Artifact to its suite verdict. It is
# injected so this module stays sandbox-agnostic; the real runner is the
# semantic sandbox. It may return INFRASTRUCTURE_UNKNOWN — that is expected
# domain output, not an exception.
CandidateRunner = Callable[[CodeArtifact], CandidateVerdict]


def evaluate_candidate_correctness(
    candidate_set: CodeCandidateSet,
    *,
    runner: CandidateRunner,
) -> CorrectnessResult:
    """Apply the all-candidates any-passing policy over a candidate set.

    ``candidate_set`` is the typed dr-code :class:`CodeCandidateSet` emitted by
    preprocessing (a **present, non-absent** outcome; the absence causes are
    surfaced by :func:`correctness_absent` before this point). Each candidate
    is compiled; non-compiling candidates are recorded as compile misses.
    Compiled candidates are run through ``runner``.

    Policy:

    * If **any** compiled candidate's verdict is ``PASSED`` -> ``PASSED``.
    * Else if **any** compiled candidate's verdict is
      ``INFRASTRUCTURE_UNKNOWN`` -> ``INFRASTRUCTURE_FAILURE`` (the rollout
      fails; never score 0). Infrastructure uncertainty over-rides a
      definitive failure so an unknown is never silently read as a fail.
    * Else if there was at least one compiled candidate (all definitively
      ``FAILED``) -> ``DEFINITIVE_TEST_FAILURE``.
    * Else (candidates existed but none compiled) -> ``COMPILE_FAILURE``.
    * The empty-candidate-set case is a distinct absence outcome and is
      surfaced here only when ``candidate_set`` is empty.

    Every candidate is examined; the policy never stops at the first failure.
    """

    if candidate_set.is_empty:
        return correctness_absent(AbsenceMode.EMPTY_CANDIDATE_SET)

    evaluations: list[CandidateEvaluation] = []
    saw_pass = False
    saw_unknown = False
    saw_compiled = False

    for candidate in candidate_set.candidates:
        artifact = CodeArtifact.try_from_candidate(candidate)
        if artifact is None:
            # Compile miss: recorded so it is examined, but a compile miss is
            # not a test verdict. It only matters if *no* candidate compiles.
            continue
        saw_compiled = True
        verdict = runner(artifact)
        evaluations.append(
            CandidateEvaluation(position=candidate.position, verdict=verdict)
        )
        if verdict is CandidateVerdict.PASSED:
            saw_pass = True
        elif verdict is CandidateVerdict.INFRASTRUCTURE_UNKNOWN:
            saw_unknown = True

    evaluated = tuple(evaluations)

    if saw_pass:
        return CorrectnessResult(
            outcome=CorrectnessOutcome.PASSED, evaluations=evaluated
        )
    if saw_unknown:
        return CorrectnessResult(
            outcome=CorrectnessOutcome.INFRASTRUCTURE_FAILURE,
            evaluations=evaluated,
        )
    if saw_compiled:
        return CorrectnessResult(
            outcome=CorrectnessOutcome.DEFINITIVE_TEST_FAILURE,
            evaluations=evaluated,
        )
    return CorrectnessResult(
        outcome=CorrectnessOutcome.COMPILE_FAILURE, evaluations=evaluated
    )


def correctness_absent(absence_mode: AbsenceMode) -> CorrectnessResult:
    """Build a Candidate Correctness result for a dr-code absence cause.

    ``absence_mode`` is the exact dr-code :class:`AbsenceMode`; the outcome
    crosswalks to the matching distinguishable :class:`CorrectnessOutcome`.
    Use this for NO_INPUT, NO_TRACE, MISSING_TRACE_KEY, PREPROCESSING_FAILURE,
    and EMPTY_CANDIDATE_SET (the last is also produced automatically by
    :func:`evaluate_candidate_correctness` for an empty set).
    """

    return CorrectnessResult(
        outcome=_ABSENCE_TO_OUTCOME[absence_mode],
        absence_mode=absence_mode,
    )


__all__ = [
    "CandidateEvaluation",
    "CandidateRunner",
    "CandidateVerdict",
    "CorrectnessOutcome",
    "CorrectnessResult",
    "correctness_absent",
    "evaluate_candidate_correctness",
]
