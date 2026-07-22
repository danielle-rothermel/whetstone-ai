"""Candidate Correctness: all-candidates any-passing + distinguishability.

Proves:

* The all-candidates any-passing policy (any pass => PASSED; all definitive
  fails => DEFINITIVE_TEST_FAILURE).
* Infrastructure-unknown fails the *rollout* (INFRASTRUCTURE_FAILURE, not
  scorable) rather than collapsing to Binary Test Pass Score 0, and over-rides
  a definitive failure.
* All eight distinguishable causes remain distinct: no-input, no-trace,
  missing-trace-key, Preprocessing Failure, empty candidate set, compile
  failure, definitive test failure, infrastructure failure.
"""

from __future__ import annotations

import pytest
from dr_code.eval import AbsenceMode, CodeCandidateSet

from whetstone.code_eval import (
    CandidateVerdict,
    CorrectnessOutcome,
    correctness_absent,
    evaluate_candidate_correctness,
)

GOOD = "def f():\n    return 1\n"
ALSO_GOOD = "def g():\n    return 2\n"
BAD_COMPILE = "def f(:\n    return\n"


def _runner(verdict: CandidateVerdict):
    return lambda _artifact: verdict


class _PositionRunner:
    """A runner returning a verdict per invocation order, recording calls."""

    def __init__(self, mapping: dict[int, CandidateVerdict]) -> None:
        self._mapping = mapping
        self.calls: list[int] = []

    def __call__(self, _artifact: object) -> CandidateVerdict:
        pos = len(self.calls)
        self.calls.append(pos)
        return self._mapping[pos]


def _runner_by_position(
    mapping: dict[int, CandidateVerdict],
) -> _PositionRunner:
    return _PositionRunner(mapping)


def test_any_passing_succeeds() -> None:
    cs = CodeCandidateSet.from_sources((GOOD, ALSO_GOOD), origin="t")
    # Second candidate passes; first fails. Any-passing => PASSED.
    run = _runner_by_position(
        {0: CandidateVerdict.FAILED, 1: CandidateVerdict.PASSED}
    )
    result = evaluate_candidate_correctness(cs, runner=run)
    assert result.outcome is CorrectnessOutcome.PASSED
    assert result.passed
    assert result.is_scorable
    # Every candidate examined (both ran).
    assert len(run.calls) == 2


def test_all_definitive_fail_is_definitive_test_failure() -> None:
    cs = CodeCandidateSet.from_sources((GOOD, ALSO_GOOD), origin="t")
    result = evaluate_candidate_correctness(
        cs, runner=_runner(CandidateVerdict.FAILED)
    )
    assert result.outcome is CorrectnessOutcome.DEFINITIVE_TEST_FAILURE
    assert result.is_scorable
    assert result.absence_mode is None


def test_infrastructure_unknown_fails_rollout_not_score_zero() -> None:
    cs = CodeCandidateSet.from_sources((GOOD, ALSO_GOOD), origin="t")
    # A definitive fail AND an infrastructure-unknown: the unknown wins, so
    # the outcome is not a definitive failure (which would score 0).
    run = _runner_by_position(
        {
            0: CandidateVerdict.FAILED,
            1: CandidateVerdict.INFRASTRUCTURE_UNKNOWN,
        }
    )
    result = evaluate_candidate_correctness(cs, runner=run)
    assert result.outcome is CorrectnessOutcome.INFRASTRUCTURE_FAILURE
    assert result.is_infrastructure_failure
    # Not scorable: the rollout fails, no Binary Test Pass Score.
    assert not result.is_scorable
    assert result.outcome is not CorrectnessOutcome.DEFINITIVE_TEST_FAILURE


def test_pass_beats_infrastructure_unknown() -> None:
    cs = CodeCandidateSet.from_sources((GOOD, ALSO_GOOD), origin="t")
    run = _runner_by_position(
        {
            0: CandidateVerdict.INFRASTRUCTURE_UNKNOWN,
            1: CandidateVerdict.PASSED,
        }
    )
    result = evaluate_candidate_correctness(cs, runner=run)
    # A real pass exists => PASSED, even though another candidate was unknown.
    assert result.outcome is CorrectnessOutcome.PASSED


def test_compile_failure_when_no_candidate_compiles() -> None:
    cs = CodeCandidateSet.from_sources((BAD_COMPILE, BAD_COMPILE), origin="t")
    # Runner should never be called (nothing compiles).
    def run(_artifact):
        raise AssertionError("runner must not run on non-compiling candidates")

    result = evaluate_candidate_correctness(cs, runner=run)
    assert result.outcome is CorrectnessOutcome.COMPILE_FAILURE
    assert result.is_scorable


def test_compile_miss_does_not_hide_a_passing_candidate() -> None:
    cs = CodeCandidateSet.from_sources((BAD_COMPILE, GOOD), origin="t")
    # First candidate does not compile (skipped, runner not called for it);
    # second compiles and passes.
    result = evaluate_candidate_correctness(
        cs, runner=_runner(CandidateVerdict.PASSED)
    )
    assert result.outcome is CorrectnessOutcome.PASSED
    assert len(result.evaluations) == 1


def test_empty_candidate_set_is_distinct() -> None:
    result = evaluate_candidate_correctness(
        CodeCandidateSet(), runner=_runner(CandidateVerdict.PASSED)
    )
    assert result.outcome is CorrectnessOutcome.EMPTY_CANDIDATE_SET
    assert result.absence_mode is AbsenceMode.EMPTY_CANDIDATE_SET
    assert result.is_scorable


@pytest.mark.parametrize(
    ("absence_mode", "outcome"),
    [
        (AbsenceMode.NO_INPUT, CorrectnessOutcome.NO_INPUT),
        (AbsenceMode.NO_TRACE, CorrectnessOutcome.NO_TRACE),
        (
            AbsenceMode.MISSING_TRACE_KEY,
            CorrectnessOutcome.MISSING_TRACE_KEY,
        ),
        (
            AbsenceMode.PREPROCESSING_FAILURE,
            CorrectnessOutcome.PREPROCESSING_FAILURE,
        ),
        (
            AbsenceMode.EMPTY_CANDIDATE_SET,
            CorrectnessOutcome.EMPTY_CANDIDATE_SET,
        ),
    ],
)
def test_absence_causes_crosswalk_from_dr_code(
    absence_mode: AbsenceMode, outcome: CorrectnessOutcome
) -> None:
    result = correctness_absent(absence_mode)
    assert result.outcome is outcome
    assert result.absence_mode is absence_mode
    assert result.is_scorable


def test_all_eight_causes_are_mutually_distinct() -> None:
    # The eight distinguishable causes the design requires: no two are equal.
    outcomes = {
        evaluate_candidate_correctness(
            CodeCandidateSet(), runner=_runner(CandidateVerdict.PASSED)
        ).outcome,  # empty candidate set
        correctness_absent(AbsenceMode.NO_INPUT).outcome,
        correctness_absent(AbsenceMode.NO_TRACE).outcome,
        correctness_absent(AbsenceMode.MISSING_TRACE_KEY).outcome,
        correctness_absent(AbsenceMode.PREPROCESSING_FAILURE).outcome,
        evaluate_candidate_correctness(
            CodeCandidateSet.from_sources((BAD_COMPILE,), origin="t"),
            runner=_runner(CandidateVerdict.PASSED),
        ).outcome,  # compile failure
        evaluate_candidate_correctness(
            CodeCandidateSet.from_sources((GOOD,), origin="t"),
            runner=_runner(CandidateVerdict.FAILED),
        ).outcome,  # definitive test failure
        evaluate_candidate_correctness(
            CodeCandidateSet.from_sources((GOOD,), origin="t"),
            runner=_runner(CandidateVerdict.INFRASTRUCTURE_UNKNOWN),
        ).outcome,  # infrastructure failure
    }
    assert len(outcomes) == 8


def test_absence_mode_must_crosswalk() -> None:
    from whetstone.code_eval.correctness import CorrectnessResult

    # A mismatched absence_mode / outcome pair is rejected.
    with pytest.raises(ValueError):
        CorrectnessResult(
            outcome=CorrectnessOutcome.NO_INPUT,
            absence_mode=AbsenceMode.NO_TRACE,
        )
    # A definitive outcome carrying an absence_mode is rejected.
    with pytest.raises(ValueError):
        CorrectnessResult(
            outcome=CorrectnessOutcome.PASSED,
            absence_mode=AbsenceMode.NO_INPUT,
        )
    # An absence outcome without its mode is rejected.
    with pytest.raises(ValueError):
        CorrectnessResult(outcome=CorrectnessOutcome.NO_INPUT)
