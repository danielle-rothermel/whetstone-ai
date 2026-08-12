from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

from dr_code.humaneval import (
    HarnessFailure,
    HumanEvalSubmissionScore,
    SubmissionOutcome,
)
from dr_code.humaneval.task import EvaluationCaseResult

from whetstone.envs.code_comp.mutant.dataset import (
    ExpectedOutcome,
    MutantRecord,
)
from whetstone.envs.code_comp.mutant.oracle import (
    MutantInputOutcome,
    MutantScore,
)


@dataclass(frozen=True, slots=True)
class CodeScore:
    """The reward-facing correctness scalar for one code submission.

    ``passed`` is true exactly for a definitive HumanEval pass or full mutant
    fidelity. ``infrastructure_unknown`` is true for harness/oracle failures.
    ED1M may carry fractional ``fidelity_to_mutant`` and reported
    ``attractor_pull``.
    """

    passed: bool
    infrastructure_unknown: bool
    outcome: str
    fidelity_to_mutant: float | None = None
    attractor_pull: float | None = None

    @property
    def row_value(self) -> float:
        return (
            self.fidelity_to_mutant
            if self.fidelity_to_mutant is not None
            else float(self.passed)
        )


@dataclass(frozen=True, slots=True)
class CodeCaseResult:
    """One HumanEval test-case outcome projected for Whetstone consumers."""

    case_id: str
    status: str
    message: str
    input_repr: str
    expected_output_repr: str
    actual_output_repr: str


@dataclass(frozen=True, slots=True)
class CodeSubmissionResult:
    """Shared submission-result envelope for code_comp scoring modes."""

    score: CodeScore
    outcome: str
    function_names: tuple[str, ...]
    best_function_name: str | None
    total_cases: int


@dataclass(frozen=True, slots=True)
class HumanEvalSubmissionResult(CodeSubmissionResult):
    """HumanEval suite scoring with per-case attribution."""

    cases: tuple[CodeCaseResult, ...] = ()


@dataclass(frozen=True, slots=True)
class MutantInputResult:
    """One mutant-oracle input outcome."""

    input_index: int
    input_repr: str
    expected_mutant_repr: str
    expected_canonical_repr: str
    actual_kind: str
    actual_output_repr: str
    matched_mutant: bool
    matched_canonical: bool
    is_distinct: bool


@dataclass(frozen=True, slots=True)
class MutantSubmissionResult(CodeSubmissionResult):
    """Mutant-oracle scoring with per-input attribution."""

    input_results: tuple[MutantInputResult, ...] = ()


def _case_from_dr_code(result: EvaluationCaseResult) -> CodeCaseResult:
    return CodeCaseResult(
        case_id=result.case_id,
        status=result.status.value,
        message=result.message,
        input_repr=result.input_repr,
        expected_output_repr=result.expected_output_repr,
        actual_output_repr=result.actual_output_repr,
    )


def _score_from_outcome(
    outcome: SubmissionOutcome | str,
    *,
    infrastructure_unknown: bool = False,
    fidelity_to_mutant: float | None = None,
    attractor_pull: float | None = None,
) -> CodeScore:
    if isinstance(outcome, SubmissionOutcome):
        label = outcome.value
        passed = outcome is SubmissionOutcome.PASSED
    else:
        label = outcome
        passed = False
    return CodeScore(
        passed=passed,
        infrastructure_unknown=infrastructure_unknown,
        outcome=label,
        fidelity_to_mutant=fidelity_to_mutant,
        attractor_pull=attractor_pull,
    )


def project_humaneval_submission_result(
    result: HumanEvalSubmissionScore,
) -> HumanEvalSubmissionResult:
    """Project one dr-code submission score into Whetstone-owned results."""

    if isinstance(result, HarnessFailure) or getattr(result, "kind", None) == (
        "harness_failure"
    ):
        evaluation = getattr(result, "evaluation", None)
        return HumanEvalSubmissionResult(
            score=_score_from_outcome(
                getattr(result, "kind", "harness_failure"),
                infrastructure_unknown=True,
            ),
            outcome=getattr(result, "kind", "harness_failure"),
            function_names=tuple(evaluation.function_names)
            if evaluation is not None
            else (),
            best_function_name=(
                evaluation.best_function_name
                if evaluation is not None
                else None
            ),
            total_cases=evaluation.total_cases
            if evaluation is not None
            else 0,
            cases=tuple(
                _case_from_dr_code(case)
                for case in (
                    evaluation.results if evaluation is not None else []
                )
            ),
        )

    outcome = result.outcome
    evaluation = getattr(result, "evaluation", None)
    cases: tuple[CodeCaseResult, ...] = ()
    function_names: tuple[str, ...] = ()
    best_function_name: str | None = None
    total_cases = 0
    if evaluation is not None:
        function_names = tuple(evaluation.function_names)
        best_function_name = evaluation.best_function_name
        total_cases = evaluation.total_cases
        if best_function_name is not None:
            cases = tuple(
                _case_from_dr_code(case)
                for case in evaluation.results
                if case.function_name == best_function_name
            )
    return HumanEvalSubmissionResult(
        score=_score_from_outcome(outcome),
        outcome=str(outcome),
        function_names=function_names,
        best_function_name=best_function_name,
        total_cases=total_cases,
        cases=cases,
    )


def project_submission_score(result: HumanEvalSubmissionScore) -> CodeScore:
    """Thin scalar projection from a dr-code submission score."""

    return project_humaneval_submission_result(result).score


def project_mutant_submission_result(
    *,
    score: MutantScore,
    mutant: MutantRecord,
    outcomes: tuple[MutantInputOutcome, ...],
) -> MutantSubmissionResult:
    """Project one mutant-oracle run into Whetstone submission results."""

    distinct = frozenset(mutant.distinct_input_indices)
    observed = tuple(
        ExpectedOutcome(
            kind=cast(
                Literal["value", "error"],
                outcome.kind.value
                if hasattr(outcome.kind, "value")
                else str(outcome.kind),
            ),
            output_repr=outcome.output_repr,
        )
        for outcome in outcomes
    )
    input_results = tuple(
        MutantInputResult(
            input_index=index,
            input_repr=mutant.input_reprs[index],
            expected_mutant_repr=repr(mutant.mutant_expected[index]),
            expected_canonical_repr=repr(mutant.canonical_expected[index]),
            actual_kind=(
                outcome.kind.value
                if hasattr(outcome.kind, "value")
                else str(outcome.kind)
            ),
            actual_output_repr=outcome.output_repr,
            matched_mutant=observed[index] == mutant.mutant_expected[index],
            matched_canonical=observed[index]
            == mutant.canonical_expected[index],
            is_distinct=index in distinct,
        )
        for index, outcome in enumerate(outcomes)
    )
    if score.infrastructure_unknown or score.fidelity_to_mutant is None:
        code_score = _score_from_outcome(
            "mutant_oracle_infrastructure_unknown",
            infrastructure_unknown=True,
        )
    else:
        code_score = CodeScore(
            passed=score.fidelity_to_mutant >= 1.0,
            infrastructure_unknown=False,
            outcome="mutant_scored",
            fidelity_to_mutant=score.fidelity_to_mutant,
            attractor_pull=score.attractor_pull,
        )
    return MutantSubmissionResult(
        score=code_score,
        outcome=code_score.outcome,
        function_names=(mutant.entry_point,),
        best_function_name=mutant.entry_point,
        total_cases=len(mutant.input_reprs),
        input_results=input_results,
    )


SubmissionResultKind = Literal["humaneval", "mutant"]


def submission_result_to_record(
    result: CodeSubmissionResult | None,
):
    """Serialize one in-memory submission result for evaluation evidence."""

    if result is None:
        return None

    from whetstone.evaluation.schema import (
        CodeCaseResultRecord,
        CodeScoreRecord,
        HumanEvalSubmissionResultRecord,
        MutantInputResultRecord,
        MutantSubmissionResultRecord,
    )

    score_record = CodeScoreRecord(
        passed=result.score.passed,
        infrastructure_unknown=result.score.infrastructure_unknown,
        outcome=result.score.outcome,
        fidelity_to_mutant=result.score.fidelity_to_mutant,
        attractor_pull=result.score.attractor_pull,
    )
    if isinstance(result, HumanEvalSubmissionResult):
        return HumanEvalSubmissionResultRecord(
            score=score_record,
            outcome=result.outcome,
            function_names=result.function_names,
            best_function_name=result.best_function_name,
            total_cases=result.total_cases,
            cases=tuple(
                CodeCaseResultRecord(
                    case_id=case.case_id,
                    status=case.status,
                    message=case.message,
                    input_repr=case.input_repr,
                    expected_output_repr=case.expected_output_repr,
                    actual_output_repr=case.actual_output_repr,
                )
                for case in result.cases
            ),
        )
    input_results = getattr(result, "input_results", ())
    if isinstance(result, MutantSubmissionResult) or input_results:
        return MutantSubmissionResultRecord(
            score=score_record,
            outcome=result.outcome,
            function_names=result.function_names,
            best_function_name=result.best_function_name,
            total_cases=result.total_cases,
            input_results=tuple(
                MutantInputResultRecord(
                    input_index=item.input_index,
                    input_repr=item.input_repr,
                    expected_mutant_repr=item.expected_mutant_repr,
                    expected_canonical_repr=item.expected_canonical_repr,
                    actual_kind=item.actual_kind,
                    actual_output_repr=item.actual_output_repr,
                    matched_mutant=item.matched_mutant,
                    matched_canonical=item.matched_canonical,
                    is_distinct=item.is_distinct,
                )
                for item in input_results
            ),
        )
    cases = getattr(result, "cases", ())
    if cases:
        return HumanEvalSubmissionResultRecord(
            score=score_record,
            outcome=result.outcome,
            function_names=result.function_names,
            best_function_name=result.best_function_name,
            total_cases=result.total_cases,
            cases=tuple(
                CodeCaseResultRecord(
                    case_id=case.case_id,
                    status=case.status,
                    message=case.message,
                    input_repr=case.input_repr,
                    expected_output_repr=case.expected_output_repr,
                    actual_output_repr=case.actual_output_repr,
                )
                for case in cases
            ),
        )
    return HumanEvalSubmissionResultRecord(
        score=score_record,
        outcome=result.outcome,
        function_names=result.function_names,
        best_function_name=result.best_function_name,
        total_cases=result.total_cases,
        cases=(),
    )


def submission_result_from_record(
    record: object,
) -> CodeSubmissionResult | None:
    """Restore one persisted submission result for in-process consumers."""

    if record is None:
        return None
    from whetstone.evaluation.schema import (
        CodeScoreRecord,
        HumanEvalSubmissionResultRecord,
        MutantSubmissionResultRecord,
    )

    if isinstance(record, dict):
        kind = record.get("kind")
        if kind == "mutant":
            parsed = MutantSubmissionResultRecord.model_validate(record)
        elif kind == "humaneval":
            parsed = HumanEvalSubmissionResultRecord.model_validate(record)
        else:
            raise TypeError(
                f"unsupported submission result record kind: {kind!r}"
            )
    elif isinstance(record, MutantSubmissionResultRecord):
        parsed = record
    elif isinstance(record, HumanEvalSubmissionResultRecord):
        parsed = record
    else:
        raise TypeError(
            f"unsupported submission result record: {type(record)!r}"
        )

    def _score(score: CodeScoreRecord) -> CodeScore:
        return CodeScore(
            passed=score.passed,
            infrastructure_unknown=score.infrastructure_unknown,
            outcome=score.outcome,
            fidelity_to_mutant=score.fidelity_to_mutant,
            attractor_pull=score.attractor_pull,
        )

    code_score = _score(parsed.score)
    if isinstance(parsed, MutantSubmissionResultRecord):
        return MutantSubmissionResult(
            score=code_score,
            outcome=parsed.outcome,
            function_names=parsed.function_names,
            best_function_name=parsed.best_function_name,
            total_cases=parsed.total_cases,
            input_results=tuple(
                MutantInputResult(
                    input_index=item.input_index,
                    input_repr=item.input_repr,
                    expected_mutant_repr=item.expected_mutant_repr,
                    expected_canonical_repr=item.expected_canonical_repr,
                    actual_kind=item.actual_kind,
                    actual_output_repr=item.actual_output_repr,
                    matched_mutant=item.matched_mutant,
                    matched_canonical=item.matched_canonical,
                    is_distinct=item.is_distinct,
                )
                for item in parsed.input_results
            ),
        )
    return HumanEvalSubmissionResult(
        score=code_score,
        outcome=parsed.outcome,
        function_names=parsed.function_names,
        best_function_name=parsed.best_function_name,
        total_cases=parsed.total_cases,
        cases=tuple(
            CodeCaseResult(
                case_id=case.case_id,
                status=case.status,
                message=case.message,
                input_repr=case.input_repr,
                expected_output_repr=case.expected_output_repr,
                actual_output_repr=case.actual_output_repr,
            )
            for case in parsed.cases
        ),
    )


__all__ = [
    "CodeCaseResult",
    "CodeScore",
    "CodeSubmissionResult",
    "HumanEvalSubmissionResult",
    "MutantInputResult",
    "MutantSubmissionResult",
    "SubmissionResultKind",
    "project_humaneval_submission_result",
    "project_mutant_submission_result",
    "project_submission_score",
    "submission_result_from_record",
    "submission_result_to_record",
]
