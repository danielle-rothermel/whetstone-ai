from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
from dr_code.humaneval import HumanEvalSubmissionScore, SubmissionOutcome
from dr_code.humaneval.task import (
    EvaluationCaseResult,
    EvaluationCaseStatus,
    HumanEvalTestCaseKind,
)

from whetstone.envs.code_comp.mutant.dataset import (
    ExpectedOutcome,
    MutantRecord,
)
from whetstone.envs.code_comp.mutant.oracle import (
    MutantInputOutcome,
    MutantScore,
    _OutcomeKind,
)
from whetstone.envs.code_comp.scoring import _project_submission_score
from whetstone.envs.code_comp.submission_result import (
    CodeCaseResult,
    project_humaneval_submission_result,
    project_mutant_submission_result,
    project_submission_score,
)


def _completed(
    *,
    outcome: SubmissionOutcome,
    cases: tuple[EvaluationCaseResult, ...] = (),
) -> HumanEvalSubmissionScore:
    return cast(
        HumanEvalSubmissionScore,
        SimpleNamespace(
            outcome=outcome,
            evaluation=SimpleNamespace(
                function_names=["candidate"],
                best_function_name="candidate",
                total_cases=len(cases),
                results=list(cases),
            ),
        ),
    )


def test_project_humaneval_pass_includes_case_rows() -> None:
    case = EvaluationCaseResult(
        task_id="HumanEval/0",
        case_id="case_0",
        function_name="candidate",
        status=EvaluationCaseStatus.PASSED,
        test_type=HumanEvalTestCaseKind.INPUT_RESULT,
        input_repr="(1,)",
        expected_output_repr="2",
        actual_output_repr="2",
    )
    result = project_humaneval_submission_result(
        _completed(outcome=SubmissionOutcome.PASSED, cases=(case,))
    )

    assert result.score.passed is True
    assert result.cases == (
        CodeCaseResult(
            case_id="case_0",
            status="passed",
            message="",
            input_repr="(1,)",
            expected_output_repr="2",
            actual_output_repr="2",
        ),
    )
    assert (
        project_submission_score(
            _completed(outcome=SubmissionOutcome.PASSED, cases=(case,))
        )
        == result.score
    )
    assert (
        _project_submission_score(
            _completed(outcome=SubmissionOutcome.PASSED, cases=(case,))
        )
        == result.score
    )


@pytest.mark.parametrize(
    "outcome",
    [
        SubmissionOutcome.TESTS_FAILED,
        SubmissionOutcome.EXTRACTION_FAILED,
        SubmissionOutcome.TIMED_OUT,
    ],
)
def test_project_humaneval_completed_failure_as_zero(
    outcome: SubmissionOutcome,
) -> None:
    result = project_humaneval_submission_result(_completed(outcome=outcome))

    assert result.score.outcome == outcome.value
    assert result.score.passed is False
    assert result.score.infrastructure_unknown is False


def test_project_humaneval_harness_failure_is_infrastructure_unknown() -> None:
    result = project_humaneval_submission_result(
        cast(
            HumanEvalSubmissionScore,
            SimpleNamespace(
                kind="harness_failure",
                evaluation=None,
            ),
        )
    )

    assert result.score.infrastructure_unknown is True
    assert result.cases == ()


def test_project_mutant_submission_result() -> None:
    mutant = SimpleNamespace(
        entry_point="f",
        input_reprs=("(1,)",),
        distinct_input_indices=(0,),
        mutant_expected=(ExpectedOutcome(kind="value", output_repr="1"),),
        canonical_expected=(ExpectedOutcome(kind="value", output_repr="2"),),
    )
    score = MutantScore(
        fidelity_to_mutant=0.0,
        attractor_pull=0.0,
        matched_mutant=0,
        matched_canonical_on_distinct=0,
        total_inputs=1,
        distinct_inputs=1,
        infrastructure_unknown=False,
    )
    outcomes = (MutantInputOutcome(kind=_OutcomeKind.VALUE, output_repr="9"),)

    result = project_mutant_submission_result(
        score=score,
        mutant=cast(MutantRecord, mutant),
        outcomes=outcomes,
    )

    assert result.score.passed is False
    assert result.input_results[0].matched_mutant is False
    assert result.total_cases == 1
