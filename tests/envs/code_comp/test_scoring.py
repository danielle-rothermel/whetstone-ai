from __future__ import annotations

from collections.abc import Sequence

import pytest

from tests.envs.support import synthetic_code_comp_tasks
from whetstone.envs.code_comp.scoring import (
    CodeScoringInput,
    run_encdec_scoring_preflight,
)
from whetstone.envs.code_comp.submission_result import (
    CodeScore,
    HumanEvalSubmissionResult,
)


def _passing_result() -> HumanEvalSubmissionResult:
    return HumanEvalSubmissionResult(
        score=CodeScore(
            passed=True,
            infrastructure_unknown=False,
            outcome="passed",
        ),
        outcome="passed",
        function_names=(),
        best_function_name=None,
        total_cases=0,
    )


def _score(
    inputs: Sequence[CodeScoringInput],
    *,
    max_wall_seconds: float | None = None,
) -> tuple[HumanEvalSubmissionResult, ...]:
    del max_wall_seconds
    return tuple(_passing_result() for _ in inputs)


def test_run_encdec_scoring_preflight_passes_for_ground_truth() -> None:
    tasks = synthetic_code_comp_tasks(1)
    result = run_encdec_scoring_preflight(tasks, _score)

    assert result.passed is True
    assert result.task_id == tasks[0].humaneval_task.task_id


def test_run_encdec_scoring_preflight_rejects_failed_score() -> None:
    tasks = synthetic_code_comp_tasks(1)

    def failing_score(
        inputs: Sequence[CodeScoringInput],
        *,
        max_wall_seconds: float | None = None,
    ) -> tuple[HumanEvalSubmissionResult, ...]:
        del max_wall_seconds
        return (
            HumanEvalSubmissionResult(
                score=CodeScore(
                    passed=False,
                    infrastructure_unknown=False,
                    outcome="tests_failed",
                ),
                outcome="tests_failed",
                function_names=(),
                best_function_name=None,
                total_cases=0,
            ),
        )

    with pytest.raises(RuntimeError, match="preflight did not pass"):
        run_encdec_scoring_preflight(tasks, failing_score)
