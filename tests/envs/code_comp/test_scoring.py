from __future__ import annotations

from collections.abc import Sequence

import pytest

from tests.envs.support import synthetic_code_comp_tasks
from whetstone.envs.code_comp.scoring import (
    CodeScore,
    CodeScoringInput,
    run_encdec_scoring_preflight,
)


def _score(
    inputs: Sequence[CodeScoringInput],
    *,
    max_wall_seconds: float | None = None,
) -> tuple[CodeScore, ...]:
    del max_wall_seconds
    return tuple(
        CodeScore(
            passed=True,
            infrastructure_unknown=False,
            outcome="passed",
        )
        for _ in inputs
    )


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
    ) -> tuple[CodeScore, ...]:
        del max_wall_seconds
        return (
            CodeScore(
                passed=False,
                infrastructure_unknown=False,
                outcome="tests_failed",
            ),
        )

    with pytest.raises(RuntimeError, match="preflight did not pass"):
        run_encdec_scoring_preflight(tasks, failing_score)
