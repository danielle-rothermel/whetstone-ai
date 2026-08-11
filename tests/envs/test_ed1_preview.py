from __future__ import annotations

from collections.abc import Sequence

import pytest

from tests.envs.support import synthetic_ed1_tasks
from whetstone.envs.ed1_preview import (
    Ed1ScoringRuntimeSummary,
    ed1_environment_fingerprint,
    run_ed1_scoring_preflight,
)
from whetstone.envs.ed1_runtime import Ed1RuntimeProbe
from whetstone.envs.ed1_scoring import CodeScore, CodeScoringInput


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


def _runtime() -> Ed1ScoringRuntimeSummary:
    return Ed1ScoringRuntimeSummary(
        evaluation_python="/copied/python",
        dr_code_version="0.1.5",
        runtime_identity_hash="a" * 64,
        probe=Ed1RuntimeProbe(
            implementation="CPython",
            numpy_version="2.0.0",
            python_executable="/copied/python",
            python_version="3.13.0",
        ),
    )


def test_ed1_environment_fingerprint_copies_runtime_fields() -> None:
    runtime = _runtime()
    fingerprint = ed1_environment_fingerprint(runtime)

    assert fingerprint.dependency_versions == (
        ("dr-code", "0.1.5"),
        ("numpy", "2.0.0"),
    )
    assert fingerprint.runtime_identity == "a" * 64


def test_run_ed1_scoring_preflight_passes_for_ground_truth() -> None:
    tasks = synthetic_ed1_tasks(1)
    result = run_ed1_scoring_preflight(tasks, _score)

    assert result.passed is True
    assert result.task_id == tasks[0].humaneval_task.task_id


def test_run_ed1_scoring_preflight_rejects_failed_score() -> None:
    tasks = synthetic_ed1_tasks(1)

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
        run_ed1_scoring_preflight(tasks, failing_score)
