from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, StrictStr

from whetstone.envs.ed1 import Ed1Instance
from whetstone.envs.ed1_runtime import Ed1RuntimeProbe
from whetstone.envs.ed1_scoring import (
    CodeBatchScorer,
    CodeScore,
    CodeScoringInput,
)
from whetstone.experiment.binding import ExecutionEnvironmentFingerprint

ED1_SCORING_PREFLIGHT_TASK_ID = "HumanEval/0"

__all__ = [
    "ED1_SCORING_PREFLIGHT_TASK_ID",
    "Ed1ScoringPreflight",
    "Ed1ScoringRuntimeSummary",
    "ed1_environment_fingerprint",
    "run_ed1_scoring_preflight",
]


class Ed1ScoringRuntimeSummary(BaseModel):
    """Runtime identity displayed and persisted with a scoring preview."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evaluation_python: StrictStr
    dr_code_version: StrictStr
    runtime_identity_hash: StrictStr
    probe: Ed1RuntimeProbe


class Ed1ScoringPreflight(BaseModel):
    """Ground-truth check completed before candidate evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: StrictStr
    passed: bool
    infrastructure_unknown: bool
    outcome: StrictStr


def _one_score(scores: Sequence[CodeScore], *, context: str) -> CodeScore:
    if len(scores) != 1:
        raise ValueError(
            f"{context} returned {len(scores)} scores, expected 1"
        )
    return scores[0]


def run_ed1_scoring_preflight(
    tasks: tuple[Ed1Instance, ...],
    batch_scorer: CodeBatchScorer,
) -> Ed1ScoringPreflight:
    task = tasks[0].humaneval_task
    score = _one_score(
        batch_scorer(
            (
                CodeScoringInput(
                    raw_submission=task.ground_truth_code,
                    task=task,
                ),
            )
        ),
        context="runtime preflight",
    )
    result = Ed1ScoringPreflight(
        task_id=task.task_id,
        passed=score.passed,
        infrastructure_unknown=score.infrastructure_unknown,
        outcome=score.outcome,
    )
    if result.infrastructure_unknown or not result.passed:
        raise RuntimeError(
            "HumanEval ground-truth runtime preflight did not pass: "
            f"{result.outcome}"
        )
    return result


def ed1_environment_fingerprint(
    runtime: Ed1ScoringRuntimeSummary,
) -> ExecutionEnvironmentFingerprint:
    return ExecutionEnvironmentFingerprint(
        dependency_versions=(
            ("dr-code", runtime.dr_code_version),
            ("numpy", runtime.probe.numpy_version),
        ),
        runtime_identity=runtime.runtime_identity_hash,
    )
