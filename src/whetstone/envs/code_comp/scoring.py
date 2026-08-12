from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from time import monotonic
from types import TracebackType
from typing import Protocol

from dr_code.caching import CheckpointedExecutionCache
from dr_code.humaneval import (
    HUMANEVAL_SCORING_PROFILE_ID,
    HUMANEVAL_SCORING_PROFILE_VERSION,
    HumanEvalSubmissionRequest,
    HumanEvalTask,
    score_humaneval_submission,
    score_humaneval_submissions_batch,
)
from dr_exec import (
    CancelToken,
    CompletedExecution,
    ExecutionJob,
    Executor,
    FiniteDurationLimit,
)
from dr_serialize import IdentityDocument
from dr_store import SqliteRecordCache

from whetstone.envs.code_comp.submission_result import (
    CodeScore,
    CodeSubmissionResult,
    HumanEvalSubmissionResult,
    project_humaneval_submission_result,
    project_submission_score,
)
from whetstone.evaluation.preview.preflight import ScoringPreflight

#: The parser contract for decoder submissions. The profile id and version are
#: also folded into the ed1 evaluation procedure identity.
CODE_COMP_SCORING_PROFILE_ID = HUMANEVAL_SCORING_PROFILE_ID
CODE_COMP_SCORING_PROFILE_VERSION = HUMANEVAL_SCORING_PROFILE_VERSION
CODE_COMP_SCORING_PREFLIGHT_TASK_ID = "HumanEval/0"


class _PreflightTask(Protocol):
    @property
    def humaneval_task(self) -> HumanEvalTask: ...


def _one_preflight_result(
    results: Sequence[CodeSubmissionResult], *, context: str
) -> CodeSubmissionResult:
    if len(results) != 1:
        raise ValueError(
            f"{context} returned {len(results)} results, expected 1"
        )
    return results[0]


def run_encdec_scoring_preflight(
    tasks: tuple[_PreflightTask, ...],
    batch_scorer: CodeBatchScorer,
) -> ScoringPreflight:
    task = tasks[0].humaneval_task
    result = _one_preflight_result(
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
    score = result.score
    preflight = ScoringPreflight(
        task_id=task.task_id,
        passed=score.passed,
        infrastructure_unknown=score.infrastructure_unknown,
        outcome=score.outcome,
    )
    if preflight.infrastructure_unknown or not preflight.passed:
        raise RuntimeError(
            "HumanEval ground-truth runtime preflight did not pass: "
            f"{preflight.outcome}"
        )
    return preflight


@dataclass(frozen=True, slots=True)
class CodeScoringInput:
    """One generated submission and its exact HumanEval scoring task."""

    raw_submission: str
    task: HumanEvalTask


class BatchScoringDeadlineExceeded(RuntimeError):
    """The shared evaluation phase wall expired during code scoring."""


class CodeBatchScorer(Protocol):
    def __call__(
        self,
        inputs: Sequence[CodeScoringInput],
        *,
        max_wall_seconds: float | None = None,
    ) -> Sequence[CodeSubmissionResult]: ...


class _DeadlineExecutor:
    """Clamp every execution job to one shared batch deadline."""

    def __init__(self, executor: Executor, deadline: float) -> None:
        self._executor = executor
        self._deadline = deadline
        self.expired = False

    def run(
        self,
        job: ExecutionJob,
        /,
        *,
        cancellation: CancelToken | None = None,
    ) -> CompletedExecution:
        remaining_seconds = self._deadline - monotonic()
        if remaining_seconds <= 0:
            self.expired = True
            raise BatchScoringDeadlineExceeded
        remaining_ns = max(1, math.ceil(remaining_seconds * 1_000_000_000))
        wall_time = job.budgets.wall_time
        if isinstance(wall_time, FiniteDurationLimit):
            remaining_ns = min(remaining_ns, wall_time.max_ns)
        bounded_job = replace(
            job,
            budgets=job.budgets.model_copy(
                update={"wall_time": FiniteDurationLimit(max_ns=remaining_ns)}
            ),
        )
        try:
            return self._executor.run(
                bounded_job,
                cancellation=cancellation,
            )
        finally:
            if monotonic() >= self._deadline:
                self.expired = True


def _batch_deadline(max_wall_seconds: float | None) -> float | None:
    if max_wall_seconds is None:
        return None
    if type(max_wall_seconds) not in (int, float):
        raise ValueError(
            "max_wall_seconds must be a finite nonnegative real number"
        )
    seconds = float(max_wall_seconds)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(
            "max_wall_seconds must be a finite nonnegative real number"
        )
    return monotonic() + seconds


class CheckpointedCodeBatchScorer:
    """Own one persistent execution cache across coordinator scoring batches.

    dr-code owns request planning, execution-key derivation, restoration, and
    checkpoint scheduling. Whetstone supplies the runtime identity and cache
    lifecycle, then projects ordered HumanEval results into submission results.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        runtime_document: IdentityDocument,
        executor: Executor,
        checkpoint_entry_count: int = 1_000,
    ) -> None:
        self._path = Path(path)
        self._runtime_document = runtime_document
        self._executor = executor
        self._checkpoint_entry_count = checkpoint_entry_count
        self._store: SqliteRecordCache | None = None
        self._cache: CheckpointedExecutionCache | None = None

    def __enter__(self) -> CheckpointedCodeBatchScorer:
        store = SqliteRecordCache(self._path)
        try:
            cache = CheckpointedExecutionCache(
                store,
                runtime_identity=self._runtime_document,
                checkpoint_entry_count=self._checkpoint_entry_count,
            )
        except BaseException:
            store.close()
            raise
        self._store = store
        self._cache = cache
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()

    def __call__(
        self,
        inputs: Sequence[CodeScoringInput],
        *,
        max_wall_seconds: float | None = None,
    ) -> tuple[CodeSubmissionResult, ...]:
        cache = self._cache
        if cache is None:
            raise RuntimeError("checkpointed code batch scorer is not open")
        deadline = _batch_deadline(max_wall_seconds)
        if deadline is not None and monotonic() >= deadline:
            raise BatchScoringDeadlineExceeded
        executor: Executor = self._executor
        deadline_executor: _DeadlineExecutor | None = None
        if deadline is not None:
            deadline_executor = _DeadlineExecutor(executor, deadline)
            executor = deadline_executor
        requests = tuple(
            HumanEvalSubmissionRequest(
                raw_submission=item.raw_submission,
                task=item.task,
                scoring_profile_id=CODE_COMP_SCORING_PROFILE_ID,
                scoring_profile_version=CODE_COMP_SCORING_PROFILE_VERSION,
            )
            for item in inputs
        )
        results = score_humaneval_submissions_batch(
            requests,
            executor=executor,
            execution_cache=cache,
        )
        cache.checkpoint()
        if deadline is not None and (
            monotonic() >= deadline
            or (deadline_executor is not None and deadline_executor.expired)
        ):
            raise BatchScoringDeadlineExceeded
        projected = tuple(
            project_humaneval_submission_result(result) for result in results
        )
        if len(projected) != len(inputs):
            raise ValueError(
                "dr-code batch scorer returned the wrong result count"
            )
        return projected

    def close(self) -> None:
        cache = self._cache
        store = self._store
        if cache is None or store is None:
            return
        self._cache = None
        self._store = None
        try:
            cache.close()
        finally:
            store.close()


def score_code_comp_submission(
    *,
    raw_submission: str,
    task: HumanEvalTask,
    executor: Executor,
) -> HumanEvalSubmissionResult:
    """Score one decoder submission into a HumanEval submission result."""

    result = score_humaneval_submission(
        raw_submission=raw_submission,
        task=task,
        scoring_profile_id=CODE_COMP_SCORING_PROFILE_ID,
        scoring_profile_version=CODE_COMP_SCORING_PROFILE_VERSION,
        executor=executor,
    )
    return project_humaneval_submission_result(result)


# Backward-compatible alias used by tests and legacy imports.
_project_submission_score = project_submission_score


__all__ = [
    "CODE_COMP_SCORING_PREFLIGHT_TASK_ID",
    "CODE_COMP_SCORING_PROFILE_ID",
    "CODE_COMP_SCORING_PROFILE_VERSION",
    "BatchScoringDeadlineExceeded",
    "CheckpointedCodeBatchScorer",
    "CodeBatchScorer",
    "CodeScore",
    "CodeScoringInput",
    "CodeSubmissionResult",
    "HumanEvalSubmissionResult",
    "_project_submission_score",
    "project_humaneval_submission_result",
    "project_submission_score",
    "run_encdec_scoring_preflight",
    "score_code_comp_submission",
]
