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
    HarnessFailure,
    HumanEvalSubmissionRequest,
    HumanEvalSubmissionScore,
    HumanEvalTask,
    SubmissionOutcome,
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

#: The parser contract for decoder submissions. The profile id and version are
#: also folded into the ed1 evaluation procedure identity.
ED1_SCORING_PROFILE_ID = HUMANEVAL_SCORING_PROFILE_ID
ED1_SCORING_PROFILE_VERSION = HUMANEVAL_SCORING_PROFILE_VERSION


@dataclass(frozen=True, slots=True)
class CodeScore:
    """The ed1 correctness outcome for one decoder submission.

    ``passed`` is the natural typed boolean projection of dr-code's
    :class:`SubmissionOutcome`: it is true exactly for ``PASSED``.
    ``infrastructure_unknown`` is true only for dr-code's typed
    :class:`HarnessFailure`. Every :class:`CompletedScore` retains its
    definitive zero/one projection, including candidate timeouts and
    incomplete candidate coverage. ``outcome`` retains the dr-code label.

    ED1M extends the projection with ``fidelity_to_mutant``, the fractional
    reward-bearing row value (fraction of inputs matching the mutant).
    ``attractor_pull`` is the reported contamination measurement (fraction of
    discriminating inputs that snapped to canonical), never a reward. Both are
    ``None`` for ED1/D1, whose HumanEval Submission Score is ``float(passed)``.
    """

    passed: bool
    infrastructure_unknown: bool
    outcome: str
    fidelity_to_mutant: float | None = None
    attractor_pull: float | None = None

    @property
    def row_value(self) -> float:
        """The environment's primary row metric value.

        ED1M returns fractional ``fidelity_to_mutant``. ED1/D1 return the
        HumanEval Submission Score, the 0.0/1.0 numeric projection of
        ``passed``.
        """
        return (
            self.fidelity_to_mutant
            if self.fidelity_to_mutant is not None
            else float(self.passed)
        )


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
    ) -> Sequence[CodeScore]: ...


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
    lifecycle, then projects the ordered HumanEval results into ``CodeScore``.
    The batch adapter and cache are supplied by the pinned dr-code dependency.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        runtime_identity: IdentityDocument,
        executor: Executor,
        checkpoint_entry_count: int = 1_000,
    ) -> None:
        self._path = Path(path)
        self._runtime_identity = runtime_identity
        self._executor = executor
        self._checkpoint_entry_count = checkpoint_entry_count
        self._store: SqliteRecordCache | None = None
        self._cache: CheckpointedExecutionCache | None = None

    def __enter__(self) -> CheckpointedCodeBatchScorer:
        store = SqliteRecordCache(self._path)
        try:
            cache = CheckpointedExecutionCache(
                store,
                runtime_identity=self._runtime_identity,
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
    ) -> tuple[CodeScore, ...]:
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
                scoring_profile_id=ED1_SCORING_PROFILE_ID,
                scoring_profile_version=ED1_SCORING_PROFILE_VERSION,
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
            _project_submission_score(result) for result in results
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


def _project_submission_score(result: HumanEvalSubmissionScore) -> CodeScore:
    if isinstance(result, HarnessFailure):
        return CodeScore(
            passed=False,
            infrastructure_unknown=True,
            outcome=result.kind,
        )
    outcome = result.outcome
    return CodeScore(
        passed=outcome is SubmissionOutcome.PASSED,
        infrastructure_unknown=False,
        outcome=str(outcome),
    )


def score_ed1_submission(
    *,
    raw_submission: str,
    task: HumanEvalTask,
    executor: Executor,
) -> CodeScore:
    """Score one decoder submission -> :class:`CodeScore`.

    Delegates to dr-code's ``score_humaneval_submission`` (preprocessing +
    subprocess test run) and projects its typed outcome onto the ed1
    correctness
    invariant: ``PASSED`` -> passed; a typed harness failure -> infrastructure
    unknown (the rollout fails); every completed outcome (tests failed,
    candidate timeout, incomplete candidate coverage, extraction failed, no
    top-level functions, ...) -> definitive fail (score 0).
    """
    result = score_humaneval_submission(
        raw_submission=raw_submission,
        task=task,
        scoring_profile_id=ED1_SCORING_PROFILE_ID,
        scoring_profile_version=ED1_SCORING_PROFILE_VERSION,
        executor=executor,
    )
    return _project_submission_score(result)


__all__ = [
    "ED1_SCORING_PROFILE_ID",
    "ED1_SCORING_PROFILE_VERSION",
    "BatchScoringDeadlineExceeded",
    "CheckpointedCodeBatchScorer",
    "CodeBatchScorer",
    "CodeScore",
    "CodeScoringInput",
    "score_ed1_submission",
]
