from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

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
from dr_exec import Executor
from dr_serialize import IdentityDocument
from dr_store import SqliteRecordCache

#: The parser contract for decoder submissions. The profile id and version are
#: also folded into the ed1 evaluation procedure identity.
ED1_SCORING_PROFILE_ID = HUMANEVAL_SCORING_PROFILE_ID
ED1_SCORING_PROFILE_VERSION = HUMANEVAL_SCORING_PROFILE_VERSION

#: The outcomes that are INFRASTRUCTURE-UNKNOWN (no definitive pass/fail): the
#: rollout fails, never scores 0. A :class:`HarnessFailure` is handled
#: separately because it is not a completed submission outcome.
_INFRASTRUCTURE_UNKNOWN_OUTCOMES = frozenset(
    {
        SubmissionOutcome.EVALUATION_INCOMPLETE,
        SubmissionOutcome.TIMED_OUT,
    }
)


@dataclass(frozen=True, slots=True)
class CodeScore:
    """The ed1 correctness outcome for one decoder submission.

    ``passed`` is the natural typed boolean projection of dr-code's
    :class:`SubmissionOutcome`: it is true exactly for ``PASSED``.
    ``infrastructure_unknown`` is true when dr-code could not deliver a
    definitive verdict (harness failure / evaluation incomplete / timeout);
    the rollout must fail, never score 0. ``outcome`` retains the dr-code
    label.

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


type CodeBatchScorer = Callable[
    [Sequence[CodeScoringInput]], Sequence[CodeScore]
]


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
        self, inputs: Sequence[CodeScoringInput]
    ) -> tuple[CodeScore, ...]:
        cache = self._cache
        if cache is None:
            raise RuntimeError("checkpointed code batch scorer is not open")
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
            executor=self._executor,
            execution_cache=cache,
        )
        projected = tuple(
            _project_submission_score(result) for result in results
        )
        if len(projected) != len(inputs):
            raise ValueError(
                "dr-code batch scorer returned the wrong result count"
            )
        cache.checkpoint()
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
    if outcome in _INFRASTRUCTURE_UNKNOWN_OUTCOMES:
        return CodeScore(
            passed=False,
            infrastructure_unknown=True,
            outcome=str(outcome),
        )
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
    invariant: ``PASSED`` -> passed; a typed harness failure,
    ``EVALUATION_INCOMPLETE``, or ``TIMED_OUT`` -> infrastructure unknown (the
    rollout fails); every other completed outcome (tests failed, extraction
    failed, no top-level functions, ...) -> definitive fail (score 0).
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
    "CheckpointedCodeBatchScorer",
    "CodeBatchScorer",
    "CodeScore",
    "CodeScoringInput",
    "score_ed1_submission",
]
