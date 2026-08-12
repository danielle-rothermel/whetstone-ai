from __future__ import annotations

from dataclasses import dataclass

from whetstone.evaluation.aggregate import Aggregate, TaskRows
from whetstone.evaluation.drivers.row_common import GenerationRowOutput
from whetstone.experiment.reward import Reward


@dataclass(frozen=True, slots=True)
class InternalEvalResult:
    """One candidate's evaluation outcome over a split."""

    aggregate: Aggregate
    reward: Reward | None
    per_task_scores: tuple[float, ...]
    per_task_counts: tuple[int, ...]
    outputs: tuple[GenerationRowOutput, ...]
    supplemental_aggregates: tuple[Aggregate, ...] = ()
    request_identities: frozenset[str] = frozenset()
    concurrency_halved: bool = False
    deadline_reached: bool = False
    guard_timeouts: int = 0


def per_task_score(task: TaskRows, num_samples: int) -> float:
    completed = task.completed_rows(num_samples)
    if not completed:
        return 0.0
    total = sum(
        float(row.value or 0.0) if row.is_present else 0.0 for row in completed
    )
    return total / len(completed)


def per_task_count(task: TaskRows, num_samples: int) -> int:
    """Count of completed repeats behind this task's mean."""
    return len(task.completed_rows(num_samples))


__all__ = [
    "InternalEvalResult",
    "per_task_count",
    "per_task_score",
]
