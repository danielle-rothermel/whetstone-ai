from __future__ import annotations

from dataclasses import dataclass

from whetstone.eval.aggregate import Aggregate, TaskRows
from whetstone.eval.drivers.row_common import RolloutRowOutput
from whetstone.experiment.reward import Reward


@dataclass(frozen=True, slots=True)
class InternalEvalResult:
    """One candidate's evaluation outcome over a split."""

    aggregate: Aggregate
    reward: Reward | None
    per_task_scores: tuple[float, ...]
    per_task_counts: tuple[int, ...]
    outputs: tuple[RolloutRowOutput, ...]
    supplemental_aggregates: tuple[Aggregate, ...] = ()
    request_identities: frozenset[str] = frozenset()
    deadline_reached: bool = False


def per_task_score(task: TaskRows, num_seeds: int) -> float:
    completed = task.completed_rows(num_seeds)
    if not completed:
        return 0.0
    total = sum(
        float(row.value or 0.0) if row.is_present else 0.0 for row in completed
    )
    return total / len(completed)


def per_task_count(task: TaskRows, num_seeds: int) -> int:
    """Count of completed repeats behind this task's mean."""
    return len(task.completed_rows(num_seeds))


__all__ = [
    "InternalEvalResult",
    "per_task_count",
    "per_task_score",
]
