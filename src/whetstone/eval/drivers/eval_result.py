from __future__ import annotations

from dataclasses import dataclass

from whetstone.eval.aggregate import Aggregate, TaskRows
from whetstone.eval.aggregation import AggregationStatus, aggregate
from whetstone.eval.config import AggregationConfig
from whetstone.eval.drivers.row_common import RolloutRowOutput
from whetstone.experiment.reward import Reward


@dataclass(frozen=True, slots=True)
class InternalEvalResult:
    """One candidate's evaluation outcome over a split."""

    aggregate: Aggregate
    reward: Reward | None
    per_task_scores: tuple[float | None, ...]
    per_task_counts: tuple[int, ...]
    outputs: tuple[RolloutRowOutput, ...]
    supplemental_aggregates: tuple[Aggregate, ...] = ()
    request_identities: frozenset[str] = frozenset()
    deadline_reached: bool = False


def per_task_score(
    task: TaskRows,
    num_seeds: int,
    config: AggregationConfig,
) -> float | None:
    """This task's score under the plan's own aggregation policy.

    This is the *same* reduction ``unweighted_task_mean`` applies to build its
    per-task inputs, called through the identical ``aggregate`` entry point on
    the identical rows. The per-task vector and the evaluation-level aggregate
    therefore read one definition of a task's score rather than two that can
    drift apart.

    Concretely, the score is the mean over this task's **present** rows.
    A non-present repeat (failed, missing, or invalid) is not a zero: under a
    missing-row-tolerant policy it is skipped, and under a propagating policy
    it withholds the task's score entirely.

    ``None`` means *unobserved*, not *scored zero*. It is returned when no
    present row backs the task -- every repeat lost, or the policy propagated
    the loss. A consumer that cannot represent an unobserved task must decide
    that explicitly; silently reading it as 0.0 mints a false hard task.
    """
    completed = task.completed_rows(num_seeds)
    output = aggregate(
        config,
        tuple(row.to_aggregation_input() for row in completed),
    )
    if output.status is not AggregationStatus.OK:
        return None
    return output.value


def per_task_count(task: TaskRows, num_seeds: int) -> int:
    """Count of **present** repeats behind this task's score.

    Rows that failed, went missing, or scored invalid are excluded, so this
    drops below ``num_seeds`` exactly when the task lost repeats. It is the
    row-completeness weight for this task, and it reads zero precisely when
    ``per_task_score`` is unobserved for want of data.
    """
    return sum(1 for row in task.completed_rows(num_seeds) if row.is_present)


__all__ = [
    "InternalEvalResult",
    "per_task_count",
    "per_task_score",
]
