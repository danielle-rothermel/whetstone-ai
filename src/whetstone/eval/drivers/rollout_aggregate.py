from __future__ import annotations

from whetstone.eval.aggregate import EvalMatrixPlan, TaskRows, unweighted_task_mean
from whetstone.eval.attribution import attribute_generated_row
from whetstone.eval.drivers.eval_result import (
    InternalEvalResult,
    per_task_count,
    per_task_score,
)
from whetstone.eval.drivers.row_common import RolloutRowOutput
from whetstone.experiment.reward import Reward

__all__ = ["aggregate_rollout_outputs"]


def aggregate_rollout_outputs(
    *,
    outputs: tuple[RolloutRowOutput, ...],
    task_hashes: tuple[str, ...],
    num_seeds: int,
    graph_hash: str,
    matrix_plan: EvalMatrixPlan,
    aggregate_name: str,
    request_identities: frozenset[str] = frozenset(),
    reward: Reward | None = None,
    deadline_reached: bool = False,
) -> InternalEvalResult:
    task_rows: list[TaskRows] = []
    for task_index, task_hash in enumerate(task_hashes):
        row_values = tuple(
            attribute_generated_row(
                row_state=output.row_state,
                score=output.score,
                failure_code=output.failure_code or None,
            )
            for output in outputs
            if output.task_index == task_index
        )
        task_rows.append(TaskRows(task_hash=task_hash, rows=row_values))

    aggregate = unweighted_task_mean(
        aggregate_name=aggregate_name,
        graph_hash=graph_hash,
        task_rows=tuple(task_rows),
        plan=matrix_plan,
    )
    return InternalEvalResult(
        aggregate=aggregate,
        reward=reward,
        per_task_scores=tuple(
            per_task_score(
                task_row, num_seeds, matrix_plan.aggregation_config
            )
            for task_row in task_rows
        ),
        per_task_counts=tuple(
            per_task_count(task_row, num_seeds) for task_row in task_rows
        ),
        outputs=outputs,
        request_identities=request_identities,
        deadline_reached=deadline_reached,
    )
