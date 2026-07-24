"""Shared builders for official-selection tests.

Constructs real Whetstone :class:`RolloutAggregate` values through the public
aggregate constructors so selection runs against certified, complete evidence
rather than stand-ins.
"""

from __future__ import annotations

from whetstone.code_eval.aggregate import (
    RolloutAggregate,
    RowValue,
    TaskRows,
    average_binary_test_pass_rate,
    mean_compression_ratio,
)

GRAPH_A = "a" * 64
GRAPH_B = "b" * 64
EVAL_HASH = "c" * 64
CONTEXT_ID = "ctx-official"


def pass_rate_aggregate(
    *,
    graph_hash: str = GRAPH_A,
    value: float = 1.0,
    tasks: int = 2,
    repeats: int = 2,
) -> RolloutAggregate:
    """A complete, OK Average Binary Test Pass Rate aggregate.

    Every planned cell is present with the same value, so the two staged
    reductions produce ``value`` and the pure status is OK.
    """
    task_rows = tuple(
        TaskRows(
            task_identity=f"task-{t}",
            expected_repeats=repeats,
            rows=tuple(RowValue(value=value) for _ in range(repeats)),
        )
        for t in range(tasks)
    )
    return average_binary_test_pass_rate(
        graph_hash=graph_hash,
        eval_config_hash=EVAL_HASH,
        evaluation_context_id=CONTEXT_ID,
        task_rows=task_rows,
        repeat_count=repeats,
    )


def compression_aggregate(
    *,
    graph_hash: str = GRAPH_A,
    value: float = 2.0,
    tasks: int = 2,
    repeats: int = 2,
) -> RolloutAggregate:
    """A complete, OK Mean Compression Ratio aggregate."""
    rows = tuple(RowValue(value=value) for _ in range(tasks * repeats))
    return mean_compression_ratio(
        graph_hash=graph_hash,
        eval_config_hash=EVAL_HASH,
        evaluation_context_id=CONTEXT_ID,
        rows=rows,
        task_count=tasks,
        repeat_count=repeats,
    )


def incomplete_pass_rate_aggregate(
    *,
    graph_hash: str = GRAPH_A,
    tasks: int = 2,
    repeats: int = 2,
) -> RolloutAggregate:
    """An incomplete Average Binary Test Pass Rate aggregate (missing rows).

    One task is short a repeat, so under the default PROPAGATE policy the pure
    reduction is not OK (MISSING_DATA) — exactly the incomplete evidence
    official selection must refuse.
    """
    task_rows = (
        TaskRows(
            task_identity="task-0",
            expected_repeats=repeats,
            rows=(RowValue(value=1.0),),  # short one repeat -> missing padded
        ),
        *(
            TaskRows(
                task_identity=f"task-{t}",
                expected_repeats=repeats,
                rows=tuple(RowValue(value=1.0) for _ in range(repeats)),
            )
            for t in range(1, tasks)
        ),
    )
    return average_binary_test_pass_rate(
        graph_hash=graph_hash,
        eval_config_hash=EVAL_HASH,
        evaluation_context_id=CONTEXT_ID,
        task_rows=task_rows,
        repeat_count=repeats,
    )
