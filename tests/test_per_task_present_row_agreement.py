"""The per-task vector and the evaluation-level aggregate read the same rows.

``per_task_score``/``per_task_count`` and ``unweighted_task_mean`` are two
views of one evaluation. Before this contract they disagreed: the per-task
reduction scored every non-present row as 0.0 and divided by ``num_seeds``,
while the aggregate applied the plan's own missing-row policy. Under a
tolerant (``skip``) policy a task with three present rows at 1.0 and one lost
repeat therefore reported 0.75 in the vector beside 1.0 in the aggregate.

``per_task_count`` was worse than merely wrong: it returned
``len(completed_rows(num_seeds))``, and ``completed_rows`` pads missing
repeats *in*, so it always equalled ``num_seeds`` and any downstream
row-completeness weighting built on it was inert.

These tests drive the real rollout aggregation path -- the same
``aggregate_rollout_outputs`` the drivers call -- rather than reimplementing
the reduction, so they fail if the two views drift apart again.
"""

from __future__ import annotations

import pytest

from whetstone.eval.aggregate import aggregation_definition
from whetstone.eval.drivers.rollout_aggregate import aggregate_rollout_outputs
from whetstone.eval.drivers.row_common import RolloutRowOutput
from whetstone.eval.traces import ExecutedRowState
from whetstone.experiment.sampling import derive_eval_split

NAMESPACE = "pertask-toy"
DATASET_REVISION = "pertask-toy-rev"
GRAPH_HASH = "c" * 64
AGGREGATE_NAME = "score"
NUM_SEEDS = 2


class _Task:
    """Minimal sampling task: an id plus the hash the split indexes it by."""

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id


def _task_hash(task: _Task) -> str:
    # A stable 64-hex identity per task id; the exact digest is irrelevant.
    return (task.task_id.encode().hex() * 64)[:64]


def _split(missing_data: str, *, max_skip_fraction: str = "0.5"):
    """A two-task, two-repeat split under the named missing-row policy.

    ``skip`` alone is not yet tolerant: ``max_skip_fraction`` defaults to 0.0,
    which voids the aggregate as soon as a single row is skipped. A tolerant
    policy is ``skip`` *plus* a nonzero tolerance, so these tests set one.
    """

    from whetstone.testing.toy.experiment import _reference_procedure

    procedure, _hash = _reference_procedure()
    aggregation = aggregation_definition(
        f"{NAMESPACE}.aggregation"
    ).materialize(
        {
            "reduction": "mean",
            "missing_data": missing_data,
            "max_skip_fraction": max_skip_fraction,
        }
    )
    return derive_eval_split(
        namespace=NAMESPACE,
        dataset_revision=DATASET_REVISION,
        split_role="internal_eval",
        tasks=(_Task("task-a"), _Task("task-b")),
        task_hash_of=_task_hash,
        procedure=procedure,
        aggregation=aggregation,
        num_seeds=NUM_SEEDS,
    )


def _row(
    *,
    task_index: int,
    seed_index: int,
    score: float | None,
    failed: bool = False,
) -> RolloutRowOutput:
    return RolloutRowOutput(
        candidate_id="cand",
        task_id=f"task-{task_index}",
        task_index=task_index,
        seed_index=seed_index,
        row_state=(
            ExecutedRowState.FAILED if failed else ExecutedRowState.SUCCESS
        ),
        trace_steps=(),
        output_text=None if failed else "out",
        score=score,
        failure_code="provider_error" if failed else "",
    )


def _evaluate(
    outputs: tuple[RolloutRowOutput, ...],
    missing_data: str,
    *,
    max_skip_fraction: str = "0.5",
):
    split = _split(missing_data, max_skip_fraction=max_skip_fraction)
    return aggregate_rollout_outputs(
        outputs=outputs,
        task_hashes=split.task_set.task_hashes,
        num_seeds=NUM_SEEDS,
        graph_hash=GRAPH_HASH,
        matrix_plan=split.evaluation_matrix_plan,
        aggregate_name=AGGREGATE_NAME,
    )


def test_one_failed_repeat_agrees_across_the_two_views() -> None:
    """The headline case: a tolerant policy, one lost repeat of one task.

    Task A scores 1.0 on both repeats. Task B scores 1.0 on its present repeat
    and loses the other. Under ``skip`` the lost repeat contributes nothing, so
    both tasks are 1.0 and so is the evaluation.

    The old reduction returned 0.5 for task B (1.0 + 0.0, over ``num_seeds``)
    and an evaluation-level *vector mean* of 0.75, disagreeing with the 1.0 the
    aggregate reported off the very same rows.
    """

    result = _evaluate(
        (
            _row(task_index=0, seed_index=0, score=1.0),
            _row(task_index=0, seed_index=1, score=1.0),
            _row(task_index=1, seed_index=0, score=1.0),
            _row(task_index=1, seed_index=1, score=None, failed=True),
        ),
        "skip",
    )

    assert result.per_task_scores == (1.0, 1.0)
    # The lost repeat is visible as a count below ``num_seeds`` -- the signal
    # the row-completeness weighting downstream needs, and which the old
    # padded count could never produce.
    assert result.per_task_counts == (2, 1)

    aggregate_value = result.aggregate.aggregation_output.value
    assert aggregate_value == pytest.approx(1.0)
    # The vector and the aggregate are two views of one evaluation.
    scored = [value for value in result.per_task_scores if value is not None]
    assert sum(scored) / len(scored) == pytest.approx(aggregate_value)
    # Row accounting independently corroborates the counts.
    assert result.aggregate.rows_present == 3
    assert result.aggregate.rows_failed == 1


def test_all_present_rows_agree() -> None:
    result = _evaluate(
        (
            _row(task_index=0, seed_index=0, score=1.0),
            _row(task_index=0, seed_index=1, score=0.0),
            _row(task_index=1, seed_index=0, score=1.0),
            _row(task_index=1, seed_index=1, score=1.0),
        ),
        "skip",
    )

    assert result.per_task_scores == (0.5, 1.0)
    assert result.per_task_counts == (2, 2)
    assert result.aggregate.aggregation_output.value == pytest.approx(0.75)
    assert result.aggregate.rows_present == 4


def test_a_fully_lost_task_is_unobserved_not_a_hard_task() -> None:
    """Every repeat of task B fails: it has no score, rather than 0.0.

    Reporting 0.0 would enter anchor calibration as a *measured* hard task and
    drag the evaluation mean down with a number no row ever produced. Under
    ``skip`` the evaluation is task A's 1.0 alone.
    """

    result = _evaluate(
        (
            _row(task_index=0, seed_index=0, score=1.0),
            _row(task_index=0, seed_index=1, score=1.0),
            _row(task_index=1, seed_index=0, score=None, failed=True),
            _row(task_index=1, seed_index=1, score=None, failed=True),
        ),
        "skip",
    )

    assert result.per_task_scores == (1.0, None)
    assert result.per_task_counts == (2, 0)
    # The unobserved task is skipped, not averaged in as a zero (which would
    # have reported 0.5).
    assert result.aggregate.aggregation_output.value == pytest.approx(1.0)
    assert result.aggregate.rows_failed == 2


def test_propagate_policy_withholds_partial_tasks_in_both_views() -> None:
    """Under ``propagate`` a lost repeat withholds that task's score.

    The vector says so with ``None`` instead of quietly substituting a
    zero-padded mean, and the count still reports the one row observed.
    """

    result = _evaluate(
        (
            _row(task_index=0, seed_index=0, score=1.0),
            _row(task_index=0, seed_index=1, score=1.0),
            _row(task_index=1, seed_index=0, score=1.0),
            _row(task_index=1, seed_index=1, score=None, failed=True),
        ),
        "propagate",
    )

    assert result.per_task_scores == (1.0, None)
    assert result.per_task_counts == (2, 1)
    # Propagation carries the incompleteness up: the evaluation withholds a
    # value too, rather than reporting a number built on a task it could not
    # score.
    assert result.aggregate.aggregation_output.value is None
    assert result.aggregate.aggregation_output.status.value == "missing_data"


def test_zero_tolerance_skip_voids_the_aggregate_but_not_the_vector() -> None:
    """Tolerance governs the aggregate; the per-task vector still reports.

    ``skip`` with the default 0.0 tolerance means "no lost row is acceptable",
    so the evaluation withholds a value. The per-task vector is unaffected: it
    reports what each task actually measured, and the counts say which task
    lost a row. Consumers read the incompleteness from the counts rather than
    from a score silently pulled toward zero.
    """

    result = _evaluate(
        (
            _row(task_index=0, seed_index=0, score=1.0),
            _row(task_index=0, seed_index=1, score=1.0),
            _row(task_index=1, seed_index=0, score=1.0),
            _row(task_index=1, seed_index=1, score=None, failed=True),
        ),
        "skip",
        max_skip_fraction="0.0",
    )

    assert result.per_task_scores == (1.0, 1.0)
    assert result.per_task_counts == (2, 1)
    assert result.aggregate.aggregation_output.value is None
    assert result.aggregate.aggregation_output.status.value == "missing_data"
