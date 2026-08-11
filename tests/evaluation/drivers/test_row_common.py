from __future__ import annotations

import pytest
from whetstone_envs.core import Instance

from whetstone.evaluation.aggregate import RowValue, TaskRows
from whetstone.evaluation.drivers.eval_result import (
    per_task_count,
    per_task_score,
)
from whetstone.evaluation.drivers.row_common import (
    ProcessTask,
    RolloutOutput,
    process_request_hash,
    start_phase_deadline,
)
from whetstone.evaluation.traces import ExecutedRowState


def test_process_task_round_trips_instance() -> None:
    instance = Instance(
        id="task-1",
        seed=7,
        strata=("code_comp",),
        prompt_inputs={"input_code": "def f(): pass"},
        gold="def f():\n    pass\n",
    )
    wire = ProcessTask.from_instance(instance)
    restored = wire.to_instance()
    assert restored == instance
    assert process_request_hash(wire) == process_request_hash(
        ProcessTask.from_instance(restored)
    )


def test_rollout_output_row_state_properties() -> None:
    success = RolloutOutput(
        candidate_id="c1",
        task_id="task-1",
        task_index=0,
        sample_index=0,
        row_state=ExecutedRowState.SUCCESS,
        executed_component_steps=(),
        output_text="ok",
        score=1.0,
    )
    failed = RolloutOutput(
        candidate_id="c1",
        task_id="task-1",
        task_index=0,
        sample_index=0,
        row_state=ExecutedRowState.FAILED,
        executed_component_steps=(),
        output_text=None,
        score=None,
    )
    missing = RolloutOutput(
        candidate_id="c1",
        task_id="task-1",
        task_index=0,
        sample_index=0,
        row_state=ExecutedRowState.MISSING,
        executed_component_steps=(),
        output_text=None,
        score=None,
    )

    assert success.failed is False
    assert success.missing is False
    assert success.invalid is False
    assert failed.failed is True
    assert missing.missing is True


def test_per_task_score_and_count_use_completed_rows_only() -> None:
    task = TaskRows(
        task_hash="task-1",
        rows=(
            RowValue(value=1.0),
            RowValue(failed=True),
            RowValue(missing=True),
        ),
    )
    assert per_task_score(task, num_samples=3) == pytest.approx(1 / 3)
    assert per_task_count(task, num_samples=3) == 3


def test_start_phase_deadline_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="finite nonnegative"):
        start_phase_deadline(float("nan"))
    with pytest.raises(ValueError, match="finite nonnegative"):
        start_phase_deadline(-1.0)
