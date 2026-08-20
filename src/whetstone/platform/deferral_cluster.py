from __future__ import annotations

from typing import TYPE_CHECKING

from dr_platform._core.ledger.states import StageExecutionState
from dr_platform.inspection.work_items import (
    PredecessorStageOutput,
    get_work_item_stages,
    list_predecessor_stage_outputs,
)

from whetstone.platform.contracts import STAGE_EVAL_ROW, STAGE_OPTIM_STEP

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine


def deferring_optim_step_index(
    work_item_id: int,
    fanin_stage_index: int,
    *,
    engine: Engine,
) -> int:
    """Return the ledger index of the optim_step that triggered this fan-in."""
    stages = get_work_item_stages(work_item_id, engine=engine)
    optim_step_indexes = [
        summary.execution.stage_index
        for summary in stages
        if summary.execution.stage_key.value == STAGE_OPTIM_STEP
        and summary.execution.stage_index < fanin_stage_index
        and summary.execution.state is StageExecutionState.SUCCEEDED
    ]
    if not optim_step_indexes:
        raise ValueError(
            "deferral episode is missing a succeeded optim_step predecessor"
        )
    return max(optim_step_indexes)


def optim_step_predecessor_output(
    work_item_id: int,
    fanin_stage_index: int,
    *,
    engine: Engine,
) -> str:
    """Return the output_reference of the deferring optim_step stage."""
    origin = deferring_optim_step_index(
        work_item_id,
        fanin_stage_index,
        engine=engine,
    )
    stages = get_work_item_stages(work_item_id, engine=engine)
    for summary in stages:
        execution = summary.execution
        if (
            execution.stage_key.value == STAGE_OPTIM_STEP
            and execution.stage_index == origin
        ):
            if execution.output_reference is None:
                raise ValueError(
                    "deferring optim_step predecessor is missing an output reference"
                )
            return execution.output_reference
    raise ValueError(
        f"deferring optim_step predecessor not found at stage_index={origin}"
    )


def list_episode_eval_row_predecessors(
    work_item_id: int,
    *,
    deferral_origin: int,
    fanin_stage_index: int,
    engine: Engine,
) -> tuple[PredecessorStageOutput, ...]:
    """Return eval_row predecessor outputs for one deferral episode."""
    return list_predecessor_stage_outputs(
        work_item_id,
        fanin_stage_index,
        engine=engine,
        stage_key=STAGE_EVAL_ROW,
        min_stage_index=deferral_origin,
    )


__all__ = [
    "deferring_optim_step_index",
    "list_episode_eval_row_predecessors",
    "optim_step_predecessor_output",
]
