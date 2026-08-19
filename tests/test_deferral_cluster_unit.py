from __future__ import annotations

from unittest.mock import MagicMock, patch

from dr_platform._core.identities import StageKey
from dr_platform._core.ledger.states import StageExecutionState
from dr_platform.inspection.work_items import PredecessorStageOutput

from whetstone.platform.contracts import STAGE_EVAL_FANIN, STAGE_EVAL_ROW, STAGE_OPTIM_STEP
from whetstone.platform.deferral_cluster import (
    deferring_optim_step_index,
    list_episode_eval_row_predecessors,
    optim_step_predecessor_output,
)


def _execution(*, stage_key: str, stage_index: int, output_reference: str | None = "out"):
    execution = MagicMock()
    execution.stage_key = StageKey(stage_key)
    execution.stage_index = stage_index
    execution.state = StageExecutionState.SUCCEEDED
    execution.output_reference = output_reference
    return execution


def _summary(execution: MagicMock) -> MagicMock:
    summary = MagicMock()
    summary.execution = execution
    return summary


def test_deferring_optim_step_index_returns_max_below_fanin() -> None:
    engine = MagicMock()
    stages = (
        _summary(_execution(stage_key=STAGE_OPTIM_STEP, stage_index=0, output_reference="ws-0")),
        _summary(_execution(stage_key=STAGE_EVAL_ROW, stage_index=1)),
        _summary(_execution(stage_key=STAGE_EVAL_FANIN, stage_index=2)),
        _summary(_execution(stage_key=STAGE_OPTIM_STEP, stage_index=3, output_reference="ws-3")),
        _summary(_execution(stage_key=STAGE_EVAL_ROW, stage_index=4)),
    )
    with patch(
        "whetstone.platform.deferral_cluster.get_work_item_stages",
        return_value=stages,
    ):
        assert deferring_optim_step_index(1, 5, engine=engine) == 3


def test_optim_step_predecessor_output_returns_matching_output() -> None:
    engine = MagicMock()
    stages = (
        _summary(_execution(stage_key=STAGE_OPTIM_STEP, stage_index=0, output_reference="ws-0")),
        _summary(_execution(stage_key=STAGE_EVAL_ROW, stage_index=1)),
    )
    with patch(
        "whetstone.platform.deferral_cluster.get_work_item_stages",
        return_value=stages,
    ):
        assert optim_step_predecessor_output(1, 2, engine=engine) == "ws-0"


def test_list_episode_eval_row_predecessors_filters_by_origin() -> None:
    engine = MagicMock()
    predecessors = (
        PredecessorStageOutput(
            stage_index=1,
            stage_key=StageKey(STAGE_EVAL_ROW),
            input_reference="row-in-1",
            output_reference="row-out-1",
        ),
    )
    with patch(
        "whetstone.platform.deferral_cluster.list_predecessor_stage_outputs",
        return_value=predecessors,
    ) as list_predecessors:
        result = list_episode_eval_row_predecessors(
            7,
            deferral_origin=0,
            fanin_stage_index=2,
            engine=engine,
        )
    assert result == predecessors
    list_predecessors.assert_called_once_with(
        7,
        2,
        engine=engine,
        stage_key=STAGE_EVAL_ROW,
        min_stage_index=0,
    )
