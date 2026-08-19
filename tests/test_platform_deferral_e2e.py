from __future__ import annotations

from dr_store.content_addressing import parse_object_reference

from whetstone.coordination.eval_service import EvalDispatchMode
from whetstone.platform.contracts import (
    STAGE_EVAL_FANIN,
    STAGE_EVAL_ROW,
    STAGE_OPTIM_STEP,
    OptimWorkInput,
    persist_work_input,
)
from whetstone.platform.eval_fanin import execute_eval_fanin_sync, execute_eval_row_sync
from whetstone.platform.step_executor import (
    execute_optim_step_sync,
    execute_run_completion_sync,
)


def test_platform_deferral_fanout_fanin_resume(copro_launch) -> None:
    runtime, launch = copro_launch

    control = launch.control
    assert control is not None
    runtime.controller.bind_launch(launch)
    work_input = OptimWorkInput(
        run_id=launch.run.run_id,
        controller_identity_hash=runtime.controller.runtime_hash,
        control_identity_hash=control.identity_hash(),
        dispatch_mode=EvalDispatchMode.PLATFORM,
    )
    current_ref = persist_work_input(runtime.store, work_input)

    completion = execute_optim_step_sync(
        runtime,
        input_reference=current_ref,
        stage_index=0,
    )
    row_successors = [
        successor
        for successor in completion.successors
        if successor.stage_key.value == STAGE_EVAL_ROW
    ]
    fanin_successors = [
        successor
        for successor in completion.successors
        if successor.stage_key.value == STAGE_EVAL_FANIN
    ]
    assert row_successors
    assert len(fanin_successors) == 1
    assert fanin_successors[0].barrier is True

    for row_successor in row_successors:
        row_completion = execute_eval_row_sync(
            runtime,
            input_reference=row_successor.input_reference,
            stage_index=row_successor.stage_index,
        )
        assert row_completion.output_reference

    fanin_completion = execute_eval_fanin_sync(
        runtime,
        input_reference=fanin_successors[0].input_reference,
        stage_index=fanin_successors[0].stage_index,
    )
    assert fanin_completion.successors
    successor = fanin_completion.successors[0]
    assert successor.stage_key.value == STAGE_OPTIM_STEP

    resumed = execute_optim_step_sync(
        runtime,
        input_reference=successor.input_reference,
        stage_index=successor.stage_index,
    )
    assert resumed.output_reference
    assert parse_object_reference(resumed.output_reference)
    assert not resumed.successors

    terminal_ref = execute_run_completion_sync(
        runtime,
        input_reference=resumed.output_reference,
    )
    assert parse_object_reference(terminal_ref)


def test_platform_deferral_assembles_row_evidence(copro_launch) -> None:
    runtime, launch = copro_launch
    from whetstone.platform.eval_fanin import build_platform_row_executor

    control = launch.control
    assert control is not None
    runtime.controller.bind_launch(launch)
    work_input = OptimWorkInput(
        run_id=launch.run.run_id,
        controller_identity_hash=runtime.controller.runtime_hash,
        control_identity_hash=control.identity_hash(),
        dispatch_mode=EvalDispatchMode.PLATFORM,
    )
    current_ref = persist_work_input(runtime.store, work_input)
    completion = execute_optim_step_sync(
        runtime,
        input_reference=current_ref,
        stage_index=0,
    )
    row_successors = [
        successor
        for successor in completion.successors
        if successor.stage_key.value == STAGE_EVAL_ROW
    ]
    fanin_successors = [
        successor
        for successor in completion.successors
        if successor.stage_key.value == STAGE_EVAL_FANIN
    ]
    row_executor = build_platform_row_executor(runtime)
    for row_successor in row_successors:
        execute_eval_row_sync(
            runtime,
            input_reference=row_successor.input_reference,
            stage_index=row_successor.stage_index,
            row_executor=row_executor,
        )
    execute_eval_fanin_sync(
        runtime,
        input_reference=fanin_successors[0].input_reference,
        stage_index=fanin_successors[0].stage_index,
    )
