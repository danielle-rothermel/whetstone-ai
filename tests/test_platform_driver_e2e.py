from __future__ import annotations

from dr_store.content_addressing import parse_object_reference

from whetstone.optim.contracts import OPTIM_RESULT_SCHEMA, OptimResult
from whetstone.platform.contracts import (
    STAGE_OPTIM_STEP,
    STAGE_RUN_COMPLETION,
    OptimWorkInput,
    persist_work_input,
)
from whetstone.platform.step_executor import (
    execute_optim_step_sync,
    execute_run_completion_sync,
)


def test_platform_driver_inline_copro_chain(copro_launch) -> None:
    runtime, launch = copro_launch
    control = launch.control
    assert control is not None
    run_id = launch.run.run_id
    work_input = OptimWorkInput(
        run_id=run_id,
        controller_identity_hash=runtime.controller.runtime_hash,
        control_identity_hash=control.identity_hash(),
    )
    input_reference = persist_work_input(runtime.store, work_input)
    stage_index = 0
    current_ref = input_reference

    while True:
        completion = execute_optim_step_sync(
            runtime,
            input_reference=current_ref,
        )
        assert completion.output_reference
        if not completion.successors:
            break
        successor = completion.successors[0]
        assert successor.stage_index == stage_index + 1
        if successor.stage_key.value == STAGE_RUN_COMPLETION:
            terminal_ref = execute_run_completion_sync(
                runtime,
                input_reference=successor.input_reference,
            )
            parsed = parse_object_reference(terminal_ref)
            assert parsed.schema == OPTIM_RESULT_SCHEMA
            result = OptimResult.model_validate(
                runtime.store.get(parsed)
            )
            assert result.run.record.run_id == run_id
            assert len(result.proposals) == 1
            return
        assert successor.stage_key.value == STAGE_OPTIM_STEP
        current_ref = successor.input_reference
        stage_index = successor.stage_index

    raise AssertionError("platform driver chain did not reach run completion")
