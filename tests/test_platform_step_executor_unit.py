from __future__ import annotations

import pytest

from whetstone.coordination.eval_service import EvalDispatchMode
from whetstone.platform.contracts import OptimWorkInput, persist_work_input
from whetstone.platform.step_executor import (
    STAGE_EVAL_FANIN,
    STAGE_EVAL_ROW,
    OptimWorkState,
    _load_work_state,
    execute_optim_step_sync,
)


def test_platform_deferral_recovers_when_cached_step_clears_deferred(
    copro_launch,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    input_reference = persist_work_input(runtime.store, work_input)

    first = execute_optim_step_sync(
        runtime,
        input_reference=input_reference,
        stage_index=0,
    )
    assert any(
        successor.stage_key.value == STAGE_EVAL_ROW for successor in first.successors
    )
    pending_state = _load_work_state(runtime, first.output_reference)
    saved_intents = pending_state.pending_deferred_intents
    assert saved_intents

    real_run_step = runtime.harness.run_step

    def cached_run_step(step_request, *, eval_context=None):
        result, result_ref = real_run_step(step_request, eval_context=eval_context)
        runtime.harness._last_deferred_platform_intents = ()  # noqa: SLF001
        return result, result_ref

    original_load = _load_work_state

    def load_with_saved_intents(runtime_arg, ref: str):
        state = original_load(runtime_arg, ref)
        if ref == input_reference and not state.pending_deferred_intents:
            return OptimWorkState(
                work_input=state.work_input,
                step_index=state.step_index,
                step_result_refs=state.step_result_refs,
                terminal=state.terminal,
                pending_step_result_ref=state.pending_step_result_ref,
                deferral_optim_step_stage_index=state.deferral_optim_step_stage_index,
                pending_deferred_intents=saved_intents,
            )
        return state

    monkeypatch.setattr(runtime.harness, "run_step", cached_run_step)
    monkeypatch.setattr(
        "whetstone.platform.step_executor._load_work_state",
        load_with_saved_intents,
    )

    recovered = execute_optim_step_sync(
        runtime,
        input_reference=input_reference,
        stage_index=0,
    )
    row_successors = [
        successor
        for successor in recovered.successors
        if successor.stage_key.value == STAGE_EVAL_ROW
    ]
    fanin_successors = [
        successor
        for successor in recovered.successors
        if successor.stage_key.value == STAGE_EVAL_FANIN
    ]
    assert row_successors
    assert len(fanin_successors) == 1
