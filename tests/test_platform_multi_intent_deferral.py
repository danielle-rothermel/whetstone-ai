from __future__ import annotations

from uuid import uuid4

from dr_store.content_addressing import parse_object_reference

from whetstone.coordination.eval_service import EvalDispatchMode
from whetstone.testing.runtime import (
    build_toy_copro_control,
    prepare_toy_copro_run,
    register_toy_runtime,
)
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.optim.contracts import OptimStepResult
from whetstone.platform.contracts import (
    STAGE_EVAL_FANIN,
    STAGE_EVAL_ROW,
    OptimWorkInput,
    persist_work_input,
)
from whetstone.platform.eval_fanin import execute_eval_fanin_sync, execute_eval_row_sync
from whetstone.platform.step_executor import execute_optim_step_sync


def test_platform_multi_intent_deferral_merges_all_resolutions(sqlite_store) -> None:
    eval_engine = ReferenceEvalRuntimeConfig().build_engine(sqlite_store)
    control = build_toy_copro_control(breadth=3, depth=1, engine=eval_engine)
    runtime = register_toy_runtime(
        store=sqlite_store,
        engine=eval_engine,
        copro_control=control,
    )
    launch = prepare_toy_copro_run(
        runtime,
        run_id=f"multi-intent-{uuid4().hex[:8]}",
        control=control,
        terminal_top_k=1,
    )

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

    internal_task_count = 2
    seed_count = 1
    deferred_intent_count = len(row_successors) // (internal_task_count * seed_count)
    assert deferred_intent_count >= 2

    for row_successor in row_successors:
        execute_eval_row_sync(
            runtime,
            input_reference=row_successor.input_reference,
            stage_index=row_successor.stage_index,
        )

    fanin_completion = execute_eval_fanin_sync(
        runtime,
        input_reference=fanin_successors[0].input_reference,
        stage_index=fanin_successors[0].stage_index,
    )
    fanin_record = runtime.store.get(
        parse_object_reference(fanin_completion.output_reference)
    )
    assert len(fanin_record["resolutions"]) == deferred_intent_count

    resumed = execute_optim_step_sync(
        runtime,
        input_reference=fanin_completion.successors[0].input_reference,
        stage_index=fanin_completion.successors[0].stage_index,
    )
    assert not resumed.successors

    binding = runtime.store.resolve(
        runtime.harness._result_binding_key(launch.run.run_id, 0)  # noqa: SLF001
    )
    assert binding is not None
    step_result = OptimStepResult.model_validate(runtime.store.get(binding))
    assert len(step_result.resolved_intents) == deferred_intent_count
