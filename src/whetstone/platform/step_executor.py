from __future__ import annotations

from typing import Any

from dr_platform._core.identities import StageKey
from dr_platform.execution.stage_completion import StageCompletion, StageSuccessor
from dr_store.content_addressing import ObjectReference, format_object_reference, parse_object_reference

from whetstone.coordination.eval_service import EvalDispatchMode, EvalExecutionContext
from whetstone.coordination.harness_run_controller import (
    RUN_LAUNCH_BINDING_PREFIX,
    OptimRunLaunch,
)
from whetstone.coordination.runtime_bootstrap import RegisteredRuntime
from whetstone.coordination.step_request_builder import StepRequestBuilder
from whetstone.core.identity import TypedRef
from whetstone.eval.runtime_engine import _task_id
from whetstone.optim.contracts import (
    OPTIM_RESULT_SCHEMA,
    OptimEvalRequest,
    OptimStepResult,
    OptimStepResultRef,
    StepStatus,
    step_result_reference,
)
from whetstone.platform.contracts import (
    STAGE_EVAL_FANIN,
    STAGE_EVAL_ROW,
    STAGE_OPTIM_STEP,
    STAGE_RUN_COMPLETION,
    EvalBatch,
    EvalFaninInput,
    EvalRowInput,
    OPTIM_WORK_INPUT_SCHEMA,
    load_work_input,
    new_batch_id,
    persist_eval_batch,
    persist_eval_fanin_input,
    persist_eval_row_input,
)

OPTIM_WORK_STATE_SCHEMA = "whetstone.optim_work_state"
OPTIM_WORK_STATE_SCHEMA_VERSION = 1
STEP_RESULT_BINDING_PREFIX = "whetstone.optim_step_result:"


class OptimWorkState:
    """Mutable harness progress for one platform member."""

    __slots__ = (
        "work_input",
        "step_index",
        "step_result_refs",
        "terminal",
        "pending_eval_batch_ref",
    )

    def __init__(
        self,
        *,
        work_input: Any,
        step_index: int = 0,
        step_result_refs: tuple[OptimStepResultRef, ...] = (),
        terminal: bool = False,
        pending_eval_batch_ref: str | None = None,
    ) -> None:
        self.work_input = work_input
        self.step_index = step_index
        self.step_result_refs = step_result_refs
        self.terminal = terminal
        self.pending_eval_batch_ref = pending_eval_batch_ref


def _load_launch(runtime: RegisteredRuntime, run_id: str) -> OptimRunLaunch:
    return runtime.controller._load_launch(run_id)  # noqa: SLF001


def _load_work_state(
    runtime: RegisteredRuntime,
    input_reference: str,
) -> OptimWorkState:
    parsed = parse_object_reference(input_reference)
    if parsed.schema == OPTIM_WORK_STATE_SCHEMA:
        payload = runtime.store.get(parsed)
        if not isinstance(payload, dict):
            raise ValueError("work state record must be an object")
        from whetstone.platform.contracts import OptimWorkInput

        work_input = OptimWorkInput.model_validate(payload["work_input"])
        step_result_refs = tuple(
            OptimStepResultRef.model_validate(ref)
            for ref in payload["step_result_refs"]
        )
        pending = payload.get("pending_eval_batch_ref")
        return OptimWorkState(
            work_input=work_input,
            step_index=int(payload["step_index"]),
            step_result_refs=step_result_refs,
            terminal=bool(payload["terminal"]),
            pending_eval_batch_ref=(
                None if pending is None else str(pending)
            ),
        )

    work_input = load_work_input(runtime.store, input_reference)
    run_binding = runtime.store.resolve(
        f"{RUN_LAUNCH_BINDING_PREFIX}{work_input.run_id}"
    )
    if run_binding is None:
        raise ValueError(
            f"optimization run launch is not bound: {work_input.run_id!r}"
        )
    step_index = 0
    step_result_refs: list[OptimStepResultRef] = []
    while True:
        binding = runtime.store.resolve(
            f"{STEP_RESULT_BINDING_PREFIX}{work_input.run_id}:{step_index}"
        )
        if binding is None:
            break
        result = OptimStepResult.model_validate(runtime.store.get(binding))
        step_result_refs.append(step_result_reference(result))
        if result.status is not StepStatus.CONTINUE:
            return OptimWorkState(
                work_input=work_input,
                step_index=step_index + 1,
                step_result_refs=tuple(step_result_refs),
                terminal=True,
            )
        step_index += 1
    return OptimWorkState(
        work_input=work_input,
        step_index=step_index,
        step_result_refs=tuple(step_result_refs),
        terminal=False,
    )


def _persist_work_state(
    runtime: RegisteredRuntime,
    state: OptimWorkState,
) -> str:
    payload: dict[str, Any] = {
        "schema_version": OPTIM_WORK_STATE_SCHEMA_VERSION,
        "run_id": state.work_input.run_id,
        "step_index": state.step_index,
        "terminal": state.terminal,
        "pending_eval_batch_ref": state.pending_eval_batch_ref,
        "step_result_refs": [
            ref.model_dump(mode="json")
            for ref in state.step_result_refs
        ],
        "work_input": state.work_input.record_content(),
    }
    reference, _ = runtime.store.put(OPTIM_WORK_STATE_SCHEMA, payload)
    return format_object_reference(reference)


def _bind_step_result(
    runtime: RegisteredRuntime,
    *,
    run_id: str,
    step_index: int,
    result_ref: TypedRef,
) -> None:
    runtime.store.bind(
        f"{STEP_RESULT_BINDING_PREFIX}{run_id}:{step_index}",
        result_ref.reference,
    )


def _next_work_input(state: OptimWorkState, *, platform_stage_index: int) -> Any:
    return state.work_input.model_copy(
        update={"platform_stage_index": platform_stage_index},
    )


def _expand_eval_rows(
    runtime: RegisteredRuntime,
    intents: tuple[OptimEvalRequest, ...],
    *,
    batch_id: str,
) -> tuple[EvalRowInput, ...]:
    engine = runtime.eval_service._engine  # noqa: SLF001
    sampling = engine.sampling
    task_ids = tuple(_task_id(task) for task in sampling.tasks)
    num_seeds = sampling.num_seeds
    rows: list[EvalRowInput] = []
    for intent in intents:
        for task_id in task_ids:
            for seed_index in range(num_seeds):
                rows.append(
                    EvalRowInput(
                        batch_id=batch_id,
                        optim_eval_request=intent,
                        task_id=task_id,
                        seed_index=seed_index,
                    )
                )
    return tuple(rows)


def _platform_deferred_successors(
    runtime: RegisteredRuntime,
    *,
    state: OptimWorkState,
    deferred_intents: tuple[OptimEvalRequest, ...],
    current_stage_index: int,
) -> tuple[StageSuccessor, ...]:
    batch_id = new_batch_id()
    row_inputs = _expand_eval_rows(
        runtime,
        deferred_intents,
        batch_id=batch_id,
    )
    row_refs = [
        persist_eval_row_input(runtime.store, row_input)
        for row_input in row_inputs
    ]
    primary_intent = deferred_intents[0]
    fanin_input = EvalFaninInput(
        batch_id=batch_id,
        optim_eval_request=primary_intent,
    )
    fanin_ref = persist_eval_fanin_input(runtime.store, fanin_input)
    pending_state = OptimWorkState(
        work_input=_next_work_input(
            state,
            platform_stage_index=current_stage_index + len(row_refs) + 1,
        ),
        step_index=state.step_index,
        step_result_refs=state.step_result_refs,
        terminal=False,
        pending_eval_batch_ref=batch_id,
    )
    work_state_ref = _persist_work_state(runtime, pending_state)
    batch = EvalBatch(
        batch_id=batch_id,
        run_id=state.work_input.run_id,
        step_index=state.step_index,
        optim_step_stage_index=current_stage_index,
        row_input_refs=tuple(row_refs),
        fanin_input_ref=fanin_ref,
        work_state_ref=work_state_ref,
    )
    batch_ref = persist_eval_batch(runtime.store, batch)
    _ = batch_ref
    successors: list[StageSuccessor] = []
    next_index = current_stage_index + 1
    for row_ref in row_refs:
        successors.append(
            StageSuccessor(
                stage_key=StageKey(STAGE_EVAL_ROW),
                stage_index=next_index,
                input_reference=row_ref,
            )
        )
        next_index += 1
    successors.append(
        StageSuccessor(
            stage_key=StageKey(STAGE_EVAL_FANIN),
            stage_index=next_index,
            input_reference=fanin_ref,
        )
    )
    return tuple(successors)


def execute_optim_step_sync(
    runtime: RegisteredRuntime,
    *,
    input_reference: str,
) -> StageCompletion:
    """Run exactly one harness step for a platform member."""
    state = _load_work_state(runtime, input_reference)
    current_stage_index = state.work_input.platform_stage_index
    if state.terminal:
        output_ref = _persist_work_state(runtime, state)
        return StageCompletion(
            output_reference=output_ref,
            successors=(),
        )
    work_input = state.work_input
    launch = _load_launch(runtime, work_input.run_id)
    if launch.control is not None:
        if launch.control.identity_hash() != work_input.control_identity_hash:
            raise ValueError("work input control hash does not match launch")
    elif (
        launch.run.optimizer_config.record_hash
        != work_input.control_identity_hash
    ):
        raise ValueError("work input control hash does not match run config")

    eval_context = EvalExecutionContext(dispatch_mode=work_input.dispatch_mode)
    bound = runtime.harness.bind_run(launch.run)
    adapter_key = bound.record.adapter_key
    control = launch.control
    step_builder = StepRequestBuilder(store=runtime.store)

    if state.step_index == 0:
        step_request = step_builder.build_first(
            run=bound,
            adapter_key=adapter_key,
            initial_candidate=launch.initial_candidate,
            control=control,
        )
    else:
        prior_ref = state.step_result_refs[-1].record_ref
        prior = OptimStepResult.model_validate(
            runtime.store.get(prior_ref.reference)
        )
        prior_results = tuple(
            OptimStepResult.model_validate(
                runtime.store.get(ref.record_ref.reference)
            )
            for ref in state.step_result_refs
        )
        if control is None:
            raise ValueError("continuing run requires a bound optimizer control")
        step_request = step_builder.build_next(
            prior=prior,
            prior_ref=prior_ref,
            prior_results=prior_results,
            control=control,
            mutation_field=str(bound.record.mutation_field),
        )

    result, result_ref = runtime.harness.run_step(
        step_request,
        eval_context=eval_context,
    )
    _bind_step_result(
        runtime,
        run_id=work_input.run_id,
        step_index=state.step_index,
        result_ref=result_ref,
    )
    updated = OptimWorkState(
        work_input=_next_work_input(
            state,
            platform_stage_index=current_stage_index + 1,
        ),
        step_index=state.step_index + 1,
        step_result_refs=state.step_result_refs + (step_result_reference(result),),
        terminal=result.status is not StepStatus.CONTINUE,
    )
    output_ref = _persist_work_state(runtime, updated)

    if updated.terminal:
        return StageCompletion(
            output_reference=output_ref,
            successors=(),
        )

    deferred = runtime.harness.last_deferred_platform_intents
    if (
        work_input.dispatch_mode is EvalDispatchMode.PLATFORM
        and not result.resolved_intents
        and deferred
    ):
        return StageCompletion(
            output_reference=output_ref,
            successors=_platform_deferred_successors(
                runtime,
                state=updated,
                deferred_intents=deferred,
                current_stage_index=current_stage_index,
            ),
        )

    return StageCompletion(
        output_reference=output_ref,
        successors=(
            StageSuccessor(
                stage_key=StageKey(STAGE_OPTIM_STEP),
                stage_index=current_stage_index + 1,
                input_reference=output_ref,
            ),
        ),
    )


def execute_run_completion_sync(
    runtime: RegisteredRuntime,
    *,
    input_reference: str,
) -> str:
    """Terminalize a completed harness loop and return the OptimResult ref."""
    state = _load_work_state(runtime, input_reference)
    if not state.terminal:
        raise ValueError("run completion requires a terminal harness loop")
    if not state.step_result_refs:
        raise ValueError("run completion requires at least one step result")
    launch = _load_launch(runtime, state.work_input.run_id)
    bound = runtime.harness.bind_run(launch.run)
    _terminal, terminal_ref = runtime.harness.terminalize(
        run=bound,
        step_results=tuple(state.step_result_refs),
    )
    if terminal_ref.schema_name != OPTIM_RESULT_SCHEMA:
        raise ValueError("terminalize must return an Optimization Result ref")
    return format_object_reference(
        ObjectReference(
            schema=terminal_ref.schema_name,
            content_hash=terminal_ref.content_hash,
        )
    )


__all__ = [
    "OptimWorkState",
    "STEP_RESULT_BINDING_PREFIX",
    "execute_optim_step_sync",
    "execute_run_completion_sync",
]
