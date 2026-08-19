from __future__ import annotations

from typing import Any

from dr_platform._core.identities import StageKey
from dr_platform.completion.execution import RunCompletionPayload
from dr_platform.execution.stage_completion import StageCompletion, StageSuccessor
from dr_store.content_addressing import ObjectReference, format_object_reference, parse_object_reference

from whetstone.coordination.eval_service import EvalDispatchMode, EvalEngineService, EvalExecutionContext
from whetstone.coordination.harness_run_controller import (
    RUN_LAUNCH_BINDING_PREFIX,
    OptimRunLaunch,
)
from whetstone.coordination.runtime_bootstrap import RegisteredRuntime
from whetstone.coordination.step_request_builder import StepRequestBuilder
from whetstone.core.identity import TypedRef
from whetstone.eval.runtime_engine import _task_id
from whetstone.platform.deferred_intents import (
    load_persisted_deferred_intents,
    persist_deferred_intents,
)
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
    EvalBatch,
    EvalFaninInput,
    EvalRowInput,
    OPTIM_WORK_INPUT_SCHEMA,
    OptimWorkInput,
    load_eval_batch_by_id,
    load_eval_row_input,
    load_run_manifest,
    load_work_input,
    new_batch_id,
    persist_eval_batch,
    persist_eval_fanin_input,
    persist_eval_row_input,
)

OPTIM_WORK_STATE_SCHEMA = "whetstone.optim_work_state"
OPTIM_WORK_STATE_SCHEMA_VERSION = 1
RUN_MEMBER_TERMINAL_BINDING_PREFIX = "whetstone.run_member_terminal:"


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


def _validate_platform_stage_index(
    *,
    stage_index: int,
    expected: int,
    stage_key: str,
) -> None:
    if stage_index != expected:
        raise ValueError(
            f"platform stage_index mismatch for {stage_key}: "
            f"admission payload has {stage_index}, work state has {expected}"
        )


def _require_controller_identity(
    runtime: RegisteredRuntime,
    work_input: OptimWorkInput,
) -> None:
    if work_input.controller_identity_hash != runtime.controller.runtime_hash:
        raise ValueError(
            "work input controller identity does not match bound runtime"
        )


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
            _step_result_binding_key(runtime, work_input.run_id, step_index)
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


def _step_result_binding_key(
    runtime: RegisteredRuntime,
    run_id: str,
    step_index: int,
) -> str:
    return runtime.harness._result_binding_key(run_id, step_index)


def _evict_step_result_binding(
    runtime: RegisteredRuntime,
    *,
    run_id: str,
    step_index: int,
) -> None:
    runtime.store.evict_bindings(
        [_step_result_binding_key(runtime, run_id, step_index)]
    )


def _bind_step_result(
    runtime: RegisteredRuntime,
    *,
    run_id: str,
    step_index: int,
    result_ref: TypedRef,
) -> None:
    runtime.store.bind(
        _step_result_binding_key(runtime, run_id, step_index),
        result_ref.reference,
    )


def _bind_run_member_terminal(
    runtime: RegisteredRuntime,
    *,
    state: OptimWorkState,
    work_state_ref: str,
) -> None:
    run_key = state.work_input.platform_run_key
    work_key = state.work_input.work_key
    if not run_key or not work_key:
        return
    runtime.store.bind(
        f"{RUN_MEMBER_TERMINAL_BINDING_PREFIX}{run_key}:{work_key}",
        parse_object_reference(work_state_ref),
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
    pending_step_result_ref: str,
) -> tuple[tuple[StageSuccessor, ...], str]:
    # Fan-in resolves every unique deferred intent registered on the batch rows.
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
        pending_step_result_ref=pending_step_result_ref,
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
            barrier=True,
        )
    )
    return tuple(successors), work_state_ref


def _platform_deferred_resume_from_batch(
    *,
    batch: EvalBatch,
) -> tuple[tuple[StageSuccessor, ...], str]:
    successors: list[StageSuccessor] = []
    next_index = batch.optim_step_stage_index + 1
    for row_ref in batch.row_input_refs:
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
            input_reference=batch.fanin_input_ref,
            barrier=True,
        )
    )
    return tuple(successors), batch.work_state_ref


def _unique_deferred_intents_from_batch(
    runtime: RegisteredRuntime,
    batch: EvalBatch,
) -> tuple[OptimEvalRequest, ...]:
    seen: set[str] = set()
    intents: list[OptimEvalRequest] = []
    for row_ref in batch.row_input_refs:
        row = load_eval_row_input(runtime.store, row_ref)
        key = EvalEngineService._intent_ref(row.optim_eval_request).content_hash
        if key in seen:
            continue
        seen.add(key)
        intents.append(row.optim_eval_request)
    return tuple(intents)


def _recover_deferred_platform_intents(
    runtime: RegisteredRuntime,
    *,
    state: OptimWorkState,
    result: OptimStepResult,
) -> tuple[OptimEvalRequest, ...]:
    if state.pending_eval_batch_ref is not None:
        batch = load_eval_batch_by_id(runtime.store, state.pending_eval_batch_ref)
        return _unique_deferred_intents_from_batch(runtime, batch)

    return load_persisted_deferred_intents(
        runtime.store,
        run_id=state.work_input.run_id,
        step_index=state.step_index,
    )


def execute_optim_step_sync(
    runtime: RegisteredRuntime,
    *,
    input_reference: str,
    stage_index: int | None = None,
) -> StageCompletion:
    """Run exactly one harness step for a platform member."""
    state = _load_work_state(runtime, input_reference)
    _require_controller_identity(runtime, state.work_input)
    current_stage_index = state.work_input.platform_stage_index
    if stage_index is not None:
        _validate_platform_stage_index(
            stage_index=stage_index,
            expected=current_stage_index,
            stage_key=STAGE_OPTIM_STEP,
        )
    if state.terminal:
        output_ref = _persist_work_state(runtime, state)
        _bind_run_member_terminal(runtime, state=state, work_state_ref=output_ref)
        return StageCompletion(
            output_reference=output_ref,
            successors=(),
        )
    work_input = state.work_input
    if (
        work_input.dispatch_mode is EvalDispatchMode.PLATFORM
        and state.pending_eval_batch_ref is not None
    ):
        batch = load_eval_batch_by_id(runtime.store, state.pending_eval_batch_ref)
        if batch.step_index == state.step_index:
            successors, output_ref = _platform_deferred_resume_from_batch(
                batch=batch,
            )
            return StageCompletion(
                output_reference=output_ref,
                successors=successors,
            )
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

    deferred = runtime.harness.last_deferred_platform_intents
    if (
        work_input.dispatch_mode is EvalDispatchMode.PLATFORM
        and not result.resolved_intents
        and result.status is StepStatus.CONTINUE
        and not deferred
    ):
        deferred = _recover_deferred_platform_intents(
            runtime,
            state=state,
            result=result,
        )
    if (
        work_input.dispatch_mode is EvalDispatchMode.PLATFORM
        and not result.resolved_intents
        and deferred
        and result.status is StepStatus.CONTINUE
    ):
        persist_deferred_intents(
            runtime.store,
            run_id=work_input.run_id,
            step_index=state.step_index,
            intents=deferred,
        )
        pending_step_result_ref = format_object_reference(
            ObjectReference(
                schema=result_ref.schema_name,
                content_hash=result_ref.content_hash,
            )
        )
        successors, output_ref = _platform_deferred_successors(
            runtime,
            state=state,
            deferred_intents=deferred,
            current_stage_index=current_stage_index,
            pending_step_result_ref=pending_step_result_ref,
        )
        _evict_step_result_binding(
            runtime,
            run_id=work_input.run_id,
            step_index=state.step_index,
        )
        return StageCompletion(
            output_reference=output_ref,
            successors=successors,
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
        _bind_run_member_terminal(runtime, state=updated, work_state_ref=output_ref)
        return StageCompletion(
            output_reference=output_ref,
            successors=(),
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
    _require_controller_identity(runtime, state.work_input)
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


def execute_run_completion_for_run_sync(
    runtime: RegisteredRuntime,
    *,
    payload: RunCompletionPayload,
) -> str:
    """Terminalize every member in a released platform run."""
    manifest = load_run_manifest(runtime.store, payload.manifest_reference)
    if manifest.membership_digest != payload.membership_digest:
        raise ValueError("run completion membership digest does not match manifest")
    if manifest.platform_run_key != str(payload.run_key):
        raise ValueError("run completion run_key does not match manifest")
    if len(manifest.members) != payload.member_count:
        raise ValueError("run completion member_count does not match manifest")

    result_refs: list[str] = []
    for member in manifest.members:
        binding = runtime.store.resolve(
            f"{RUN_MEMBER_TERMINAL_BINDING_PREFIX}"
            f"{manifest.platform_run_key}:{member.work_key}"
        )
        if binding is None:
            raise ValueError(
                "run completion requires a terminal work state binding for "
                f"{member.work_key!r}"
            )
        work_state_ref = format_object_reference(binding)
        result_refs.append(
            execute_run_completion_sync(
                runtime,
                input_reference=work_state_ref,
            )
        )

    if len(result_refs) != 1:
        raise ValueError(
            "v1 run completion supports exactly one member; "
            f"got {len(result_refs)}"
        )
    return result_refs[0]


__all__ = [
    "OptimWorkState",
    "RUN_MEMBER_TERMINAL_BINDING_PREFIX",
    "_bind_step_result",
    "_load_work_state",
    "_persist_work_state",
    "_platform_deferred_successors",
    "_require_controller_identity",
    "_validate_platform_stage_index",
    "execute_optim_step_sync",
    "execute_run_completion_for_run_sync",
    "execute_run_completion_sync",
]
