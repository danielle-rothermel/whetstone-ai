from __future__ import annotations

from typing import Any

from dr_platform._core.identities import StageKey
from dr_platform.completion.execution import RunCompletionPayload
from dr_platform.execution.stage_completion import StageCompletion, StageSuccessor
from dr_store.content_addressing import (
    ObjectReference,
    format_object_reference,
    parse_object_reference,
)

from whetstone.coordination.eval_service import EvalDispatchMode, EvalExecutionContext
from whetstone.coordination.harness_run_controller import (
    RUN_LAUNCH_BINDING_PREFIX,
    OptimRunLaunch,
)
from whetstone.coordination.runtime_bootstrap import RegisteredRuntime
from whetstone.coordination.step_request_builder import StepRequestBuilder
from whetstone.core.identity import TypedRef
from whetstone.eval.runtime_engine import _task_id
from whetstone.optim.miprov2.engine_binding import engine_for_task_hashes
from whetstone.platform.deferred_intents import (
    evict_deferred_intents,
    load_persisted_deferred_intents,
    persist_deferred_intents,
)
from whetstone.platform.work_state_head import (
    bind_work_state_head,
    resolve_work_state_head,
)
from whetstone.optim.gepa.harness_adapter import GEPA_ADAPTER_KEY
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
    DeferralJoinInput,
    EvalRowInput,
    OptimPlatformRunResult,
    OptimRunMemberResult,
    OptimWorkInput,
    load_run_manifest,
    load_work_input,
    persist_deferral_join_input,
    persist_eval_row_input,
    persist_run_result,
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
        "pending_step_result_ref",
        "deferral_optim_step_stage_index",
        "pending_deferred_intents",
    )

    def __init__(
        self,
        *,
        work_input: Any,
        step_index: int = 0,
        step_result_refs: tuple[OptimStepResultRef, ...] = (),
        terminal: bool = False,
        pending_step_result_ref: str | None = None,
        deferral_optim_step_stage_index: int | None = None,
        pending_deferred_intents: tuple[OptimEvalRequest, ...] = (),
    ) -> None:
        self.work_input = work_input
        self.step_index = step_index
        self.step_result_refs = step_result_refs
        self.terminal = terminal
        self.pending_step_result_ref = pending_step_result_ref
        self.deferral_optim_step_stage_index = deferral_optim_step_stage_index
        self.pending_deferred_intents = pending_deferred_intents


def _load_launch(runtime: RegisteredRuntime, run_id: str) -> OptimRunLaunch:
    return runtime.controller.load_launch(run_id)


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
        raise ValueError("work input controller identity does not match bound runtime")


def _serialize_deferred_intents(
    intents: tuple[OptimEvalRequest, ...],
) -> list[dict[str, object]]:
    return [intent.model_dump(mode="json") for intent in intents]


def _deserialize_deferred_intents(
    payload: object,
) -> tuple[OptimEvalRequest, ...]:
    if not isinstance(payload, list):
        return ()
    return tuple(OptimEvalRequest.model_validate(item) for item in payload)


def _work_state_from_payload(payload: dict[str, Any]) -> OptimWorkState:
    work_input = OptimWorkInput.model_validate(payload["work_input"])
    step_result_refs = tuple(
        OptimStepResultRef.model_validate(ref) for ref in payload["step_result_refs"]
    )
    pending_step = payload.get("pending_step_result_ref")
    deferral_origin = payload.get("deferral_optim_step_stage_index")
    return OptimWorkState(
        work_input=work_input,
        step_index=int(payload["step_index"]),
        step_result_refs=step_result_refs,
        terminal=bool(payload["terminal"]),
        pending_step_result_ref=(None if pending_step is None else str(pending_step)),
        deferral_optim_step_stage_index=(
            None if deferral_origin is None else int(deferral_origin)
        ),
        pending_deferred_intents=_deserialize_deferred_intents(
            payload.get("pending_deferred_intents", ())
        ),
    )


def _accept_work_state_head(
    head: OptimWorkState,
    work_input: OptimWorkInput,
) -> bool:
    if head.work_input.run_id != work_input.run_id:
        return False
    if head.work_input.controller_identity_hash != work_input.controller_identity_hash:
        return False
    if head.work_input.control_identity_hash != work_input.control_identity_hash:
        return False
    if head.work_input.platform_stage_index < work_input.platform_stage_index:
        return False
    if head.pending_step_result_ref is not None:
        return True
    return head.work_input.platform_stage_index == work_input.platform_stage_index


def _load_work_state(
    runtime: RegisteredRuntime,
    input_reference: str,
) -> OptimWorkState:
    parsed = parse_object_reference(input_reference)
    if parsed.schema == OPTIM_WORK_STATE_SCHEMA:
        payload = runtime.store.get(parsed)
        if not isinstance(payload, dict):
            raise ValueError("work state record must be an object")
        return _work_state_from_payload(payload)

    work_input = load_work_input(runtime.store, input_reference)
    run_binding = runtime.store.resolve(
        f"{RUN_LAUNCH_BINDING_PREFIX}{work_input.run_id}"
    )
    if run_binding is None:
        raise ValueError(f"optimization run launch is not bound: {work_input.run_id!r}")
    head_ref = resolve_work_state_head(
        runtime.store,
        run_id=work_input.run_id,
        work_key=work_input.work_key,
    )
    if head_ref is not None:
        head_payload = runtime.store.get(parse_object_reference(head_ref))
        if isinstance(head_payload, dict):
            head_state = _work_state_from_payload(head_payload)
            if _accept_work_state_head(head_state, work_input):
                return head_state
    # step_index becomes the next harness step to run; in-flight deferral binds may
    # belong to step_index (OptimWorkState retry) or step_index - 1 (work-input replay).
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
        "pending_step_result_ref": state.pending_step_result_ref,
        "deferral_optim_step_stage_index": state.deferral_optim_step_stage_index,
        "pending_deferred_intents": _serialize_deferred_intents(
            state.pending_deferred_intents
        ),
        "step_result_refs": [
            ref.model_dump(mode="json") for ref in state.step_result_refs
        ],
        "work_input": state.work_input.record_content(),
    }
    reference, _ = runtime.store.put(OPTIM_WORK_STATE_SCHEMA, payload)
    work_state_ref = format_object_reference(reference)
    bind_work_state_head(
        runtime.store,
        run_id=state.work_input.run_id,
        work_key=state.work_input.work_key,
        work_state_ref=work_state_ref,
    )
    return work_state_ref


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


def _bound_unresolved_deferral_step_index(
    runtime: RegisteredRuntime,
    *,
    run_id: str,
    state: OptimWorkState,
) -> int | None:
    candidates: list[int] = []
    if state.step_index >= 0:
        candidates.append(state.step_index)
    if state.step_index > 0:
        candidates.append(state.step_index - 1)
    for deferral_step_index in candidates:
        binding = runtime.store.resolve(
            _step_result_binding_key(runtime, run_id, deferral_step_index)
        )
        if binding is None:
            continue
        result = OptimStepResult.model_validate(runtime.store.get(binding))
        if result.status is StepStatus.CONTINUE and not result.resolved_intents:
            if load_persisted_deferred_intents(
                runtime.store,
                run_id=run_id,
                step_index=deferral_step_index,
            ):
                return deferral_step_index
    return None


def _recover_crash_window_deferral(
    runtime: RegisteredRuntime,
    *,
    state: OptimWorkState,
    current_stage_index: int,
) -> StageCompletion | None:
    work_input = state.work_input
    if work_input.dispatch_mode is not EvalDispatchMode.PLATFORM:
        return None
    deferral_step_index = _bound_unresolved_deferral_step_index(
        runtime,
        run_id=work_input.run_id,
        state=state,
    )
    if deferral_step_index is None:
        return None
    deferred = load_persisted_deferred_intents(
        runtime.store,
        run_id=work_input.run_id,
        step_index=deferral_step_index,
    )
    if not deferred:
        return None
    binding = runtime.store.resolve(
        _step_result_binding_key(runtime, work_input.run_id, deferral_step_index)
    )
    if binding is None:
        return None
    result = OptimStepResult.model_validate(runtime.store.get(binding))
    result_ref = step_result_reference(result).record_ref
    pending_step_result_ref = format_object_reference(
        ObjectReference(
            schema=result_ref.schema_name,
            content_hash=result_ref.content_hash,
        )
    )
    emit_state = OptimWorkState(
        work_input=state.work_input,
        step_index=deferral_step_index,
        step_result_refs=state.step_result_refs[:deferral_step_index],
        terminal=False,
    )
    persist_deferred_intents(
        runtime.store,
        run_id=work_input.run_id,
        step_index=deferral_step_index,
        intents=deferred,
    )
    successors, output_ref = _platform_deferred_successors(
        runtime,
        state=emit_state,
        deferred_intents=deferred,
        current_stage_index=current_stage_index,
        pending_step_result_ref=pending_step_result_ref,
    )
    _evict_step_result_binding(
        runtime,
        run_id=work_input.run_id,
        step_index=deferral_step_index,
    )
    evict_deferred_intents(
        runtime.store,
        run_id=work_input.run_id,
        step_index=deferral_step_index,
    )
    return StageCompletion(
        output_reference=output_ref,
        successors=successors,
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


def _task_ids_for_intent(
    runtime: RegisteredRuntime,
    intent: OptimEvalRequest,
) -> tuple[str, ...]:
    engine = runtime.eval_service._engine  # noqa: SLF001
    if intent.task_hashes is None:
        return tuple(_task_id(task) for task in engine.sampling.tasks)
    subset = engine_for_task_hashes(engine, intent.task_hashes)
    return tuple(_task_id(task) for task in subset.sampling.tasks)


def _expand_eval_rows(
    runtime: RegisteredRuntime,
    intents: tuple[OptimEvalRequest, ...],
    *,
    deferral_origin_stage_index: int,
    work_state_ref: str,
) -> tuple[EvalRowInput, ...]:
    engine = runtime.eval_service._engine  # noqa: SLF001
    num_seeds = engine.sampling.num_seeds
    rows: list[EvalRowInput] = []
    row_ordinal = 0
    for intent in intents:
        for task_id in _task_ids_for_intent(runtime, intent):
            for seed_index in range(num_seeds):
                rows.append(
                    EvalRowInput(
                        work_state_ref=work_state_ref,
                        deferral_origin_stage_index=deferral_origin_stage_index,
                        row_ordinal=row_ordinal,
                        optim_eval_request=intent,
                        task_id=task_id,
                        seed_index=seed_index,
                    )
                )
                row_ordinal += 1
    return tuple(rows)


def _emit_deferred_successors(
    runtime: RegisteredRuntime,
    *,
    state: OptimWorkState,
    deferred_intents: tuple[OptimEvalRequest, ...],
    current_stage_index: int,
    pending_step_result_ref: str,
    work_state_ref: str,
) -> tuple[tuple[StageSuccessor, ...], str]:
    row_inputs = _expand_eval_rows(
        runtime,
        deferred_intents,
        deferral_origin_stage_index=current_stage_index,
        work_state_ref=work_state_ref,
    )
    row_refs = [
        persist_eval_row_input(runtime.store, row_input) for row_input in row_inputs
    ]
    join_ref = persist_deferral_join_input(
        runtime.store,
        DeferralJoinInput(
            work_state_ref=work_state_ref,
            deferral_optim_step_stage_index=current_stage_index,
            primary_optim_eval_request=deferred_intents[0],
            row_input_refs=tuple(row_refs),
        ),
    )
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
            input_reference=join_ref,
            barrier=True,
        )
    )
    return tuple(successors), work_state_ref


def _deferred_row_count(
    runtime: RegisteredRuntime,
    deferred_intents: tuple[OptimEvalRequest, ...],
) -> int:
    engine = runtime.eval_service._engine  # noqa: SLF001
    num_seeds = engine.sampling.num_seeds
    return sum(
        len(_task_ids_for_intent(runtime, intent)) * num_seeds
        for intent in deferred_intents
    )


def _platform_deferred_successors(
    runtime: RegisteredRuntime,
    *,
    state: OptimWorkState,
    deferred_intents: tuple[OptimEvalRequest, ...],
    current_stage_index: int,
    pending_step_result_ref: str,
) -> tuple[tuple[StageSuccessor, ...], str]:
    row_count = _deferred_row_count(runtime, deferred_intents)
    pending_state = OptimWorkState(
        work_input=_next_work_input(
            state,
            platform_stage_index=current_stage_index + row_count + 1,
        ),
        step_index=state.step_index,
        step_result_refs=state.step_result_refs,
        terminal=False,
        pending_step_result_ref=pending_step_result_ref,
        deferral_optim_step_stage_index=current_stage_index,
        pending_deferred_intents=deferred_intents,
    )
    work_state_ref = _persist_work_state(runtime, pending_state)
    successors, output_ref = _emit_deferred_successors(
        runtime,
        state=state,
        deferred_intents=deferred_intents,
        current_stage_index=current_stage_index,
        pending_step_result_ref=pending_step_result_ref,
        work_state_ref=work_state_ref,
    )
    return successors, output_ref


def _platform_deferred_resume(
    runtime: RegisteredRuntime,
    *,
    state: OptimWorkState,
) -> tuple[tuple[StageSuccessor, ...], str]:
    if state.pending_step_result_ref is None:
        raise ValueError("deferred resume requires a pending step result")
    if state.deferral_optim_step_stage_index is None:
        raise ValueError("deferred resume requires a deferral origin stage index")
    if not state.pending_deferred_intents:
        raise ValueError("deferred resume requires pending deferred intents")
    work_state_ref = _persist_work_state(runtime, state)
    successors, output_ref = _emit_deferred_successors(
        runtime,
        state=state,
        deferred_intents=state.pending_deferred_intents,
        current_stage_index=state.deferral_optim_step_stage_index,
        pending_step_result_ref=state.pending_step_result_ref,
        work_state_ref=work_state_ref,
    )
    return successors, output_ref


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
        expected_stage_index = current_stage_index
        if (
            state.pending_step_result_ref is not None
            and state.deferral_optim_step_stage_index is not None
        ):
            expected_stage_index = state.deferral_optim_step_stage_index
        _validate_platform_stage_index(
            stage_index=stage_index,
            expected=expected_stage_index,
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
        and state.pending_step_result_ref is not None
    ):
        successors, output_ref = _platform_deferred_resume(
            runtime,
            state=state,
        )
        evict_deferred_intents(
            runtime.store,
            run_id=work_input.run_id,
            step_index=state.step_index,
        )
        return StageCompletion(
            output_reference=output_ref,
            successors=successors,
        )
    crash_recovery = _recover_crash_window_deferral(
        runtime,
        state=state,
        current_stage_index=current_stage_index,
    )
    if crash_recovery is not None:
        return crash_recovery
    if work_input.dispatch_mode is EvalDispatchMode.PLATFORM:
        bound_unresolved = _bound_unresolved_deferral_step_index(
            runtime,
            run_id=work_input.run_id,
            state=state,
        )
        if bound_unresolved is not None:
            raise ValueError(
                "bound deferral with persisted intents but recovery failed"
            )
    launch = _load_launch(runtime, work_input.run_id)
    if launch.control is not None:
        if launch.control.identity_hash() != work_input.control_identity_hash:
            raise ValueError("work input control hash does not match launch")
    elif launch.run.optimizer_config.record_hash != work_input.control_identity_hash:
        raise ValueError("work input control hash does not match run config")

    eval_context = EvalExecutionContext(dispatch_mode=work_input.dispatch_mode)
    bound = runtime.harness.bind_run(launch.run)
    adapter_key = bound.record.adapter_key
    control = launch.control
    step_builder = StepRequestBuilder(store=runtime.store)
    adapter = runtime.adapter_registry.resolve(adapter_key)
    bind_evaluation_service = getattr(adapter, "bind_evaluation_service", None)
    if callable(bind_evaluation_service):
        # PLATFORM fan-in completes intents on runtime.eval_service. GEPA
        # must acquire those claims with the same owner, not a fresh
        # EvalEngineService minted inside evaluate().
        bind_evaluation_service(runtime.eval_service)

    extra_pools = None
    if adapter_key == GEPA_ADAPTER_KEY:
        # Same-step resume must not replay the deferred adapter effect.
        extra_pools = {"platform_stage_index": current_stage_index}
    if state.step_index == 0:
        step_request = step_builder.build_first(
            run=bound,
            adapter_key=adapter_key,
            initial_candidate=launch.initial_candidate,
            control=control,
            extra_pools=extra_pools,
        )
    else:
        prior_ref = state.step_result_refs[-1].record_ref
        prior = OptimStepResult.model_validate(runtime.store.get(prior_ref.reference))
        prior_results = tuple(
            OptimStepResult.model_validate(runtime.store.get(ref.record_ref.reference))
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
            extra_pools=extra_pools,
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
        if state.pending_deferred_intents:
            deferred = state.pending_deferred_intents
        else:
            deferred = load_persisted_deferred_intents(
                runtime.store,
                run_id=work_input.run_id,
                step_index=step_request.step_index,
            )
    if (
        work_input.dispatch_mode is EvalDispatchMode.PLATFORM
        and not result.resolved_intents
        and deferred
        and result.status is StepStatus.CONTINUE
    ):
        deferral_step_index = step_request.step_index
        persist_deferred_intents(
            runtime.store,
            run_id=work_input.run_id,
            step_index=deferral_step_index,
            intents=deferred,
        )
        pending_step_result_ref = format_object_reference(
            ObjectReference(
                schema=result_ref.schema_name,
                content_hash=result_ref.content_hash,
            )
        )
        emit_state = OptimWorkState(
            work_input=state.work_input,
            step_index=deferral_step_index,
            step_result_refs=state.step_result_refs[:deferral_step_index],
            terminal=False,
        )
        successors, output_ref = _platform_deferred_successors(
            runtime,
            state=emit_state,
            deferred_intents=deferred,
            current_stage_index=current_stage_index,
            pending_step_result_ref=pending_step_result_ref,
        )
        _evict_step_result_binding(
            runtime,
            run_id=work_input.run_id,
            step_index=deferral_step_index,
        )
        evict_deferred_intents(
            runtime.store,
            run_id=work_input.run_id,
            step_index=deferral_step_index,
        )
        return StageCompletion(
            output_reference=output_ref,
            successors=successors,
        )

    _bind_step_result(
        runtime,
        run_id=work_input.run_id,
        step_index=step_request.step_index,
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

    run_result = OptimPlatformRunResult(
        platform_run_key=manifest.platform_run_key,
        membership_digest=manifest.membership_digest,
        member_results=tuple(
            OptimRunMemberResult(
                work_key=member.work_key,
                run_id=member.run_id,
                result_reference=result_ref,
            )
            for member, result_ref in zip(
                manifest.members,
                result_refs,
                strict=True,
            )
        ),
    )
    return persist_run_result(runtime.store, run_result)


__all__ = [
    "OptimWorkState",
    "RUN_MEMBER_TERMINAL_BINDING_PREFIX",
    "_bind_step_result",
    "_load_work_state",
    "_persist_work_state",
    "_deferred_row_count",
    "_platform_deferred_successors",
    "_task_ids_for_intent",
    "_require_controller_identity",
    "_validate_platform_stage_index",
    "execute_optim_step_sync",
    "execute_run_completion_for_run_sync",
    "execute_run_completion_sync",
]
