from __future__ import annotations

from typing import Any

from dr_store.content_addressing import ObjectReference, format_object_reference

from whetstone.coordination.harness_run_controller import (
    RUN_LAUNCH_BINDING_PREFIX,
    OptimRunLaunch,
)
from whetstone.coordination.runtime_bootstrap import RegisteredRuntime
from whetstone.coordination.step_request_builder import StepRequestBuilder
from whetstone.core.identity import TypedRef
from whetstone.optim.contracts import (
    OPTIM_RESULT_SCHEMA,
    OptimStepResult,
    OptimStepResultRef,
    StepStatus,
    step_result_reference,
)
from whetstone.coordination.eval_service import EvalDispatchMode
from whetstone.platform.contracts import load_work_input

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
    )

    def __init__(
        self,
        *,
        work_input: Any,
        step_index: int = 0,
        step_result_refs: tuple[OptimStepResultRef, ...] = (),
        terminal: bool = False,
    ) -> None:
        self.work_input = work_input
        self.step_index = step_index
        self.step_result_refs = step_result_refs
        self.terminal = terminal


def _load_launch(runtime: RegisteredRuntime, run_id: str) -> OptimRunLaunch:
    return runtime.controller._load_launch(run_id)  # noqa: SLF001


def _load_work_state(
    runtime: RegisteredRuntime,
    input_reference: str,
) -> OptimWorkState:
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
        "step_result_refs": [
            ref.record_ref.model_dump(mode="json")
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


def execute_optim_step_sync(
    runtime: RegisteredRuntime,
    *,
    input_reference: str,
) -> str:
    """Run exactly one harness step for a platform member."""
    state = _load_work_state(runtime, input_reference)
    if state.terminal:
        return _persist_work_state(runtime, state)
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

    previous_dispatch = runtime.eval_service.set_dispatch_mode(
        work_input.dispatch_mode
    )
    try:
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

        result, result_ref = runtime.harness.run_step(step_request)
        if (
            work_input.dispatch_mode is EvalDispatchMode.PLATFORM
            and result.status is StepStatus.CONTINUE
            and not result.resolved_intents
        ):
            raise ValueError(
                "PLATFORM dispatch defers evaluation to eval_row/eval_fanin "
                "stages; complete fan-in before scheduling the next optim_step"
            )
    finally:
        runtime.eval_service.set_dispatch_mode(previous_dispatch)
    _bind_step_result(
        runtime,
        run_id=work_input.run_id,
        step_index=state.step_index,
        result_ref=result_ref,
    )
    updated = OptimWorkState(
        work_input=work_input,
        step_index=state.step_index + 1,
        step_result_refs=state.step_result_refs + (step_result_reference(result),),
        terminal=result.status is not StepStatus.CONTINUE,
    )
    return _persist_work_state(runtime, updated)


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
