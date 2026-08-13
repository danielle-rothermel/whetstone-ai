from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dr_platform._core.identities import StageKey
from dr_platform.execution.stage_completion import StageCompletion, StageSuccessor
from dr_store.content_addressing import format_object_reference

from whetstone.coordination.eval_service import (
    EvalDispatchMode,
    EvalEngineService,
    EvalExecutionContext,
)
from whetstone.optim.contracts import (
    INTENT_RESOLUTION_SCHEMA_VERSION,
    IntentOutcome,
    IntentResolution,
    OptimEvalRequest,
    ResolutionClass,
    ResolutionDetail,
)
from whetstone.platform.contracts import (
    STAGE_OPTIM_STEP,
    load_eval_batch_by_id,
    load_eval_fanin_input,
    load_eval_row_input,
)
from whetstone.platform.step_executor import _load_work_state, _persist_work_state

if TYPE_CHECKING:
    from whetstone.coordination.runtime_bootstrap import RegisteredRuntime

PLATFORM_EVAL_ROW_SCHEMA = "whetstone.platform_eval_row"
PLATFORM_EVAL_ROW_SCHEMA_VERSION = 1
PLATFORM_EVAL_FANIN_SCHEMA = "whetstone.platform_eval_fanin"
PLATFORM_EVAL_FANIN_SCHEMA_VERSION = 1


def _require_platform_eval_service(runtime: RegisteredRuntime) -> EvalEngineService:
    service = runtime.eval_service
    if not isinstance(service, EvalEngineService):
        raise TypeError("platform eval stages require EvalEngineService")
    return service


def build_inline_row_executor(runtime: RegisteredRuntime):
    """Resolve one deferred intent in-process (platform unit tests)."""

    def executor(
        *,
        intent: OptimEvalRequest,
        task_id: str,
        seed_index: int,
    ) -> None:
        _ = (task_id, seed_index)
        service = runtime.eval_service
        service.resolve_optim_eval_request(
            intent,
            context=EvalExecutionContext(dispatch_mode=EvalDispatchMode.INLINE),
        )

    return executor


def execute_eval_row_sync(
    runtime: RegisteredRuntime,
    *,
    input_reference: str,
    row_executor: Any | None = None,
) -> StageCompletion:
    """Execute one platform eval row work item."""
    _require_platform_eval_service(runtime)
    row_input = load_eval_row_input(runtime.store, input_reference)
    resolved_executor = row_executor or build_inline_row_executor(runtime)
    resolved_executor(
        intent=row_input.optim_eval_request,
        task_id=row_input.task_id,
        seed_index=row_input.seed_index,
    )
    row_record = {
        "schema_version": PLATFORM_EVAL_ROW_SCHEMA_VERSION,
        "optim_eval_request": row_input.optim_eval_request.model_dump(mode="json"),
        "task_id": row_input.task_id,
        "seed_index": row_input.seed_index,
        "batch_id": row_input.batch_id,
        "completed": True,
    }
    reference, _ = runtime.store.put(PLATFORM_EVAL_ROW_SCHEMA, row_record)
    output_ref = format_object_reference(reference)
    return StageCompletion(output_reference=output_ref)


def execute_eval_fanin_sync(
    runtime: RegisteredRuntime,
    *,
    input_reference: str,
    row_loader: Any | None = None,
) -> StageCompletion:
    """Resolve a deferred platform eval intent after row execution."""
    service = _require_platform_eval_service(runtime)
    fanin_input = load_eval_fanin_input(runtime.store, input_reference)
    batch = load_eval_batch_by_id(runtime.store, fanin_input.batch_id)
    pending = service.load_platform_intent(fanin_input.optim_eval_request)
    if pending is None:
        raise ValueError("platform eval intent is not pending")
    if row_loader is not None:
        row_loader(intent=fanin_input.optim_eval_request)
    for row_ref in batch.row_input_refs:
        load_eval_row_input(runtime.store, row_ref)
    resolution = service.resolve_optim_eval_request(
        fanin_input.optim_eval_request,
        context=EvalExecutionContext(dispatch_mode=EvalDispatchMode.INLINE),
    )
    fanin_record = {
        "schema_version": PLATFORM_EVAL_FANIN_SCHEMA_VERSION,
        "batch_id": fanin_input.batch_id,
        "optim_eval_request": fanin_input.optim_eval_request.model_dump(mode="json"),
        "resolution": resolution.model_dump(mode="json"),
    }
    reference, _ = runtime.store.put(PLATFORM_EVAL_FANIN_SCHEMA, fanin_record)
    output_ref = format_object_reference(reference)
    work_state = _load_work_state(runtime, batch.work_state_ref)
    resumed_ref = _persist_work_state(runtime, work_state)
    return StageCompletion(
        output_reference=output_ref,
        successors=(
            StageSuccessor(
                stage_key=StageKey(STAGE_OPTIM_STEP),
                stage_index=work_state.work_input.platform_stage_index,
                input_reference=resumed_ref,
            ),
        ),
    )


def serialize_platform_eval_intent(intent: OptimEvalRequest) -> dict[str, object]:
    return {
        "optim_eval_request": intent.model_dump(mode="json"),
        "pending": True,
    }


def pending_resolution_detail() -> ResolutionDetail:
    return ResolutionDetail(
        classification=ResolutionClass.INFRASTRUCTURE,
        message="evaluation deferred to platform eval stages",
    )


def pending_platform_resolution(intent: OptimEvalRequest) -> IntentResolution:
    return IntentResolution(
        schema_version=INTENT_RESOLUTION_SCHEMA_VERSION,
        optim_eval_request=intent,
        outcome=IntentOutcome.REJECTED,
        detail=pending_resolution_detail(),
    )


__all__ = [
    "PLATFORM_EVAL_FANIN_SCHEMA",
    "PLATFORM_EVAL_ROW_SCHEMA",
    "build_inline_row_executor",
    "execute_eval_fanin_sync",
    "execute_eval_row_sync",
    "pending_platform_resolution",
    "serialize_platform_eval_intent",
]
