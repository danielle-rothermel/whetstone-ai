from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dr_store.content_addressing import format_object_reference, parse_object_reference

from whetstone.coordination.eval_service import EvalDispatchMode, EvalEngineService
from whetstone.optim.contracts import (
    INTENT_RESOLUTION_SCHEMA_VERSION,
    IntentOutcome,
    IntentResolution,
    OptimEvalRequest,
    ResolutionClass,
    ResolutionDetail,
)

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
    if service.dispatch_mode is not EvalDispatchMode.PLATFORM:
        raise ValueError("platform eval stages require PLATFORM dispatch mode")
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
        previous = service.set_dispatch_mode(EvalDispatchMode.INLINE)
        try:
            service.resolve_optim_eval_request(intent)
        finally:
            service.set_dispatch_mode(previous)

    return executor


def execute_eval_row_sync(
    runtime: RegisteredRuntime,
    *,
    input_reference: str,
    row_executor: Any | None = None,
) -> str:
    """Execute one platform eval row work item."""
    _require_platform_eval_service(runtime)
    parsed = parse_object_reference(input_reference)
    payload = runtime.store.get(parsed)
    if not isinstance(payload, dict):
        raise ValueError("eval row input must be an object record")
    intent = OptimEvalRequest.model_validate(payload["optim_eval_request"])
    task_id = str(payload["task_id"])
    seed_index = int(payload["seed_index"])
    resolved_executor = row_executor or build_inline_row_executor(runtime)
    resolved_executor(intent=intent, task_id=task_id, seed_index=seed_index)
    row_record = {
        "schema_version": PLATFORM_EVAL_ROW_SCHEMA_VERSION,
        "optim_eval_request": intent.model_dump(mode="json"),
        "task_id": task_id,
        "seed_index": seed_index,
        "completed": True,
    }
    reference, _ = runtime.store.put(PLATFORM_EVAL_ROW_SCHEMA, row_record)
    return format_object_reference(reference)


def execute_eval_fanin_sync(
    runtime: RegisteredRuntime,
    *,
    input_reference: str,
    row_loader: Any | None = None,
) -> str:
    """Resolve a deferred platform eval intent after row execution."""
    service = _require_platform_eval_service(runtime)
    parsed = parse_object_reference(input_reference)
    payload = runtime.store.get(parsed)
    if not isinstance(payload, dict):
        raise ValueError("eval fan-in input must be an object record")
    intent = OptimEvalRequest.model_validate(payload["optim_eval_request"])
    pending = service.load_platform_intent(intent)
    if pending is None:
        raise ValueError("platform eval intent is not pending")
    if row_loader is not None:
        row_loader(intent=intent)
    original_mode = service._dispatch_mode  # noqa: SLF001
    object.__setattr__(service, "_dispatch_mode", EvalDispatchMode.INLINE)
    try:
        resolution = service.resolve_optim_eval_request(intent)
    finally:
        object.__setattr__(service, "_dispatch_mode", original_mode)
    fanin_record = {
        "schema_version": PLATFORM_EVAL_FANIN_SCHEMA_VERSION,
        "optim_eval_request": intent.model_dump(mode="json"),
        "resolution": resolution.model_dump(mode="json"),
    }
    reference, _ = runtime.store.put(PLATFORM_EVAL_FANIN_SCHEMA, fanin_record)
    return format_object_reference(reference)


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
