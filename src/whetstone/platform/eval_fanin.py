from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dr_platform._core.identities import StageKey
from dr_platform.execution.stage_completion import StageCompletion, StageSuccessor
from dr_store.content_addressing import format_object_reference, parse_object_reference

from whetstone.coordination.eval_service import (
    EvalDispatchMode,
    EvalEngineService,
    EvalExecutionContext,
)
from whetstone.core.identity import TypedRef
from whetstone.eval.protocol import EvalEvidenceWithRef, EvalRejected, EvalRequest
from whetstone.eval.row_slice import RowEvalSlice
from whetstone.eval.schema import EvalEvidence, EvalFailureEvidence
from whetstone.eval.schema_names import EVAL_EVIDENCE_SCHEMA, EVAL_FAILURE_SCHEMA
from whetstone.optim.contracts import (
    INTENT_RESOLUTION_SCHEMA_VERSION,
    IntentOutcome,
    IntentResolution,
    OptimEvalRequest,
    OptimStepResult,
    OptimStepResultRef,
    ResolutionClass,
    ResolutionDetail,
)
from whetstone.platform.contracts import (
    STAGE_EVAL_ROW,
    STAGE_OPTIM_STEP,
    load_eval_batch_by_id,
    load_eval_fanin_input,
    load_eval_row_input,
)
from whetstone.platform.step_executor import (
    _bind_step_result,
    _load_work_state,
    _persist_work_state,
    _validate_platform_stage_index,
    OptimWorkState,
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
    return service


def build_inline_row_executor(runtime: RegisteredRuntime):
    """Resolve one deferred intent in-process (platform unit tests)."""

    def executor(
        *,
        intent: OptimEvalRequest,
        task_id: str,
        seed_index: int,
    ) -> TypedRef | None:
        _ = (task_id, seed_index)
        service = runtime.eval_service
        service.resolve_optim_eval_request(
            intent,
            context=EvalExecutionContext(dispatch_mode=EvalDispatchMode.INLINE),
        )
        return None

    return executor


def build_platform_row_executor(runtime: RegisteredRuntime):
    """Run one task slice for a deferred intent without claiming platform resolution."""

    def executor(
        *,
        intent: OptimEvalRequest,
        task_id: str,
        seed_index: int,
    ) -> TypedRef | None:
        service = _require_platform_eval_service(runtime)
        scoped_engine = service._engine.for_task_seed(task_id, seed_index)  # noqa: SLF001
        result = scoped_engine.evaluate(
            EvalRequest(
                request_id=intent.eval_request.request_id,
                candidate=intent.eval_request.candidate,
                metadata=intent.eval_request.metadata,
            )
        )
        if isinstance(result, EvalRejected):
            raise ValueError(result.detail.message)
        if not isinstance(result, EvalEvidenceWithRef):
            raise TypeError(f"unexpected platform row evaluation result: {result!r}")
        return result.evidence_ref

    return executor


def execute_eval_row_sync(
    runtime: RegisteredRuntime,
    *,
    input_reference: str,
    stage_index: int | None = None,
    row_executor: Any | None = None,
) -> StageCompletion:
    """Execute one platform eval row work item."""
    _require_platform_eval_service(runtime)
    row_input = load_eval_row_input(runtime.store, input_reference)
    if stage_index is not None:
        batch = load_eval_batch_by_id(runtime.store, row_input.batch_id)
        try:
            row_offset = batch.row_input_refs.index(input_reference)
        except ValueError as error:
            raise ValueError(
                "eval row input is not registered in its batch"
            ) from error
        expected = batch.optim_step_stage_index + 1 + row_offset
        _validate_platform_stage_index(
            stage_index=stage_index,
            expected=expected,
            stage_key="eval_row",
        )
    resolved_executor = row_executor or build_inline_row_executor(runtime)
    evidence_ref = resolved_executor(
        intent=row_input.optim_eval_request,
        task_id=row_input.task_id,
        seed_index=row_input.seed_index,
    )
    row_record: dict[str, object] = {
        "schema_version": PLATFORM_EVAL_ROW_SCHEMA_VERSION,
        "optim_eval_request": row_input.optim_eval_request.model_dump(mode="json"),
        "task_id": row_input.task_id,
        "seed_index": row_input.seed_index,
        "batch_id": row_input.batch_id,
        "completed": True,
    }
    if evidence_ref is not None:
        row_record["evidence_ref"] = evidence_ref.model_dump(mode="json")
    reference, _ = runtime.store.put(PLATFORM_EVAL_ROW_SCHEMA, row_record)
    runtime.store.bind(input_reference, reference)
    output_ref = format_object_reference(reference)
    return StageCompletion(output_reference=output_ref)


def _unique_batch_intents(batch, store) -> tuple[OptimEvalRequest, ...]:
    seen: set[str] = set()
    intents: list[OptimEvalRequest] = []
    for row_ref in batch.row_input_refs:
        row = load_eval_row_input(store, row_ref)
        key = EvalEngineService._intent_ref(row.optim_eval_request).content_hash
        if key in seen:
            continue
        seen.add(key)
        intents.append(row.optim_eval_request)
    return tuple(intents)


def _load_completed_row_record(store, row_input_ref: str) -> dict[str, object]:
    binding = store.resolve(row_input_ref)
    if binding is None:
        raise ValueError("eval row input is not bound to a completion record")
    record = store.get(binding)
    if not isinstance(record, dict) or not record.get("completed"):
        raise ValueError("eval row completion record is missing or incomplete")
    return record


def _row_evidence_ref(record: dict[str, object]) -> TypedRef | None:
    raw = record.get("evidence_ref")
    if raw is None:
        return None
    return TypedRef.model_validate(raw)


def _load_row_evidence(store, evidence_ref: TypedRef) -> EvalEvidence | EvalFailureEvidence:
    if evidence_ref.schema_name == EVAL_EVIDENCE_SCHEMA:
        return EvalEvidence.model_validate(store.get(evidence_ref.reference))
    if evidence_ref.schema_name == EVAL_FAILURE_SCHEMA:
        return EvalFailureEvidence.model_validate(store.get(evidence_ref.reference))
    raise ValueError(
        "platform row evidence has the wrong schema: "
        f"{evidence_ref.schema_name!r}"
    )


def _batch_row_slices(
    runtime: RegisteredRuntime,
    batch,
) -> dict[str, tuple[RowEvalSlice, ...]]:
    grouped: dict[str, list[RowEvalSlice]] = {}
    for row_ref in batch.row_input_refs:
        row_input = load_eval_row_input(runtime.store, row_ref)
        row_record = _load_completed_row_record(runtime.store, row_ref)
        evidence_ref = _row_evidence_ref(row_record)
        if evidence_ref is None:
            raise ValueError("eval batch row is missing evidence_ref")
        evidence = _load_row_evidence(runtime.store, evidence_ref)
        if not isinstance(evidence, EvalEvidence):
            raise ValueError("platform row assembly requires success evidence")
        intent_key = EvalEngineService._intent_ref(
            row_input.optim_eval_request
        ).content_hash
        grouped.setdefault(intent_key, []).append(
            RowEvalSlice(
                task_id=row_input.task_id,
                seed_index=row_input.seed_index,
                evidence=evidence,
            )
        )
    return {key: tuple(slices) for key, slices in grouped.items()}


def _batch_has_row_evidence(runtime: RegisteredRuntime, batch) -> bool:
    for row_ref in batch.row_input_refs:
        row_record = _load_completed_row_record(runtime.store, row_ref)
        if _row_evidence_ref(row_record) is None:
            return False
    return True


def _eval_row_batch_id(runtime: RegisteredRuntime, output_reference: str) -> str:
    parsed = parse_object_reference(output_reference)
    if parsed.schema != PLATFORM_EVAL_ROW_SCHEMA:
        raise ValueError(
            "eval row predecessor output has the wrong schema: "
            f"{parsed.schema!r}"
        )
    record = runtime.store.get(parsed)
    if not isinstance(record, dict):
        raise ValueError("eval row predecessor output is not an object")
    batch_id = record.get("batch_id")
    if not isinstance(batch_id, str) or not batch_id:
        raise ValueError("eval row predecessor output is missing batch_id")
    return batch_id


def _verify_eval_row_predecessors(
    runtime: RegisteredRuntime,
    *,
    work_item_id: int,
    stage_index: int,
    batch,
) -> None:
    from dr_platform.inspection.work_items import list_predecessor_stage_outputs

    if runtime.ledger_engine is None:
        return
    predecessors = list_predecessor_stage_outputs(
        work_item_id,
        below_stage_index=stage_index,
        engine=runtime.ledger_engine,
    )
    eval_row_outputs = [
        predecessor
        for predecessor in predecessors
        if predecessor.stage_key.value == STAGE_EVAL_ROW
    ]
    batch_eval_row_outputs = [
        predecessor
        for predecessor in eval_row_outputs
        if predecessor.output_reference is not None
        and _eval_row_batch_id(runtime, predecessor.output_reference) == batch.batch_id
    ]
    if len(batch_eval_row_outputs) != len(batch.row_input_refs):
        raise ValueError(
            "eval fan-in ledger predecessors do not match batch row count: "
            f"expected {len(batch.row_input_refs)}, got {len(batch_eval_row_outputs)}"
        )
    for predecessor in batch_eval_row_outputs:
        parsed = parse_object_reference(predecessor.output_reference)
        if parsed.schema != PLATFORM_EVAL_ROW_SCHEMA:
            raise ValueError(
                "eval row predecessor output has the wrong schema: "
                f"{parsed.schema!r}"
            )
        record = runtime.store.get(parsed)
        if not isinstance(record, dict) or not record.get("completed"):
            raise ValueError("eval row predecessor output is not completed")


def _finalize_deferred_step(
    runtime: RegisteredRuntime,
    *,
    batch,
    resolutions: tuple[IntentResolution, ...],
) -> OptimWorkState:
    pending = OptimStepResult.model_validate(
        runtime.store.get(parse_object_reference(batch.pending_step_result_ref))
    )
    merged = pending.model_copy(update={"resolved_intents": resolutions})
    merged_ref = runtime.harness._put_result(merged)  # noqa: SLF001
    _bind_step_result(
        runtime,
        run_id=batch.run_id,
        step_index=batch.step_index,
        result_ref=merged_ref,
    )
    work_state = _load_work_state(runtime, batch.work_state_ref)
    resumed = OptimWorkState(
        work_input=work_state.work_input,
        step_index=work_state.step_index + 1,
        step_result_refs=work_state.step_result_refs
        + (OptimStepResultRef(record=merged, record_ref=merged_ref),),
        terminal=False,
        pending_eval_batch_ref=None,
    )
    return resumed


def execute_eval_fanin_sync(
    runtime: RegisteredRuntime,
    *,
    input_reference: str,
    stage_index: int | None = None,
    work_item_id: int | None = None,
    row_loader: Any | None = None,
) -> StageCompletion:
    """Resolve a deferred platform eval intent after row execution."""
    service = _require_platform_eval_service(runtime)
    fanin_input = load_eval_fanin_input(runtime.store, input_reference)
    batch = load_eval_batch_by_id(runtime.store, fanin_input.batch_id)
    work_state = _load_work_state(runtime, batch.work_state_ref)
    if stage_index is not None:
        _validate_platform_stage_index(
            stage_index=stage_index,
            expected=work_state.work_input.platform_stage_index,
            stage_key="eval_fanin",
        )
    batch_intents = _unique_batch_intents(batch, runtime.store)
    if not batch_intents:
        raise ValueError("eval batch has no deferred intents")
    if service.load_platform_intent(fanin_input.optim_eval_request) is None:
        raise ValueError("platform eval intent is not pending")
    if row_loader is not None:
        row_loader(intent=fanin_input.optim_eval_request)
    if work_item_id is not None and stage_index is not None:
        _verify_eval_row_predecessors(
            runtime,
            work_item_id=work_item_id,
            stage_index=stage_index,
            batch=batch,
        )
    for row_ref in batch.row_input_refs:
        load_eval_row_input(runtime.store, row_ref)
    resolutions: list[IntentResolution] = []
    if _batch_has_row_evidence(runtime, batch):
        row_slices_by_intent = _batch_row_slices(runtime, batch)
        for intent in batch_intents:
            intent_key = EvalEngineService._intent_ref(intent).content_hash
            row_slices = row_slices_by_intent.get(intent_key)
            if row_slices is None:
                raise ValueError("eval batch is missing row evidence for intent")
            resolutions.append(
                service.resolve_platform_intent_from_row_slices(
                    intent,
                    row_slices=row_slices,
                )
            )
    else:
        inline = EvalExecutionContext(dispatch_mode=EvalDispatchMode.INLINE)
        for intent in batch_intents:
            resolutions.append(
                service.resolve_optim_eval_request(intent, context=inline)
            )
    resolution = resolutions[0]
    fanin_record = {
        "schema_version": PLATFORM_EVAL_FANIN_SCHEMA_VERSION,
        "batch_id": fanin_input.batch_id,
        "optim_eval_request": fanin_input.optim_eval_request.model_dump(mode="json"),
        "resolution": resolution.model_dump(mode="json"),
        "resolutions": [item.model_dump(mode="json") for item in resolutions],
    }
    reference, _ = runtime.store.put(PLATFORM_EVAL_FANIN_SCHEMA, fanin_record)
    output_ref = format_object_reference(reference)
    resumed = _finalize_deferred_step(
        runtime,
        batch=batch,
        resolutions=tuple(resolutions),
    )
    resumed_stage_index = resumed.work_input.platform_stage_index + 1
    resumed = OptimWorkState(
        work_input=resumed.work_input.model_copy(
            update={"platform_stage_index": resumed_stage_index},
        ),
        step_index=resumed.step_index,
        step_result_refs=resumed.step_result_refs,
        terminal=resumed.terminal,
        pending_eval_batch_ref=resumed.pending_eval_batch_ref,
    )
    resumed_ref = _persist_work_state(runtime, resumed)
    return StageCompletion(
        output_reference=output_ref,
        successors=(
            StageSuccessor(
                stage_key=StageKey(STAGE_OPTIM_STEP),
                stage_index=resumed.work_input.platform_stage_index,
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
    "build_platform_row_executor",
    "execute_eval_fanin_sync",
    "execute_eval_row_sync",
    "pending_platform_resolution",
    "serialize_platform_eval_intent",
]
