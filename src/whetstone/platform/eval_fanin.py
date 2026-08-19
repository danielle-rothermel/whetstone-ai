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
from whetstone.core.identity import TypedRef, typed_ref_for_record
from whetstone.eval.protocol import EvalRequest
from whetstone.eval.row_slice import RowEvalCompletion, RowEvalOutcome
from whetstone.eval.schema import EvalEvidence, EvalFailureEvidence
from whetstone.eval.schema_names import EVAL_EVIDENCE_SCHEMA, EVAL_FAILURE_SCHEMA
from whetstone.optim.contracts import (
    INTENT_RESOLUTION_SCHEMA,
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
    load_deferral_join_input,
    load_eval_row_input,
)
from whetstone.platform.deferral_cluster import list_episode_eval_row_predecessors
from whetstone.platform.step_executor import (
    _bind_step_result,
    _load_work_state,
    _persist_work_state,
    _require_controller_identity,
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
    ) -> RowEvalCompletion | None:
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
    ) -> RowEvalCompletion | None:
        service = _require_platform_eval_service(runtime)
        scoped_engine = service._engine.for_task_seed(task_id, seed_index)  # noqa: SLF001
        return scoped_engine.evaluate_row(
            EvalRequest(
                request_id=intent.eval_request.request_id,
                candidate=intent.eval_request.candidate,
                metadata=intent.eval_request.metadata,
            )
        )

    return executor


def _persist_row_completion(
    row_record: dict[str, object],
    completion: RowEvalCompletion | None,
) -> None:
    if completion is None:
        return
    if completion.evidence_ref is not None:
        row_record["evidence_ref"] = completion.evidence_ref.model_dump(mode="json")
    if completion.rejected_detail is not None:
        row_record["rejected_detail"] = completion.rejected_detail.model_dump(
            mode="json"
        )
    if completion.supplemental_aggregate_refs:
        row_record["supplemental_aggregate_refs"] = [
            item.model_dump(mode="json")
            for item in completion.supplemental_aggregate_refs
        ]


def execute_eval_row_sync(
    runtime: RegisteredRuntime,
    *,
    input_reference: str,
    stage_index: int | None = None,
    row_executor: Any | None = None,
) -> StageCompletion:
    """Execute one platform eval row work item."""
    _require_platform_eval_service(runtime)
    existing_binding = runtime.store.resolve(input_reference)
    if existing_binding is not None:
        existing_record = runtime.store.get(existing_binding)
        if isinstance(existing_record, dict) and existing_record.get("completed"):
            return StageCompletion(
                output_reference=format_object_reference(existing_binding)
            )
    row_input = load_eval_row_input(runtime.store, input_reference)
    work_state = _load_work_state(runtime, row_input.work_state_ref)
    _require_controller_identity(runtime, work_state.work_input)
    if stage_index is not None:
        expected = row_input.deferral_origin_stage_index + 1 + row_input.row_ordinal
        _validate_platform_stage_index(
            stage_index=stage_index,
            expected=expected,
            stage_key="eval_row",
        )
    resolved_executor = row_executor or build_inline_row_executor(runtime)
    completion = resolved_executor(
        intent=row_input.optim_eval_request,
        task_id=row_input.task_id,
        seed_index=row_input.seed_index,
    )
    row_record: dict[str, object] = {
        "schema_version": PLATFORM_EVAL_ROW_SCHEMA_VERSION,
        "optim_eval_request": row_input.optim_eval_request.model_dump(mode="json"),
        "task_id": row_input.task_id,
        "seed_index": row_input.seed_index,
        "deferral_origin_stage_index": row_input.deferral_origin_stage_index,
        "row_ordinal": row_input.row_ordinal,
        "completed": True,
    }
    _persist_row_completion(row_record, completion)
    reference, _ = runtime.store.put(PLATFORM_EVAL_ROW_SCHEMA, row_record)
    runtime.store.bind(input_reference, reference)
    output_ref = format_object_reference(reference)
    return StageCompletion(output_reference=output_ref)


def _load_completed_row_record_from_output(
    runtime: RegisteredRuntime,
    output_reference: str,
) -> dict[str, object]:
    parsed = parse_object_reference(output_reference)
    if parsed.schema != PLATFORM_EVAL_ROW_SCHEMA:
        raise ValueError(
            "eval row predecessor output has the wrong schema: "
            f"{parsed.schema!r}"
        )
    record = runtime.store.get(parsed)
    if not isinstance(record, dict) or not record.get("completed"):
        raise ValueError("eval row predecessor output is not completed")
    return record


def _unique_intents_from_row_records(
    records: tuple[dict[str, object], ...],
) -> tuple[OptimEvalRequest, ...]:
    seen: set[str] = set()
    intents: list[OptimEvalRequest] = []
    for record in records:
        intent = OptimEvalRequest.model_validate(record["optim_eval_request"])
        key = EvalEngineService._intent_ref(intent).content_hash
        if key in seen:
            continue
        seen.add(key)
        intents.append(intent)
    return tuple(intents)


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


def _row_rejected_detail(record: dict[str, object]) -> ResolutionDetail | None:
    raw = record.get("rejected_detail")
    if raw is None:
        return None
    return ResolutionDetail.model_validate(raw)


def _row_supplemental_aggregate_refs(
    record: dict[str, object],
) -> tuple[TypedRef, ...]:
    raw = record.get("supplemental_aggregate_refs")
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError("supplemental_aggregate_refs must be a list")
    return tuple(TypedRef.model_validate(item) for item in raw)


def _row_outcomes_from_records(
    runtime: RegisteredRuntime,
    records: tuple[dict[str, object], ...],
) -> dict[str, tuple[RowEvalOutcome, ...]]:
    grouped: dict[str, list[RowEvalOutcome]] = {}
    for record in records:
        task_id = str(record["task_id"])
        seed_index = int(record["seed_index"])
        rejected_detail = _row_rejected_detail(record)
        supplemental_refs = _row_supplemental_aggregate_refs(record)
        if rejected_detail is not None:
            outcome = RowEvalOutcome(
                task_id=task_id,
                seed_index=seed_index,
                rejected_detail=rejected_detail,
            )
        else:
            evidence_ref = _row_evidence_ref(record)
            if evidence_ref is None:
                raise ValueError("eval row output is missing row outcome")
            loaded = _load_row_evidence(runtime.store, evidence_ref)
            if isinstance(loaded, EvalFailureEvidence):
                outcome = RowEvalOutcome(
                    task_id=task_id,
                    seed_index=seed_index,
                    evidence_ref=evidence_ref,
                    failure=loaded,
                )
            else:
                outcome = RowEvalOutcome(
                    task_id=task_id,
                    seed_index=seed_index,
                    evidence_ref=evidence_ref,
                    evidence=loaded,
                    supplemental_aggregate_refs=supplemental_refs,
                )
        intent = OptimEvalRequest.model_validate(record["optim_eval_request"])
        intent_key = EvalEngineService._intent_ref(intent).content_hash
        grouped.setdefault(intent_key, []).append(outcome)
    return {key: tuple(outcomes) for key, outcomes in grouped.items()}


def _records_have_row_outcomes(records: tuple[dict[str, object], ...]) -> bool:
    for record in records:
        if _row_evidence_ref(record) is not None:
            return True
        if _row_rejected_detail(record) is not None:
            return True
    return False


def _expected_episode_row_count(state: OptimWorkState, runtime: RegisteredRuntime) -> int:
    if not state.pending_deferred_intents:
        raise ValueError("deferral episode is missing pending deferred intents")
    engine = runtime.eval_service._engine  # noqa: SLF001
    task_count = len(engine.sampling.tasks)
    seed_count = engine.sampling.num_seeds
    return len(state.pending_deferred_intents) * task_count * seed_count


def _verify_episode_eval_row_predecessors(
    runtime: RegisteredRuntime,
    *,
    work_item_id: int,
    stage_index: int,
    deferral_origin: int,
    expected_row_count: int,
) -> tuple[dict[str, object], ...]:
    if runtime.ledger_engine is None:
        return ()
    predecessors = list_episode_eval_row_predecessors(
        work_item_id,
        deferral_origin=deferral_origin,
        fanin_stage_index=stage_index,
        engine=runtime.ledger_engine,
    )
    if len(predecessors) != expected_row_count:
        raise ValueError(
            "eval fan-in ledger predecessors do not match episode row count: "
            f"expected {expected_row_count}, got {len(predecessors)}"
        )
    records: list[dict[str, object]] = []
    for predecessor in predecessors:
        if predecessor.output_reference is None:
            raise ValueError("eval row predecessor output is missing")
        records.append(
            _load_completed_row_record_from_output(
                runtime,
                predecessor.output_reference,
            )
        )
    return tuple(records)


def _load_bound_resolution(
    service: EvalEngineService,
    intent: OptimEvalRequest,
) -> IntentResolution | None:
    bound = service._store.resolve(service._key(intent))  # noqa: SLF001
    if bound is None:
        return None
    return service._load(bound, expected_optim_eval_request=intent)  # noqa: SLF001


def _resolution_identity_hashes(
    resolutions: tuple[IntentResolution, ...],
) -> tuple[str, ...]:
    return tuple(
        typed_ref_for_record(
            INTENT_RESOLUTION_SCHEMA,
            resolution.model_dump(mode="json"),
        ).content_hash
        for resolution in resolutions
    )


def _load_fanin_resolutions_from_binding(
    store,
    binding: Any,
) -> tuple[IntentResolution, ...] | None:
    record = store.get(binding)
    if not isinstance(record, dict):
        return None
    if record.get("schema_version") != PLATFORM_EVAL_FANIN_SCHEMA_VERSION:
        return None
    raw_resolutions = record.get("resolutions")
    if isinstance(raw_resolutions, list) and raw_resolutions:
        return tuple(
            IntentResolution.model_validate(item) for item in raw_resolutions
        )
    raw_resolution = record.get("resolution")
    if raw_resolution is not None:
        return (IntentResolution.model_validate(raw_resolution),)
    return None


def _load_fanin_resolutions_for_input(
    runtime: RegisteredRuntime,
    input_reference: str,
) -> tuple[IntentResolution, ...] | None:
    binding = runtime.store.resolve(input_reference)
    if binding is None:
        return None
    return _load_fanin_resolutions_from_binding(runtime.store, binding)


def _fanin_entry_allowed(
    service: EvalEngineService,
    *,
    primary_intent: OptimEvalRequest,
    episode_intents: tuple[OptimEvalRequest, ...],
    input_reference: str,
    runtime: RegisteredRuntime,
) -> tuple[IntentResolution, ...] | None:
    cached = _load_fanin_resolutions_for_input(runtime, input_reference)
    if cached is not None:
        return cached
    bound_resolutions = tuple(
        _load_bound_resolution(service, intent) for intent in episode_intents
    )
    if all(resolution is not None for resolution in bound_resolutions):
        return bound_resolutions  # type: ignore[return-value]
    if service.load_platform_intent(primary_intent) is not None:
        return None
    if any(resolution is not None for resolution in bound_resolutions):
        return None
    return None


def _resolve_episode_intent(
    service: EvalEngineService,
    *,
    intent: OptimEvalRequest,
    row_outcomes: tuple[RowEvalOutcome, ...] | None,
) -> IntentResolution:
    existing = _load_bound_resolution(service, intent)
    if existing is not None:
        return existing
    if row_outcomes is None:
        raise ValueError("deferral episode is missing row outcomes for intent")
    return service.resolve_platform_intent_from_row_outcomes(
        intent,
        row_outcomes=row_outcomes,
    )


def _finalize_deferred_step(
    runtime: RegisteredRuntime,
    *,
    work_state: OptimWorkState,
    work_state_ref: str,
    pending_step_result_ref: str,
    run_id: str,
    step_index: int,
    resolutions: tuple[IntentResolution, ...],
) -> OptimWorkState:
    pending = OptimStepResult.model_validate(
        runtime.store.get(parse_object_reference(pending_step_result_ref))
    )
    if pending.resolved_intents:
        if _resolution_identity_hashes(pending.resolved_intents) != _resolution_identity_hashes(
            resolutions
        ):
            raise ValueError("fan-in retry conflicts with merged step resolutions")
        loaded = _load_work_state(runtime, work_state_ref)
        if loaded.step_index > step_index:
            return loaded
    binding = runtime.store.resolve(
        _step_result_binding_key(runtime, run_id, step_index)
    )
    if binding is not None:
        bound_result = OptimStepResult.model_validate(runtime.store.get(binding))
        if bound_result.resolved_intents and (
            _resolution_identity_hashes(bound_result.resolved_intents)
            == _resolution_identity_hashes(resolutions)
        ):
            loaded = _load_work_state(runtime, work_state_ref)
            if loaded.step_index > step_index:
                return loaded
    merged = pending.model_copy(update={"resolved_intents": resolutions})
    merged_ref = runtime.harness._put_result(merged)  # noqa: SLF001
    _bind_step_result(
        runtime,
        run_id=run_id,
        step_index=step_index,
        result_ref=merged_ref,
    )
    loaded = _load_work_state(runtime, work_state_ref)
    if loaded.step_index > step_index:
        return loaded
    return OptimWorkState(
        work_input=loaded.work_input,
        step_index=loaded.step_index + 1,
        step_result_refs=loaded.step_result_refs
        + (OptimStepResultRef(record=merged, record_ref=merged_ref),),
        terminal=False,
    )


def _step_result_binding_key(
    runtime: RegisteredRuntime,
    run_id: str,
    step_index: int,
) -> str:
    return runtime.harness._result_binding_key(run_id, step_index)  # noqa: SLF001


def _complete_fanin_handoff(
    runtime: RegisteredRuntime,
    *,
    work_state: OptimWorkState,
    work_state_ref: str,
    join_input,
    input_reference: str,
    resolutions: tuple[IntentResolution, ...],
) -> StageCompletion:
    resolution = resolutions[0]
    fanin_record = {
        "schema_version": PLATFORM_EVAL_FANIN_SCHEMA_VERSION,
        "optim_eval_request": join_input.primary_optim_eval_request.model_dump(
            mode="json"
        ),
        "resolution": resolution.model_dump(mode="json"),
        "resolutions": [item.model_dump(mode="json") for item in resolutions],
    }
    input_binding = runtime.store.resolve(input_reference)
    if input_binding is not None:
        cached = _load_fanin_resolutions_from_binding(runtime.store, input_binding)
        if cached is not None:
            resolutions = cached
        output_ref = format_object_reference(input_binding)
    else:
        reference, _ = runtime.store.put(PLATFORM_EVAL_FANIN_SCHEMA, fanin_record)
        runtime.store.bind(input_reference, reference)
        output_ref = format_object_reference(reference)
    resumed = _finalize_deferred_step(
        runtime,
        work_state=work_state,
        work_state_ref=work_state_ref,
        pending_step_result_ref=work_state.pending_step_result_ref or "",
        run_id=work_state.work_input.run_id,
        step_index=work_state.step_index,
        resolutions=resolutions,
    )
    resumed_stage_index = resumed.work_input.platform_stage_index + 1
    resumed = OptimWorkState(
        work_input=resumed.work_input.model_copy(
            update={"platform_stage_index": resumed_stage_index},
        ),
        step_index=resumed.step_index,
        step_result_refs=resumed.step_result_refs,
        terminal=resumed.terminal,
    )
    resumed_ref = _persist_work_state(runtime, resumed)
    return StageCompletion(
        output_reference=output_ref,
        successors=(
            StageSuccessor(
                stage_key=StageKey("optim_step"),
                stage_index=resumed.work_input.platform_stage_index,
                input_reference=resumed_ref,
            ),
        ),
    )


def _collect_episode_row_records(
    runtime: RegisteredRuntime,
    *,
    work_state: OptimWorkState,
    work_state_ref: str,
    work_item_id: int | None,
    stage_index: int | None,
    deferral_origin: int,
) -> tuple[dict[str, object], ...]:
    if work_item_id is not None and stage_index is not None:
        return _verify_episode_eval_row_predecessors(
            runtime,
            work_item_id=work_item_id,
            stage_index=stage_index,
            deferral_origin=deferral_origin,
            expected_row_count=_expected_episode_row_count(work_state, runtime),
        )
    records: list[dict[str, object]] = []
    for row_input_ref in _row_input_refs_for_episode(
        runtime,
        work_state=work_state,
        deferral_origin=deferral_origin,
        work_state_ref=work_state_ref,
    ):
        binding = runtime.store.resolve(row_input_ref)
        if binding is None:
            raise ValueError("eval row input is not bound to a completion record")
        record = runtime.store.get(binding)
        if not isinstance(record, dict) or not record.get("completed"):
            raise ValueError("eval row completion record is missing or incomplete")
        records.append(record)
    return tuple(records)


def _row_input_refs_for_episode(
    runtime: RegisteredRuntime,
    *,
    work_state: OptimWorkState,
    deferral_origin: int,
    work_state_ref: str,
) -> tuple[str, ...]:
    from whetstone.platform.contracts import persist_eval_row_input
    from whetstone.platform.step_executor import _expand_eval_rows

    row_inputs = _expand_eval_rows(
        runtime,
        work_state.pending_deferred_intents,
        deferral_origin_stage_index=deferral_origin,
        work_state_ref=work_state_ref,
    )
    return tuple(
        persist_eval_row_input(runtime.store, row_input) for row_input in row_inputs
    )


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
    join_input = load_deferral_join_input(runtime.store, input_reference)
    work_state_ref = join_input.work_state_ref
    work_state = _load_work_state(runtime, work_state_ref)
    _require_controller_identity(runtime, work_state.work_input)
    if stage_index is not None:
        _validate_platform_stage_index(
            stage_index=stage_index,
            expected=work_state.work_input.platform_stage_index,
            stage_key="eval_fanin",
        )
    deferral_origin = join_input.deferral_optim_step_stage_index
    if work_state.pending_step_result_ref is None:
        raise ValueError("deferral fan-in requires a pending step result")
    row_records = _collect_episode_row_records(
        runtime,
        work_state=work_state,
        work_state_ref=work_state_ref,
        work_item_id=work_item_id,
        stage_index=stage_index,
        deferral_origin=deferral_origin,
    )
    episode_intents = _unique_intents_from_row_records(row_records)
    if not episode_intents:
        raise ValueError("deferral episode has no deferred intents")
    cached_resolutions = _fanin_entry_allowed(
        service,
        primary_intent=join_input.primary_optim_eval_request,
        episode_intents=episode_intents,
        input_reference=input_reference,
        runtime=runtime,
    )
    if row_loader is not None:
        row_loader(intent=join_input.primary_optim_eval_request)
    if cached_resolutions is not None:
        return _complete_fanin_handoff(
            runtime,
            work_state=work_state,
            work_state_ref=work_state_ref,
            join_input=join_input,
            input_reference=input_reference,
            resolutions=cached_resolutions,
        )
    if not _records_have_row_outcomes(row_records):
        raise ValueError(
            "deferral episode is incomplete: row outcomes are required for platform fan-in"
        )
    row_outcomes_by_intent = _row_outcomes_from_records(runtime, row_records)
    resolutions: list[IntentResolution] = []
    for intent in episode_intents:
        intent_key = EvalEngineService._intent_ref(intent).content_hash
        row_outcomes = row_outcomes_by_intent.get(intent_key)
        resolutions.append(
            _resolve_episode_intent(
                service,
                intent=intent,
                row_outcomes=row_outcomes,
            )
        )
    return _complete_fanin_handoff(
        runtime,
        work_state=work_state,
        work_state_ref=work_state_ref,
        join_input=join_input,
        input_reference=input_reference,
        resolutions=tuple(resolutions),
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
