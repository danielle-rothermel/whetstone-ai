from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

from dr_code.humaneval import (
    HUMANEVAL_SCORING_PROFILE_ID,
    HUMANEVAL_SCORING_PROFILE_VERSION,
    resolve_humaneval_scoring_profile,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
)
from sqlalchemy.engine import Connection, Engine

from whetstone.db import io
from whetstone.eval_failures import failure_metadata_from_exception
from whetstone.platform.scoring_workflow import (
    ScheduledScoreSubmissionWorkflow,
    await_scheduled_score_workflows,
    platform_scoring_workflow_id,
    schedule_score_submission_workflow,
)
from whetstone.platform.scoring_workflow_state import (
    ScoringWorkflowPresence,
    classify_scoring_workflow_presence,
)
from whetstone.records import (
    DEFAULT_SCORE_DATASET_NAME,
    DEFAULT_SCORE_DATASET_SPLIT,
    FailureMetadataPayload,
    GenerationRunStatus,
    stable_score_attempt_id,
)

if TYPE_CHECKING:
    from dr_platform import OperationProgress

DEFAULT_RESCORE_CHUNK_SIZE = 500
DEFAULT_MAX_IN_FLIGHT = 100
DEFAULT_RESCORE_SUBMISSION_STATUSES = (
    GenerationRunStatus.SUCCESS,
    GenerationRunStatus.PARTIAL,
)


class BatchRescoreItemStatus(StrEnum):
    ALREADY_SCORED = "already_scored"
    WOULD_SCHEDULE = "would_schedule"
    SCHEDULED = "scheduled"
    RECOVERED = "recovered"
    WORKFLOW_IN_FLIGHT = "workflow_in_flight"
    WORKFLOW_ORPHAN = "workflow_orphan"
    FAILED = "failed"


class RescoreGenerationCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_id: StrictStr
    fair_order_key: StrictStr
    generation_run_id: StrictStr
    score_attempt_index: StrictInt
    existing_score_attempt_id: StrictStr | None = None


class BatchRescoreItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_id: StrictStr
    fair_order_key: StrictStr
    generation_run_id: StrictStr
    score_attempt_id: StrictStr
    workflow_id: StrictStr
    status: BatchRescoreItemStatus
    existing_score_attempt_id: StrictStr | None = None
    failure: FailureMetadataPayload | None = None


class BatchRescoreResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_name: StrictStr
    generation_statuses: tuple[GenerationRunStatus, ...]
    generation_attempt_index: StrictInt | None
    scoring_profile_id: StrictStr
    scoring_profile_version: StrictStr
    parser_profile_id: StrictStr
    parser_version: StrictStr
    score_attempt_index: StrictInt
    dataset_name: StrictStr
    dataset_split: StrictStr
    dry_run: StrictBool
    max_in_flight: StrictInt
    total_candidates: StrictInt
    selected_count: StrictInt
    already_scored_count: StrictInt
    needs_score_count: StrictInt
    scheduled_count: StrictInt
    recovered_count: StrictInt
    in_flight_count: StrictInt
    orphan_count: StrictInt
    failed_count: StrictInt
    items: tuple[BatchRescoreItem, ...] = Field(default_factory=tuple)


class BatchRescoreExecution(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    result: BatchRescoreResult
    workflow_handles: tuple[Any, ...] = Field(
        default_factory=tuple,
        exclude=True,
    )


class ScheduleScoreWorkflow(Protocol):
    def __call__(
        self,
        *,
        database_url: str,
        generation_run_id: str,
        score_attempt_index: int,
        scoring_profile_id: str,
        scoring_profile_version: str,
        dataset_name: str,
        dataset_split: str,
        recover_orphans: bool = True,
    ) -> ScheduledScoreSubmissionWorkflow: ...


def _await_oldest_handle(
    pending_handles: list[Any],
    *,
    await_workflows: Any,
) -> Any:
    oldest = pending_handles.pop(0)
    await_workflows([oldest])
    return oldest


def _wait_for_in_flight_slot(
    pending_handles: list[Any],
    *,
    max_in_flight: int,
    await_workflows: Any,
    progress: OperationProgress | None = None,
    selected: int,
    items: Sequence[BatchRescoreItem],
    slots_released: int,
) -> int:
    while len(pending_handles) >= max_in_flight:
        if progress is not None:
            progress.update(
                phase="awaiting",
                selected=selected,
                pending=len(pending_handles),
                slots_released=slots_released,
                **_rescore_count_metrics(items),
            )
            progress.event(
                "awaiting slot",
                {
                    "in_flight": len(pending_handles),
                    "selected": selected,
                    "slots_released": slots_released + 1,
                },
            )
        _await_oldest_handle(pending_handles, await_workflows=await_workflows)
        slots_released += 1
    return slots_released


def _await_remaining_handles(
    pending_handles: list[Any],
    *,
    await_workflows: Any,
    progress: OperationProgress | None = None,
    selected: int,
    items: Sequence[BatchRescoreItem],
    slots_released: int,
) -> int:
    if not pending_handles:
        return slots_released
    if progress is not None:
        progress.update(
            phase="awaiting",
            selected=selected,
            pending=len(pending_handles),
            slots_released=slots_released,
            **_rescore_count_metrics(items),
        )
        progress.event(
            "awaiting remaining",
            {
                "remaining": len(pending_handles),
                "selected": selected,
            },
        )
    while pending_handles:
        _await_oldest_handle(pending_handles, await_workflows=await_workflows)
        slots_released += 1
    return slots_released


def rescore_submission_runs(
    engine: Engine,
    *,
    database_url: str,
    experiment_name: str,
    generation_statuses: Sequence[GenerationRunStatus] = (
        DEFAULT_RESCORE_SUBMISSION_STATUSES
    ),
    generation_attempt_index: int | None = None,
    scoring_profile_id: str = HUMANEVAL_SCORING_PROFILE_ID,
    scoring_profile_version: str = HUMANEVAL_SCORING_PROFILE_VERSION,
    score_attempt_index: int = 0,
    dataset_name: str = DEFAULT_SCORE_DATASET_NAME,
    dataset_split: str = DEFAULT_SCORE_DATASET_SPLIT,
    chunk_size: int = DEFAULT_RESCORE_CHUNK_SIZE,
    limit: int | None = None,
    dry_run: bool = False,
    recover_orphans: bool = True,
    max_in_flight: int = DEFAULT_MAX_IN_FLIGHT,
    schedule_workflow: ScheduleScoreWorkflow = (
        schedule_score_submission_workflow
    ),
    await_workflows: Any = await_scheduled_score_workflows,
    progress: OperationProgress | None = None,
) -> BatchRescoreExecution:
    validate_rescore_request(
        chunk_size=chunk_size,
        limit=limit,
        generation_attempt_index=generation_attempt_index,
        score_attempt_index=score_attempt_index,
        generation_statuses=generation_statuses,
        max_in_flight=max_in_flight,
    )
    scoring_profile = resolve_humaneval_scoring_profile(
        scoring_profile_id=scoring_profile_id,
        scoring_profile_version=scoring_profile_version,
    )
    items: list[BatchRescoreItem] = []
    pending_handles: list[Any] = []
    slots_released = 0
    offset = 0
    with engine.begin() as connection:
        total_candidates = count_rescore_submission_candidates(
            connection,
            experiment_name=experiment_name,
            generation_statuses=generation_statuses,
            generation_attempt_index=generation_attempt_index,
            scoring_profile_id=scoring_profile.profile_id,
            scoring_profile_version=scoring_profile.version,
            parser_profile_id=scoring_profile.parser_profile.profile_id,
            parser_version=scoring_profile.parser_profile.version,
            score_attempt_index=score_attempt_index,
            dataset_name=dataset_name,
            dataset_split=dataset_split,
        )
    if limit is not None:
        total_candidates = min(total_candidates, limit)
    if progress is not None:
        progress.event(
            "started",
            {
                "experiment": experiment_name,
                "max_in_flight": max_in_flight,
                "chunk_size": chunk_size,
                "dry_run": dry_run,
                "limit": limit,
                "total_candidates": total_candidates,
            },
        )
        progress.update(
            phase="selecting",
            experiment=experiment_name,
            max_in_flight=max_in_flight,
            chunk_size=chunk_size,
            dry_run=dry_run,
            limit=limit,
            total_candidates=total_candidates,
            selected=0,
        )
    while limit is None or offset < limit:
        page_limit = (
            chunk_size if limit is None else min(chunk_size, limit - offset)
        )
        with engine.begin() as connection:
            candidates = load_rescore_submission_candidates(
                connection,
                experiment_name=experiment_name,
                generation_statuses=generation_statuses,
                generation_attempt_index=generation_attempt_index,
                scoring_profile_id=scoring_profile.profile_id,
                scoring_profile_version=scoring_profile.version,
                parser_profile_id=scoring_profile.parser_profile.profile_id,
                parser_version=scoring_profile.parser_profile.version,
                score_attempt_index=score_attempt_index,
                dataset_name=dataset_name,
                dataset_split=dataset_split,
                limit=page_limit,
                offset=offset,
            )
        if not candidates:
            break
        for candidate in candidates:
            item, handle = plan_or_schedule_rescore_item(
                candidate,
                database_url=database_url,
                score_attempt_index=score_attempt_index,
                scoring_profile_id=scoring_profile.profile_id,
                scoring_profile_version=scoring_profile.version,
                parser_profile_id=scoring_profile.parser_profile.profile_id,
                parser_version=scoring_profile.parser_profile.version,
                dataset_name=dataset_name,
                dataset_split=dataset_split,
                dry_run=dry_run,
                recover_orphans=recover_orphans,
                schedule_workflow=schedule_workflow,
            )
            items.append(item)
            if handle is not None:
                if not dry_run:
                    slots_released = _wait_for_in_flight_slot(
                        pending_handles,
                        max_in_flight=max_in_flight,
                        await_workflows=await_workflows,
                        progress=progress,
                        selected=len(items),
                        items=items,
                        slots_released=slots_released,
                    )
                pending_handles.append(handle)
            if progress is not None:
                progress.update(
                    phase="scheduling",
                    selected=len(items),
                    pending=len(pending_handles),
                    slots_released=slots_released,
                    **_rescore_count_metrics(items),
                )
        offset += len(candidates)
        if len(candidates) < page_limit:
            break

    if not dry_run and pending_handles:
        slots_released = _await_remaining_handles(
            pending_handles,
            await_workflows=await_workflows,
            progress=progress,
            selected=len(items),
            items=items,
            slots_released=slots_released,
        )

    result = batch_rescore_result(
        experiment_name=experiment_name,
        generation_statuses=tuple(generation_statuses),
        generation_attempt_index=generation_attempt_index,
        scoring_profile_id=scoring_profile.profile_id,
        scoring_profile_version=scoring_profile.version,
        parser_profile_id=scoring_profile.parser_profile.profile_id,
        parser_version=scoring_profile.parser_profile.version,
        score_attempt_index=score_attempt_index,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        dry_run=dry_run,
        max_in_flight=max_in_flight,
        total_candidates=total_candidates,
        items=tuple(items),
    )
    if progress is not None:
        progress.complete(
            {
                "total_candidates": total_candidates,
                "selected": result.selected_count,
                "scheduled": result.scheduled_count,
                "recovered": result.recovered_count,
                "failed": result.failed_count,
                "slots_released": slots_released,
                "max_in_flight": max_in_flight,
            }
        )
    return BatchRescoreExecution(
        result=result,
        workflow_handles=(),
    )


def _rescore_count_metrics(
    items: Sequence[BatchRescoreItem],
) -> dict[str, int]:
    scheduled = sum(
        item.status
        in {
            BatchRescoreItemStatus.SCHEDULED,
            BatchRescoreItemStatus.RECOVERED,
        }
        for item in items
    )
    already_scored = sum(
        item.status is BatchRescoreItemStatus.ALREADY_SCORED for item in items
    )
    failed = sum(
        item.status is BatchRescoreItemStatus.FAILED for item in items
    )
    in_flight = sum(
        item.status is BatchRescoreItemStatus.WORKFLOW_IN_FLIGHT
        for item in items
    )
    return {
        "scheduled": scheduled,
        "already_scored": already_scored,
        "failed": failed,
        "in_flight": in_flight,
    }


def count_rescore_submission_candidates(
    connection: Connection,
    *,
    experiment_name: str,
    generation_statuses: Sequence[GenerationRunStatus],
    generation_attempt_index: int | None,
    scoring_profile_id: str,
    scoring_profile_version: str,
    parser_profile_id: str,
    parser_version: str,
    score_attempt_index: int,
    dataset_name: str,
    dataset_split: str,
) -> int:
    return int(
        connection.execute(
            io.count_rescore_submission_candidates(
                experiment_name=experiment_name,
                generation_statuses=tuple(generation_statuses),
                generation_attempt_index=generation_attempt_index,
                scoring_profile_id=scoring_profile_id,
                scoring_profile_version=scoring_profile_version,
                parser_profile_id=parser_profile_id,
                parser_version=parser_version,
                score_attempt_index=score_attempt_index,
                dataset_name=dataset_name,
                dataset_split=dataset_split,
            )
        ).scalar_one()
    )


def load_rescore_submission_candidates(
    connection: Connection,
    *,
    experiment_name: str,
    generation_statuses: Sequence[GenerationRunStatus],
    generation_attempt_index: int | None,
    scoring_profile_id: str,
    scoring_profile_version: str,
    parser_profile_id: str,
    parser_version: str,
    score_attempt_index: int,
    dataset_name: str,
    dataset_split: str,
    limit: int,
    offset: int,
) -> tuple[RescoreGenerationCandidate, ...]:
    rows = connection.execute(
        io.select_rescore_submission_candidates(
            experiment_name=experiment_name,
            generation_statuses=tuple(generation_statuses),
            generation_attempt_index=generation_attempt_index,
            scoring_profile_id=scoring_profile_id,
            scoring_profile_version=scoring_profile_version,
            parser_profile_id=parser_profile_id,
            parser_version=parser_version,
            score_attempt_index=score_attempt_index,
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            limit=limit,
            offset=offset,
        )
    ).mappings()
    return tuple(rescore_submission_candidate_from_row(row) for row in rows)


def plan_or_schedule_rescore_item(
    candidate: RescoreGenerationCandidate,
    *,
    database_url: str,
    score_attempt_index: int,
    scoring_profile_id: str,
    scoring_profile_version: str,
    parser_profile_id: str,
    parser_version: str,
    dataset_name: str,
    dataset_split: str,
    dry_run: bool,
    recover_orphans: bool,
    schedule_workflow: ScheduleScoreWorkflow,
) -> tuple[BatchRescoreItem, Any | None]:
    score_attempt_id = stable_score_attempt_id(
        generation_run_id=candidate.generation_run_id,
        scoring_profile_id=scoring_profile_id,
        scoring_profile_version=scoring_profile_version,
        parser_profile_id=parser_profile_id,
        parser_version=parser_version,
        attempt_index=candidate.score_attempt_index,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
    )
    workflow_id = platform_scoring_workflow_id(score_attempt_id)
    if candidate.existing_score_attempt_id is not None:
        return (
            batch_rescore_item(
                candidate,
                score_attempt_id=score_attempt_id,
                workflow_id=workflow_id,
                status=BatchRescoreItemStatus.ALREADY_SCORED,
            ),
            None,
        )
    if dry_run:
        return (
            batch_rescore_item(
                candidate,
                score_attempt_id=score_attempt_id,
                workflow_id=workflow_id,
                status=BatchRescoreItemStatus.WOULD_SCHEDULE,
            ),
            None,
        )
    try:
        scheduled = schedule_workflow(
            database_url=database_url,
            generation_run_id=candidate.generation_run_id,
            score_attempt_index=candidate.score_attempt_index,
            scoring_profile_id=scoring_profile_id,
            scoring_profile_version=scoring_profile_version,
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            recover_orphans=recover_orphans,
        )
    except Exception as error:
        return (
            batch_rescore_item(
                candidate,
                score_attempt_id=score_attempt_id,
                workflow_id=workflow_id,
                status=BatchRescoreItemStatus.FAILED,
                failure=failure_metadata_from_exception(error),
            ),
            None,
        )
    if scheduled.scheduled:
        status = (
            BatchRescoreItemStatus.RECOVERED
            if scheduled.recovered
            else BatchRescoreItemStatus.SCHEDULED
        )
        return (
            batch_rescore_item(
                candidate,
                score_attempt_id=scheduled.score_attempt_id,
                workflow_id=scheduled.workflow_id,
                status=status,
            ),
            scheduled.workflow_handle,
        )
    presence = classify_scoring_workflow_presence(
        database_url=database_url,
        score_attempt_id=score_attempt_id,
        workflow_id=workflow_id,
    )
    if presence is ScoringWorkflowPresence.COMPLETE:
        return (
            batch_rescore_item(
                candidate,
                score_attempt_id=score_attempt_id,
                workflow_id=workflow_id,
                status=BatchRescoreItemStatus.ALREADY_SCORED,
            ),
            None,
        )
    if presence is ScoringWorkflowPresence.ORPHAN:
        return (
            batch_rescore_item(
                candidate,
                score_attempt_id=score_attempt_id,
                workflow_id=workflow_id,
                status=BatchRescoreItemStatus.WORKFLOW_ORPHAN,
            ),
            None,
        )
    return (
        batch_rescore_item(
            candidate,
            score_attempt_id=scheduled.score_attempt_id,
            workflow_id=scheduled.workflow_id,
            status=BatchRescoreItemStatus.WORKFLOW_IN_FLIGHT,
        ),
        None,
    )


def batch_rescore_item(
    candidate: RescoreGenerationCandidate,
    *,
    score_attempt_id: str,
    workflow_id: str,
    status: BatchRescoreItemStatus,
    failure: FailureMetadataPayload | None = None,
) -> BatchRescoreItem:
    return BatchRescoreItem(
        prediction_id=candidate.prediction_id,
        fair_order_key=candidate.fair_order_key,
        generation_run_id=candidate.generation_run_id,
        score_attempt_id=score_attempt_id,
        workflow_id=workflow_id,
        status=status,
        existing_score_attempt_id=candidate.existing_score_attempt_id,
        failure=failure,
    )


def batch_rescore_result(
    *,
    experiment_name: str,
    generation_statuses: tuple[GenerationRunStatus, ...],
    generation_attempt_index: int | None,
    scoring_profile_id: str,
    scoring_profile_version: str,
    parser_profile_id: str,
    parser_version: str,
    score_attempt_index: int,
    dataset_name: str,
    dataset_split: str,
    dry_run: bool,
    max_in_flight: int,
    total_candidates: int,
    items: tuple[BatchRescoreItem, ...],
) -> BatchRescoreResult:
    already_scored_count = sum(
        item.status is BatchRescoreItemStatus.ALREADY_SCORED
        for item in items
    )
    scheduled_count = sum(
        item.status is BatchRescoreItemStatus.SCHEDULED for item in items
    )
    recovered_count = sum(
        item.status is BatchRescoreItemStatus.RECOVERED for item in items
    )
    in_flight_count = sum(
        item.status is BatchRescoreItemStatus.WORKFLOW_IN_FLIGHT
        for item in items
    )
    orphan_count = sum(
        item.status is BatchRescoreItemStatus.WORKFLOW_ORPHAN for item in items
    )
    failed_count = sum(
        item.status is BatchRescoreItemStatus.FAILED for item in items
    )
    needs_score_count = sum(
        item.status
        in {
            BatchRescoreItemStatus.WOULD_SCHEDULE,
            BatchRescoreItemStatus.SCHEDULED,
            BatchRescoreItemStatus.RECOVERED,
            BatchRescoreItemStatus.WORKFLOW_ORPHAN,
            BatchRescoreItemStatus.FAILED,
        }
        for item in items
    )
    return BatchRescoreResult(
        experiment_name=experiment_name,
        generation_statuses=generation_statuses,
        generation_attempt_index=generation_attempt_index,
        scoring_profile_id=scoring_profile_id,
        scoring_profile_version=scoring_profile_version,
        parser_profile_id=parser_profile_id,
        parser_version=parser_version,
        score_attempt_index=score_attempt_index,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        dry_run=dry_run,
        max_in_flight=max_in_flight,
        total_candidates=total_candidates,
        selected_count=len(items),
        already_scored_count=already_scored_count,
        needs_score_count=needs_score_count,
        scheduled_count=scheduled_count,
        recovered_count=recovered_count,
        in_flight_count=in_flight_count,
        orphan_count=orphan_count,
        failed_count=failed_count,
        items=items,
    )


def rescore_submission_candidate_from_row(
    row: Any,
) -> RescoreGenerationCandidate:
    return RescoreGenerationCandidate(
        prediction_id=row["prediction_id"],
        fair_order_key=row["fair_order_key"],
        generation_run_id=row["generation_run_id"],
        score_attempt_index=row["score_attempt_index"],
        existing_score_attempt_id=row["existing_score_attempt_id"],
    )
def parse_rescore_producer_statuses(
    values: Sequence[str] | None,
) -> tuple[GenerationRunStatus, ...]:
    if not values:
        return DEFAULT_RESCORE_SUBMISSION_STATUSES
    resolved: list[GenerationRunStatus] = []
    for value in values:
        try:
            resolved.append(GenerationRunStatus(value))
        except ValueError as error:
            allowed = ", ".join(status.value for status in GenerationRunStatus)
            raise ValueError(
                f"producer-status must be one of: {allowed}"
            ) from error
    return tuple(resolved)


def validate_rescore_request(
    *,
    chunk_size: int,
    limit: int | None,
    generation_attempt_index: int | None,
    score_attempt_index: int,
    generation_statuses: Sequence[GenerationRunStatus],
    max_in_flight: int,
) -> None:
    if not generation_statuses:
        raise ValueError("generation_statuses must not be empty")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if max_in_flight < 1:
        raise ValueError("max_in_flight must be positive")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive when provided")
    if generation_attempt_index is not None and generation_attempt_index < 0:
        raise ValueError("generation_attempt_index must be non-negative")
    if score_attempt_index < 0:
        raise ValueError("score_attempt_index must be non-negative")
