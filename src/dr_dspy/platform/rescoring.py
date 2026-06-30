from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
)
from sqlalchemy.engine import Connection, Engine

from dr_dspy.db import io
from dr_dspy.eval_failures import summarize_exception
from dr_dspy.humaneval.profiles import (
    HUMANEVAL_SCORING_PROFILE_ID,
    HUMANEVAL_SCORING_PROFILE_VERSION,
    resolve_humaneval_scoring_profile,
)
from dr_dspy.platform.scoring_workflow import (
    DEFAULT_HUMANEVAL_DATASET_NAME,
    DEFAULT_HUMANEVAL_DATASET_SPLIT,
    ScheduledScoreGenerationWorkflow,
    platform_scoring_workflow_id,
    schedule_score_generation_workflow,
)
from dr_dspy.records import (
    FailureMetadataPayload,
    GenerationRunStatus,
    stable_score_attempt_id,
)

DEFAULT_RESCORE_CHUNK_SIZE = 500


class BatchRescoreItemStatus(StrEnum):
    ALREADY_SCORED = "already_scored"
    WOULD_SCHEDULE = "would_schedule"
    SCHEDULED = "scheduled"
    WORKFLOW_ALREADY_PRESENT = "workflow_already_present"
    FAILED = "failed"


class RescoreGenerationCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_id: StrictStr
    fair_order_key: StrictStr
    generation_run_id: StrictStr
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
    generation_status: GenerationRunStatus
    generation_attempt_index: StrictInt | None
    scoring_profile_id: StrictStr
    scoring_profile_version: StrictStr
    parser_profile_id: StrictStr
    parser_version: StrictStr
    score_attempt_index: StrictInt
    dataset_name: StrictStr
    dataset_split: StrictStr
    dry_run: StrictBool
    selected_count: StrictInt
    already_scored_count: StrictInt
    pending_score_count: StrictInt
    scheduled_count: StrictInt
    already_scheduled_count: StrictInt
    failed_count: StrictInt
    items: tuple[BatchRescoreItem, ...] = Field(default_factory=tuple)


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
    ) -> ScheduledScoreGenerationWorkflow: ...


def rescore_generation_runs(
    engine: Engine,
    *,
    database_url: str,
    experiment_name: str,
    generation_status: GenerationRunStatus = GenerationRunStatus.SUCCESS,
    generation_attempt_index: int | None = None,
    scoring_profile_id: str = HUMANEVAL_SCORING_PROFILE_ID,
    scoring_profile_version: str = HUMANEVAL_SCORING_PROFILE_VERSION,
    score_attempt_index: int = 0,
    dataset_name: str = DEFAULT_HUMANEVAL_DATASET_NAME,
    dataset_split: str = DEFAULT_HUMANEVAL_DATASET_SPLIT,
    chunk_size: int = DEFAULT_RESCORE_CHUNK_SIZE,
    limit: int | None = None,
    dry_run: bool = False,
    schedule_workflow: ScheduleScoreWorkflow = (
        schedule_score_generation_workflow
    ),
) -> BatchRescoreResult:
    validate_rescore_request(
        chunk_size=chunk_size,
        limit=limit,
        generation_attempt_index=generation_attempt_index,
        score_attempt_index=score_attempt_index,
    )
    scoring_profile = resolve_humaneval_scoring_profile(
        scoring_profile_id=scoring_profile_id,
        scoring_profile_version=scoring_profile_version,
    )
    items: list[BatchRescoreItem] = []
    offset = 0
    while limit is None or offset < limit:
        page_limit = (
            chunk_size if limit is None else min(chunk_size, limit - offset)
        )
        with engine.begin() as connection:
            candidates = load_rescore_generation_candidates(
                connection,
                experiment_name=experiment_name,
                generation_status=generation_status,
                generation_attempt_index=generation_attempt_index,
                scoring_profile_id=scoring_profile.profile_id,
                scoring_profile_version=scoring_profile.version,
                parser_profile_id=scoring_profile.parser_profile.profile_id,
                parser_version=scoring_profile.parser_profile.version,
                score_attempt_index=score_attempt_index,
                limit=page_limit,
                offset=offset,
            )
        if not candidates:
            break
        for candidate in candidates:
            items.append(
                plan_or_schedule_rescore_item(
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
                    schedule_workflow=schedule_workflow,
                )
            )
        offset += len(candidates)
        if len(candidates) < page_limit:
            break

    return batch_rescore_result(
        experiment_name=experiment_name,
        generation_status=generation_status,
        generation_attempt_index=generation_attempt_index,
        scoring_profile_id=scoring_profile.profile_id,
        scoring_profile_version=scoring_profile.version,
        parser_profile_id=scoring_profile.parser_profile.profile_id,
        parser_version=scoring_profile.parser_profile.version,
        score_attempt_index=score_attempt_index,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        dry_run=dry_run,
        items=tuple(items),
    )


def load_rescore_generation_candidates(
    connection: Connection,
    *,
    experiment_name: str,
    generation_status: GenerationRunStatus,
    generation_attempt_index: int | None,
    scoring_profile_id: str,
    scoring_profile_version: str,
    parser_profile_id: str,
    parser_version: str,
    score_attempt_index: int,
    limit: int,
    offset: int,
) -> tuple[RescoreGenerationCandidate, ...]:
    rows = connection.execute(
        io.select_rescore_generation_candidates(
            experiment_name=experiment_name,
            generation_status=generation_status,
            generation_attempt_index=generation_attempt_index,
            scoring_profile_id=scoring_profile_id,
            scoring_profile_version=scoring_profile_version,
            parser_profile_id=parser_profile_id,
            parser_version=parser_version,
            score_attempt_index=score_attempt_index,
            limit=limit,
            offset=offset,
        )
    ).mappings()
    return tuple(rescore_generation_candidate_from_row(row) for row in rows)


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
    schedule_workflow: ScheduleScoreWorkflow,
) -> BatchRescoreItem:
    score_attempt_id = stable_score_attempt_id(
        generation_run_id=candidate.generation_run_id,
        scoring_profile_id=scoring_profile_id,
        scoring_profile_version=scoring_profile_version,
        parser_profile_id=parser_profile_id,
        parser_version=parser_version,
        attempt_index=score_attempt_index,
    )
    workflow_id = platform_scoring_workflow_id(score_attempt_id)
    if candidate.existing_score_attempt_id is not None:
        return batch_rescore_item(
            candidate,
            score_attempt_id=score_attempt_id,
            workflow_id=workflow_id,
            status=BatchRescoreItemStatus.ALREADY_SCORED,
        )
    if dry_run:
        return batch_rescore_item(
            candidate,
            score_attempt_id=score_attempt_id,
            workflow_id=workflow_id,
            status=BatchRescoreItemStatus.WOULD_SCHEDULE,
        )
    try:
        scheduled = schedule_workflow(
            database_url=database_url,
            generation_run_id=candidate.generation_run_id,
            score_attempt_index=score_attempt_index,
            scoring_profile_id=scoring_profile_id,
            scoring_profile_version=scoring_profile_version,
            dataset_name=dataset_name,
            dataset_split=dataset_split,
        )
    except Exception as error:
        return batch_rescore_item(
            candidate,
            score_attempt_id=score_attempt_id,
            workflow_id=workflow_id,
            status=BatchRescoreItemStatus.FAILED,
            failure=failure_payload_from_exception(error),
        )
    return batch_rescore_item(
        candidate,
        score_attempt_id=scheduled.score_attempt_id,
        workflow_id=scheduled.workflow_id,
        status=(
            BatchRescoreItemStatus.SCHEDULED
            if scheduled.scheduled
            else BatchRescoreItemStatus.WORKFLOW_ALREADY_PRESENT
        ),
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
    generation_status: GenerationRunStatus,
    generation_attempt_index: int | None,
    scoring_profile_id: str,
    scoring_profile_version: str,
    parser_profile_id: str,
    parser_version: str,
    score_attempt_index: int,
    dataset_name: str,
    dataset_split: str,
    dry_run: bool,
    items: tuple[BatchRescoreItem, ...],
) -> BatchRescoreResult:
    already_scored_count = sum(
        item.status is BatchRescoreItemStatus.ALREADY_SCORED
        for item in items
    )
    scheduled_count = sum(
        item.status is BatchRescoreItemStatus.SCHEDULED for item in items
    )
    already_scheduled_count = sum(
        item.status is BatchRescoreItemStatus.WORKFLOW_ALREADY_PRESENT
        for item in items
    )
    failed_count = sum(
        item.status is BatchRescoreItemStatus.FAILED for item in items
    )
    return BatchRescoreResult(
        experiment_name=experiment_name,
        generation_status=generation_status,
        generation_attempt_index=generation_attempt_index,
        scoring_profile_id=scoring_profile_id,
        scoring_profile_version=scoring_profile_version,
        parser_profile_id=parser_profile_id,
        parser_version=parser_version,
        score_attempt_index=score_attempt_index,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        dry_run=dry_run,
        selected_count=len(items),
        already_scored_count=already_scored_count,
        pending_score_count=len(items) - already_scored_count,
        scheduled_count=scheduled_count,
        already_scheduled_count=already_scheduled_count,
        failed_count=failed_count,
        items=items,
    )


def rescore_generation_candidate_from_row(
    row: Any,
) -> RescoreGenerationCandidate:
    return RescoreGenerationCandidate(
        prediction_id=row["prediction_id"],
        fair_order_key=row["fair_order_key"],
        generation_run_id=row["generation_run_id"],
        existing_score_attempt_id=row["existing_score_attempt_id"],
    )


def failure_payload_from_exception(
    error: BaseException,
) -> FailureMetadataPayload:
    summary = summarize_exception(error)
    return FailureMetadataPayload(
        failure_class=summary.failure_class,
        error_type=summary.failure_exception_type,
        message=summary.message,
        metadata=summary.failure_metadata,
    )


def validate_rescore_request(
    *,
    chunk_size: int,
    limit: int | None,
    generation_attempt_index: int | None,
    score_attempt_index: int,
) -> None:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive when provided")
    if generation_attempt_index is not None and generation_attempt_index < 0:
        raise ValueError("generation_attempt_index must be non-negative")
    if score_attempt_index < 0:
        raise ValueError("score_attempt_index must be non-negative")
