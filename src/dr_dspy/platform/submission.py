from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr
from sqlalchemy import Select, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Connection, Engine

from dr_dspy.db import io, schema
from dr_dspy.eval_failures import summarize_exception
from dr_dspy.hashing import sha256_json_digest
from dr_dspy.platform.fairness import fair_ordered_specs
from dr_dspy.platform.queue_worker import (
    PLATFORM_GENERATION_QUEUE_NAME,
    EnqueuedPredictionWorkflow,
    enqueue_prediction_graph_workflow,
)
from dr_dspy.records import (
    BatchSubmitItemEnqueueStatus,
    BatchSubmitItemInsertStatus,
    BatchSubmitItemRecord,
    BatchSubmitOperationRecord,
    BatchSubmitOperationStatus,
    ExperimentRecord,
    FailureMetadataPayload,
    PredictionSpecRecord,
)

DEFAULT_SUBMIT_CHUNK_SIZE = 500
BATCH_SUBMIT_ITEM_ID_LENGTH = 32
WORKFLOW_ID_METADATA_KEY = "workflow_id"
GENERATION_RUN_ID_METADATA_KEY = "generation_run_id"

type EnqueueWorkflow = Callable[
    [str, str, int, str],
    EnqueuedPredictionWorkflow,
]


class SubmittedPredictionItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_id: StrictStr
    fair_order_key: StrictStr
    insert_status: BatchSubmitItemInsertStatus
    enqueue_status: BatchSubmitItemEnqueueStatus
    workflow_id: StrictStr | None = None
    generation_run_id: StrictStr | None = None
    failure: FailureMetadataPayload | None = None


class SubmitPredictionSpecsResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation_key: StrictStr
    experiment_name: StrictStr
    queue_name: StrictStr
    requested_count: StrictInt
    inserted_count: StrictInt
    already_present_count: StrictInt
    enqueued_count: StrictInt
    already_scheduled_count: StrictInt
    failed_count: StrictInt
    items: tuple[SubmittedPredictionItem, ...] = Field(default_factory=tuple)


class EnqueueCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_id: StrictStr
    fair_order_key: StrictStr
    item_index: StrictInt
    insert_status: BatchSubmitItemInsertStatus


def submit_prediction_specs(
    engine: Engine,
    *,
    database_url: str,
    operation_key: str,
    experiment_name: str,
    specs: Iterable[PredictionSpecRecord],
    submit_spec: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    chunk_size: int = DEFAULT_SUBMIT_CHUNK_SIZE,
    attempt_index: int = 0,
    queue_name: str = PLATFORM_GENERATION_QUEUE_NAME,
    enqueue_workflow: EnqueueWorkflow | None = None,
) -> SubmitPredictionSpecsResult:
    validate_chunk_size(chunk_size)
    resolved_enqueue_workflow = enqueue_workflow or _enqueue_workflow
    ordered_specs = fair_ordered_specs(specs)
    validate_submit_specs(
        experiment_name=experiment_name,
        specs=ordered_specs,
    )

    with engine.begin() as connection:
        prepare_submission_records(
            connection,
            operation_key=operation_key,
            experiment_name=experiment_name,
            ordered_specs=ordered_specs,
            submit_spec=submit_spec,
            metadata=metadata,
            chunk_size=chunk_size,
        )

    for chunk in chunked(ordered_specs, chunk_size):
        with engine.begin() as connection:
            candidates = load_enqueue_candidates(
                connection,
                operation_key=operation_key,
                prediction_ids=tuple(spec.prediction_id for spec in chunk),
            )
        for candidate in candidates:
            try:
                workflow = resolved_enqueue_workflow(
                    database_url,
                    candidate.prediction_id,
                    attempt_index,
                    queue_name,
                )
                item = SubmittedPredictionItem(
                    prediction_id=candidate.prediction_id,
                    fair_order_key=candidate.fair_order_key,
                    insert_status=candidate.insert_status,
                    enqueue_status=(
                        BatchSubmitItemEnqueueStatus.ENQUEUED
                        if workflow.enqueued
                        else (
                            BatchSubmitItemEnqueueStatus
                            .WORKFLOW_ALREADY_PRESENT
                        )
                    ),
                    workflow_id=workflow.workflow_id,
                    generation_run_id=workflow.generation_run_id,
                )
            except Exception as error:
                item = SubmittedPredictionItem(
                    prediction_id=candidate.prediction_id,
                    fair_order_key=candidate.fair_order_key,
                    insert_status=candidate.insert_status,
                    enqueue_status=BatchSubmitItemEnqueueStatus.FAILED,
                    failure=failure_payload_from_exception(error),
                )
            with engine.begin() as connection:
                update_batch_item_outcome(
                    connection,
                    operation_key=operation_key,
                    item=item,
                )

    with engine.begin() as connection:
        return update_operation_summary(
            connection,
            operation_key=operation_key,
            experiment_name=experiment_name,
            queue_name=queue_name,
        )


def prepare_submission_records(
    connection: Connection,
    *,
    operation_key: str,
    experiment_name: str,
    ordered_specs: Sequence[PredictionSpecRecord],
    submit_spec: dict[str, Any] | None,
    metadata: dict[str, Any] | None,
    chunk_size: int = DEFAULT_SUBMIT_CHUNK_SIZE,
) -> None:
    validate_chunk_size(chunk_size)
    created_at = datetime.now(UTC)
    connection.execute(
        idempotent_insert_experiment(
            ExperimentRecord(
                experiment_name=experiment_name,
                created_at=created_at,
            )
        )
    )
    operation = BatchSubmitOperationRecord(
        operation_key=operation_key,
        experiment_name=experiment_name,
        status=BatchSubmitOperationStatus.PREPARED,
        requested_count=len(ordered_specs),
        spec=submit_spec or {},
        metadata=metadata or {},
        created_at=created_at,
    )
    connection.execute(idempotent_insert_batch_operation(operation))

    item_index_by_prediction_id = {
        spec.prediction_id: item_index
        for item_index, spec in enumerate(ordered_specs)
    }
    for chunk in chunked(ordered_specs, chunk_size):
        inserted_ids = bulk_insert_prediction_specs(connection, chunk)
        for spec in chunk:
            insert_status = (
                BatchSubmitItemInsertStatus.INSERTED
                if spec.prediction_id in inserted_ids
                else BatchSubmitItemInsertStatus.ALREADY_PRESENT
            )
            item = SubmittedPredictionItem(
                prediction_id=spec.prediction_id,
                fair_order_key=spec.fair_order_key,
                insert_status=insert_status,
                enqueue_status=BatchSubmitItemEnqueueStatus.PENDING,
            )
            insert_batch_item(
                connection,
                record=batch_item_record(
                    operation_key=operation_key,
                    item_index=item_index_by_prediction_id[
                        spec.prediction_id
                    ],
                    item=item,
                ),
            )


def validate_submit_specs(
    *,
    experiment_name: str,
    specs: Sequence[PredictionSpecRecord],
) -> None:
    for spec in specs:
        if spec.experiment_name != experiment_name:
            raise ValueError(
                "prediction spec experiment_name must match submit operation"
            )


def bulk_insert_prediction_specs(
    connection: Connection,
    specs: Sequence[PredictionSpecRecord],
) -> set[str]:
    if not specs:
        return set()
    rows = [io.prediction_spec_row(spec) for spec in specs]
    inserted = connection.execute(
        insert(schema.prediction_specs)
        .values(rows)
        .on_conflict_do_nothing(index_elements=["prediction_id"])
        .returning(schema.prediction_specs.c.prediction_id)
    )
    return {str(row[0]) for row in inserted}


def idempotent_insert_experiment(record: ExperimentRecord) -> Any:
    return (
        insert(schema.experiments)
        .values(io.experiment_row(record))
        .on_conflict_do_nothing(index_elements=["experiment_name"])
    )


def idempotent_insert_batch_operation(
    record: BatchSubmitOperationRecord,
) -> Any:
    return (
        insert(schema.batch_submit_operations)
        .values(io.batch_submit_operation_row(record))
        .on_conflict_do_nothing(index_elements=["operation_key"])
    )


def insert_batch_item(
    connection: Connection,
    *,
    record: BatchSubmitItemRecord,
) -> None:
    connection.execute(
        insert(schema.batch_submit_items)
        .values(io.batch_submit_item_row(record))
        .on_conflict_do_nothing(
            index_elements=["operation_key", "prediction_id"]
        )
    )


def load_enqueue_candidates(
    connection: Connection,
    *,
    operation_key: str,
    prediction_ids: Sequence[str],
) -> tuple[EnqueueCandidate, ...]:
    if not prediction_ids:
        return ()
    rows = connection.execute(
        select_batch_items_for_predictions(
            operation_key=operation_key,
            prediction_ids=prediction_ids,
        )
    ).mappings()
    return tuple(
        EnqueueCandidate(
            prediction_id=row["prediction_id"],
            fair_order_key=row["fair_order_key"],
            item_index=row["item_index"],
            insert_status=BatchSubmitItemInsertStatus(row["insert_status"]),
        )
        for row in rows
        if item_needs_enqueue(
            enqueue_status=BatchSubmitItemEnqueueStatus(
                row["enqueue_status"]
            ),
            enqueue_metadata=row["enqueue_metadata"],
        )
    )


def select_batch_items_for_predictions(
    *,
    operation_key: str,
    prediction_ids: Sequence[str],
) -> Select[tuple[Any, ...]]:
    return (
        select(schema.batch_submit_items)
        .where(schema.batch_submit_items.c.operation_key == operation_key)
        .where(schema.batch_submit_items.c.prediction_id.in_(prediction_ids))
        .order_by(
            schema.batch_submit_items.c.fair_order_key,
            schema.batch_submit_items.c.prediction_id,
        )
    )


def item_needs_enqueue(
    *,
    enqueue_status: BatchSubmitItemEnqueueStatus,
    enqueue_metadata: dict[str, Any],
) -> bool:
    if enqueue_status is BatchSubmitItemEnqueueStatus.ENQUEUED:
        return False
    if (
        enqueue_status is BatchSubmitItemEnqueueStatus.WORKFLOW_ALREADY_PRESENT
        and WORKFLOW_ID_METADATA_KEY in enqueue_metadata
    ):
        return False
    return True


def update_batch_item_outcome(
    connection: Connection,
    *,
    operation_key: str,
    item: SubmittedPredictionItem,
) -> None:
    existing = connection.execute(
        select(schema.batch_submit_items)
        .where(schema.batch_submit_items.c.operation_key == operation_key)
        .where(schema.batch_submit_items.c.prediction_id == item.prediction_id)
    ).mappings().one()
    updated_item = item.model_copy(
        update={
            "insert_status": (
                BatchSubmitItemInsertStatus(existing["insert_status"])
            )
        }
    )
    connection.execute(
        update(schema.batch_submit_items)
        .where(schema.batch_submit_items.c.operation_key == operation_key)
        .where(schema.batch_submit_items.c.prediction_id == item.prediction_id)
        .values(
            insert_status=updated_item.insert_status.value,
            enqueue_status=updated_item.enqueue_status.value,
            enqueue_metadata=enqueue_metadata_for_item(updated_item),
            failure=(
                updated_item.failure.model_dump(mode="json")
                if updated_item.failure is not None
                else None
            ),
        )
    )


def update_operation_summary(
    connection: Connection,
    *,
    operation_key: str,
    experiment_name: str,
    queue_name: str,
) -> SubmitPredictionSpecsResult:
    rows = tuple(
        connection.execute(
            select(schema.batch_submit_items)
            .where(schema.batch_submit_items.c.operation_key == operation_key)
            .order_by(
                schema.batch_submit_items.c.fair_order_key,
                schema.batch_submit_items.c.prediction_id,
            )
        ).mappings()
    )
    items = tuple(submitted_item_from_row(dict(row)) for row in rows)
    requested_count = len(items)
    inserted_count = sum(
        item.insert_status is BatchSubmitItemInsertStatus.INSERTED
        for item in items
    )
    already_present_count = sum(
        item.insert_status is BatchSubmitItemInsertStatus.ALREADY_PRESENT
        for item in items
    )
    enqueued_count = sum(
        item.enqueue_status is BatchSubmitItemEnqueueStatus.ENQUEUED
        for item in items
    )
    already_scheduled_count = sum(
        (
            item.enqueue_status
            is BatchSubmitItemEnqueueStatus.WORKFLOW_ALREADY_PRESENT
        )
        for item in items
    )
    failed_count = sum(
        item.enqueue_status is BatchSubmitItemEnqueueStatus.FAILED
        for item in items
    )
    status = operation_status(
        requested_count=requested_count,
        failed_count=failed_count,
    )
    connection.execute(
        update(schema.batch_submit_operations)
        .where(schema.batch_submit_operations.c.operation_key == operation_key)
        .values(
            status=status.value,
            requested_count=requested_count,
            inserted_count=inserted_count,
            already_present_count=already_present_count,
            enqueued_count=enqueued_count,
            already_scheduled_count=already_scheduled_count,
            failed_count=failed_count,
            completed_at=datetime.now(UTC),
        )
    )
    return SubmitPredictionSpecsResult(
        operation_key=operation_key,
        experiment_name=experiment_name,
        queue_name=queue_name,
        requested_count=requested_count,
        inserted_count=inserted_count,
        already_present_count=already_present_count,
        enqueued_count=enqueued_count,
        already_scheduled_count=already_scheduled_count,
        failed_count=failed_count,
        items=items,
    )


def batch_item_record(
    *,
    operation_key: str,
    item_index: int,
    item: SubmittedPredictionItem,
) -> BatchSubmitItemRecord:
    return BatchSubmitItemRecord(
        batch_submit_item_id=batch_submit_item_id(
            operation_key=operation_key,
            prediction_id=item.prediction_id,
        ),
        operation_key=operation_key,
        item_index=item_index,
        prediction_id=item.prediction_id,
        fair_order_key=item.fair_order_key,
        insert_status=item.insert_status,
        enqueue_status=item.enqueue_status,
        enqueue_metadata=enqueue_metadata_for_item(item),
        failure=item.failure,
    )


def enqueue_metadata_for_item(
    item: SubmittedPredictionItem,
) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            WORKFLOW_ID_METADATA_KEY: item.workflow_id,
            GENERATION_RUN_ID_METADATA_KEY: item.generation_run_id,
        }.items()
        if value is not None
    }


def submitted_item_from_row(row: dict[str, Any]) -> SubmittedPredictionItem:
    metadata = dict(row["enqueue_metadata"])
    failure = row["failure"]
    return SubmittedPredictionItem(
        prediction_id=row["prediction_id"],
        fair_order_key=row["fair_order_key"],
        insert_status=BatchSubmitItemInsertStatus(row["insert_status"]),
        enqueue_status=BatchSubmitItemEnqueueStatus(
            row["enqueue_status"]
        ),
        workflow_id=metadata.get(WORKFLOW_ID_METADATA_KEY),
        generation_run_id=metadata.get(GENERATION_RUN_ID_METADATA_KEY),
        failure=(
            FailureMetadataPayload.model_validate(failure)
            if failure is not None
            else None
        ),
    )


def batch_submit_item_id(
    *,
    operation_key: str,
    prediction_id: str,
) -> str:
    return sha256_json_digest(
        {
            "operation_key": operation_key,
            "prediction_id": prediction_id,
        },
        length=BATCH_SUBMIT_ITEM_ID_LENGTH,
    )


def operation_status(
    *,
    requested_count: int,
    failed_count: int,
) -> BatchSubmitOperationStatus:
    if failed_count == 0:
        return BatchSubmitOperationStatus.COMPLETED
    if failed_count >= requested_count:
        return BatchSubmitOperationStatus.ERROR
    return BatchSubmitOperationStatus.PARTIAL


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


def chunked(
    values: Sequence[PredictionSpecRecord],
    chunk_size: int,
) -> tuple[tuple[PredictionSpecRecord, ...], ...]:
    validate_chunk_size(chunk_size)
    return tuple(
        tuple(values[index:index + chunk_size])
        for index in range(0, len(values), chunk_size)
    )


def validate_chunk_size(chunk_size: int) -> None:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")


def _enqueue_workflow(
    database_url: str,
    prediction_id: str,
    attempt_index: int,
    queue_name: str,
) -> EnqueuedPredictionWorkflow:
    return enqueue_prediction_graph_workflow(
        database_url=database_url,
        prediction_id=prediction_id,
        attempt_index=attempt_index,
        queue_name=queue_name,
    )
