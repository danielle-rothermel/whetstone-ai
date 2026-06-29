from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr
from sqlalchemy import update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Connection

from dr_dspy.db import io, schema
from dr_dspy.eval_failures import summarize_exception
from dr_dspy.hashing import sha256_json_digest
from dr_dspy.platform.fairness import fair_ordered_specs
from dr_dspy.platform.queue_worker import (
    PLATFORM_GENERATION_QUEUE_NAME,
    EnqueuePredictionWorkflowsResult,
    enqueue_prediction_graph_workflows,
)
from dr_dspy.records import (
    BatchSubmitItemRecord,
    BatchSubmitItemStatus,
    BatchSubmitOperationRecord,
    BatchSubmitOperationStatus,
    FailureMetadataPayload,
    PredictionSpecRecord,
)

DEFAULT_SUBMIT_CHUNK_SIZE = 500
BATCH_SUBMIT_ITEM_ID_LENGTH = 32
INSERT_STATUS_METADATA_KEY = "insert_status"
WORKFLOWS_METADATA_KEY = "workflows"

type EnqueueWorkflows = Callable[
    [str, Sequence[str], int, str],
    EnqueuePredictionWorkflowsResult,
]


class SubmittedPredictionItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_id: StrictStr
    fair_order_key: StrictStr
    status: BatchSubmitItemStatus
    insert_status: BatchSubmitItemStatus | None = None
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
    failed_count: StrictInt
    items: tuple[SubmittedPredictionItem, ...] = Field(default_factory=tuple)


def submit_prediction_specs(
    connection: Connection,
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
    enqueue_workflows: EnqueueWorkflows | None = None,
) -> SubmitPredictionSpecsResult:
    resolved_enqueue_workflows = enqueue_workflows or _enqueue_workflows
    ordered_specs = fair_ordered_specs(specs)
    validate_submit_specs(
        experiment_name=experiment_name,
        specs=ordered_specs,
    )
    created_at = datetime.now(UTC)
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

    items: list[SubmittedPredictionItem] = []
    inserted_count = 0
    already_present_count = 0
    enqueued_count = 0
    failed_count = 0
    item_index = 0
    for chunk in chunked(ordered_specs, chunk_size):
        inserted_ids = bulk_insert_prediction_specs(connection, chunk)
        insert_status_by_prediction_id = {
            spec.prediction_id: (
                BatchSubmitItemStatus.INSERTED
                if spec.prediction_id in inserted_ids
                else BatchSubmitItemStatus.ALREADY_PRESENT
            )
            for spec in chunk
        }
        inserted_count += len(inserted_ids)
        already_present_count += len(chunk) - len(inserted_ids)

        try:
            enqueue_result = resolved_enqueue_workflows(
                database_url,
                [spec.prediction_id for spec in chunk],
                attempt_index,
                queue_name,
            )
        except Exception as error:
            failure = failure_payload_from_exception(error)
            for spec in chunk:
                item = SubmittedPredictionItem(
                    prediction_id=spec.prediction_id,
                    fair_order_key=spec.fair_order_key,
                    status=BatchSubmitItemStatus.FAILED,
                    insert_status=insert_status_by_prediction_id[
                        spec.prediction_id
                    ],
                    failure=failure,
                )
                insert_batch_item(
                    connection,
                    record=batch_item_record(
                        operation_key=operation_key,
                        item_index=item_index,
                        item=item,
                    ),
                )
                items.append(item)
                item_index += 1
                failed_count += 1
            continue

        workflow_by_prediction_id = {
            workflow.prediction_id: workflow
            for workflow in enqueue_result.workflows
        }
        for spec in chunk:
            workflow = workflow_by_prediction_id[spec.prediction_id]
            status = (
                BatchSubmitItemStatus.ENQUEUED
                if workflow.enqueued
                else BatchSubmitItemStatus.ALREADY_PRESENT
            )
            if workflow.enqueued:
                enqueued_count += 1
            item = SubmittedPredictionItem(
                prediction_id=spec.prediction_id,
                fair_order_key=spec.fair_order_key,
                status=status,
                insert_status=insert_status_by_prediction_id[
                    spec.prediction_id
                ],
                workflow_id=workflow.workflow_id,
                generation_run_id=workflow.generation_run_id,
            )
            insert_batch_item(
                connection,
                record=batch_item_record(
                    operation_key=operation_key,
                    item_index=item_index,
                    item=item,
                ),
            )
            items.append(item)
            item_index += 1

    status = operation_status(
        requested_count=len(ordered_specs),
        failed_count=failed_count,
    )
    connection.execute(
        update(schema.batch_submit_operations)
        .where(schema.batch_submit_operations.c.operation_key == operation_key)
        .values(
            status=status.value,
            requested_count=len(ordered_specs),
            inserted_count=inserted_count,
            already_present_count=already_present_count,
            enqueued_count=enqueued_count,
            failed_count=failed_count,
            completed_at=datetime.now(UTC),
        )
    )
    return SubmitPredictionSpecsResult(
        operation_key=operation_key,
        experiment_name=experiment_name,
        queue_name=queue_name,
        requested_count=len(ordered_specs),
        inserted_count=inserted_count,
        already_present_count=already_present_count,
        enqueued_count=enqueued_count,
        failed_count=failed_count,
        items=tuple(items),
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
        .on_conflict_do_nothing(index_elements=["batch_submit_item_id"])
    )


def batch_item_record(
    *,
    operation_key: str,
    item_index: int,
    item: SubmittedPredictionItem,
) -> BatchSubmitItemRecord:
    enqueue_metadata = {
        key: value
        for key, value in {
            INSERT_STATUS_METADATA_KEY: (
                item.insert_status.value
                if item.insert_status is not None
                else None
            ),
            "workflow_id": item.workflow_id,
            "generation_run_id": item.generation_run_id,
        }.items()
        if value is not None
    }
    return BatchSubmitItemRecord(
        batch_submit_item_id=batch_submit_item_id(
            operation_key=operation_key,
            prediction_id=item.prediction_id,
        ),
        operation_key=operation_key,
        item_index=item_index,
        prediction_id=item.prediction_id,
        fair_order_key=item.fair_order_key,
        status=item.status,
        enqueue_metadata=enqueue_metadata,
        failure=item.failure,
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
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    return tuple(
        tuple(values[index:index + chunk_size])
        for index in range(0, len(values), chunk_size)
    )


def _enqueue_workflows(
    database_url: str,
    prediction_ids: Sequence[str],
    attempt_index: int,
    queue_name: str,
) -> EnqueuePredictionWorkflowsResult:
    return enqueue_prediction_graph_workflows(
        database_url=database_url,
        prediction_ids=prediction_ids,
        attempt_index=attempt_index,
        queue_name=queue_name,
    )
