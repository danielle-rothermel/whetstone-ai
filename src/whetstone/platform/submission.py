"""Batch submission: whetstone's instantiation of the dr-platform facade.

The claim/lease machinery lives in dr-platform; this module wires the
domain in — experiment/spec seeding inside the registration
transaction, the generation-workflow enqueue target, whetstone's
failure classifier, and the frozen physical naming
(``PLATFORM_SCHEMA``). ``PredictionSpecRecord`` satisfies the
``SubmittableItem`` protocol directly (prediction_id / fair_order_key /
experiment_name).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from dr_platform import (
    BatchSubmitResult,
    EnqueueFailure,
    EnqueueOutcome,
    JsonlFieldNames,
    submit_batch,
    submit_batch_jsonl,
)
from sqlalchemy.dialects.postgresql import insert

from whetstone.db import io as db_io
from whetstone.db import schema
from whetstone.platform.platform_db import PLATFORM_SCHEMA
from whetstone.platform.queue_worker import (
    PLATFORM_GENERATION_QUEUE_NAME,
    EnqueuedPredictionWorkflow,
    enqueue_prediction_graph_workflow,
)
from whetstone.records import ExperimentRecord, PredictionSpecRecord

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
    from pathlib import Path

    from dr_platform.items import SubmittableItem
    from sqlalchemy.engine import Connection, Engine

DEFAULT_SUBMIT_CHUNK_SIZE = 500
GENERATION_RUN_ID_METADATA_KEY = "generation_run_id"
WHETSTONE_JSONL_FIELDS = JsonlFieldNames(
    item_id="prediction_id",
    order_key="fair_order_key",
    group_key="experiment_name",
)

type EnqueueWorkflow = Callable[
    [str, str, int, str],
    EnqueuedPredictionWorkflow,
]


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
) -> BatchSubmitResult:
    return submit_batch(
        engine,
        operation_key=operation_key,
        group_key=experiment_name,
        items=list(specs),
        enqueue=_enqueue_item(
            database_url=database_url,
            attempt_index=attempt_index,
            queue_name=queue_name,
            enqueue_workflow=enqueue_workflow,
        ),
        schema=PLATFORM_SCHEMA,
        seed=_seed_experiment_and_specs(experiment_name),
        submit_spec=submit_spec,
        metadata=metadata,
        chunk_size=chunk_size,
        classify_error=enqueue_failure_from_whetstone_exception,
    )


def submit_prediction_specs_jsonl(
    engine: Engine,
    *,
    database_url: str,
    operation_key: str,
    experiment_name: str,
    specs_file: Path,
    submit_spec: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    chunk_size: int = DEFAULT_SUBMIT_CHUNK_SIZE,
    attempt_index: int = 0,
    queue_name: str = PLATFORM_GENERATION_QUEUE_NAME,
    enqueue_workflow: EnqueueWorkflow | None = None,
) -> BatchSubmitResult:
    return submit_batch_jsonl(
        engine,
        operation_key=operation_key,
        group_key=experiment_name,
        items_file=specs_file,
        parse=PredictionSpecRecord.model_validate_json,
        fields=WHETSTONE_JSONL_FIELDS,
        enqueue=_enqueue_item(
            database_url=database_url,
            attempt_index=attempt_index,
            queue_name=queue_name,
            enqueue_workflow=enqueue_workflow,
        ),
        schema=PLATFORM_SCHEMA,
        seed=_seed_experiment_and_specs(experiment_name),
        submit_spec=submit_spec,
        metadata=metadata,
        chunk_size=chunk_size,
        classify_error=enqueue_failure_from_whetstone_exception,
    )


def _seed_experiment_and_specs(
    experiment_name: str,
) -> Callable[[Connection, Sequence[SubmittableItem]], set[str]]:
    def seed(
        connection: Connection,
        items: Sequence[SubmittableItem],
    ) -> set[str]:
        specs = cast("Sequence[PredictionSpecRecord]", items)
        connection.execute(
            idempotent_insert_experiment(
                ExperimentRecord(experiment_name=experiment_name)
            )
        )
        return bulk_insert_prediction_specs(connection, specs)

    return seed


def _enqueue_item(
    *,
    database_url: str,
    attempt_index: int,
    queue_name: str,
    enqueue_workflow: EnqueueWorkflow | None,
) -> Callable[[str], EnqueueOutcome]:
    resolved = enqueue_workflow or _enqueue_generation_workflow

    def enqueue(item_id: str) -> EnqueueOutcome:
        workflow = resolved(database_url, item_id, attempt_index, queue_name)
        return EnqueueOutcome(
            workflow_id=workflow.workflow_id,
            enqueued=workflow.enqueued,
            metadata={
                GENERATION_RUN_ID_METADATA_KEY: workflow.generation_run_id,
            },
        )

    return enqueue


def _enqueue_generation_workflow(
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


def enqueue_failure_from_whetstone_exception(
    error: BaseException,
) -> EnqueueFailure:
    """Classify through whetstone's policy chain (psycopg/DBOS aware).

    Field mapping mirrors the pre-extraction persisted JSONB shape
    (failure_class / error_type / message / metadata; no
    underlying_exception_type — parity with
    failure_metadata_from_exception).
    """
    from whetstone.eval_failures import summarize_exception

    summary = summarize_exception(error)
    return EnqueueFailure(
        failure_class=summary.failure_class,
        error_type=summary.failure_exception_type,
        message=summary.message,
        metadata=summary.failure_metadata,
    )


def bulk_insert_prediction_specs(
    connection: Connection,
    specs: Sequence[PredictionSpecRecord],
) -> set[str]:
    if not specs:
        return set()
    rows = [db_io.prediction_spec_row(spec) for spec in specs]
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
        .values(db_io.experiment_row(record))
        .on_conflict_do_nothing(index_elements=["experiment_name"])
    )
