from __future__ import annotations

import hashlib
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from dr_dspy.eval_failures.recording import recordable_jsonb
from dr_dspy.harness import dbos as dbos_runtime
from dr_dspy.harness import logging as eval_logging
from dr_dspy.harness.reporting import validate_sql_identifier
from dr_dspy.lm.utils import stable_json

BATCH_OPERATION_TABLE_NAME = "dr_dspy_batch_operations"
BATCH_OPERATION_ITEM_TABLE_NAME = "dr_dspy_batch_operation_items"
OPERATION_KEY_DIGEST_LENGTH = 32
DEFAULT_OPERATION_TAIL_INTERVAL_SECONDS = 2.0


class BatchOperationKind(StrEnum):
    SUBMIT = "submit"
    ENQUEUE_SCORES = "enqueue_scores"
    REPAIR = "repair"


class BatchOperationStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class BatchOperationItemKind(StrEnum):
    SAMPLE = "sample"


class BatchDispatcherCompletionMode(StrEnum):
    OFFSET_TOTAL = "offset_total"
    EMPTY_BATCH = "empty_batch"


class BatchOperationProgress(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation_kind: BatchOperationKind
    operation_key: StrictStr
    experiment_name: StrictStr
    script_kind: StrictStr
    workflow_id: StrictStr
    attempt: StrictInt
    status: BatchOperationStatus
    total_items: StrictInt
    next_offset: StrictInt
    metadata: dict[StrictStr, Any] = Field(default_factory=dict)
    processed_count: StrictInt
    inserted_count: StrictInt
    enqueued_count: StrictInt
    existing_workflow_count: StrictInt
    marked_count: StrictInt
    batch_count: StrictInt
    counters: dict[StrictStr, StrictInt] = Field(default_factory=dict)
    last_error: str | None = None
    log_file: StrictStr


class BatchOperationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_offset: StrictInt
    next_offset: StrictInt
    batch_size: StrictInt
    processed: StrictInt = 0
    inserted: StrictInt = 0
    enqueued: StrictInt = 0
    existing_workflows: StrictInt = 0
    marked: StrictInt = 0
    counters: dict[StrictStr, StrictInt] = Field(default_factory=dict)


class BatchOperationItemWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_index: StrictInt
    item_count: StrictInt


class RepairApplyResultLike(Protocol):
    stranded_generations_marked: int
    generation_retries_enqueued: int
    generation_retries_existing: int
    generation_retries_reset: int
    stranded_scoring_marked: int
    pending_scoring_enqueued: int
    pending_scoring_existing: int
    pending_scoring_marked_queued: int
    scoring_retries_enqueued: int
    scoring_retries_existing: int
    scoring_retries_marked_queued: int

    @property
    def generation_retries_selected(self) -> int: ...

    @property
    def pending_scoring_selected(self) -> int: ...

    @property
    def scoring_retries_selected(self) -> int: ...


def operation_key(spec: Mapping[str, Any]) -> str:
    raw = stable_json(dict(spec))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[
        :OPERATION_KEY_DIGEST_LENGTH
    ]


def new_operation_key() -> str:
    return uuid.uuid4().hex[:OPERATION_KEY_DIGEST_LENGTH]


def operation_workflow_id(
    operation_kind: BatchOperationKind, operation_key: str, attempt: int
) -> str:
    return f"{operation_kind.value}:{operation_key}:{attempt}"


def resolve_operation_log_path(
    *,
    log_root: Path,
    experiment_name: str,
    operation_kind: BatchOperationKind,
    log_file: Path | None,
    hash_length: int,
) -> Path:
    return eval_logging.resolve_detail_log_path(
        log_root=log_root,
        experiment_name=experiment_name,
        log_kind=operation_kind.value,
        log_file=log_file,
        hash_length=hash_length,
    )


def configure_operation_file_logging(
    log_file: Path, *, logger_name: str
) -> None:
    eval_logging.configure_detail_file_logging(
        log_file, logger_name=logger_name
    )


def emit_operation_log(
    event: str,
    payload: Mapping[str, Any],
    *,
    logger_name: str,
) -> None:
    eval_logging.emit_detail_log(event, payload, logger_name=logger_name)


def validate_operation_table(
    table_name: str = BATCH_OPERATION_TABLE_NAME,
) -> None:
    validate_sql_identifier(table_name)


def operation_table_sql(
    table_name: str = BATCH_OPERATION_TABLE_NAME,
) -> str:
    validate_operation_table(table_name)
    return f"""
CREATE TABLE IF NOT EXISTS {table_name} (
    operation_kind          TEXT        NOT NULL,
    operation_key           TEXT        NOT NULL,
    experiment_name         TEXT        NOT NULL,
    script_kind             TEXT        NOT NULL,
    workflow_id             TEXT        NOT NULL,
    attempt                 INTEGER     NOT NULL,
    spec                    JSONB       NOT NULL,
    metadata                JSONB       NOT NULL DEFAULT '{{}}'::jsonb,
    total_items             INTEGER     NOT NULL,
    next_offset             INTEGER     NOT NULL DEFAULT 0,
    processed_count         INTEGER     NOT NULL DEFAULT 0,
    inserted_count          INTEGER     NOT NULL DEFAULT 0,
    enqueued_count          INTEGER     NOT NULL DEFAULT 0,
    existing_workflow_count INTEGER     NOT NULL DEFAULT 0,
    marked_count            INTEGER     NOT NULL DEFAULT 0,
    batch_count             INTEGER     NOT NULL DEFAULT 0,
    counters                JSONB       NOT NULL DEFAULT '{{}}'::jsonb,
    status                  TEXT        NOT NULL DEFAULT 'pending',
    last_error              TEXT,
    log_file                TEXT        NOT NULL,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at            TIMESTAMPTZ,
    PRIMARY KEY (operation_kind, operation_key)
)
"""


def operation_index_sql(
    table_name: str = BATCH_OPERATION_TABLE_NAME,
) -> tuple[str, ...]:
    validate_operation_table(table_name)
    index_name = f"idx_{table_name.replace('.', '_')}_experiment_status"
    return (
        f"CREATE INDEX IF NOT EXISTS {index_name} "
        f"ON {table_name}(experiment_name, operation_kind, status)",
    )


def operation_item_table_sql(
    *,
    table_name: str = BATCH_OPERATION_ITEM_TABLE_NAME,
    operation_table_name: str = BATCH_OPERATION_TABLE_NAME,
) -> str:
    validate_operation_table(table_name)
    validate_operation_table(operation_table_name)
    return f"""
CREATE TABLE IF NOT EXISTS {table_name} (
    operation_kind TEXT        NOT NULL,
    operation_key  TEXT        NOT NULL,
    item_kind      TEXT        NOT NULL,
    item_index     INTEGER     NOT NULL,
    payload        JSONB       NOT NULL,
    payload_digest TEXT        NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (operation_kind, operation_key, item_kind, item_index),
    FOREIGN KEY (operation_kind, operation_key)
        REFERENCES {operation_table_name}(operation_kind, operation_key)
        ON DELETE CASCADE
)
"""


def operation_item_index_sql(
    table_name: str = BATCH_OPERATION_ITEM_TABLE_NAME,
) -> tuple[str, ...]:
    validate_operation_table(table_name)
    index_name = f"idx_{table_name.replace('.', '_')}_operation_kind"
    return (
        f"CREATE INDEX IF NOT EXISTS {index_name} "
        f"ON {table_name}(operation_kind, operation_key, item_kind)",
    )


def payload_digest(payload: Mapping[str, Any]) -> str:
    raw = stable_json(dict(payload))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def record_operation_items(
    database_url: str,
    *,
    operation_kind: BatchOperationKind,
    operation_key: str,
    item_kind: BatchOperationItemKind,
    payloads: Sequence[Mapping[str, Any]],
    table_name: str = BATCH_OPERATION_ITEM_TABLE_NAME,
) -> None:
    validate_operation_table(table_name)
    expected_digests = [payload_digest(payload) for payload in payloads]
    with dbos_runtime.connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                cast(
                    Any,
                    f"""
                    SELECT item_index, payload_digest
                    FROM {table_name}
                    WHERE
                        operation_kind = %s
                        AND operation_key = %s
                        AND item_kind = %s
                    ORDER BY item_index
                    """,
                ),
                (operation_kind.value, operation_key, item_kind.value),
            )
            existing_rows = cur.fetchall()
            if existing_rows:
                existing_digests = [row[1] for row in existing_rows]
                expected_indexes = list(range(len(expected_digests)))
                existing_indexes = [row[0] for row in existing_rows]
                if (
                    existing_indexes != expected_indexes
                    or existing_digests != expected_digests
                ):
                    raise ValueError(
                        "operation item manifest mismatch: "
                        f"{operation_kind.value}:{operation_key}:"
                        f"{item_kind.value}"
                    )
                return
            if not expected_digests:
                return
            cur.executemany(
                cast(
                    Any,
                    f"""
                    INSERT INTO {table_name} (
                        operation_kind,
                        operation_key,
                        item_kind,
                        item_index,
                        payload,
                        payload_digest
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                ),
                [
                    (
                        operation_kind.value,
                        operation_key,
                        item_kind.value,
                        index,
                        recordable_jsonb(dict(payload)),
                        digest,
                    )
                    for index, (payload, digest) in enumerate(
                        zip(payloads, expected_digests, strict=True)
                    )
                ],
            )


def fetch_operation_items(
    database_url: str,
    *,
    operation_kind: BatchOperationKind,
    operation_key: str,
    item_kind: BatchOperationItemKind,
    start_index: int,
    limit: int,
    table_name: str = BATCH_OPERATION_ITEM_TABLE_NAME,
) -> list[dict[str, Any]]:
    validate_operation_table(table_name)
    if limit <= 0:
        return []
    with dbos_runtime.connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                cast(
                    Any,
                    f"""
                    SELECT payload
                    FROM {table_name}
                    WHERE
                        operation_kind = %s
                        AND operation_key = %s
                        AND item_kind = %s
                        AND item_index >= %s
                    ORDER BY item_index
                    LIMIT %s
                    """,
                ),
                (
                    operation_kind.value,
                    operation_key,
                    item_kind.value,
                    start_index,
                    limit,
                ),
            )
            rows = cur.fetchall()
    return [dict(row[0]) for row in rows]


def operation_item_window(
    *,
    start_offset: int,
    limit: int,
    total_items: int,
    items_per_group: int,
) -> BatchOperationItemWindow:
    if items_per_group <= 0:
        raise ValueError("items_per_group must be positive")
    if limit <= 0 or start_offset >= total_items:
        return BatchOperationItemWindow(start_index=0, item_count=0)
    end_offset = min(start_offset + limit, total_items)
    start_index = start_offset // items_per_group
    end_index = (end_offset - 1) // items_per_group
    return BatchOperationItemWindow(
        start_index=start_index,
        item_count=end_index - start_index + 1,
    )


def row_to_progress(row: tuple[Any, ...]) -> BatchOperationProgress:
    return BatchOperationProgress(
        operation_kind=row[0],
        operation_key=row[1],
        experiment_name=row[2],
        script_kind=row[3],
        workflow_id=row[4],
        attempt=row[5],
        status=row[6],
        total_items=row[7],
        next_offset=row[8],
        metadata=dict(row[9]),
        processed_count=row[10],
        inserted_count=row[11],
        enqueued_count=row[12],
        existing_workflow_count=row[13],
        marked_count=row[14],
        batch_count=row[15],
        counters=dict(row[16]),
        last_error=row[17],
        log_file=row[18],
    )


def fetch_operation_progress(
    database_url: str,
    *,
    operation_kind: BatchOperationKind,
    operation_key: str,
    table_name: str = BATCH_OPERATION_TABLE_NAME,
) -> BatchOperationProgress:
    validate_operation_table(table_name)
    with dbos_runtime.connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                cast(
                    Any,
                    f"""
                    SELECT
                        operation_kind,
                        operation_key,
                        experiment_name,
                        script_kind,
                        workflow_id,
                        attempt,
                        status,
                        total_items,
                        next_offset,
                        metadata,
                        processed_count,
                        inserted_count,
                        enqueued_count,
                        existing_workflow_count,
                        marked_count,
                        batch_count,
                        counters,
                        last_error,
                        log_file
                    FROM {table_name}
                    WHERE
                        operation_kind = %s
                        AND operation_key = %s
                    """,
                ),
                (operation_kind.value, operation_key),
            )
            row = cur.fetchone()
    if row is None:
        raise ValueError(
            f"operation not found: {operation_kind.value}:{operation_key}"
        )
    return row_to_progress(row)


def fetch_operation_spec(
    database_url: str,
    *,
    operation_kind: BatchOperationKind,
    operation_key: str,
    table_name: str = BATCH_OPERATION_TABLE_NAME,
) -> dict[str, Any]:
    validate_operation_table(table_name)
    with dbos_runtime.connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                cast(
                    Any,
                    f"""
                    SELECT spec
                    FROM {table_name}
                    WHERE
                        operation_kind = %s
                        AND operation_key = %s
                    """,
                ),
                (operation_kind.value, operation_key),
            )
            row = cur.fetchone()
    if row is None:
        raise ValueError(
            f"operation not found: {operation_kind.value}:{operation_key}"
        )
    return dict(row[0])


def prepare_operation(
    database_url: str,
    *,
    operation_kind: BatchOperationKind,
    operation_key: str,
    experiment_name: str,
    script_kind: str,
    spec: Mapping[str, Any],
    metadata: Mapping[str, Any],
    total_items: int,
    log_file: Path,
    table_name: str = BATCH_OPERATION_TABLE_NAME,
) -> BatchOperationProgress:
    validate_operation_table(table_name)
    with dbos_runtime.connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                cast(
                    Any,
                    f"""
                    SELECT attempt, status
                    FROM {table_name}
                    WHERE
                        operation_kind = %s
                        AND operation_key = %s
                    """,
                ),
                (operation_kind.value, operation_key),
            )
            row = cur.fetchone()
            if row is None:
                attempt = 1
                workflow_id = operation_workflow_id(
                    operation_kind, operation_key, attempt
                )
                cur.execute(
                    cast(
                        Any,
                        f"""
                        INSERT INTO {table_name} (
                            operation_kind,
                            operation_key,
                            experiment_name,
                            script_kind,
                            workflow_id,
                            attempt,
                            spec,
                            metadata,
                            total_items,
                            log_file
                        )
                        VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        )
                        """,
                    ),
                    (
                        operation_kind.value,
                        operation_key,
                        experiment_name,
                        script_kind,
                        workflow_id,
                        attempt,
                        recordable_jsonb(dict(spec)),
                        recordable_jsonb(dict(metadata)),
                        total_items,
                        str(log_file),
                    ),
                )
            elif row[1] == BatchOperationStatus.FAILED.value:
                attempt = int(row[0]) + 1
                workflow_id = operation_workflow_id(
                    operation_kind, operation_key, attempt
                )
                cur.execute(
                    cast(
                        Any,
                        f"""
                        UPDATE {table_name}
                        SET
                            workflow_id = %s,
                            attempt = %s,
                            metadata = %s,
                            status = 'pending',
                            last_error = NULL,
                            log_file = %s,
                            updated_at = now(),
                            completed_at = NULL
                        WHERE
                            operation_kind = %s
                            AND operation_key = %s
                        """,
                    ),
                    (
                        workflow_id,
                        attempt,
                        recordable_jsonb(dict(metadata)),
                        str(log_file),
                        operation_kind.value,
                        operation_key,
                    ),
                )
    return fetch_operation_progress(
        database_url,
        operation_kind=operation_kind,
        operation_key=operation_key,
        table_name=table_name,
    )


def mark_operation_running(
    database_url: str,
    *,
    operation_kind: BatchOperationKind,
    operation_key: str,
    table_name: str = BATCH_OPERATION_TABLE_NAME,
) -> None:
    validate_operation_table(table_name)
    with dbos_runtime.connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                cast(
                    Any,
                    f"""
                    UPDATE {table_name}
                    SET status = 'running',
                        last_error = NULL,
                        updated_at = now()
                    WHERE
                        operation_kind = %s
                        AND operation_key = %s
                    """,
                ),
                (operation_kind.value, operation_key),
            )


def merged_counters(
    current: Mapping[str, int], delta: Mapping[str, int]
) -> dict[str, int]:
    merged = dict(current)
    for key, value in delta.items():
        merged[key] = merged.get(key, 0) + value
    return merged


def record_operation_batch_success(
    database_url: str,
    *,
    operation_kind: BatchOperationKind,
    operation_key: str,
    result: BatchOperationResult,
    table_name: str = BATCH_OPERATION_TABLE_NAME,
) -> None:
    progress = fetch_operation_progress(
        database_url,
        operation_kind=operation_kind,
        operation_key=operation_key,
        table_name=table_name,
    )
    counters = merged_counters(progress.counters, result.counters)
    validate_operation_table(table_name)
    with dbos_runtime.connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                cast(
                    Any,
                    f"""
                    UPDATE {table_name}
                    SET
                        next_offset = GREATEST(next_offset, %s),
                        processed_count = processed_count + %s,
                        inserted_count = inserted_count + %s,
                        enqueued_count = enqueued_count + %s,
                        existing_workflow_count = (
                            existing_workflow_count + %s
                        ),
                        marked_count = marked_count + %s,
                        batch_count = batch_count + 1,
                        counters = %s,
                        updated_at = now()
                    WHERE
                        operation_kind = %s
                        AND operation_key = %s
                    """,
                ),
                (
                    result.next_offset,
                    result.processed,
                    result.inserted,
                    result.enqueued,
                    result.existing_workflows,
                    result.marked,
                    recordable_jsonb(counters),
                    operation_kind.value,
                    operation_key,
                ),
            )


def mark_operation_completed(
    database_url: str,
    *,
    operation_kind: BatchOperationKind,
    operation_key: str,
    table_name: str = BATCH_OPERATION_TABLE_NAME,
) -> None:
    validate_operation_table(table_name)
    with dbos_runtime.connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                cast(
                    Any,
                    f"""
                    UPDATE {table_name}
                    SET status = 'completed',
                        last_error = NULL,
                        updated_at = now(),
                        completed_at = now()
                    WHERE
                        operation_kind = %s
                        AND operation_key = %s
                    """,
                ),
                (operation_kind.value, operation_key),
            )


def mark_operation_failed(
    database_url: str,
    *,
    operation_kind: BatchOperationKind,
    operation_key: str,
    error: str,
    table_name: str = BATCH_OPERATION_TABLE_NAME,
) -> None:
    validate_operation_table(table_name)
    with dbos_runtime.connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                cast(
                    Any,
                    f"""
                    UPDATE {table_name}
                    SET status = 'failed',
                        last_error = %s,
                        updated_at = now()
                    WHERE
                        operation_kind = %s
                        AND operation_key = %s
                    """,
                ),
                (error, operation_kind.value, operation_key),
            )


def _is_dispatcher_complete(
    progress: BatchOperationProgress,
    completion_mode: BatchDispatcherCompletionMode,
) -> bool:
    if completion_mode is BatchDispatcherCompletionMode.OFFSET_TOTAL:
        return progress.next_offset >= progress.total_items
    return False


def _emit_completed_event(
    *,
    emit_log: Callable[[str, Mapping[str, Any]], None],
    completed_event: str | None,
    completed_payload: Callable[
        [BatchOperationProgress], Mapping[str, Any]
    ]
    | None,
    progress: BatchOperationProgress,
) -> None:
    if completed_event is None or completed_payload is None:
        return
    emit_log(completed_event, completed_payload(progress))


def run_operation_dispatcher(
    database_url: str,
    *,
    operation_kind: BatchOperationKind,
    operation_key: str,
    configure_logging: Callable[[Path], None],
    emit_log: Callable[[str, Mapping[str, Any]], None],
    started_event: str,
    started_payload: Callable[[BatchOperationProgress], Mapping[str, Any]],
    failed_event: str,
    batch_step: Callable[[str, str], BatchOperationResult],
    completion_mode: BatchDispatcherCompletionMode,
    completed_event: str | None = None,
    completed_payload: Callable[
        [BatchOperationProgress], Mapping[str, Any]
    ]
    | None = None,
    table_name: str = BATCH_OPERATION_TABLE_NAME,
) -> str:
    progress = fetch_operation_progress(
        database_url,
        operation_kind=operation_kind,
        operation_key=operation_key,
        table_name=table_name,
    )
    configure_logging(Path(progress.log_file))
    emit_log(started_event, started_payload(progress))
    mark_operation_running(
        database_url,
        operation_kind=operation_kind,
        operation_key=operation_key,
        table_name=table_name,
    )
    try:
        while True:
            progress = fetch_operation_progress(
                database_url,
                operation_kind=operation_kind,
                operation_key=operation_key,
                table_name=table_name,
            )
            if _is_dispatcher_complete(progress, completion_mode):
                mark_operation_completed(
                    database_url,
                    operation_kind=operation_kind,
                    operation_key=operation_key,
                    table_name=table_name,
                )
                _emit_completed_event(
                    emit_log=emit_log,
                    completed_event=completed_event,
                    completed_payload=completed_payload,
                    progress=progress,
                )
                return "completed"
            result = batch_step(database_url, operation_key)
            if (
                completion_mode is BatchDispatcherCompletionMode.EMPTY_BATCH
                and result.processed == 0
            ):
                mark_operation_completed(
                    database_url,
                    operation_kind=operation_kind,
                    operation_key=operation_key,
                    table_name=table_name,
                )
                _emit_completed_event(
                    emit_log=emit_log,
                    completed_event=completed_event,
                    completed_payload=completed_payload,
                    progress=progress,
                )
                return "completed"
    except Exception as error:
        error_text = repr(error)
        mark_operation_failed(
            database_url,
            operation_kind=operation_kind,
            operation_key=operation_key,
            error=error_text,
            table_name=table_name,
        )
        emit_log(
            failed_event, {"operation_key": operation_key, "error": error_text}
        )
        raise


def repair_batch_operation_result(
    *,
    progress: BatchOperationProgress,
    batch_size: int,
    repair_result: RepairApplyResultLike,
) -> BatchOperationResult:
    generation_batch_count = repair_result.generation_retries_selected
    scoring_batch_count = (
        repair_result.pending_scoring_selected
        + repair_result.scoring_retries_selected
    )
    processed = generation_batch_count + scoring_batch_count
    return BatchOperationResult(
        start_offset=progress.processed_count,
        next_offset=progress.processed_count + processed,
        batch_size=batch_size,
        processed=processed,
        enqueued=(
            repair_result.generation_retries_enqueued
            + repair_result.pending_scoring_enqueued
            + repair_result.scoring_retries_enqueued
        ),
        existing_workflows=(
            repair_result.generation_retries_existing
            + repair_result.pending_scoring_existing
            + repair_result.scoring_retries_existing
        ),
        marked=(
            repair_result.stranded_generations_marked
            + repair_result.generation_retries_reset
            + repair_result.stranded_scoring_marked
            + repair_result.pending_scoring_marked_queued
            + repair_result.scoring_retries_marked_queued
        ),
        counters={
            "generation_processed": generation_batch_count,
            "scoring_processed": scoring_batch_count,
            "stranded_generations_marked": (
                repair_result.stranded_generations_marked
            ),
            "generation_retries_reset": repair_result.generation_retries_reset,
            "stranded_scoring_marked": repair_result.stranded_scoring_marked,
            "pending_scoring_marked_queued": (
                repair_result.pending_scoring_marked_queued
            ),
            "scoring_retries_marked_queued": (
                repair_result.scoring_retries_marked_queued
            ),
        },
    )


def ensure_operation_workflow(
    *,
    workflow_id: str,
    workflow: Callable[[str, str], str],
    database_url: str,
    operation_key: str,
) -> bool:
    if dbos_runtime.DBOS.get_workflow_status(workflow_id) is not None:
        return False
    with dbos_runtime.SetWorkflowID(workflow_id):
        try:
            dbos_runtime.DBOS.start_workflow(
                workflow, database_url, operation_key
            )
        except dbos_runtime.WORKFLOW_START_RACE_ERRORS:
            return False
        except Exception as error:
            if dbos_runtime.workflow_start_raced(
                workflow_id=workflow_id,
                error=error,
            ):
                return False
            raise
    return True


def _fetch_phase_counts(
    database_url: str,
    *,
    prediction_table: str,
    experiment_name: str,
) -> dict[str, int]:
    validate_sql_identifier(prediction_table)
    with dbos_runtime.connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                cast(
                    Any,
                    f"""
                    SELECT generation_status, COUNT(*)
                    FROM {prediction_table}
                    WHERE experiment_name = %s
                    GROUP BY generation_status
                    """,
                ),
                (experiment_name,),
            )
            generation_rows = cur.fetchall()
            cur.execute(
                cast(
                    Any,
                    f"""
                    SELECT scoring_status, COUNT(*)
                    FROM {prediction_table}
                    WHERE experiment_name = %s
                    GROUP BY scoring_status
                    """,
                ),
                (experiment_name,),
            )
            scoring_rows = cur.fetchall()
    counts = {row[0]: row[1] for row in generation_rows}
    counts.update({row[0]: row[1] for row in scoring_rows})
    return counts


def format_counters(counters: Mapping[str, int]) -> str:
    if not counters:
        return "-"
    return ",".join(
        f"{key}:{value}" for key, value in sorted(counters.items())
    )


def format_operation_total(
    operation_kind: BatchOperationKind, total_items: int
) -> str:
    if (
        operation_kind
        in (BatchOperationKind.ENQUEUE_SCORES, BatchOperationKind.REPAIR)
        and total_items == 0
    ):
        return "open"
    return str(total_items)


def operation_progress_lines(
    *,
    operation_kind: BatchOperationKind,
    progress: BatchOperationProgress,
    counts: Mapping[str, int],
) -> tuple[str, ...]:
    total_label = format_operation_total(operation_kind, progress.total_items)
    main_line = (
        f"{operation_kind.value} {progress.status.value:<9} | "
        f"total={total_label:>6} | "
        f"offset={progress.next_offset:>6} | "
        f"processed={progress.processed_count:>6} | "
        f"inserted={progress.inserted_count:>6} | "
        f"enqueued={progress.enqueued_count:>6} | "
        f"existing={progress.existing_workflow_count:>6} | "
        f"marked={progress.marked_count:>6} | "
        f"batches={progress.batch_count:>4} | "
        f"gen_done={counts.get('generated', 0):>6} | "
        f"score_done={counts.get('scored', 0):>6}"
    )
    if progress.last_error:
        main_line = f"{main_line} | error={progress.last_error}"
    counter_line = format_counters(progress.counters)
    if counter_line == "-":
        return (main_line,)
    return (main_line, f"{operation_kind.value} counters | {counter_line}")


def tail_operation_progress(
    *,
    database_url: str,
    operation_kind: BatchOperationKind,
    operation_key: str,
    prediction_table: str,
    experiment_name: str,
    operator_log: Callable[..., None],
    interval_seconds: float = DEFAULT_OPERATION_TAIL_INTERVAL_SECONDS,
    table_name: str = BATCH_OPERATION_TABLE_NAME,
) -> BatchOperationProgress:
    last_lines: tuple[str, ...] = ()
    while True:
        progress = fetch_operation_progress(
            database_url,
            operation_kind=operation_kind,
            operation_key=operation_key,
            table_name=table_name,
        )
        counts = _fetch_phase_counts(
            database_url,
            prediction_table=prediction_table,
            experiment_name=experiment_name,
        )
        lines = operation_progress_lines(
            operation_kind=operation_kind,
            progress=progress,
            counts=counts,
        )
        if lines != last_lines:
            style = "green"
            if progress.status is BatchOperationStatus.FAILED:
                style = "red"
            elif progress.status in (
                BatchOperationStatus.PENDING,
                BatchOperationStatus.RUNNING,
            ):
                style = "cyan"
            for line in lines:
                operator_log(line, style=style)
            last_lines = lines
        if progress.status in (
            BatchOperationStatus.COMPLETED,
            BatchOperationStatus.FAILED,
        ):
            return progress
        time.sleep(interval_seconds)
