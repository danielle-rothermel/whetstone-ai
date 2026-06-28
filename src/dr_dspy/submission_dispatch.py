from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from psycopg.types.json import Jsonb
from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr

from dr_dspy import dbos_runtime, eval_logging
from dr_dspy.eval_reporting import validate_sql_identifier
from dr_dspy.lm_utils import stable_json

SUBMIT_LOG_KIND = "submit"
SUBMIT_KEY_DIGEST_LENGTH = 32
SUBMIT_WORKFLOW_PREFIX = "submit"
DEFAULT_SUBMIT_TAIL_INTERVAL_SECONDS = 2.0


class SubmissionStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class SubmissionProgress(BaseModel):
    model_config = ConfigDict(extra="forbid")

    submit_key: StrictStr
    experiment_name: StrictStr
    workflow_id: StrictStr
    attempt: StrictInt
    submission_id: StrictStr
    status: SubmissionStatus
    total_jobs: StrictInt
    next_offset: StrictInt
    inserted_count: StrictInt
    enqueued_count: StrictInt
    existing_workflow_count: StrictInt
    batch_count: StrictInt
    last_error: str | None = None
    log_file: StrictStr


class SubmitBatchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_offset: StrictInt
    next_offset: StrictInt
    batch_size: StrictInt
    inserted: StrictInt
    enqueued: StrictInt
    existing_workflows: StrictInt


def submit_key(spec: Mapping[str, Any]) -> str:
    raw = stable_json(dict(spec))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[
        :SUBMIT_KEY_DIGEST_LENGTH
    ]


def submit_workflow_id(key: str, attempt: int) -> str:
    return f"{SUBMIT_WORKFLOW_PREFIX}:{key}:{attempt}"


def resolve_submit_log_path(
    *,
    log_root: Path,
    experiment_name: str,
    log_file: Path | None,
    hash_length: int,
) -> Path:
    return eval_logging.resolve_detail_log_path(
        log_root=log_root,
        experiment_name=experiment_name,
        log_kind=SUBMIT_LOG_KIND,
        log_file=log_file,
        hash_length=hash_length,
    )


def configure_submit_file_logging(
    log_file: Path, *, logger_name: str
) -> None:
    eval_logging.configure_detail_file_logging(
        log_file, logger_name=logger_name
    )


def emit_submit_log(
    event: str,
    payload: Mapping[str, Any],
    *,
    logger_name: str,
) -> None:
    eval_logging.emit_detail_log(event, payload, logger_name=logger_name)


def validate_submission_table(table_name: str) -> None:
    validate_sql_identifier(table_name)


def submission_table_sql(table_name: str) -> str:
    validate_submission_table(table_name)
    return f"""
CREATE TABLE IF NOT EXISTS {table_name} (
    submit_key              TEXT PRIMARY KEY,
    experiment_name         TEXT        NOT NULL,
    script_kind             TEXT        NOT NULL,
    workflow_id             TEXT        NOT NULL,
    attempt                 INTEGER     NOT NULL,
    submission_id           TEXT        NOT NULL,
    spec                    JSONB       NOT NULL,
    metadata                JSONB       NOT NULL DEFAULT '{{}}'::jsonb,
    total_jobs              INTEGER     NOT NULL,
    next_offset             INTEGER     NOT NULL DEFAULT 0,
    inserted_count          INTEGER     NOT NULL DEFAULT 0,
    enqueued_count          INTEGER     NOT NULL DEFAULT 0,
    existing_workflow_count INTEGER     NOT NULL DEFAULT 0,
    batch_count             INTEGER     NOT NULL DEFAULT 0,
    status                  TEXT        NOT NULL DEFAULT 'pending',
    last_error              TEXT,
    log_file                TEXT        NOT NULL,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at            TIMESTAMPTZ
)
"""


def submission_index_sql(table_name: str) -> tuple[str, ...]:
    validate_submission_table(table_name)
    index_name = f"idx_{table_name.replace('.', '_')}_experiment_status"
    return (
        f"CREATE INDEX IF NOT EXISTS {index_name} "
        f"ON {table_name}(experiment_name, status)",
    )


def row_to_progress(row: tuple[Any, ...]) -> SubmissionProgress:
    return SubmissionProgress(
        submit_key=row[0],
        experiment_name=row[1],
        workflow_id=row[2],
        attempt=row[3],
        submission_id=row[4],
        status=row[5],
        total_jobs=row[6],
        next_offset=row[7],
        inserted_count=row[8],
        enqueued_count=row[9],
        existing_workflow_count=row[10],
        batch_count=row[11],
        last_error=row[12],
        log_file=row[13],
    )


def fetch_submission_progress(
    database_url: str, *, table_name: str, key: str
) -> SubmissionProgress:
    validate_submission_table(table_name)
    with dbos_runtime.connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                cast(
                    Any,
                    f"""
                    SELECT
                        submit_key,
                        experiment_name,
                        workflow_id,
                        attempt,
                        submission_id,
                        status,
                        total_jobs,
                        next_offset,
                        inserted_count,
                        enqueued_count,
                        existing_workflow_count,
                        batch_count,
                        last_error,
                        log_file
                    FROM {table_name}
                    WHERE submit_key = %s
                    """,
                ),
                (key,),
            )
            row = cur.fetchone()
    if row is None:
        raise ValueError(f"submit_key not found: {key}")
    return row_to_progress(row)


def fetch_submission_spec(
    database_url: str, *, table_name: str, key: str
) -> dict[str, Any]:
    validate_submission_table(table_name)
    with dbos_runtime.connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                cast(
                    Any,
                    f"""
                    SELECT spec
                    FROM {table_name}
                    WHERE submit_key = %s
                    """,
                ),
                (key,),
            )
            row = cur.fetchone()
    if row is None:
        raise ValueError(f"submit_key not found: {key}")
    return dict(row[0])


def prepare_submission(
    database_url: str,
    *,
    table_name: str,
    key: str,
    experiment_name: str,
    script_kind: str,
    submission_id: str,
    spec: Mapping[str, Any],
    metadata: Mapping[str, Any],
    total_jobs: int,
    log_file: Path,
) -> SubmissionProgress:
    validate_submission_table(table_name)
    with dbos_runtime.connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                cast(
                    Any,
                    f"""
                    SELECT attempt, status, submission_id
                    FROM {table_name}
                    WHERE submit_key = %s
                    """,
                ),
                (key,),
            )
            row = cur.fetchone()
            if row is None:
                attempt = 1
                workflow_id = submit_workflow_id(key, attempt)
                cur.execute(
                    cast(
                        Any,
                        f"""
                        INSERT INTO {table_name} (
                            submit_key,
                            experiment_name,
                            script_kind,
                            workflow_id,
                            attempt,
                            submission_id,
                            spec,
                            metadata,
                            total_jobs,
                            log_file
                        )
                        VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        )
                        """,
                    ),
                    (
                        key,
                        experiment_name,
                        script_kind,
                        workflow_id,
                        attempt,
                        submission_id,
                        Jsonb(dict(spec)),
                        Jsonb(dict(metadata)),
                        total_jobs,
                        str(log_file),
                    ),
                )
            elif row[1] == SubmissionStatus.FAILED.value:
                attempt = int(row[0]) + 1
                workflow_id = submit_workflow_id(key, attempt)
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
                        WHERE submit_key = %s
                        """,
                    ),
                    (
                        workflow_id,
                        attempt,
                        Jsonb(dict(metadata)),
                        str(log_file),
                        key,
                    ),
                )
    return fetch_submission_progress(
        database_url, table_name=table_name, key=key
    )


def mark_submission_running(
    database_url: str, *, table_name: str, key: str
) -> None:
    validate_submission_table(table_name)
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
                    WHERE submit_key = %s
                    """,
                ),
                (key,),
            )


def record_batch_success(
    database_url: str,
    *,
    table_name: str,
    key: str,
    result: SubmitBatchResult,
) -> None:
    validate_submission_table(table_name)
    with dbos_runtime.connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                cast(
                    Any,
                    f"""
                    UPDATE {table_name}
                    SET
                        next_offset = GREATEST(next_offset, %s),
                        inserted_count = inserted_count + %s,
                        enqueued_count = enqueued_count + %s,
                        existing_workflow_count = (
                            existing_workflow_count + %s
                        ),
                        batch_count = batch_count + 1,
                        updated_at = now()
                    WHERE submit_key = %s
                    """,
                ),
                (
                    result.next_offset,
                    result.inserted,
                    result.enqueued,
                    result.existing_workflows,
                    key,
                ),
            )


def mark_submission_completed(
    database_url: str, *, table_name: str, key: str
) -> None:
    validate_submission_table(table_name)
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
                    WHERE submit_key = %s
                    """,
                ),
                (key,),
            )


def mark_submission_failed(
    database_url: str, *, table_name: str, key: str, error: str
) -> None:
    validate_submission_table(table_name)
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
                    WHERE submit_key = %s
                    """,
                ),
                (error, key),
            )


def ensure_submit_workflow(
    *,
    workflow_id: str,
    workflow: Callable[[str, str], str],
    database_url: str,
    submit_key: str,
) -> bool:
    if dbos_runtime.DBOS.get_workflow_status(workflow_id) is not None:
        return False
    with dbos_runtime.SetWorkflowID(workflow_id):
        try:
            dbos_runtime.DBOS.start_workflow(
                workflow, database_url, submit_key
            )
        except Exception:
            if dbos_runtime.DBOS.get_workflow_status(workflow_id) is not None:
                return False
            raise
    return True


def tail_submission_progress(
    *,
    database_url: str,
    table_name: str,
    submit_key: str,
    prediction_table: str,
    experiment_name: str,
    operator_log: Callable[..., None],
    interval_seconds: float = DEFAULT_SUBMIT_TAIL_INTERVAL_SECONDS,
) -> SubmissionProgress:
    last_line: str | None = None
    while True:
        progress = fetch_submission_progress(
            database_url, table_name=table_name, key=submit_key
        )
        counts = _fetch_phase_counts(
            database_url,
            prediction_table=prediction_table,
            experiment_name=experiment_name,
        )
        line = (
            f"submit {progress.status.value:<9} | "
            f"planned={progress.total_jobs:>6} | "
            f"offset={progress.next_offset:>6} | "
            f"inserted={progress.inserted_count:>6} | "
            f"enqueued={progress.enqueued_count:>6} | "
            f"existing={progress.existing_workflow_count:>6} | "
            f"batches={progress.batch_count:>4} | "
            f"gen_done={counts.get('generated', 0):>6} | "
            f"score_done={counts.get('scored', 0):>6}"
        )
        if progress.last_error:
            line = f"{line} | error={progress.last_error}"
        if line != last_line:
            style = "green"
            if progress.status is SubmissionStatus.FAILED:
                style = "red"
            elif progress.status in (
                SubmissionStatus.PENDING,
                SubmissionStatus.RUNNING,
            ):
                style = "cyan"
            operator_log(line, style=style)
            last_line = line
        if progress.status in (
            SubmissionStatus.COMPLETED,
            SubmissionStatus.FAILED,
        ):
            return progress
        time.sleep(interval_seconds)


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
