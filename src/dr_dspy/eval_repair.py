from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from typing import Any, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from dr_dspy import dbos_runtime
from dr_dspy.eval_reporting import validate_sql_identifier

GENERATION_REPAIR_ERROR = "Reconciled from DBOS failed generation workflow."
SCORING_REPAIR_ERROR = (
    "Reconciled from missing or failed DBOS scoring workflow."
)


class RepairCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_id: StrictStr
    task_id: StrictStr
    sample_index: StrictInt
    repetition_seed: StrictInt
    dbos_status: StrictStr
    dimensions: dict[str, Any] = Field(default_factory=dict)
    scoring_status: str | None = None


class RepairPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stranded_generations: list[RepairCandidate] = Field(default_factory=list)
    generation_retry_prediction_ids: list[StrictStr] = Field(
        default_factory=list
    )
    pending_scoring_prediction_ids: list[StrictStr] = Field(
        default_factory=list
    )
    stranded_scoring: list[RepairCandidate] = Field(default_factory=list)
    scoring_retry_prediction_ids: list[StrictStr] = Field(
        default_factory=list
    )


class RepairApplyResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repair_token: StrictStr
    stranded_generations_marked: StrictInt
    generation_retries_enqueued: StrictInt
    generation_retries_reset: StrictInt
    stranded_scoring_marked: StrictInt
    pending_scoring_enqueued: StrictInt
    scoring_retries_enqueued: StrictInt
    scoring_retries_marked_queued: StrictInt


class EvalDbosConfigLike(Protocol):
    database_url: str
    dbos_system_database_url: str


def validate_prediction_table(prediction_table: str) -> None:
    validate_sql_identifier(prediction_table)


def validate_columns(columns: Sequence[str]) -> None:
    for column in columns:
        validate_sql_identifier(column)


def order_clause(order_columns: Sequence[str]) -> str:
    validate_columns(order_columns)
    return ", ".join(order_columns)


def dimension_select_clause(dimension_columns: Sequence[str]) -> str:
    validate_columns(dimension_columns)
    if not dimension_columns:
        return ""
    return ",\n                    " + ",\n                    ".join(
        dimension_columns
    )


def fetch_prediction_ids_by_status(
    database_url: str,
    *,
    prediction_table: str,
    experiment_name: str,
    generation_status: str,
    scoring_statuses: Sequence[str] | None = None,
    order_columns: Sequence[str],
    limit: int,
) -> list[str]:
    validate_prediction_table(prediction_table)
    scoring_clause = ""
    params: list[Any] = [experiment_name, generation_status]
    if scoring_statuses is not None:
        scoring_clause = "AND scoring_status = ANY(%s)"
        params.append(list(scoring_statuses))
    params.append(limit)
    query = f"""
        SELECT prediction_id
        FROM {prediction_table}
        WHERE
            experiment_name = %s
            AND generation_status = %s
            {scoring_clause}
        ORDER BY {order_clause(order_columns)}
        LIMIT %s
    """
    with dbos_runtime.connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(cast(Any, query), tuple(params))
            rows = cur.fetchall()
    return [row[0] for row in rows]


def fetch_generation_error_prediction_ids(
    database_url: str,
    *,
    prediction_table: str,
    experiment_name: str,
    order_columns: Sequence[str],
    limit: int,
) -> list[str]:
    return fetch_prediction_ids_by_status(
        database_url,
        prediction_table=prediction_table,
        experiment_name=experiment_name,
        generation_status="generation_error",
        scoring_statuses=None,
        order_columns=order_columns,
        limit=limit,
    )


def fetch_pending_scoring_prediction_ids(
    database_url: str,
    *,
    prediction_table: str,
    experiment_name: str,
    order_columns: Sequence[str],
    limit: int,
) -> list[str]:
    return fetch_prediction_ids_by_status(
        database_url,
        prediction_table=prediction_table,
        experiment_name=experiment_name,
        generation_status="generated",
        scoring_statuses=("pending",),
        order_columns=order_columns,
        limit=limit,
    )


def fetch_score_error_prediction_ids(
    database_url: str,
    *,
    prediction_table: str,
    experiment_name: str,
    order_columns: Sequence[str],
    limit: int,
) -> list[str]:
    return fetch_prediction_ids_by_status(
        database_url,
        prediction_table=prediction_table,
        experiment_name=experiment_name,
        generation_status="generated",
        scoring_statuses=("score_error",),
        order_columns=order_columns,
        limit=limit,
    )


def fetch_scoreable_prediction_ids(
    database_url: str,
    *,
    prediction_table: str,
    experiment_name: str,
    order_columns: Sequence[str],
    limit: int,
) -> list[str]:
    return fetch_prediction_ids_by_status(
        database_url,
        prediction_table=prediction_table,
        experiment_name=experiment_name,
        generation_status="generated",
        scoring_statuses=("pending", "score_error"),
        order_columns=order_columns,
        limit=limit,
    )


def fetch_started_generation_repair_candidates(
    database_url: str,
    *,
    dbos_system_database_url: str,
    prediction_table: str,
    experiment_name: str,
    dimension_columns: Sequence[str],
    order_columns: Sequence[str],
) -> list[RepairCandidate]:
    validate_prediction_table(prediction_table)
    select_dimensions = dimension_select_clause(dimension_columns)
    query = f"""
        SELECT
            prediction_id,
            task_id,
            sample_index,
            repetition_seed
            {select_dimensions}
        FROM {prediction_table}
        WHERE
            experiment_name = %s
            AND generation_status = 'started'
        ORDER BY {order_clause(order_columns)}
    """
    with dbos_runtime.connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(cast(Any, query), (experiment_name,))
            app_rows = cur.fetchall()

    if not app_rows:
        return []

    prediction_ids = [row[0] for row in app_rows]
    workflow_ids = [
        dbos_runtime.generation_workflow_id(prediction_id)
        for prediction_id in prediction_ids
    ]
    with dbos_runtime.connect_db(dbos_system_database_url) as conn:
        with conn.cursor() as cur:
            failed_statuses = list(dbos_runtime.DBOS_FAILED_WORKFLOW_STATUSES)
            cur.execute(
                """
                SELECT workflow_uuid, status
                FROM dbos.workflow_status
                WHERE
                    workflow_uuid = ANY(%s)
                    AND status = ANY(%s)
                """,
                (workflow_ids, failed_statuses),
            )
            dbos_rows = cur.fetchall()

    dbos_status_by_prediction_id = {
        workflow_uuid.removeprefix("generate:"): status
        for workflow_uuid, status in dbos_rows
    }
    return [
        RepairCandidate(
            prediction_id=row[0],
            task_id=row[1],
            sample_index=row[2],
            repetition_seed=row[3],
            dbos_status=dbos_status_by_prediction_id[row[0]],
            dimensions={
                column: row[index + 4]
                for index, column in enumerate(dimension_columns)
            },
        )
        for row in app_rows
        if row[0] in dbos_status_by_prediction_id
    ]


def fetch_stranded_scoring_repair_candidates(
    database_url: str,
    *,
    dbos_system_database_url: str,
    prediction_table: str,
    experiment_name: str,
    dimension_columns: Sequence[str],
    order_columns: Sequence[str],
    limit: int,
) -> list[RepairCandidate]:
    validate_prediction_table(prediction_table)
    select_dimensions = dimension_select_clause(dimension_columns)
    query = f"""
        SELECT
            prediction_id,
            task_id,
            sample_index,
            repetition_seed,
            scoring_status
            {select_dimensions}
        FROM {prediction_table}
        WHERE
            experiment_name = %s
            AND generation_status = 'generated'
            AND scoring_status IN ('started', 'queued')
        ORDER BY {order_clause(order_columns)}
        LIMIT %s
    """
    with dbos_runtime.connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(cast(Any, query), (experiment_name, limit))
            app_rows = cur.fetchall()

    if not app_rows:
        return []

    prediction_ids = [row[0] for row in app_rows]
    stable_workflow_ids = [
        dbos_runtime.score_workflow_id(prediction_id)
        for prediction_id in prediction_ids
    ]
    retry_workflow_suffixes = [
        f":{prediction_id}" for prediction_id in prediction_ids
    ]
    with dbos_runtime.connect_db(dbos_system_database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT workflow_uuid, status
                FROM dbos.workflow_status
                WHERE
                    workflow_uuid = ANY(%s)
                    OR workflow_uuid LIKE ANY(%s)
                """,
                (
                    stable_workflow_ids,
                    [
                        f"score-retry:%{suffix}"
                        for suffix in retry_workflow_suffixes
                    ],
                ),
            )
            dbos_rows = cur.fetchall()

    active_prediction_ids: set[str] = set()
    failed_status_by_prediction_id: dict[str, str] = {}
    seen_prediction_ids: set[str] = set()
    for workflow_uuid, status in dbos_rows:
        prediction_id = workflow_uuid.rsplit(":", 1)[-1]
        seen_prediction_ids.add(prediction_id)
        if status in dbos_runtime.DBOS_ACTIVE_WORKFLOW_STATUSES:
            active_prediction_ids.add(prediction_id)
        if status in dbos_runtime.DBOS_FAILED_WORKFLOW_STATUSES:
            failed_status_by_prediction_id[prediction_id] = status

    candidates: list[RepairCandidate] = []
    for row in app_rows:
        prediction_id = row[0]
        if prediction_id in active_prediction_ids:
            continue
        dbos_status = failed_status_by_prediction_id.get(prediction_id)
        if dbos_status is None and prediction_id in seen_prediction_ids:
            continue
        candidates.append(
            RepairCandidate(
                prediction_id=prediction_id,
                task_id=row[1],
                sample_index=row[2],
                repetition_seed=row[3],
                scoring_status=row[4],
                dbos_status=(
                    dbos_status or dbos_runtime.MISSING_DBOS_WORKFLOW_STATUS
                ),
                dimensions={
                    column: row[index + 5]
                    for index, column in enumerate(dimension_columns)
                },
            )
        )
    return candidates


def mark_started_generations_as_repaired_errors(
    database_url: str,
    *,
    prediction_table: str,
    prediction_ids: Sequence[str],
) -> int:
    if not prediction_ids:
        return 0
    validate_prediction_table(prediction_table)
    query = f"""
        UPDATE {prediction_table}
        SET
            generation_status = 'generation_error',
            generation_error = %s,
            updated_at = now()
        WHERE
            prediction_id = ANY(%s)
            AND generation_status = 'started'
    """
    with dbos_runtime.connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                cast(Any, query),
                (GENERATION_REPAIR_ERROR, list(prediction_ids)),
            )
            return cur.rowcount if cur.rowcount is not None else 0


def mark_stranded_scoring_as_errors(
    database_url: str,
    *,
    prediction_table: str,
    prediction_ids: Sequence[str],
) -> int:
    if not prediction_ids:
        return 0
    validate_prediction_table(prediction_table)
    query = f"""
        UPDATE {prediction_table}
        SET
            scoring_status = 'score_error',
            scoring_error = %s,
            updated_at = now()
        WHERE
            prediction_id = ANY(%s)
            AND scoring_status IN ('started', 'queued')
    """
    with dbos_runtime.connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                cast(Any, query),
                (SCORING_REPAIR_ERROR, list(prediction_ids)),
            )
            return cur.rowcount if cur.rowcount is not None else 0


def mark_scoring_queued(
    database_url: str,
    *,
    prediction_table: str,
    prediction_ids: Sequence[str],
) -> int:
    if not prediction_ids:
        return 0
    validate_prediction_table(prediction_table)
    query = f"""
        UPDATE {prediction_table}
        SET
            scoring_status = 'queued',
            scoring_error = NULL,
            updated_at = now()
        WHERE prediction_id = ANY(%s)
            AND scoring_status IN ('pending', 'score_error')
    """
    with dbos_runtime.connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(cast(Any, query), (list(prediction_ids),))
            return cur.rowcount if cur.rowcount is not None else 0


def build_repair_plan(
    database_url: str,
    *,
    dbos_system_database_url: str,
    prediction_table: str,
    experiment_name: str,
    dimension_columns: Sequence[str],
    order_columns: Sequence[str],
    generation_limit: int,
    scoring_limit: int,
) -> RepairPlan:
    return RepairPlan(
        stranded_generations=fetch_started_generation_repair_candidates(
            database_url,
            dbos_system_database_url=dbos_system_database_url,
            prediction_table=prediction_table,
            experiment_name=experiment_name,
            dimension_columns=dimension_columns,
            order_columns=order_columns,
        ),
        generation_retry_prediction_ids=fetch_generation_error_prediction_ids(
            database_url,
            prediction_table=prediction_table,
            experiment_name=experiment_name,
            order_columns=order_columns,
            limit=generation_limit,
        ),
        pending_scoring_prediction_ids=fetch_pending_scoring_prediction_ids(
            database_url,
            prediction_table=prediction_table,
            experiment_name=experiment_name,
            order_columns=order_columns,
            limit=scoring_limit,
        ),
        stranded_scoring=fetch_stranded_scoring_repair_candidates(
            database_url,
            dbos_system_database_url=dbos_system_database_url,
            prediction_table=prediction_table,
            experiment_name=experiment_name,
            dimension_columns=dimension_columns,
            order_columns=order_columns,
            limit=scoring_limit,
        ),
        scoring_retry_prediction_ids=fetch_score_error_prediction_ids(
            database_url,
            prediction_table=prediction_table,
            experiment_name=experiment_name,
            order_columns=order_columns,
            limit=scoring_limit,
        ),
    )


def apply_repair(
    config: EvalDbosConfigLike,
    *,
    prediction_table: str,
    experiment_name: str,
    dimension_columns: Sequence[str],
    order_columns: Sequence[str],
    generation_limit: int,
    scoring_limit: int,
    score_timeout: float,
    fetch_generation_jobs: Callable[[Sequence[str]], Sequence[Any]],
    reset_generation_errors: Callable[[Sequence[str]], int],
    configure_runtime: Callable[[], None],
    enqueue_generation_jobs: Callable[[Sequence[Any], str], object],
    enqueue_score_jobs: Callable[[Sequence[str], float, str | None], None],
    repair_token: str | None = None,
) -> RepairApplyResult:
    resolved_repair_token = repair_token or uuid.uuid4().hex

    stranded_generations = fetch_started_generation_repair_candidates(
        config.database_url,
        dbos_system_database_url=config.dbos_system_database_url,
        prediction_table=prediction_table,
        experiment_name=experiment_name,
        dimension_columns=dimension_columns,
        order_columns=order_columns,
    )
    stranded_generations_marked = (
        mark_started_generations_as_repaired_errors(
            config.database_url,
            prediction_table=prediction_table,
            prediction_ids=[
                candidate.prediction_id
                for candidate in stranded_generations
            ],
        )
    )

    generation_retry_prediction_ids = fetch_generation_error_prediction_ids(
        config.database_url,
        prediction_table=prediction_table,
        experiment_name=experiment_name,
        order_columns=order_columns,
        limit=generation_limit,
    )
    generation_retry_jobs = fetch_generation_jobs(
        generation_retry_prediction_ids
    )
    configure_runtime()
    enqueue_generation_jobs(generation_retry_jobs, resolved_repair_token)
    generation_retries_reset = reset_generation_errors(
        generation_retry_prediction_ids
    )

    stranded_scoring = fetch_stranded_scoring_repair_candidates(
        config.database_url,
        dbos_system_database_url=config.dbos_system_database_url,
        prediction_table=prediction_table,
        experiment_name=experiment_name,
        dimension_columns=dimension_columns,
        order_columns=order_columns,
        limit=scoring_limit,
    )
    stranded_scoring_marked = mark_stranded_scoring_as_errors(
        config.database_url,
        prediction_table=prediction_table,
        prediction_ids=[
            candidate.prediction_id for candidate in stranded_scoring
        ],
    )

    pending_scoring_prediction_ids = fetch_pending_scoring_prediction_ids(
        config.database_url,
        prediction_table=prediction_table,
        experiment_name=experiment_name,
        order_columns=order_columns,
        limit=scoring_limit,
    )
    enqueue_score_jobs(pending_scoring_prediction_ids, score_timeout, None)
    pending_scoring_enqueued = mark_scoring_queued(
        config.database_url,
        prediction_table=prediction_table,
        prediction_ids=pending_scoring_prediction_ids,
    )

    scoring_retry_prediction_ids = fetch_score_error_prediction_ids(
        config.database_url,
        prediction_table=prediction_table,
        experiment_name=experiment_name,
        order_columns=order_columns,
        limit=scoring_limit,
    )
    enqueue_score_jobs(
        scoring_retry_prediction_ids,
        score_timeout,
        resolved_repair_token,
    )
    scoring_retries_marked_queued = mark_scoring_queued(
        config.database_url,
        prediction_table=prediction_table,
        prediction_ids=scoring_retry_prediction_ids,
    )

    return RepairApplyResult(
        repair_token=resolved_repair_token,
        stranded_generations_marked=stranded_generations_marked,
        generation_retries_enqueued=len(generation_retry_jobs),
        generation_retries_reset=generation_retries_reset,
        stranded_scoring_marked=stranded_scoring_marked,
        pending_scoring_enqueued=pending_scoring_enqueued,
        scoring_retries_enqueued=len(scoring_retry_prediction_ids),
        scoring_retries_marked_queued=scoring_retries_marked_queued,
    )
