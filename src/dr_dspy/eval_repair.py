from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from dr_dspy import dbos_runtime
from dr_dspy import job_ordering as shared_job_ordering
from dr_dspy.eval_reporting import validate_sql_identifier
from dr_dspy.failure_policy import RECOVERABLE_FAILURE_CLASSES, FailureClass
from dr_dspy.prediction_status import (
    GENERATION_RETRY_STATUSES,
    SCORING_QUEUEABLE_STATUSES,
    SCORING_RETRY_STATUSES,
    STRANDED_SCORING_STATUSES,
    GenerationStatus,
    ScoringStatus,
)

GENERATION_REPAIR_ERROR = "Reconciled from DBOS failed generation workflow."
SCORING_REPAIR_ERROR = (
    "Reconciled from missing or failed DBOS scoring workflow."
)
GENERATION_REPAIR_EXCEPTION_TYPE = "dr_dspy.repair.stranded_generation"
SCORING_REPAIR_EXCEPTION_TYPE = "dr_dspy.repair.stranded_scoring"
REPAIR_PLAN_COUNT_PAGE_SIZE = 10_000


class RepairCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_id: StrictStr
    task_id: StrictStr
    sample_index: StrictInt
    repetition_seed: StrictInt
    dbos_status: StrictStr
    dimensions: dict[str, Any] = Field(default_factory=dict)
    scoring_status: str | None = None


class RepairRetryCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_id: StrictStr
    failure_class: StrictStr


class RepairRetrySummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recoverable_count: StrictInt = 0
    excluded_count: StrictInt = 0

    @property
    def retryable_count(self) -> int:
        return self.recoverable_count


class RepairRetrySelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[RepairRetryCandidate] = Field(default_factory=list)
    excluded_count: StrictInt = 0

    @property
    def prediction_ids(self) -> list[str]:
        return [candidate.prediction_id for candidate in self.candidates]

    @property
    def summary(self) -> RepairRetrySummary:
        return RepairRetrySummary(
            recoverable_count=len(self.candidates),
            excluded_count=self.excluded_count,
        )


class RepairPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stranded_generation_count: StrictInt = 0
    generation_retry_summary: RepairRetrySummary = Field(
        default_factory=RepairRetrySummary
    )
    pending_scoring_count: StrictInt = 0
    stranded_scoring_count: StrictInt = 0
    scoring_retry_summary: RepairRetrySummary = Field(
        default_factory=RepairRetrySummary
    )


class RepairApplyResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repair_token: StrictStr
    stranded_generations_marked: StrictInt
    generation_retries_enqueued: StrictInt
    generation_retries_existing: StrictInt = 0
    generation_retries_reset: StrictInt
    stranded_scoring_marked: StrictInt
    pending_scoring_enqueued: StrictInt
    pending_scoring_existing: StrictInt = 0
    pending_scoring_marked_queued: StrictInt = 0
    scoring_retries_enqueued: StrictInt
    scoring_retries_existing: StrictInt = 0
    scoring_retries_marked_queued: StrictInt

    @property
    def generation_retries_selected(self) -> int:
        return (
            self.generation_retries_enqueued
            + self.generation_retries_existing
        )

    @property
    def pending_scoring_selected(self) -> int:
        return self.pending_scoring_enqueued + self.pending_scoring_existing

    @property
    def scoring_retries_selected(self) -> int:
        return self.scoring_retries_enqueued + self.scoring_retries_existing


class EvalDbosConfigLike(Protocol):
    database_url: str
    dbos_system_database_url: str


def validate_prediction_table(prediction_table: str) -> None:
    validate_sql_identifier(prediction_table)


def validate_columns(columns: Sequence[str]) -> None:
    for column in columns:
        validate_sql_identifier(column)


def order_clause(
    order_columns: Sequence[str], *, shuffle_key: str | None = None
) -> str:
    validate_columns(order_columns)
    ordered_columns = ", ".join(order_columns)
    if shuffle_key is None:
        return ordered_columns
    return f"md5(prediction_id || %s), {ordered_columns}"


def repair_order_shuffle_key(*parts: object) -> str:
    return shared_job_ordering.stable_order_key("repair", *parts)


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
    shuffle_key = repair_order_shuffle_key(
        "status",
        prediction_table,
        experiment_name,
        generation_status,
        ",".join(scoring_statuses or ()),
    )
    params.extend([shuffle_key, limit])
    query = f"""
        SELECT prediction_id
        FROM {prediction_table}
        WHERE
            experiment_name = %s
            AND generation_status = %s
            {scoring_clause}
        ORDER BY {order_clause(order_columns, shuffle_key=shuffle_key)}
        LIMIT %s
    """
    with dbos_runtime.connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(cast(Any, query), tuple(params))
            rows = cur.fetchall()
    return [row[0] for row in rows]


def recoverable_failure_class_values() -> list[str]:
    return [
        failure_class.value
        for failure_class in sorted(
            RECOVERABLE_FAILURE_CLASSES, key=lambda item: item.value
        )
    ]


def fetch_prediction_retry_selection(
    database_url: str,
    *,
    prediction_table: str,
    experiment_name: str,
    status_column: str,
    failure_class_column: str,
    retry_statuses: Sequence[str],
    generation_status: str | None = None,
    order_columns: Sequence[str],
    limit: int,
) -> RepairRetrySelection:
    validate_prediction_table(prediction_table)
    validate_columns([status_column, failure_class_column])
    generation_clause = ""
    recoverable_failure_classes = recoverable_failure_class_values()
    params: list[Any] = [
        experiment_name,
        list(retry_statuses),
        recoverable_failure_classes,
    ]
    if generation_status is not None:
        generation_clause = "AND generation_status = %s"
        params.append(generation_status)
    shuffle_key = repair_order_shuffle_key(
        "retry",
        prediction_table,
        experiment_name,
        status_column,
        failure_class_column,
        ",".join(retry_statuses),
        generation_status or "",
    )
    params.extend([shuffle_key, limit])
    query = f"""
        SELECT prediction_id, {failure_class_column}
        FROM {prediction_table}
        WHERE
            experiment_name = %s
            AND {status_column} = ANY(%s)
            AND {failure_class_column} = ANY(%s)
            {generation_clause}
        ORDER BY {order_clause(order_columns, shuffle_key=shuffle_key)}
        LIMIT %s
    """
    excluded_params: list[Any] = [
        experiment_name,
        list(retry_statuses),
        recoverable_failure_classes,
    ]
    if generation_status is not None:
        excluded_params.append(generation_status)
    excluded_query = f"""
        SELECT COUNT(*)
        FROM {prediction_table}
        WHERE
            experiment_name = %s
            AND {status_column} = ANY(%s)
            AND (
                {failure_class_column} IS NULL
                OR NOT ({failure_class_column} = ANY(%s))
            )
            {generation_clause}
    """
    with dbos_runtime.connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                cast(Any, query),
                tuple(params),
            )
            rows = cur.fetchall()
            cur.execute(cast(Any, excluded_query), tuple(excluded_params))
            excluded_row = cur.fetchone()
            excluded_count = int(excluded_row[0]) if excluded_row else 0
    return RepairRetrySelection(
        candidates=[
            RepairRetryCandidate(prediction_id=row[0], failure_class=row[1])
            for row in rows
        ],
        excluded_count=excluded_count,
    )


def count_prediction_retry_candidates(
    database_url: str,
    *,
    prediction_table: str,
    experiment_name: str,
    status_column: str,
    failure_class_column: str,
    retry_statuses: Sequence[str],
    generation_status: str | None = None,
) -> RepairRetrySummary:
    validate_prediction_table(prediction_table)
    validate_columns([status_column, failure_class_column])
    generation_clause = ""
    recoverable_failure_classes = recoverable_failure_class_values()
    params: list[Any] = [
        recoverable_failure_classes,
        recoverable_failure_classes,
        experiment_name,
        list(retry_statuses),
    ]
    if generation_status is not None:
        generation_clause = "AND generation_status = %s"
        params.append(generation_status)
    query = f"""
        SELECT
            COUNT(*) FILTER (
                WHERE {failure_class_column} = ANY(%s)
            ),
            COUNT(*) FILTER (
                WHERE {failure_class_column} IS NULL
                    OR NOT ({failure_class_column} = ANY(%s))
            )
        FROM {prediction_table}
        WHERE
            experiment_name = %s
            AND {status_column} = ANY(%s)
            {generation_clause}
    """
    with dbos_runtime.connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(cast(Any, query), tuple(params))
            row = cur.fetchone()
    if row is None:
        return RepairRetrySummary()
    return RepairRetrySummary(
        recoverable_count=int(row[0]),
        excluded_count=int(row[1]),
    )


def fetch_generation_retry_selection(
    database_url: str,
    *,
    prediction_table: str,
    experiment_name: str,
    order_columns: Sequence[str],
    limit: int,
) -> RepairRetrySelection:
    return fetch_prediction_retry_selection(
        database_url,
        prediction_table=prediction_table,
        experiment_name=experiment_name,
        status_column="generation_status",
        failure_class_column="generation_failure_class",
        retry_statuses=GENERATION_RETRY_STATUSES,
        order_columns=order_columns,
        limit=limit,
    )


def count_prediction_ids_by_status(
    database_url: str,
    *,
    prediction_table: str,
    experiment_name: str,
    generation_status: str,
    scoring_statuses: Sequence[str] | None = None,
) -> int:
    validate_prediction_table(prediction_table)
    scoring_clause = ""
    params: list[Any] = [experiment_name, generation_status]
    if scoring_statuses is not None:
        scoring_clause = "AND scoring_status = ANY(%s)"
        params.append(list(scoring_statuses))
    query = f"""
        SELECT COUNT(*)
        FROM {prediction_table}
        WHERE
            experiment_name = %s
            AND generation_status = %s
            {scoring_clause}
    """
    with dbos_runtime.connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(cast(Any, query), tuple(params))
            row = cur.fetchone()
    return int(row[0]) if row else 0


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
        generation_status=GenerationStatus.GENERATED.value,
        scoring_statuses=(ScoringStatus.PENDING.value,),
        order_columns=order_columns,
        limit=limit,
    )


def fetch_scoring_retry_selection(
    database_url: str,
    *,
    prediction_table: str,
    experiment_name: str,
    order_columns: Sequence[str],
    limit: int,
) -> RepairRetrySelection:
    return fetch_prediction_retry_selection(
        database_url,
        prediction_table=prediction_table,
        experiment_name=experiment_name,
        status_column="scoring_status",
        failure_class_column="scoring_failure_class",
        retry_statuses=SCORING_RETRY_STATUSES,
        generation_status=GenerationStatus.GENERATED.value,
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
        generation_status=GenerationStatus.GENERATED.value,
        scoring_statuses=SCORING_QUEUEABLE_STATUSES,
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
    limit: int | None = None,
    offset: int = 0,
) -> list[RepairCandidate]:
    validate_prediction_table(prediction_table)
    select_dimensions = dimension_select_clause(dimension_columns)
    limit_clause = ""
    offset_clause = ""
    params: list[Any] = [experiment_name, GenerationStatus.STARTED.value]
    shuffle_key = repair_order_shuffle_key(
        "stranded-generation",
        prediction_table,
        experiment_name,
    )
    params.append(shuffle_key)
    if limit is not None:
        limit_clause = "LIMIT %s"
        params.append(limit)
    if offset:
        offset_clause = "OFFSET %s"
        params.append(offset)
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
            AND generation_status = %s
        ORDER BY {order_clause(order_columns, shuffle_key=shuffle_key)}
        {limit_clause}
        {offset_clause}
    """
    with dbos_runtime.connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                cast(Any, query),
                tuple(params),
            )
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
    limit: int | None = None,
    offset: int = 0,
) -> list[RepairCandidate]:
    validate_prediction_table(prediction_table)
    select_dimensions = dimension_select_clause(dimension_columns)
    limit_clause = ""
    offset_clause = ""
    params: list[Any] = [
        experiment_name,
        GenerationStatus.GENERATED.value,
        list(STRANDED_SCORING_STATUSES),
    ]
    shuffle_key = repair_order_shuffle_key(
        "stranded-scoring",
        prediction_table,
        experiment_name,
    )
    params.append(shuffle_key)
    if limit is not None:
        limit_clause = "LIMIT %s"
        params.append(limit)
    if offset:
        offset_clause = "OFFSET %s"
        params.append(offset)
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
            AND generation_status = %s
            AND scoring_status = ANY(%s)
        ORDER BY {order_clause(order_columns, shuffle_key=shuffle_key)}
        {limit_clause}
        {offset_clause}
    """
    with dbos_runtime.connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                cast(Any, query),
                tuple(params),
            )
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


def count_started_generation_repair_candidates(
    database_url: str,
    *,
    dbos_system_database_url: str,
    prediction_table: str,
    experiment_name: str,
    dimension_columns: Sequence[str],
    order_columns: Sequence[str],
    page_size: int = REPAIR_PLAN_COUNT_PAGE_SIZE,
) -> int:
    app_count = count_prediction_ids_by_status(
        database_url,
        prediction_table=prediction_table,
        experiment_name=experiment_name,
        generation_status=GenerationStatus.STARTED.value,
    )
    total = 0
    for offset in range(0, app_count, page_size):
        total += len(
            fetch_started_generation_repair_candidates(
                database_url,
                dbos_system_database_url=dbos_system_database_url,
                prediction_table=prediction_table,
                experiment_name=experiment_name,
                dimension_columns=dimension_columns,
                order_columns=order_columns,
                limit=page_size,
                offset=offset,
            )
        )
    return total


def count_stranded_scoring_repair_candidates(
    database_url: str,
    *,
    dbos_system_database_url: str,
    prediction_table: str,
    experiment_name: str,
    dimension_columns: Sequence[str],
    order_columns: Sequence[str],
    page_size: int = REPAIR_PLAN_COUNT_PAGE_SIZE,
) -> int:
    app_count = count_prediction_ids_by_status(
        database_url,
        prediction_table=prediction_table,
        experiment_name=experiment_name,
        generation_status=GenerationStatus.GENERATED.value,
        scoring_statuses=STRANDED_SCORING_STATUSES,
    )
    total = 0
    for offset in range(0, app_count, page_size):
        total += len(
            fetch_stranded_scoring_repair_candidates(
                database_url,
                dbos_system_database_url=dbos_system_database_url,
                prediction_table=prediction_table,
                experiment_name=experiment_name,
                dimension_columns=dimension_columns,
                order_columns=order_columns,
                limit=page_size,
                offset=offset,
            )
        )
    return total


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
            generation_status = %s,
            generation_error = %s,
            generation_failure_class = %s,
            generation_exception_type = %s,
            generation_exception_message = %s,
            updated_at = now()
        WHERE
            prediction_id = ANY(%s)
            AND generation_status = %s
    """
    with dbos_runtime.connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                cast(Any, query),
                (
                    GenerationStatus.ERROR.value,
                    GENERATION_REPAIR_ERROR,
                    FailureClass.TRANSIENT.value,
                    GENERATION_REPAIR_EXCEPTION_TYPE,
                    GENERATION_REPAIR_ERROR,
                    list(prediction_ids),
                    GenerationStatus.STARTED.value,
                ),
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
            scoring_status = %s,
            scoring_error = %s,
            scoring_failure_class = %s,
            scoring_exception_type = %s,
            scoring_exception_message = %s,
            updated_at = now()
        WHERE
            prediction_id = ANY(%s)
            AND scoring_status = ANY(%s)
    """
    with dbos_runtime.connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                cast(Any, query),
                (
                    ScoringStatus.ERROR.value,
                    SCORING_REPAIR_ERROR,
                    FailureClass.TRANSIENT.value,
                    SCORING_REPAIR_EXCEPTION_TYPE,
                    SCORING_REPAIR_ERROR,
                    list(prediction_ids),
                    list(STRANDED_SCORING_STATUSES),
                ),
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
            scoring_status = %s,
            scoring_error = NULL,
            scoring_failure_class = NULL,
            scoring_exception_type = NULL,
            scoring_exception_message = NULL,
            updated_at = now()
        WHERE prediction_id = ANY(%s)
            AND scoring_status = ANY(%s)
    """
    with dbos_runtime.connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                cast(Any, query),
                (
                    ScoringStatus.QUEUED.value,
                    list(prediction_ids),
                    list(SCORING_QUEUEABLE_STATUSES),
                ),
            )
            return cur.rowcount if cur.rowcount is not None else 0


def unique_prediction_ids(prediction_ids: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(prediction_ids))


def build_repair_plan(
    database_url: str,
    *,
    dbos_system_database_url: str,
    prediction_table: str,
    experiment_name: str,
    dimension_columns: Sequence[str],
    order_columns: Sequence[str],
) -> RepairPlan:
    generation_retry_summary = count_prediction_retry_candidates(
        database_url,
        prediction_table=prediction_table,
        experiment_name=experiment_name,
        status_column="generation_status",
        failure_class_column="generation_failure_class",
        retry_statuses=GENERATION_RETRY_STATUSES,
    )
    scoring_retry_summary = count_prediction_retry_candidates(
        database_url,
        prediction_table=prediction_table,
        experiment_name=experiment_name,
        status_column="scoring_status",
        failure_class_column="scoring_failure_class",
        retry_statuses=SCORING_RETRY_STATUSES,
        generation_status=GenerationStatus.GENERATED.value,
    )
    return RepairPlan(
        stranded_generation_count=count_started_generation_repair_candidates(
            database_url,
            dbos_system_database_url=dbos_system_database_url,
            prediction_table=prediction_table,
            experiment_name=experiment_name,
            dimension_columns=dimension_columns,
            order_columns=order_columns,
        ),
        generation_retry_summary=generation_retry_summary,
        pending_scoring_count=count_prediction_ids_by_status(
            database_url,
            prediction_table=prediction_table,
            experiment_name=experiment_name,
            generation_status=GenerationStatus.GENERATED.value,
            scoring_statuses=(ScoringStatus.PENDING.value,),
        ),
        stranded_scoring_count=count_stranded_scoring_repair_candidates(
            database_url,
            dbos_system_database_url=dbos_system_database_url,
            prediction_table=prediction_table,
            experiment_name=experiment_name,
            dimension_columns=dimension_columns,
            order_columns=order_columns,
        ),
        scoring_retry_summary=scoring_retry_summary,
    )


def apply_repair_batch(
    config: EvalDbosConfigLike,
    *,
    prediction_table: str,
    experiment_name: str,
    dimension_columns: Sequence[str],
    order_columns: Sequence[str],
    batch_size: int,
    score_timeout: float,
    fetch_generation_jobs: Callable[[Sequence[str]], Sequence[Any]],
    reset_generation_errors: Callable[[Sequence[str]], int],
    enqueue_generation_jobs: Callable[
        [Sequence[Any], str], dbos_runtime.EnqueueWorkflowsResult
    ],
    enqueue_score_jobs: Callable[
        [Sequence[str], float, str | None], dbos_runtime.EnqueueWorkflowsResult
    ],
    repair_token: str,
) -> RepairApplyResult:
    generation_capacity = batch_size
    stranded_generation_ids: list[str] = []
    if generation_capacity > 0:
        stranded_generations = fetch_started_generation_repair_candidates(
            config.database_url,
            dbos_system_database_url=config.dbos_system_database_url,
            prediction_table=prediction_table,
            experiment_name=experiment_name,
            dimension_columns=dimension_columns,
            order_columns=order_columns,
            limit=generation_capacity,
        )
        stranded_generation_ids = [
            candidate.prediction_id for candidate in stranded_generations
        ]
    stranded_generations_marked = mark_started_generations_as_repaired_errors(
        config.database_url,
        prediction_table=prediction_table,
        prediction_ids=stranded_generation_ids,
    )

    generation_retry_prediction_ids = stranded_generation_ids
    remaining_generation_capacity = generation_capacity - len(
        generation_retry_prediction_ids
    )
    if remaining_generation_capacity > 0:
        retry_selection = fetch_generation_retry_selection(
            config.database_url,
            prediction_table=prediction_table,
            experiment_name=experiment_name,
            order_columns=order_columns,
            limit=remaining_generation_capacity,
        )
        generation_retry_prediction_ids = unique_prediction_ids(
            [
                *generation_retry_prediction_ids,
                *retry_selection.prediction_ids,
            ]
        )
    generation_retry_jobs = fetch_generation_jobs(
        generation_retry_prediction_ids
    )
    generation_enqueue_result = enqueue_generation_jobs(
        generation_retry_jobs, repair_token
    )
    generation_retries_reset = reset_generation_errors(
        generation_retry_prediction_ids
    )

    scoring_capacity = batch_size
    pending_scoring_prediction_ids: list[str] = []
    if scoring_capacity > 0:
        pending_scoring_prediction_ids = fetch_pending_scoring_prediction_ids(
            config.database_url,
            prediction_table=prediction_table,
            experiment_name=experiment_name,
            order_columns=order_columns,
            limit=scoring_capacity,
        )
    pending_score_result = enqueue_score_jobs(
        pending_scoring_prediction_ids, score_timeout, None
    )
    pending_scoring_marked = mark_scoring_queued(
        config.database_url,
        prediction_table=prediction_table,
        prediction_ids=pending_scoring_prediction_ids,
    )

    remaining_scoring_capacity = scoring_capacity - len(
        pending_scoring_prediction_ids
    )
    stranded_scoring_ids: list[str] = []
    if remaining_scoring_capacity > 0:
        stranded_scoring = fetch_stranded_scoring_repair_candidates(
            config.database_url,
            dbos_system_database_url=config.dbos_system_database_url,
            prediction_table=prediction_table,
            experiment_name=experiment_name,
            dimension_columns=dimension_columns,
            order_columns=order_columns,
            limit=remaining_scoring_capacity,
        )
        stranded_scoring_ids = [
            candidate.prediction_id for candidate in stranded_scoring
        ]
    stranded_scoring_marked = mark_stranded_scoring_as_errors(
        config.database_url,
        prediction_table=prediction_table,
        prediction_ids=stranded_scoring_ids,
    )

    scoring_retry_prediction_ids = stranded_scoring_ids
    remaining_scoring_capacity -= len(scoring_retry_prediction_ids)
    if remaining_scoring_capacity > 0:
        retry_selection = fetch_scoring_retry_selection(
            config.database_url,
            prediction_table=prediction_table,
            experiment_name=experiment_name,
            order_columns=order_columns,
            limit=remaining_scoring_capacity,
        )
        scoring_retry_prediction_ids = unique_prediction_ids(
            [
                *scoring_retry_prediction_ids,
                *retry_selection.prediction_ids,
            ]
        )
    scoring_retry_result = enqueue_score_jobs(
        scoring_retry_prediction_ids,
        score_timeout,
        repair_token,
    )
    scoring_retries_marked_queued = mark_scoring_queued(
        config.database_url,
        prediction_table=prediction_table,
        prediction_ids=scoring_retry_prediction_ids,
    )

    return RepairApplyResult(
        repair_token=repair_token,
        stranded_generations_marked=stranded_generations_marked,
        generation_retries_enqueued=generation_enqueue_result.enqueued,
        generation_retries_existing=generation_enqueue_result.existing,
        generation_retries_reset=generation_retries_reset,
        stranded_scoring_marked=stranded_scoring_marked,
        pending_scoring_enqueued=pending_score_result.enqueued,
        pending_scoring_existing=pending_score_result.existing,
        pending_scoring_marked_queued=pending_scoring_marked,
        scoring_retries_enqueued=scoring_retry_result.enqueued,
        scoring_retries_existing=scoring_retry_result.existing,
        scoring_retries_marked_queued=scoring_retries_marked_queued,
    )
