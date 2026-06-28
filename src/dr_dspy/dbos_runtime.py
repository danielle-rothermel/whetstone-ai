from __future__ import annotations

import hashlib
import os
import resource
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from enum import StrEnum
from typing import Any, Protocol, cast

import psycopg
from dbos import DBOS, DBOSConfig, SetWorkflowID
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, ConfigDict, StrictBool, StrictInt, StrictStr

DATABASE_URL_ENV = "DATABASE_URL"
DBOS_SYSTEM_DATABASE_URL_ENV = "DBOS_SYSTEM_DATABASE_URL"
DB_POOL_AUTO = "auto"
DEFAULT_DB_POOL_MARGIN = 8
DEFAULT_EXPERIMENT_QUEUE_HASH_LENGTH = 8


class QueueSelection(StrEnum):
    GENERATION = "generation"
    SCORING = "scoring"
    BOTH = "both"


class DbosWorkflowStatus(StrEnum):
    PENDING = "PENDING"
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"
    MAX_RECOVERY_ATTEMPTS_EXCEEDED = "MAX_RECOVERY_ATTEMPTS_EXCEEDED"
    CANCELLED = "CANCELLED"
    ENQUEUED = "ENQUEUED"
    DELAYED = "DELAYED"


DBOS_ACTIVE_WORKFLOW_STATUSES = (
    DbosWorkflowStatus.ENQUEUED.value,
    DbosWorkflowStatus.PENDING.value,
    DbosWorkflowStatus.DELAYED.value,
)
DBOS_FAILED_WORKFLOW_STATUSES = (
    DbosWorkflowStatus.ERROR.value,
    DbosWorkflowStatus.CANCELLED.value,
    DbosWorkflowStatus.MAX_RECOVERY_ATTEMPTS_EXCEEDED.value,
)
MISSING_DBOS_WORKFLOW_STATUS = "MISSING"


class EvalDbosConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    database_url: StrictStr
    dbos_system_database_url: StrictStr
    generation_concurrency: StrictInt
    scoring_concurrency: StrictInt


class DbPoolConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_size: StrictInt


class EvalQueueNames(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generation: StrictStr
    scoring: StrictStr


class EnqueueWorkflowsResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enqueued: StrictInt
    existing: StrictInt


class QueueNameConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generation_base_name: StrictStr
    scoring_base_name: StrictStr
    hash_length: StrictInt = DEFAULT_EXPERIMENT_QUEUE_HASH_LENGTH


class OpenFileLimitResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requested: StrictInt
    original_soft: StrictInt
    original_hard: StrictInt
    active_soft: StrictInt
    active_hard: StrictInt
    changed: StrictBool


class PredictionJobLike(Protocol):
    prediction_id: str
    experiment_name: str


DB_POOLS: dict[str, ConnectionPool] = {}


def resolve_database_url(
    database_url: str | None,
    *,
    database_url_env: str = DATABASE_URL_ENV,
    error_suffix: str = "",
) -> str:
    resolved = database_url or os.environ.get(database_url_env)
    if not resolved:
        suffix = f" {error_suffix}" if error_suffix else ""
        raise ValueError(
            f"--database-url or {database_url_env} is required{suffix}"
        )
    return resolved


def build_eval_dbos_config(
    *,
    database_url: str | None,
    dbos_system_database_url: str | None,
    generation_concurrency: int,
    scoring_concurrency: int,
    database_url_env: str = DATABASE_URL_ENV,
    dbos_system_database_url_env: str = DBOS_SYSTEM_DATABASE_URL_ENV,
    database_url_error_suffix: str = "",
) -> EvalDbosConfig:
    resolved_database_url = resolve_database_url(
        database_url,
        database_url_env=database_url_env,
        error_suffix=database_url_error_suffix,
    )
    return EvalDbosConfig(
        database_url=resolved_database_url,
        dbos_system_database_url=(
            dbos_system_database_url
            or os.environ.get(dbos_system_database_url_env)
            or resolved_database_url
        ),
        generation_concurrency=generation_concurrency,
        scoring_concurrency=scoring_concurrency,
    )


def build_dbos_config(config: EvalDbosConfig, *, app_name: str) -> DBOSConfig:
    return {
        "name": app_name,
        "system_database_url": config.dbos_system_database_url,
    }


@contextmanager
def connect_db(database_url: str) -> Iterator[psycopg.Connection[Any]]:
    pool = DB_POOLS.get(database_url)
    if pool is None:
        with psycopg.connect(database_url) as conn:
            yield conn
        return
    with pool.connection() as conn:
        yield conn


def close_db_connection_pools() -> None:
    for pool in DB_POOLS.values():
        pool.close()
    DB_POOLS.clear()


def configure_db_connection_pools(
    database_urls: Sequence[str], *, max_size: int
) -> None:
    for database_url in dict.fromkeys(database_urls):
        if database_url in DB_POOLS:
            continue
        DB_POOLS[database_url] = ConnectionPool(
            conninfo=database_url,
            min_size=0,
            max_size=max_size,
            open=True,
        )


def auto_db_pool_max_size(
    *,
    queue: QueueSelection,
    generation_concurrency: int,
    scoring_concurrency: int,
    margin: int = DEFAULT_DB_POOL_MARGIN,
) -> int:
    if queue is QueueSelection.GENERATION:
        return generation_concurrency + margin
    if queue is QueueSelection.SCORING:
        return scoring_concurrency + margin
    return generation_concurrency + scoring_concurrency + margin


def resolve_db_pool_config(
    *,
    raw_max_size: str,
    queue: QueueSelection,
    generation_concurrency: int,
    scoring_concurrency: int,
) -> DbPoolConfig:
    if raw_max_size == DB_POOL_AUTO:
        return DbPoolConfig(
            max_size=auto_db_pool_max_size(
                queue=queue,
                generation_concurrency=generation_concurrency,
                scoring_concurrency=scoring_concurrency,
            ),
        )
    max_size = int(raw_max_size)
    if max_size < 1:
        raise ValueError("--db-pool-max-size must be positive or 'auto'")
    return DbPoolConfig(max_size=max_size)


def configure_worker_db_connection_pools(
    config: EvalDbosConfig,
    *,
    queue: QueueSelection,
    raw_max_size: str,
) -> DbPoolConfig:
    pool_config = resolve_db_pool_config(
        raw_max_size=raw_max_size,
        queue=queue,
        generation_concurrency=config.generation_concurrency,
        scoring_concurrency=config.scoring_concurrency,
    )
    configure_db_connection_pools(
        [config.database_url, config.dbos_system_database_url],
        max_size=pool_config.max_size,
    )
    return pool_config


def create_schema(
    database_url: str,
    *,
    statements: Sequence[str],
) -> None:
    with connect_db(database_url) as conn:
        with conn.cursor() as cur:
            for statement in statements:
                cur.execute(cast(Any, statement))


def is_unlimited_resource_limit(value: int) -> bool:
    return value == resource.RLIM_INFINITY


def target_open_file_soft_limit(requested: int, hard_limit: int) -> int:
    if is_unlimited_resource_limit(hard_limit):
        return requested
    return min(requested, hard_limit)


def raise_open_file_limit(requested: int) -> OpenFileLimitResult:
    original_soft, original_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target_soft = target_open_file_soft_limit(requested, original_hard)
    changed = False

    if original_soft < target_soft:
        resource.setrlimit(
            resource.RLIMIT_NOFILE,
            (target_soft, original_hard),
        )
        changed = True

    active_soft, active_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    return OpenFileLimitResult(
        requested=requested,
        original_soft=original_soft,
        original_hard=original_hard,
        active_soft=active_soft,
        active_hard=active_hard,
        changed=changed,
    )


def format_resource_limit(value: int) -> str:
    if is_unlimited_resource_limit(value):
        return "unlimited"
    return str(value)


def open_file_limit_line(result: OpenFileLimitResult) -> str:
    changed = "yes" if result.changed else "no"
    return (
        f"{'Open Files':<14} | "
        f"requested={result.requested:>5} | "
        f"soft={format_resource_limit(result.active_soft):>9} | "
        f"hard={format_resource_limit(result.active_hard):>9} | "
        f"changed={changed}"
    )


def open_file_limit_style(result: OpenFileLimitResult) -> str:
    if result.active_soft < result.requested:
        return "yellow"
    return "green"


def experiment_hash(
    experiment_name: str,
    *,
    hash_length: int = DEFAULT_EXPERIMENT_QUEUE_HASH_LENGTH,
) -> str:
    return hashlib.sha256(experiment_name.encode("utf-8")).hexdigest()[
        :hash_length
    ]


def eval_queue_names(
    experiment_name: str, queue_config: QueueNameConfig
) -> EvalQueueNames:
    suffix = experiment_hash(
        experiment_name, hash_length=queue_config.hash_length
    )
    return EvalQueueNames(
        generation=f"{queue_config.generation_base_name}_{suffix}",
        scoring=f"{queue_config.scoring_base_name}_{suffix}",
    )


def queue_names_for_selection(
    selection: QueueSelection,
    *,
    experiment_name: str,
    queue_config: QueueNameConfig,
) -> tuple[str, ...]:
    queue_names = eval_queue_names(experiment_name, queue_config)
    if selection is QueueSelection.GENERATION:
        return (queue_names.generation,)
    if selection is QueueSelection.SCORING:
        return (queue_names.scoring,)
    return (queue_names.generation, queue_names.scoring)


def register_eval_queues(
    config: EvalDbosConfig,
    *,
    experiment_name: str,
    queue_config: QueueNameConfig,
) -> None:
    queue_names = eval_queue_names(experiment_name, queue_config)
    DBOS.register_queue(
        queue_names.generation,
        worker_concurrency=config.generation_concurrency,
        on_conflict="always_update",
    )
    DBOS.register_queue(
        queue_names.scoring,
        worker_concurrency=config.scoring_concurrency,
        on_conflict="always_update",
    )


def eval_queue_concurrency_by_name(
    config: EvalDbosConfig,
    *,
    experiment_name: str,
    queue_config: QueueNameConfig,
) -> dict[str, int]:
    queue_names = eval_queue_names(experiment_name, queue_config)
    return {
        queue_names.generation: config.generation_concurrency,
        queue_names.scoring: config.scoring_concurrency,
    }


def sync_existing_dbos_queue_concurrency(
    config: EvalDbosConfig,
    *,
    experiment_name: str,
    queue_config: QueueNameConfig,
) -> int:
    updated_count = 0
    try:
        with connect_db(config.dbos_system_database_url) as conn:
            with conn.cursor() as cur:
                for queue_name, worker_concurrency in (
                    eval_queue_concurrency_by_name(
                        config,
                        experiment_name=experiment_name,
                        queue_config=queue_config,
                    ).items()
                ):
                    cur.execute(
                        """
                        UPDATE dbos.queues
                        SET
                            worker_concurrency = %s,
                            updated_at = (
                                EXTRACT(epoch FROM now()) * 1000.0
                            )::bigint
                        WHERE name = %s
                        """,
                        (worker_concurrency, queue_name),
                    )
                    updated_count += cur.rowcount or 0
    except (psycopg.errors.UndefinedTable, psycopg.errors.InvalidSchemaName):
        return 0
    return updated_count


def queue_config_line(
    config: EvalDbosConfig, *, experiment_name: str
) -> str:
    return (
        f"{'Queue Config':<14} | "
        f"generation={config.generation_concurrency} | "
        f"scoring={config.scoring_concurrency} | "
        f"experiment={experiment_name}"
    )


def listen_to_selected_queue(
    selection: QueueSelection,
    *,
    experiment_name: str,
    queue_config: QueueNameConfig,
) -> None:
    DBOS.listen_queues(
        list(
            queue_names_for_selection(
                selection,
                experiment_name=experiment_name,
                queue_config=queue_config,
            )
        )
    )


def configure_dbos_runtime(
    config: EvalDbosConfig,
    *,
    app_name: str,
    experiment_name: str,
    queue_config: QueueNameConfig,
    queue: QueueSelection | None = None,
    consume_queues: bool = True,
    operator_log: Callable[[str], None] | None = None,
) -> None:
    DBOS(config=build_dbos_config(config, app_name=app_name))
    sync_existing_dbos_queue_concurrency(
        config, experiment_name=experiment_name, queue_config=queue_config
    )
    if queue is not None:
        listen_to_selected_queue(
            queue, experiment_name=experiment_name, queue_config=queue_config
        )
    elif not consume_queues:
        DBOS.listen_queues([])
    DBOS.launch()
    register_eval_queues(
        config, experiment_name=experiment_name, queue_config=queue_config
    )
    if operator_log is not None:
        operator_log(
            queue_config_line(config, experiment_name=experiment_name)
        )


def configure_pooled_worker_runtime(
    config: EvalDbosConfig,
    *,
    app_name: str,
    experiment_name: str,
    queue: QueueSelection,
    queue_config: QueueNameConfig,
    raw_db_pool_max_size: str,
    operator_log: Callable[[str], None] | None = None,
) -> DbPoolConfig:
    pool_config = configure_worker_db_connection_pools(
        config,
        queue=queue,
        raw_max_size=raw_db_pool_max_size,
    )
    configure_dbos_runtime(
        config,
        app_name=app_name,
        experiment_name=experiment_name,
        queue=queue,
        queue_config=queue_config,
        operator_log=operator_log,
    )
    return pool_config


def generation_workflow_id(
    prediction_id: str, *, retry_token: str | None = None
) -> str:
    if retry_token is None:
        return f"generate:{prediction_id}"
    return f"generate-retry:{retry_token}:{prediction_id}"


def score_workflow_id(
    prediction_id: str, *, retry_token: str | None = None
) -> str:
    if retry_token is None:
        return f"score:{prediction_id}"
    return f"score-retry:{retry_token}:{prediction_id}"


def enqueue_generation_workflows(
    database_url: str,
    jobs: Sequence[PredictionJobLike],
    *,
    queue_config: QueueNameConfig,
    workflow: Callable[..., str],
    score_timeout: float,
    retry_token: str | None = None,
) -> EnqueueWorkflowsResult:
    enqueued = 0
    existing = 0
    for job in jobs:
        workflow_id = generation_workflow_id(
            job.prediction_id, retry_token=retry_token
        )
        queue_names = eval_queue_names(job.experiment_name, queue_config)
        if DBOS.get_workflow_status(workflow_id) is not None:
            existing += 1
            continue
        with SetWorkflowID(workflow_id):
            try:
                DBOS.enqueue_workflow(
                    queue_names.generation,
                    workflow,
                    database_url,
                    job.prediction_id,
                    job.experiment_name,
                    score_timeout,
                )
            except Exception:
                if DBOS.get_workflow_status(workflow_id) is not None:
                    existing += 1
                    continue
                raise
        enqueued += 1
    return EnqueueWorkflowsResult(enqueued=enqueued, existing=existing)


def enqueue_score_workflow(
    database_url: str,
    prediction_id: str,
    *,
    experiment_name: str,
    queue_config: QueueNameConfig,
    workflow: Callable[..., str],
    timeout: float,
    retry_token: str | None = None,
) -> None:
    workflow_id = score_workflow_id(prediction_id, retry_token=retry_token)
    queue_names = eval_queue_names(experiment_name, queue_config)
    with SetWorkflowID(workflow_id):
        DBOS.enqueue_workflow(
            queue_names.scoring,
            workflow,
            database_url,
            prediction_id,
            timeout,
        )


def enqueue_score_workflows(
    database_url: str,
    prediction_ids: Sequence[str],
    *,
    experiment_name: str,
    queue_config: QueueNameConfig,
    workflow: Callable[..., str],
    timeout: float,
    retry_token: str | None = None,
) -> None:
    for prediction_id in prediction_ids:
        enqueue_score_workflow(
            database_url,
            prediction_id,
            experiment_name=experiment_name,
            queue_config=queue_config,
            workflow=workflow,
            timeout=timeout,
            retry_token=retry_token,
        )
