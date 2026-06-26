from __future__ import annotations

import logging
import os
import threading
from enum import Enum
from typing import Annotated, Any

import psycopg
import typer
from dbos import DBOS, DBOSConfig
from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr

from dr_dspy.event_log import DATABASE_URL_ENV
from dr_dspy.runtime import configure_multiprocessing, load_env_file

# Configuration

SCRIPT_KIND = "humaneval_eval_only_dbos_v0"
DBOS_APP_NAME = "dr-dspy-humaneval-eval-only"
DBOS_SYSTEM_DATABASE_URL_ENV = "DBOS_SYSTEM_DATABASE_URL"
GENERATION_QUEUE_NAME = "dr_dspy_humaneval_generation"
SCORING_QUEUE_NAME = "dr_dspy_humaneval_scoring"
DEFAULT_GENERATION_CONCURRENCY = 200
DEFAULT_SCORING_CONCURRENCY = 32


EXPERIMENTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS dr_dspy_eval_experiments (
    experiment_name TEXT PRIMARY KEY,
    script_kind     TEXT        NOT NULL,
    seed            INTEGER     NOT NULL,
    sample_count    INTEGER     NOT NULL,
    instruction     TEXT        NOT NULL,
    metadata        JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""

PREDICTIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS dr_dspy_eval_predictions (
    prediction_id        TEXT PRIMARY KEY,
    experiment_name      TEXT        NOT NULL
        REFERENCES dr_dspy_eval_experiments(experiment_name),
    script_kind          TEXT        NOT NULL,
    submission_id        TEXT        NOT NULL,
    task_id              TEXT        NOT NULL,
    sample_index         INTEGER     NOT NULL,
    model                TEXT        NOT NULL,
    temperature          DOUBLE PRECISION,
    repetition_seed      INTEGER     NOT NULL,
    prompt               TEXT        NOT NULL,
    test                 TEXT        NOT NULL,
    entry_point          TEXT        NOT NULL,
    reasoning            JSONB       NOT NULL DEFAULT '{}'::jsonb,
    generation_status    TEXT        NOT NULL DEFAULT 'pending',
    generation_error     TEXT,
    raw_code             TEXT,
    response_metadata    JSONB       NOT NULL DEFAULT '{}'::jsonb,
    usage_metadata       JSONB       NOT NULL DEFAULT '{}'::jsonb,
    provider_cost        DOUBLE PRECISION,
    scoring_status       TEXT        NOT NULL DEFAULT 'pending',
    score                DOUBLE PRECISION,
    scoring_error        TEXT,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    generated_at         TIMESTAMPTZ,
    scored_at            TIMESTAMPTZ,
    UNIQUE (
        experiment_name,
        task_id,
        model,
        temperature,
        repetition_seed
    )
)
"""

PREDICTION_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_dr_dspy_eval_predictions_experiment "
    "ON dr_dspy_eval_predictions(experiment_name)",
    "CREATE INDEX IF NOT EXISTS idx_dr_dspy_eval_predictions_generation "
    "ON dr_dspy_eval_predictions(experiment_name, generation_status)",
    "CREATE INDEX IF NOT EXISTS idx_dr_dspy_eval_predictions_scoring "
    "ON dr_dspy_eval_predictions(experiment_name, scoring_status)",
    "CREATE INDEX IF NOT EXISTS idx_dr_dspy_eval_predictions_model "
    "ON dr_dspy_eval_predictions(experiment_name, model, temperature)",
)


class QueueSelection(str, Enum):
    GENERATION = "generation"
    SCORING = "scoring"
    BOTH = "both"


class EvalDbosConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    database_url: StrictStr
    dbos_system_database_url: StrictStr
    generation_concurrency: StrictInt = DEFAULT_GENERATION_CONCURRENCY
    scoring_concurrency: StrictInt = DEFAULT_SCORING_CONCURRENCY


def resolve_database_url(database_url: str | None) -> str:
    resolved = database_url or os.environ.get(DATABASE_URL_ENV)
    if not resolved:
        raise ValueError(
            f"--database-url or {DATABASE_URL_ENV} is required for this "
            "Postgres-only DBOS harness"
        )
    return resolved


def build_eval_dbos_config(
    *,
    database_url: str | None,
    dbos_system_database_url: str | None,
    generation_concurrency: int,
    scoring_concurrency: int,
) -> EvalDbosConfig:
    resolved_database_url = resolve_database_url(database_url)
    return EvalDbosConfig(
        database_url=resolved_database_url,
        dbos_system_database_url=(
            dbos_system_database_url
            or os.environ.get(DBOS_SYSTEM_DATABASE_URL_ENV)
            or resolved_database_url
        ),
        generation_concurrency=generation_concurrency,
        scoring_concurrency=scoring_concurrency,
    )


def build_dbos_config(config: EvalDbosConfig) -> DBOSConfig:
    return {
        "name": DBOS_APP_NAME,
        "system_database_url": config.dbos_system_database_url,
    }


def create_eval_schema(database_url: str) -> None:
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(EXPERIMENTS_TABLE_SQL)
            cur.execute(PREDICTIONS_TABLE_SQL)
            for statement in PREDICTION_INDEX_SQL:
                cur.execute(statement)


def register_eval_queues(config: EvalDbosConfig) -> None:
    DBOS.register_queue(
        GENERATION_QUEUE_NAME,
        worker_concurrency=config.generation_concurrency,
    )
    DBOS.register_queue(
        SCORING_QUEUE_NAME,
        worker_concurrency=config.scoring_concurrency,
    )


def listen_to_selected_queue(selection: QueueSelection) -> None:
    if selection is QueueSelection.GENERATION:
        DBOS.listen_queues([GENERATION_QUEUE_NAME])
        return
    if selection is QueueSelection.SCORING:
        DBOS.listen_queues([SCORING_QUEUE_NAME])
        return
    DBOS.listen_queues([GENERATION_QUEUE_NAME, SCORING_QUEUE_NAME])


def configure_dbos_runtime(
    config: EvalDbosConfig, *, queue: QueueSelection | None = None
) -> None:
    DBOS(config=build_dbos_config(config))
    register_eval_queues(config)
    if queue is not None:
        listen_to_selected_queue(queue)
    DBOS.launch()


def fetch_status_counts(
    database_url: str, *, experiment_name: str | None
) -> list[dict[str, Any]]:
    where_clause = ""
    params: tuple[str, ...] = ()
    if experiment_name is not None:
        where_clause = "WHERE experiment_name = %s"
        params = (experiment_name,)

    query = f"""
        SELECT
            experiment_name,
            model,
            temperature,
            generation_status,
            scoring_status,
            COUNT(*) AS count
        FROM dr_dspy_eval_predictions
        {where_clause}
        GROUP BY
            experiment_name,
            model,
            temperature,
            generation_status,
            scoring_status
        ORDER BY
            experiment_name,
            model,
            temperature,
            generation_status,
            scoring_status
    """
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
    return [
        {
            "experiment_name": row[0],
            "model": row[1],
            "temperature": row[2],
            "generation_status": row[3],
            "scoring_status": row[4],
            "count": row[5],
        }
        for row in rows
    ]


app = typer.Typer(no_args_is_help=True)


def common_config(
    *,
    database_url: str | None,
    dbos_system_database_url: str | None,
    generation_concurrency: int,
    scoring_concurrency: int,
) -> EvalDbosConfig:
    load_env_file()
    return build_eval_dbos_config(
        database_url=database_url,
        dbos_system_database_url=dbos_system_database_url,
        generation_concurrency=generation_concurrency,
        scoring_concurrency=scoring_concurrency,
    )


@app.command()
def init_db(
    database_url: Annotated[
        str | None,
        typer.Option(
            "--database-url",
            help=f"Postgres URL; defaults to {DATABASE_URL_ENV}.",
        ),
    ] = None,
) -> None:
    config = common_config(
        database_url=database_url,
        dbos_system_database_url=None,
        generation_concurrency=DEFAULT_GENERATION_CONCURRENCY,
        scoring_concurrency=DEFAULT_SCORING_CONCURRENCY,
    )
    create_eval_schema(config.database_url)
    typer.echo("initialized dr-dspy eval tables")


@app.command()
def status(
    experiment_name: Annotated[
        str | None,
        typer.Option(
            "--experiment-name",
            help="Limit status to one experiment.",
        ),
    ] = None,
    database_url: Annotated[
        str | None,
        typer.Option(
            "--database-url",
            help=f"Postgres URL; defaults to {DATABASE_URL_ENV}.",
        ),
    ] = None,
) -> None:
    config = common_config(
        database_url=database_url,
        dbos_system_database_url=None,
        generation_concurrency=DEFAULT_GENERATION_CONCURRENCY,
        scoring_concurrency=DEFAULT_SCORING_CONCURRENCY,
    )
    rows = fetch_status_counts(
        config.database_url, experiment_name=experiment_name
    )
    if not rows:
        typer.echo("no prediction rows found")
        return
    for row in rows:
        typer.echo(
            "{experiment_name} | {model} | temp={temperature} | "
            "generation={generation_status} | scoring={scoring_status} | "
            "count={count}".format(**row)
        )


@app.command()
def worker(
    queue: Annotated[
        QueueSelection,
        typer.Option("--queue", help="Queue set this worker should consume."),
    ] = QueueSelection.BOTH,
    database_url: Annotated[
        str | None,
        typer.Option(
            "--database-url",
            help=f"Postgres URL; defaults to {DATABASE_URL_ENV}.",
        ),
    ] = None,
    dbos_system_database_url: Annotated[
        str | None,
        typer.Option(
            "--dbos-system-database-url",
            help=(
                "DBOS system database URL; defaults to "
                f"{DBOS_SYSTEM_DATABASE_URL_ENV} or DATABASE_URL."
            ),
        ),
    ] = None,
    generation_concurrency: Annotated[
        int, typer.Option("--generation-concurrency")
    ] = DEFAULT_GENERATION_CONCURRENCY,
    scoring_concurrency: Annotated[
        int, typer.Option("--scoring-concurrency")
    ] = DEFAULT_SCORING_CONCURRENCY,
) -> None:
    config = common_config(
        database_url=database_url,
        dbos_system_database_url=dbos_system_database_url,
        generation_concurrency=generation_concurrency,
        scoring_concurrency=scoring_concurrency,
    )
    create_eval_schema(config.database_url)
    configure_dbos_runtime(config, queue=queue)
    typer.echo(f"worker listening on {queue.value} queue(s)")
    threading.Event().wait()


if __name__ == "__main__":
    configure_multiprocessing()
    logging.getLogger("dspy").setLevel(logging.WARNING)
    app()
