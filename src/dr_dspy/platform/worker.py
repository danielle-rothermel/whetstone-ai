from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

import typer
from dbos import DBOS
from rich.console import Console
from sqlalchemy import create_engine

from dr_dspy.harness import dbos as shared_dbos
from dr_dspy.platform.graph_workflow import (
    platform_generation_workflow_id,
    run_prediction_graph_workflow_once,
)
from dr_dspy.platform.queue_worker import (
    PLATFORM_GENERATION_QUEUE_NAME,
    listen_to_platform_generation_queue,
    register_platform_generation_queue,
)
from dr_dspy.platform.submission import (
    DEFAULT_SUBMIT_CHUNK_SIZE,
    submit_prediction_specs,
)
from dr_dspy.records import PredictionSpecRecord
from dr_dspy.runtime import load_env_file, run_typer_app

DBOS_APP_NAME = "dr-dspy-platform-graph-v1"
DEFAULT_WORKER_CONCURRENCY = 1
DBOS_SYSTEM_DATABASE_URL_HELP = (
    "DBOS system database URL; defaults to "
    f"{shared_dbos.DBOS_SYSTEM_DATABASE_URL_ENV} or the resolved app "
    "database URL."
)

CONSOLE = Console()
APP = typer.Typer(no_args_is_help=True)


def configure_platform_dbos_runtime(
    *,
    database_url: str | None,
    dbos_system_database_url: str | None,
    worker_concurrency: int = DEFAULT_WORKER_CONCURRENCY,
    consume_generation_queue: bool = False,
) -> shared_dbos.EvalDbosConfig:
    config = shared_dbos.build_eval_dbos_config(
        database_url=database_url,
        dbos_system_database_url=dbos_system_database_url,
        generation_concurrency=worker_concurrency,
        scoring_concurrency=DEFAULT_WORKER_CONCURRENCY,
        database_url_error_suffix="for platform graph workflow",
    )
    DBOS(config=shared_dbos.build_dbos_config(config, app_name=DBOS_APP_NAME))
    if consume_generation_queue:
        listen_to_platform_generation_queue()
    else:
        DBOS.listen_queues([])
    DBOS.launch()
    if consume_generation_queue:
        register_platform_generation_queue(
            worker_concurrency=worker_concurrency,
        )
    return config


@APP.command("run-one")
def run_one(
    prediction_id: Annotated[
        str,
        typer.Option(
            "--prediction-id",
            help="Existing v1 prediction spec id to execute.",
        ),
    ],
    attempt_index: Annotated[
        int,
        typer.Option("--attempt-index", min=0),
    ] = 0,
    database_url: Annotated[
        str | None,
        typer.Option(
            "--database-url",
            help="Postgres URL; defaults to DATABASE_URL.",
        ),
    ] = None,
    dbos_system_database_url: Annotated[
        str | None,
        typer.Option(
            "--dbos-system-database-url",
            help=DBOS_SYSTEM_DATABASE_URL_HELP,
        ),
    ] = None,
    env_file: Annotated[Path | None, typer.Option()] = None,
) -> None:
    load_env_file(env_file) if env_file is not None else load_env_file()
    config = configure_platform_dbos_runtime(
        database_url=database_url,
        dbos_system_database_url=dbos_system_database_url,
        consume_generation_queue=False,
    )
    try:
        generation_run_id = run_prediction_graph_workflow_once(
            database_url=config.database_url,
            prediction_id=prediction_id,
            attempt_index=attempt_index,
        )
        CONSOLE.print(
            {
                "workflow_id": platform_generation_workflow_id(
                    generation_run_id
                ),
                "generation_run_id": generation_run_id,
            }
        )
    finally:
        shared_dbos.destroy_dbos_runtime()


@APP.command(help="Launch a queue-consuming v1 generation worker.")
def worker(
    worker_concurrency: Annotated[
        int,
        typer.Option("--worker-concurrency", min=1),
    ] = DEFAULT_WORKER_CONCURRENCY,
    database_url: Annotated[
        str | None,
        typer.Option(
            "--database-url",
            help="Postgres URL; defaults to DATABASE_URL.",
        ),
    ] = None,
    dbos_system_database_url: Annotated[
        str | None,
        typer.Option(
            "--dbos-system-database-url",
            help=DBOS_SYSTEM_DATABASE_URL_HELP,
        ),
    ] = None,
    env_file: Annotated[Path | None, typer.Option()] = None,
) -> None:
    load_env_file(env_file) if env_file is not None else load_env_file()
    configure_platform_dbos_runtime(
        database_url=database_url,
        dbos_system_database_url=dbos_system_database_url,
        worker_concurrency=worker_concurrency,
        consume_generation_queue=True,
    )
    CONSOLE.print(
        {
            "queue_name": PLATFORM_GENERATION_QUEUE_NAME,
            "worker_concurrency": worker_concurrency,
            "status": "running",
        }
    )
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        CONSOLE.print("platform graph DBOS runtime stopping")
    finally:
        shared_dbos.destroy_dbos_runtime()


@APP.command("submit-jsonl")
def submit_jsonl(
    specs_file: Annotated[
        Path,
        typer.Option(
            "--specs-file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="JSONL file of PredictionSpecRecord payloads.",
        ),
    ],
    operation_key: Annotated[
        str,
        typer.Option("--operation-key", help="Stable logical submit key."),
    ],
    experiment_name: Annotated[
        str,
        typer.Option(
            "--experiment-name",
            help="Experiment name all submitted specs must match.",
        ),
    ],
    chunk_size: Annotated[
        int,
        typer.Option("--chunk-size", min=1),
    ] = DEFAULT_SUBMIT_CHUNK_SIZE,
    attempt_index: Annotated[
        int,
        typer.Option("--attempt-index", min=0),
    ] = 0,
    worker_concurrency: Annotated[
        int,
        typer.Option("--queue-worker-concurrency", min=1),
    ] = DEFAULT_WORKER_CONCURRENCY,
    database_url: Annotated[
        str | None,
        typer.Option(
            "--database-url",
            help="Postgres URL; defaults to DATABASE_URL.",
        ),
    ] = None,
    dbos_system_database_url: Annotated[
        str | None,
        typer.Option(
            "--dbos-system-database-url",
            help="DBOS system database URL; defaults to DATABASE_URL.",
        ),
    ] = None,
    env_file: Annotated[Path | None, typer.Option()] = None,
) -> None:
    load_env_file(env_file) if env_file is not None else load_env_file()
    config = configure_platform_dbos_runtime(
        database_url=database_url,
        dbos_system_database_url=dbos_system_database_url,
        worker_concurrency=worker_concurrency,
        consume_generation_queue=False,
    )
    register_platform_generation_queue(
        worker_concurrency=worker_concurrency,
    )
    engine = create_engine(config.database_url)
    try:
        specs = read_prediction_specs_jsonl(specs_file)
        with engine.begin() as connection:
            result = submit_prediction_specs(
                connection,
                database_url=config.database_url,
                operation_key=operation_key,
                experiment_name=experiment_name,
                specs=specs,
                submit_spec={"source": str(specs_file)},
                chunk_size=chunk_size,
                attempt_index=attempt_index,
            )
        CONSOLE.print(result.model_dump(mode="json"))
    finally:
        engine.dispose()
        shared_dbos.destroy_dbos_runtime()


def read_prediction_specs_jsonl(
    path: Path,
) -> tuple[PredictionSpecRecord, ...]:
    specs: list[PredictionSpecRecord] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                specs.append(PredictionSpecRecord.model_validate_json(stripped))
            except ValueError as error:
                raise ValueError(
                    f"invalid prediction spec JSON on line {line_number}"
                ) from error
    return tuple(specs)


def main() -> None:
    run_typer_app(APP)


if __name__ == "__main__":
    main()
