from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

import typer
from dbos import DBOS
from rich.console import Console

from dr_dspy.harness import dbos as shared_dbos
from dr_dspy.platform.graph_workflow import (
    platform_generation_workflow_id,
    run_prediction_graph_workflow_once,
)
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
) -> shared_dbos.EvalDbosConfig:
    config = shared_dbos.build_eval_dbos_config(
        database_url=database_url,
        dbos_system_database_url=dbos_system_database_url,
        generation_concurrency=DEFAULT_WORKER_CONCURRENCY,
        scoring_concurrency=DEFAULT_WORKER_CONCURRENCY,
        database_url_error_suffix="for platform graph workflow",
    )
    DBOS(config=shared_dbos.build_dbos_config(config, app_name=DBOS_APP_NAME))
    DBOS.listen_queues([])
    DBOS.launch()
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


@APP.command(
    help=(
        "Launch the platform DBOS runtime without queue listeners. "
        "Queue-consuming workers are deferred until batch submission lands."
    )
)
def worker(
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
    )
    CONSOLE.print(
        "platform graph DBOS runtime running without queue listeners; "
        "press Ctrl-C to stop"
    )
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        CONSOLE.print("platform graph DBOS runtime stopping")
    finally:
        shared_dbos.destroy_dbos_runtime()


def main() -> None:
    run_typer_app(APP)


if __name__ == "__main__":
    main()
