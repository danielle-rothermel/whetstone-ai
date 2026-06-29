from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated, Any

import typer
from dbos import DBOS, SetWorkflowID
from rich.console import Console

from dr_dspy.harness import dbos as shared_dbos
from dr_dspy.platform.graph_workflow import (
    WORKFLOW_ID_PREFIX,
    run_prediction_graph_workflow,
)
from dr_dspy.records import stable_generation_run_id
from dr_dspy.runtime import load_env_file, run_typer_app

DBOS_APP_NAME = "dr-dspy-platform-graph-v1"
DEFAULT_WORKER_CONCURRENCY = 1

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
            help="DBOS system database URL; defaults to DATABASE_URL.",
        ),
    ] = None,
    env_file: Annotated[Path | None, typer.Option()] = None,
) -> None:
    load_env_file(env_file) if env_file is not None else load_env_file()
    config = configure_platform_dbos_runtime(
        database_url=database_url,
        dbos_system_database_url=dbos_system_database_url,
    )
    generation_run_id = stable_generation_run_id(
        prediction_id=prediction_id,
        attempt_index=attempt_index,
    )
    workflow_id = f"{WORKFLOW_ID_PREFIX}:{generation_run_id}"
    try:
        with SetWorkflowID(workflow_id):
            handle = DBOS.start_workflow(
                run_prediction_graph_workflow,
                config.database_url,
                prediction_id,
                attempt_index,
            )
        result = _workflow_result(handle)
        CONSOLE.print(
            {
                "workflow_id": workflow_id,
                "generation_run_id": result,
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
            help="DBOS system database URL; defaults to DATABASE_URL.",
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


def _workflow_result(handle: Any) -> str:
    result = handle.get_result()
    if not isinstance(result, str):
        raise TypeError("platform graph workflow returned a non-string result")
    return result


def main() -> None:
    run_typer_app(APP)


if __name__ == "__main__":
    main()
