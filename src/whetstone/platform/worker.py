"""Whetstone DBOS worker bootstrap."""

from __future__ import annotations

import typer
from dbos import DBOS
from dr_platform.dbos_config import destroy_dbos_runtime

from whetstone.platform.cli_env import load_env_file, run_typer_app
from whetstone.platform.dbos_bootstrap import (
    build_dbos_config,
    build_eval_dbos_config,
)
from whetstone.platform.targets import (
    listen_to_execution_queues,
    register_execution_queues,
)

DBOS_APP_NAME = "whetstone"
DEFAULT_WORKER_CONCURRENCY = 1
APP = typer.Typer(no_args_is_help=True)


def configure_platform_dbos_runtime(
    *,
    database_url: str | None,
    dbos_system_database_url: str | None,
    worker_concurrency: int = DEFAULT_WORKER_CONCURRENCY,
) -> None:
    config = build_eval_dbos_config(
        database_url=database_url,
        dbos_system_database_url=dbos_system_database_url,
        generation_concurrency=worker_concurrency,
        scoring_concurrency=worker_concurrency,
        database_url_error_suffix="for Whetstone execution",
    )
    try:
        DBOS(config=build_dbos_config(config, app_name=DBOS_APP_NAME))
        listen_to_execution_queues()
        DBOS.launch()
        register_execution_queues(worker_concurrency=worker_concurrency)
    except Exception:
        destroy_dbos_runtime()
        raise


@APP.command("serve")
def serve(
    database_url: str | None = typer.Option(None, "--database-url"),
    dbos_system_database_url: str | None = typer.Option(
        None, "--dbos-system-database-url"
    ),
    worker_concurrency: int = typer.Option(DEFAULT_WORKER_CONCURRENCY, min=1),
) -> None:
    load_env_file()
    configure_platform_dbos_runtime(
        database_url=database_url,
        dbos_system_database_url=dbos_system_database_url,
        worker_concurrency=worker_concurrency,
    )
    try:
        typer.echo("Whetstone queues are registered.")
    finally:
        destroy_dbos_runtime()


def main() -> None:
    run_typer_app(APP)


if __name__ == "__main__":
    main()
