"""Explicit Whetstone publication command; workers never export implicitly."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

import typer
from dbos import DBOS, DBOSClient
from dr_platform import ExportReconciliationDependencies
from dr_platform.enqueue_runtime import (
    DbosEnqueueAdapter,
    DbosWorkflowObserver,
)
from dr_platform.reconciliation_runtime import (
    DbosLifecycleReader,
    ReconcileOptions,
)
from sqlalchemy import Engine, create_engine

from whetstone.platform.operations import WhetstoneDbosCanceller
from whetstone.platform.runtime import (
    build_whetstone_dbos_config,
    dbos_config,
    resolve_application_database_url,
    shutdown_dbos_runtime,
)
from whetstone.platform.targets import (
    listen_to_execution_queues,
    register_execution_queues,
    target_registry,
)
from whetstone.publication import export_whetstone

APP = typer.Typer(no_args_is_help=True)
PUBLICATION_RECONCILIATION_PAGE_SIZE = 100
PUBLICATION_RECONCILIATION_MAX_CYCLES = 10
PUBLICATION_RUNTIME_CONCURRENCY = 1


@contextmanager
def build_export_reconciliation_dependencies(
    *,
    application_database_url: str,
    dbos_system_database_url: str | None = None,
) -> Iterator[ExportReconciliationDependencies]:
    """Build the real bounded Whetstone reconciliation runtime."""

    config = build_whetstone_dbos_config(
        database_url=application_database_url,
        system_database_url=dbos_system_database_url,
        generation_concurrency=PUBLICATION_RUNTIME_CONCURRENCY,
        scoring_concurrency=PUBLICATION_RUNTIME_CONCURRENCY,
    )
    client: DBOSClient | None = None
    dbos_engine: Engine | None = None
    try:
        DBOS(config=dbos_config(config, app_name="whetstone-publication"))
        listen_to_execution_queues()
        DBOS.launch()
        register_execution_queues(
            worker_concurrency=PUBLICATION_RUNTIME_CONCURRENCY
        )
        client = DBOSClient(system_database_url=config.system_database_url)
        dbos_engine = create_engine(config.system_database_url)
        yield ExportReconciliationDependencies(
            resolver=target_registry(),
            queue_lookup=client,
            reader=DbosLifecycleReader(client),
            dbos_engine=dbos_engine,
            options=ReconcileOptions(
                page_size=PUBLICATION_RECONCILIATION_PAGE_SIZE
            ),
            max_cycles=PUBLICATION_RECONCILIATION_MAX_CYCLES,
            recovery_observer=DbosWorkflowObserver(),
            enqueue_adapter=DbosEnqueueAdapter(),
            compensation_canceller=WhetstoneDbosCanceller(client),
        )
    finally:
        if client is not None:
            client.destroy()
        if dbos_engine is not None:
            dbos_engine.dispose()
        shutdown_dbos_runtime()


@APP.command()
def publish(
    destination: Annotated[Path, typer.Option("--destination")],
    detail_destination: Annotated[Path | None, typer.Option()] = None,
    dbos_system_database_url: Annotated[
        str | None, typer.Option("--dbos-system-database-url")
    ] = None,
) -> None:
    """Build, validate, and promote both complete Whetstone bundles."""

    application_database_url = resolve_application_database_url()
    engine = create_engine(application_database_url)
    try:
        with build_export_reconciliation_dependencies(
            application_database_url=application_database_url,
            dbos_system_database_url=dbos_system_database_url,
        ) as reconciliation:
            analysis, detail = export_whetstone(
                engine,
                reconciliation=reconciliation,
                destination_path=destination,
                detail_destination_path=detail_destination,
            )
        typer.echo(analysis.model_dump_json())
        typer.echo(detail.model_dump_json())
    finally:
        engine.dispose()


def main() -> None:
    APP()
