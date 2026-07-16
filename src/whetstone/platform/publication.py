"""Explicit Whetstone publication command; workers never export implicitly."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, cast

import typer
from dbos import DBOSClient
from dr_platform import ExportReconciliationDependencies
from dr_platform.reconciliation_runtime import (
    DbosLifecycleReader,
    ReconcileOptions,
)
from sqlalchemy import create_engine

from whetstone.platform.enqueue_runtime import (
    InProcessDbosApi,
    platform_enqueue_runtime,
)
from whetstone.platform.integrity import (
    required_bundle_integrity_configuration,
)
from whetstone.platform.operations import WhetstoneDbosCanceller
from whetstone.platform.runtime import resolve_application_database_url
from whetstone.platform.targets import target_registry
from whetstone.publication import export_whetstone

APP = typer.Typer(no_args_is_help=True)
PUBLICATION_RECONCILIATION_PAGE_SIZE = 100
PUBLICATION_RECONCILIATION_MAX_CYCLES = 10


@contextmanager
def build_export_reconciliation_dependencies(
    *,
    application_database_url: str,
    dbos_system_database_url: str | None = None,
) -> Iterator[ExportReconciliationDependencies]:
    """Build the real bounded Whetstone reconciliation runtime.

    The runtime never listens to execution queues and never updates the
    worker-owned queue configuration, so publication can run beside the
    long-lived worker without consuming or reconfiguring paid work.  All
    DBOS observation and cancellation runs over the launched in-process
    runtime because ``DBOSClient`` cannot open SQLite system databases.
    The export contract reads ``current_database()``/``clock_timestamp()``
    from ``dbos_engine`` for its reconciled-cut proof, which is
    PostgreSQL-only SQL, so the application database supplies that clock
    (it is the exact engine the previous code built whenever the DBOS
    system database defaulted to the application database).
    """

    dbos_engine = create_engine(application_database_url)
    try:
        with platform_enqueue_runtime(
            application_database_url=application_database_url,
            system_database_url=dbos_system_database_url,
        ) as runtime:
            dbos_api = cast("DBOSClient", InProcessDbosApi())
            yield ExportReconciliationDependencies(
                resolver=target_registry(),
                queue_lookup=runtime.queue_lookup,
                reader=DbosLifecycleReader(dbos_api),
                dbos_engine=dbos_engine,
                options=ReconcileOptions(
                    page_size=PUBLICATION_RECONCILIATION_PAGE_SIZE
                ),
                max_cycles=PUBLICATION_RECONCILIATION_MAX_CYCLES,
                recovery_observer=runtime.workflow_observer,
                enqueue_adapter=runtime.enqueue_adapter,
                compensation_canceller=WhetstoneDbosCanceller(dbos_api),
            )
    finally:
        dbos_engine.dispose()


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
    integrity = required_bundle_integrity_configuration()
    engine = create_engine(application_database_url)
    try:
        with build_export_reconciliation_dependencies(
            application_database_url=application_database_url,
            dbos_system_database_url=dbos_system_database_url,
        ) as reconciliation:
            analysis, detail = export_whetstone(
                engine,
                reconciliation=reconciliation,
                integrity_signer=integrity.signer,
                destination_path=destination,
                detail_destination_path=detail_destination,
            )
        typer.echo(analysis.model_dump_json())
        typer.echo(detail.model_dump_json())
    finally:
        engine.dispose()


def main() -> None:
    APP()
