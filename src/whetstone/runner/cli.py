"""The ``whetstone-validate`` CLI: the runner's process lifecycle.

Three commands. ``cell`` runs or resumes one cell; ``status`` prints the
validated ledger lines; ``refinalize`` appends an evidence-only corrected
terminal line. Only ``cell`` needs a DBOS runtime, and it owns that runtime's
whole lifetime.

**The typed factory is the configuration boundary.** Building a runnable cell
means resolving an environment, a provider route, credentials, a proposer
transport, and an evaluation runtime -- none of which belong on a command line.
The CLI takes one ``module:callable`` that returns a fully assembled
:class:`RunnerLaunch`, so configuration is typed Python that the type checker
sees rather than a pile of flags that fail at runtime.

**Order is the whole point of the lifecycle body.** Persistence is
bootstrapped, capabilities are registered through the single startup site, and
only then is the DBOS app constructed and launched. Registration must complete
before ``launch()`` because recovery begins there: a workflow recovered against
a half-registered process fails to resolve capabilities it was built with. The
``finally: destroy()`` is unconditional so a failing run still tears its
runtime down.

**A completed cell never starts a runtime at all.** ``prepare_cell_launch``
short-circuits first, so re-running a finished wave costs neither credits nor a
database.
"""

from __future__ import annotations

import importlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, cast

import typer

from whetstone.coordination.run_workflow import RunController
from whetstone.core.identity import compute_identity_hash
from whetstone.experiment.task_selection import load_task_split_manifest
from whetstone.optimization.gepa.factory import CanonicalGepaAdapterFactory
from whetstone.optimization.proposal.proposer import ProviderProposerTransport
from whetstone.runner.cell import (
    CellConfig,
    CellOutcome,
    bind_cell_launch,
    prepare_cell_launch,
    run_cell,
)
from whetstone.runner.ledger import Ledger
from whetstone.runner.refinalize import refinalize_cell
from whetstone.runner.startup import register_runtime

__all__ = [
    "DBOS_APPLICATION_DATABASE_URL_ENV",
    "DBOS_APPLICATION_VERSION_ENV",
    "DBOS_SYSTEM_DATABASE_URL_ENV",
    "RunnerLaunch",
    "app",
    "load_task_split_manifest",
    "main",
    "run_cell_command",
]

DBOS_SYSTEM_DATABASE_URL_ENV = "WHETSTONE_DBOS_SYSTEM_DATABASE_URL"
DBOS_APPLICATION_DATABASE_URL_ENV = "WHETSTONE_DBOS_APPLICATION_DATABASE_URL"
DBOS_APPLICATION_VERSION_ENV = "WHETSTONE_DBOS_APPLICATION_VERSION"

app = typer.Typer(no_args_is_help=True)


@dataclass(frozen=True, slots=True)
class RunnerLaunch:
    """One fully assembled cell plus everything its process must register.

    A factory returns this rather than a bare cell config, because the
    capabilities a recovered workflow resolves -- the proposal transport, the
    run controllers, GEPA's adapter factories -- are process-level facts the
    cell itself does not own.
    """

    cell: CellConfig
    transport: ProviderProposerTransport
    controllers: tuple[RunController, ...] = ()
    gepa_factories: tuple[CanonicalGepaAdapterFactory, ...] = ()


def _load_factory(path: str) -> Callable[[], RunnerLaunch]:
    if ":" not in path:
        raise typer.BadParameter("factory must be 'module:callable'")
    module_name, attribute = path.split(":", 1)
    value = getattr(importlib.import_module(module_name), attribute)
    if not callable(value):
        raise typer.BadParameter(f"{path!r} is not callable")
    return cast(Callable[[], RunnerLaunch], value)


def _default_dbos_database_url(config: CellConfig) -> str:
    """A per-cell SQLite system database under the cell's own ledger root.

    Keyed by the cell identity so concurrent cells never share a system
    database, and stable across restarts so a resumed cell recovers its own
    workflows rather than starting clean.
    """
    identity = compute_identity_hash(
        schema="whetstone.runner.dbos_cell_database",
        schema_version=1,
        payload={"cell_id": config.cell_id},
    )
    directory = config.ledger.root / "dbos"
    directory.mkdir(parents=True, exist_ok=True)
    database = (directory / f"{identity}.sqlite").resolve()
    return f"sqlite:///{database}"


def _dbos_executor_id(config: CellConfig) -> str:
    return compute_identity_hash(
        schema="whetstone.runner.dbos_cell_executor",
        schema_version=1,
        payload={"cell_id": config.cell_id},
    )


def run_cell_command(
    launch: RunnerLaunch,
    *,
    system_database_url: str | None = None,
    application_database_url: str | None = None,
    application_version: str | None = None,
    environ: dict[str, str] | None = None,
) -> CellOutcome:
    """Own one cell's whole DBOS lifetime: register, launch, run, destroy."""
    env = environ if environ is not None else dict(os.environ)
    config = launch.cell
    completed = prepare_cell_launch(config)
    if completed is not None:
        return completed

    from dbos import DBOS, DBOSConfig

    register_runtime(
        transport=launch.transport,
        controllers=launch.controllers,
        gepa_factories=launch.gepa_factories,
    )
    # Bind the cell's immutable controls before launch, so DBOS recovery can
    # only ever resume a cell whose inputs are already durable.
    bind_cell_launch(config)

    dbos_config: DBOSConfig = {
        "name": f"whetstone-{config.optimizer}",
        "executor_id": _dbos_executor_id(config),
        "system_database_url": (
            system_database_url
            or env.get(DBOS_SYSTEM_DATABASE_URL_ENV)
            or _default_dbos_database_url(config)
        ),
        "run_admin_server": False,
    }
    application_database = application_database_url or env.get(
        DBOS_APPLICATION_DATABASE_URL_ENV
    )
    if application_database is not None:
        dbos_config["application_database_url"] = application_database
    resolved_version = application_version or env.get(
        DBOS_APPLICATION_VERSION_ENV
    )
    if resolved_version is not None:
        dbos_config["application_version"] = resolved_version

    DBOS(config=dbos_config)
    try:
        DBOS.launch()
        return run_cell(config)
    finally:
        DBOS.destroy()


@app.command("cell")
def cell_command(
    factory: Annotated[
        str,
        typer.Option(
            help="Import path for a zero-argument RunnerLaunch factory."
        ),
    ],
    dbos_system_database_url: Annotated[
        str | None,
        typer.Option(
            help=(
                "DBOS system database URL. Defaults to "
                "<ledger>/dbos/<cell-identity-hash>.sqlite or "
                f"${DBOS_SYSTEM_DATABASE_URL_ENV}."
            )
        ),
    ] = None,
    dbos_application_database_url: Annotated[
        str | None,
        typer.Option(
            help=(
                "Optional DBOS application database URL; defaults to "
                f"${DBOS_APPLICATION_DATABASE_URL_ENV}."
            )
        ),
    ] = None,
    dbos_application_version: Annotated[
        str | None,
        typer.Option(
            help=(
                "Optional deployment version for recovery; defaults to "
                f"${DBOS_APPLICATION_VERSION_ENV} or DBOS source hashing."
            )
        ),
    ] = None,
) -> None:
    """Run or resume one typed cell."""
    outcome = run_cell_command(
        _load_factory(factory)(),
        system_database_url=dbos_system_database_url,
        application_database_url=dbos_application_database_url,
        application_version=dbos_application_version,
    )
    typer.echo(outcome.record.model_dump_json(by_alias=True, indent=2))


@app.command("status")
def status_command(
    root: Annotated[Path, typer.Option(help="Run ledger directory.")],
) -> None:
    """Print validated cell records as stable JSON."""
    records = Ledger(root).load()
    typer.echo(
        json.dumps(
            [
                record.model_dump(mode="json", by_alias=True)
                for record in records
            ],
            indent=2,
            sort_keys=True,
        )
    )


@app.command("refinalize")
def refinalize_command(
    root: Annotated[Path, typer.Option(help="Run ledger directory.")],
    optimizer: Annotated[str, typer.Option()],
    env: Annotated[str, typer.Option()],
    attempt: Annotated[int, typer.Option(min=0)],
) -> None:
    """Append an evidence-only corrected terminal projection."""
    outcome = refinalize_cell(
        Ledger(root), optimizer=optimizer, env=env, attempt=attempt
    )
    typer.echo(
        json.dumps(
            {
                "changed": outcome.changed,
                "reason": outcome.reason,
                "record": (outcome.corrected or outcome.original).model_dump(
                    mode="json", by_alias=True
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    app()
