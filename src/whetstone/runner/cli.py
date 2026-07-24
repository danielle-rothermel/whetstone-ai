"""Operational CLI for typed cell factories and ledger projections."""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, cast

import typer

from whetstone.runner.cell import CellConfig, run_cell
from whetstone.runner.dryrun import run_dry_cell
from whetstone.runner.ledger import Ledger
from whetstone.runner.refinalize import refinalize_cell

app = typer.Typer(no_args_is_help=True)


def _load_factory(path: str) -> Callable[[], CellConfig]:
    if ":" not in path:
        raise typer.BadParameter("factory must be 'module:callable'")
    module_name, attribute = path.split(":", 1)
    value = getattr(importlib.import_module(module_name), attribute)
    if not callable(value):
        raise typer.BadParameter(f"{path!r} is not callable")
    return cast(Callable[[], CellConfig], value)


@app.command("cell")
def cell_command(
    factory: Annotated[
        str,
        typer.Option(
            help="Import path for a zero-argument typed CellConfig factory."
        ),
    ],
    dry: Annotated[
        bool,
        typer.Option(
            help="Use the scripted dry boundary injected by factory."
        ),
    ] = False,
) -> None:
    """Run or resume one typed cell."""
    config = _load_factory(factory)()
    outcome = run_dry_cell(config) if dry else run_cell(config)
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
        Ledger(root),
        optimizer=optimizer,
        env=env,
        attempt=attempt,
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


__all__ = ["app", "main"]
