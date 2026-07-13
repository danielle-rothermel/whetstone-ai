"""Explicit Whetstone publication command; workers never export implicitly."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from sqlalchemy import create_engine

from whetstone.platform.runtime import resolve_application_database_url
from whetstone.publication import export_whetstone

APP = typer.Typer(no_args_is_help=True)


@APP.command()
def publish(
    destination: Annotated[Path, typer.Option("--destination")],
    detail_destination: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Build, validate, and promote both complete Whetstone bundles."""

    engine = create_engine(resolve_application_database_url())
    analysis, detail = export_whetstone(
        engine,
        destination_path=destination,
        detail_destination_path=detail_destination,
    )
    typer.echo(analysis.model_dump_json())
    typer.echo(detail.model_dump_json())


def main() -> None:
    APP()
