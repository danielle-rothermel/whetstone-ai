"""Shared Typer option annotations for analysis scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

ExperimentNameOption = Annotated[
    list[str],
    typer.Option("--experiment-name", help="Experiment name to include."),
]
DatabaseUrlOption = Annotated[
    str | None,
    typer.Option(
        "--database-url",
        help="Postgres URL; defaults to DATABASE_URL.",
    ),
]
EnvFileOption = Annotated[
    Path | None,
    typer.Option("--env-file", help="Optional .env file path."),
]
LimitOption = Annotated[
    int | None,
    typer.Option("--limit", help="Optional row limit for debugging."),
]
RequireScoreOption = Annotated[
    bool,
    typer.Option(
        "--require-score/--include-unscored",
        help=(
            "Only include runs with a successful humaneval@v1 score attempt "
            "(excludes in-progress rescoring)."
        ),
    ),
]
