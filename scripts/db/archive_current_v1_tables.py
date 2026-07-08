from __future__ import annotations

import os

import typer
from sqlalchemy import create_engine, text

from whetstone.db.migrations.url import normalize_postgresql_driver_url

DATABASE_URL_ENV = "DATABASE_URL"
ARCHIVE_SCHEMA = "archive"
OLD_V1_TABLE_NAMES = (
    "dr_dspy_experiments",
    "dr_dspy_prediction_specs",
    "dr_dspy_generation_runs",
    "dr_dspy_node_attempts",
    "dr_dspy_score_attempts",
    "dr_dspy_prediction_projection",
    "dr_dspy_batch_submit_operations",
    "dr_dspy_batch_submit_items",
    "dr_dspy_throttle_backoff",
)

APP = typer.Typer(no_args_is_help=True)


@APP.command()
def main(
    database_url: str | None = typer.Option(
        None,
        "--database-url",
        help=f"Postgres URL; defaults to {DATABASE_URL_ENV}.",
    ),
    source_schema: str = typer.Option("public", "--source-schema"),
    archive_schema: str = typer.Option(ARCHIVE_SCHEMA, "--archive-schema"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    resolved_database_url = normalize_postgresql_driver_url(
        database_url or os.environ[DATABASE_URL_ENV]
    )
    engine = create_engine(resolved_database_url)
    try:
        with engine.begin() as connection:
            connection.execute(
                text(f"CREATE SCHEMA IF NOT EXISTS {archive_schema}")
            )
            for table_name in OLD_V1_TABLE_NAMES:
                exists = connection.execute(
                    text(
                        "SELECT to_regclass(:qualified_table)::text"
                    ),
                    {"qualified_table": f"{source_schema}.{table_name}"},
                ).scalar_one_or_none()
                if exists is None:
                    typer.echo(f"skip missing {source_schema}.{table_name}")
                    continue
                statement = (
                    f"ALTER TABLE {source_schema}.{table_name} "
                    f"SET SCHEMA {archive_schema}"
                )
                typer.echo(statement)
                if not dry_run:
                    connection.execute(text(statement))
    finally:
        engine.dispose()


if __name__ == "__main__":
    APP()
