"""Zero-spend end-to-end smoke for the composable migration.

Builds specs from an experiment config, routes every LM call through
dr-providers' ``ScriptedProvider`` (no network, no spend), then drives
the real platform path — ``submit-jsonl`` semantics, an in-process
queue worker, and ``rescore`` — against a scratch database, and prints
the append-only outcome/score evidence.

Run (scratch DB is created/migrated by the operator first — see
--help):

    createdb dr_dspy_e2e_smoke
    uv run python scripts/e2e/fixture_smoke.py \\
        --database-url postgresql:///dr_dspy_e2e_smoke
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated

import typer
from dr_platform import await_operation, destroy_dbos_runtime
from dr_providers import Provider, ScriptedOutcome, ScriptedProvider
from sqlalchemy import create_engine, text

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO_ROOT / "configs" / "experiments" / "humaneval_encdec_smoke.json"
)
OPERATION_KEY = "e2e-fixture-smoke-v1"

FIXTURE_GENERATION = '''Here is the solution:

```python
def solution(*args, **kwargs):
    return args[0] if args else None
```
'''

APP = typer.Typer(add_completion=False)


def _fixture_provider() -> Provider:
    return ScriptedProvider([ScriptedOutcome(text=FIXTURE_GENERATION)])


def _migrate(database_url: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        check=True,
        cwd=REPO_ROOT,
        env={**os.environ, "DATABASE_URL": database_url},
    )
    from whetstone.platform.platform_db import ensure_platform_schema

    ensure_platform_schema(database_url)


def _counts(connection, sql: str) -> list[tuple]:
    return [tuple(row) for row in connection.execute(text(sql))]


@APP.command()
def main(
    database_url: Annotated[
        str,
        typer.Option(
            help="Scratch Postgres URL (never a production database)."
        ),
    ],
    config_file: Annotated[
        Path,
        typer.Option(exists=True),
    ] = DEFAULT_CONFIG,
    worker_concurrency: Annotated[int, typer.Option(min=1)] = 4,
    timeout_seconds: Annotated[float, typer.Option()] = 1800.0,
) -> None:
    from whetstone.platform import node_execution
    from whetstone.platform.platform_db import PLATFORM_SCHEMA
    from whetstone.platform.rescoring import rescore_generation_runs
    from whetstone.platform.spec_builder import (
        DEFAULT_CONFIGS_ROOT,
        iter_experiment_specs_from_file,
        write_prediction_specs_jsonl,
    )
    from whetstone.platform.submission import submit_prediction_specs_jsonl
    from whetstone.platform.worker import configure_platform_dbos_runtime

    typer.echo(f"[1/6] migrating scratch database {database_url!r}")
    _migrate(database_url)

    typer.echo(f"[2/6] building specs from {config_file}")
    specs = tuple(
        iter_experiment_specs_from_file(
            config_file,
            configs_root=DEFAULT_CONFIGS_ROOT,
        )
    )
    experiment_name = specs[0].experiment_name
    workdir = Path(tempfile.mkdtemp(prefix="e2e-fixture-smoke-"))
    specs_file = workdir / "specs.jsonl"
    write_prediction_specs_jsonl(specs, specs_file)
    typer.echo(f"      spec_count={len(specs)} experiment={experiment_name}")

    typer.echo("[3/6] routing all LM calls through ScriptedProvider")
    # Deliberate module-attribute patch: the provider seam for the
    # zero-spend smoke.
    node_execution.default_http_provider = _fixture_provider  # ty: ignore[invalid-assignment]

    typer.echo("[4/6] launching in-process worker + submit-jsonl")
    config = configure_platform_dbos_runtime(
        database_url=database_url,
        dbos_system_database_url=f"sqlite:///{workdir}/dbos_system.sqlite",
        worker_concurrency=worker_concurrency,
        consume_generation_queue=True,
    )
    engine = create_engine(config.database_url)
    try:
        submit_result = submit_prediction_specs_jsonl(
            engine,
            database_url=config.database_url,
            operation_key=OPERATION_KEY,
            experiment_name=experiment_name,
            specs_file=specs_file,
            submit_spec={"source": str(specs_file)},
        )
        typer.echo(
            "      submit: "
            + str(
                submit_result.model_dump(
                    mode="json",
                    exclude={"items"},
                )
            )
        )

        typer.echo("[5/6] awaiting generation workflows, then rescore")
        breakdown = await_operation(
            engine,
            operation_key=OPERATION_KEY,
            schema=PLATFORM_SCHEMA,
            poll_interval_seconds=2.0,
            timeout_seconds=timeout_seconds,
        )
        typer.echo(f"      workflows: {breakdown.status_counts}")
        rescore = rescore_generation_runs(
            engine,
            database_url=config.database_url,
            experiment_name=experiment_name,
        )
        typer.echo(f"      rescore: {rescore.model_dump(mode='json')}")

        typer.echo("[6/6] evidence (append-only outcomes and scores)")
        with engine.connect() as connection:
            evidence = {
                "prediction_specs": _counts(
                    connection,
                    "SELECT COUNT(*) FROM dr_dspy_prediction_specs",
                ),
                "batch_operation": _counts(
                    connection,
                    "SELECT status, requested_count, enqueued_count, "
                    "failed_count FROM dr_dspy_batch_submit_operations",
                ),
                "generation_runs_by_status": _counts(
                    connection,
                    "SELECT status, COUNT(*) FROM dr_dspy_generation_runs "
                    "GROUP BY status ORDER BY status",
                ),
                "node_attempts_by_status": _counts(
                    connection,
                    "SELECT status, COUNT(*) FROM dr_dspy_node_attempts "
                    "GROUP BY status ORDER BY status",
                ),
                "score_attempts_by_outcome": _counts(
                    connection,
                    "SELECT status, generated_code_outcome, COUNT(*) "
                    "FROM dr_dspy_score_attempts "
                    "GROUP BY status, generated_code_outcome "
                    "ORDER BY status, generated_code_outcome",
                ),
                "sample_ids": _counts(
                    connection,
                    "SELECT prediction_id, generation_run_id, "
                    "score_attempt_id FROM dr_dspy_score_attempts "
                    "ORDER BY score_attempt_id LIMIT 3",
                ),
            }
        for key, rows in evidence.items():
            typer.echo(f"      {key}: {rows}")
    finally:
        engine.dispose()
        destroy_dbos_runtime()


if __name__ == "__main__":
    APP()
