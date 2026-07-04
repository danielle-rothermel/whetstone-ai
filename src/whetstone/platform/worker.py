from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

import typer
from dbos import DBOS
from rich.console import Console
from sqlalchemy import create_engine

from whetstone.humaneval.profiles import (
    HUMANEVAL_SCORING_PROFILE_ID,
    HUMANEVAL_SCORING_PROFILE_VERSION,
)
from whetstone.migration.v0_encdec_backfill import run_v0_encdec_backfill
from whetstone.platform.cli_env import load_env_file, run_typer_app
from whetstone.platform.dbos_bootstrap import (
    DBOS_SYSTEM_DATABASE_URL_ENV,
    EvalDbosConfig,
    build_dbos_config,
    build_eval_dbos_config,
    destroy_dbos_runtime,
    resolve_database_url,
)
from whetstone.platform.graph_workflow import (
    platform_generation_workflow_id,
    run_prediction_graph_workflow_once,
)
from whetstone.platform.progress_log import (
    DEFAULT_PROGRESS_INTERVAL_SECONDS,
    operation_progress,
)
from whetstone.platform.queue_worker import (
    PLATFORM_GENERATION_QUEUE_NAME,
    listen_to_platform_generation_queue,
    register_platform_generation_queue,
)
from whetstone.platform.rescoring import (
    DEFAULT_MAX_IN_FLIGHT,
    DEFAULT_RESCORE_CHUNK_SIZE,
    parse_rescore_generation_statuses,
    rescore_generation_runs,
)
from whetstone.platform.scoring_workflow import (
    platform_scoring_workflow_id,
    run_score_generation_workflow_once,
)
from whetstone.platform.spec_builder import (
    DEFAULT_CONFIGS_ROOT,
    iter_experiment_specs_from_file,
    write_prediction_specs_jsonl,
)
from whetstone.platform.submission import (
    DEFAULT_SUBMIT_CHUNK_SIZE,
    bulk_insert_prediction_specs,
    idempotent_insert_experiment,
    submit_prediction_specs_jsonl,
)
from whetstone.records import (
    DEFAULT_SCORE_DATASET_NAME,
    DEFAULT_SCORE_DATASET_SPLIT,
    ExperimentRecord,
)

DBOS_APP_NAME = "dr-dspy-platform-graph-v1"
DEFAULT_WORKER_CONCURRENCY = 1
DBOS_SYSTEM_DATABASE_URL_HELP = (
    "DBOS system database URL; defaults to "
    f"{DBOS_SYSTEM_DATABASE_URL_ENV} or the resolved app "
    "database URL."
)

CONSOLE = Console()
APP = typer.Typer(no_args_is_help=True)


def configure_platform_dbos_runtime(
    *,
    database_url: str | None,
    dbos_system_database_url: str | None,
    worker_concurrency: int = DEFAULT_WORKER_CONCURRENCY,
    consume_generation_queue: bool = False,
    database_url_error_suffix: str = "for platform graph workflow",
) -> EvalDbosConfig:
    config = build_eval_dbos_config(
        database_url=database_url,
        dbos_system_database_url=dbos_system_database_url,
        generation_concurrency=worker_concurrency,
        scoring_concurrency=DEFAULT_WORKER_CONCURRENCY,
        database_url_error_suffix=database_url_error_suffix,
    )
    try:
        DBOS(
            config=build_dbos_config(
                config, app_name=DBOS_APP_NAME
            )
        )
        if consume_generation_queue:
            listen_to_platform_generation_queue()
        else:
            DBOS.listen_queues([])
        DBOS.launch()
        if consume_generation_queue:
            register_platform_generation_queue(
                worker_concurrency=worker_concurrency,
            )
    except Exception:
        destroy_dbos_runtime()
        raise
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
            help=DBOS_SYSTEM_DATABASE_URL_HELP,
        ),
    ] = None,
    env_file: Annotated[Path | None, typer.Option()] = None,
) -> None:
    load_env_file(env_file) if env_file is not None else load_env_file()
    config = configure_platform_dbos_runtime(
        database_url=database_url,
        dbos_system_database_url=dbos_system_database_url,
        consume_generation_queue=False,
    )
    try:
        generation_run_id = run_prediction_graph_workflow_once(
            database_url=config.database_url,
            prediction_id=prediction_id,
            attempt_index=attempt_index,
        )
        CONSOLE.print(
            {
                "workflow_id": platform_generation_workflow_id(
                    generation_run_id
                ),
                "generation_run_id": generation_run_id,
            }
        )
    finally:
        destroy_dbos_runtime()


@APP.command("score-one")
def score_one(
    generation_run_id: Annotated[
        str,
        typer.Option(
            "--generation-run-id",
            help="Existing v1 generation run id to score.",
        ),
    ],
    score_attempt_index: Annotated[
        int,
        typer.Option("--score-attempt-index", min=0),
    ] = 0,
    scoring_profile_id: Annotated[
        str,
        typer.Option("--scoring-profile-id"),
    ] = HUMANEVAL_SCORING_PROFILE_ID,
    scoring_profile_version: Annotated[
        str,
        typer.Option("--scoring-profile-version"),
    ] = HUMANEVAL_SCORING_PROFILE_VERSION,
    dataset_name: Annotated[
        str,
        typer.Option("--dataset-name"),
    ] = DEFAULT_SCORE_DATASET_NAME,
    dataset_split: Annotated[
        str,
        typer.Option("--dataset-split"),
    ] = DEFAULT_SCORE_DATASET_SPLIT,
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
        consume_generation_queue=False,
        database_url_error_suffix="for platform scoring workflow",
    )
    try:
        score_result = run_score_generation_workflow_once(
            database_url=config.database_url,
            generation_run_id=generation_run_id,
            score_attempt_index=score_attempt_index,
            scoring_profile_id=scoring_profile_id,
            scoring_profile_version=scoring_profile_version,
            dataset_name=dataset_name,
            dataset_split=dataset_split,
        )
        CONSOLE.print(
            {
                "workflow_id": platform_scoring_workflow_id(
                    score_result.score_attempt_id
                ),
                "generation_run_id": generation_run_id,
                "score_attempt_id": score_result.score_attempt_id,
                "insert_status": score_result.insert_status,
            }
        )
    finally:
        destroy_dbos_runtime()


@APP.command("rescore")
def rescore(
    experiment_name: Annotated[
        str,
        typer.Option(
            "--experiment-name",
            help="Existing v1 experiment name whose generation runs to score.",
        ),
    ],
    generation_status: Annotated[
        list[str] | None,
        typer.Option(
            "--generation-status",
            help=(
                "Repeatable generation statuses to rescore. "
                "Defaults to success and partial."
            ),
        ),
    ] = None,
    generation_attempt_index: Annotated[
        int | None,
        typer.Option("--generation-attempt-index", min=0),
    ] = None,
    score_attempt_index: Annotated[
        int,
        typer.Option("--score-attempt-index", min=0),
    ] = 0,
    scoring_profile_id: Annotated[
        str,
        typer.Option("--scoring-profile-id"),
    ] = HUMANEVAL_SCORING_PROFILE_ID,
    scoring_profile_version: Annotated[
        str,
        typer.Option("--scoring-profile-version"),
    ] = HUMANEVAL_SCORING_PROFILE_VERSION,
    dataset_name: Annotated[
        str,
        typer.Option("--dataset-name"),
    ] = DEFAULT_SCORE_DATASET_NAME,
    dataset_split: Annotated[
        str,
        typer.Option("--dataset-split"),
    ] = DEFAULT_SCORE_DATASET_SPLIT,
    chunk_size: Annotated[
        int,
        typer.Option("--chunk-size", min=1),
    ] = DEFAULT_RESCORE_CHUNK_SIZE,
    max_in_flight: Annotated[
        int,
        typer.Option(
            "--max-in-flight",
            min=1,
            help=(
                "Maximum concurrent scoring workflows; when at cap, "
                "scheduling waits for the oldest to finish before "
                "starting the next."
            ),
        ),
    ] = DEFAULT_MAX_IN_FLIGHT,
    limit: Annotated[
        int | None,
        typer.Option("--limit", min=1),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run"),
    ] = False,
    recover_orphans: Annotated[
        bool,
        typer.Option(
            "--recover-orphans/--no-recover-orphans",
            help=(
                "When DBOS has a terminal scoring workflow without a "
                "persisted score attempt, replay the workflow to finish "
                "persistence."
            ),
        ),
    ] = True,
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
    progress_interval: Annotated[
        float,
        typer.Option(
            "--progress-interval",
            min=0,
            help=(
                "Seconds between Rich progress heartbeats on stderr; "
                "0 disables heartbeats but keeps event lines."
            ),
        ),
    ] = DEFAULT_PROGRESS_INTERVAL_SECONDS,
) -> None:
    load_env_file(env_file) if env_file is not None else load_env_file()
    try:
        resolved_generation_statuses = parse_rescore_generation_statuses(
            generation_status
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error

    launched_dbos = False
    if dry_run:
        resolved_database_url = resolve_database_url(
            database_url=database_url,
            error_suffix="for platform batch rescoring",
        )
    else:
        config = configure_platform_dbos_runtime(
            database_url=database_url,
            dbos_system_database_url=dbos_system_database_url,
            consume_generation_queue=False,
        )
        resolved_database_url = config.database_url
        launched_dbos = True
    engine = create_engine(resolved_database_url)
    try:
        with operation_progress(
            "rescore",
            interval_seconds=progress_interval,
        ) as progress:
            execution = rescore_generation_runs(
                engine,
                database_url=resolved_database_url,
                experiment_name=experiment_name,
                generation_statuses=resolved_generation_statuses,
                generation_attempt_index=generation_attempt_index,
                scoring_profile_id=scoring_profile_id,
                scoring_profile_version=scoring_profile_version,
                score_attempt_index=score_attempt_index,
                dataset_name=dataset_name,
                dataset_split=dataset_split,
                chunk_size=chunk_size,
                limit=limit,
                dry_run=dry_run,
                recover_orphans=recover_orphans,
                max_in_flight=max_in_flight,
                progress=progress,
            )
        CONSOLE.print(execution.result.model_dump(mode="json"))
    finally:
        engine.dispose()
        if launched_dbos:
            destroy_dbos_runtime()


@APP.command("backfill-v0-encdec")
def backfill_v0_encdec(
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run"),
    ] = False,
    limit: Annotated[
        int | None,
        typer.Option("--limit", min=1),
    ] = None,
    target_experiment_name: Annotated[
        str | None,
        typer.Option(
            "--target-experiment-name",
            help=(
                "Override experiment_name on reshaped v1 specs. "
                "When omitted, each v0 row keeps its legacy experiment_name."
            ),
        ),
    ] = None,
    chunk_size: Annotated[
        int | None,
        typer.Option(
            "--chunk-size",
            min=1,
            help=(
                "Process terminal v0 rows in chunks of this size, committing "
                "each chunk independently."
            ),
        ),
    ] = None,
    reshape_workers: Annotated[
        int,
        typer.Option(
            "--reshape-workers",
            min=1,
            help="Parallel workers for CPU-bound v0 row reshape.",
        ),
    ] = 1,
    database_url: Annotated[
        str | None,
        typer.Option(
            "--database-url",
            help="Postgres URL; defaults to DATABASE_URL.",
        ),
    ] = None,
    env_file: Annotated[Path | None, typer.Option()] = None,
    progress_interval: Annotated[
        float,
        typer.Option(
            "--progress-interval",
            min=0,
            help=(
                "Seconds between Rich progress heartbeats on stderr; "
                "0 disables heartbeats but keeps event lines."
            ),
        ),
    ] = DEFAULT_PROGRESS_INTERVAL_SECONDS,
) -> None:
    load_env_file(env_file) if env_file is not None else load_env_file()
    resolved_database_url = resolve_database_url(
        database_url=database_url,
        error_suffix="for enc-dec v0 backfill",
    )
    engine = create_engine(resolved_database_url)
    try:
        with operation_progress(
            "backfill",
            interval_seconds=progress_interval,
        ) as progress:
            result = run_v0_encdec_backfill(
                engine,
                dry_run=dry_run,
                limit=limit,
                target_experiment_name=target_experiment_name,
                chunk_size=chunk_size,
                reshape_workers=reshape_workers,
                progress=progress,
            )
        CONSOLE.print(result.model_dump(mode="json"))
    finally:
        engine.dispose()


@APP.command(help="Launch a queue-consuming v1 generation worker.")
def worker(
    worker_concurrency: Annotated[
        int,
        typer.Option("--worker-concurrency", min=1),
    ] = DEFAULT_WORKER_CONCURRENCY,
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
            help=DBOS_SYSTEM_DATABASE_URL_HELP,
        ),
    ] = None,
    env_file: Annotated[Path | None, typer.Option()] = None,
) -> None:
    load_env_file(env_file) if env_file is not None else load_env_file()
    configure_platform_dbos_runtime(
        database_url=database_url,
        dbos_system_database_url=dbos_system_database_url,
        worker_concurrency=worker_concurrency,
        consume_generation_queue=True,
    )
    CONSOLE.print(
        {
            "queue_name": PLATFORM_GENERATION_QUEUE_NAME,
            "worker_concurrency": worker_concurrency,
            "status": "running",
        }
    )
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        CONSOLE.print("platform graph DBOS runtime stopping")
    finally:
        destroy_dbos_runtime()


@APP.command("build-specs")
def build_specs(
    config_file: Annotated[
        Path,
        typer.Option(
            "--config-file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Experiment JSON config for spec generation.",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            file_okay=True,
            dir_okay=False,
            writable=True,
            help="Write JSONL here; defaults to stdout.",
        ),
    ] = None,
    insert: Annotated[
        bool,
        typer.Option(
            "--insert",
            help=(
                "Bulk-insert generated specs and experiment row "
                "into Postgres."
            ),
        ),
    ] = False,
    database_url: Annotated[
        str | None,
        typer.Option(
            "--database-url",
            help="Postgres URL; required with --insert.",
        ),
    ] = None,
    configs_root: Annotated[
        Path,
        typer.Option(
            "--configs-root",
            file_okay=False,
            dir_okay=True,
            readable=True,
            help=(
                "Root directory for composable config fragments "
                "(split, model_configs paths)."
            ),
        ),
    ] = DEFAULT_CONFIGS_ROOT,
    env_file: Annotated[Path | None, typer.Option()] = None,
) -> None:
    load_env_file(env_file) if env_file is not None else load_env_file()
    specs = tuple(
        iter_experiment_specs_from_file(
            config_file,
            configs_root=configs_root,
        )
    )
    if not specs:
        raise typer.BadParameter("config produced no prediction specs")
    experiment_name = specs[0].experiment_name
    destination = write_prediction_specs_jsonl(specs, output)
    inserted_count = 0
    if insert:
        resolved_database_url = resolve_database_url(
            database_url=database_url,
            error_suffix="for build-specs insert",
        )
        engine = create_engine(resolved_database_url)
        try:
            with engine.begin() as connection:
                connection.execute(
                    idempotent_insert_experiment(
                        ExperimentRecord(
                            experiment_name=experiment_name,
                            config_metadata={"source": str(config_file)},
                        )
                    )
                )
                inserted_count = len(
                    bulk_insert_prediction_specs(connection, specs)
                )
        finally:
            engine.dispose()
    CONSOLE.print(
        {
            "experiment_name": experiment_name,
            "spec_count": len(specs),
            "output": str(destination),
            "inserted_count": inserted_count if insert else None,
        }
    )


@APP.command("submit-jsonl")
def submit_jsonl(
    specs_file: Annotated[
        Path,
        typer.Option(
            "--specs-file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="JSONL file of PredictionSpecRecord payloads.",
        ),
    ],
    operation_key: Annotated[
        str,
        typer.Option("--operation-key", help="Stable logical submit key."),
    ],
    experiment_name: Annotated[
        str,
        typer.Option(
            "--experiment-name",
            help="Experiment name all submitted specs must match.",
        ),
    ],
    chunk_size: Annotated[
        int,
        typer.Option("--chunk-size", min=1),
    ] = DEFAULT_SUBMIT_CHUNK_SIZE,
    attempt_index: Annotated[
        int,
        typer.Option("--attempt-index", min=0),
    ] = 0,
    queue_registration_concurrency: Annotated[
        int,
        typer.Option(
            "--queue-registration-concurrency",
            min=1,
            help=(
                "Worker concurrency to register in DBOS queue metadata. "
                "submit-jsonl does not start a queue worker."
            ),
        ),
    ] = DEFAULT_WORKER_CONCURRENCY,
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
        worker_concurrency=queue_registration_concurrency,
        consume_generation_queue=False,
    )
    register_platform_generation_queue(
        worker_concurrency=queue_registration_concurrency,
    )
    engine = create_engine(config.database_url)
    try:
        result = submit_prediction_specs_jsonl(
            engine,
            database_url=config.database_url,
            operation_key=operation_key,
            experiment_name=experiment_name,
            specs_file=specs_file,
            submit_spec={"source": str(specs_file)},
            chunk_size=chunk_size,
            attempt_index=attempt_index,
        )
        CONSOLE.print(result.model_dump(mode="json"))
    finally:
        engine.dispose()
        destroy_dbos_runtime()


def main() -> None:
    run_typer_app(APP)


if __name__ == "__main__":
    main()
