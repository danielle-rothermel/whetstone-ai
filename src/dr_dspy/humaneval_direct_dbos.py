from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from dbos import DBOS
from psycopg.types.json import Jsonb
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
)
from rich.console import Console

import dspy
from dr_dspy import batch_operation as shared_batch
from dr_dspy import dbos_runtime as shared_dbos
from dr_dspy import dspy_runner as shared_dspy_runner
from dr_dspy import eval_logging as shared_eval_logging
from dr_dspy import eval_repair as shared_eval_repair
from dr_dspy import human_eval_sampling as shared_human_eval_sampling
from dr_dspy import humaneval_dbos_flow as shared_flow
from dr_dspy import job_ordering as shared_job_ordering
from dr_dspy import worker_monitor as shared_worker_monitor
from dr_dspy import worker_resources as shared_worker_resources
from dr_dspy.experiment_dimensions import (
    Dimension,
    identity_dimension_names,
)
from dr_dspy.failures import (
    FailureSummary,
    error_text,
    failure_summary_payload,
    should_retry_step,
    summarize_exception,
)
from dr_dspy.human_eval import HumanEvalTask
from dr_dspy.lm_utils import (
    LmEventBuffer,
    ModelConfig,
)
from dr_dspy.prediction_status import (
    GENERATION_RETRY_STATUSES,
    GenerationStatus,
    ScoringStatus,
)
from dr_dspy.runtime import load_env_file
from dr_dspy.scoring import (
    HumanEvalScoreResult,
    score_humaneval_prediction,
)
from dr_dspy.signatures import DspySignatureConfig
from dspy.signatures.signature import make_signature

# Configuration

DATABASE_URL_ENV = "DATABASE_URL"
DBOS_APP_NAME = "dr-dspy-humaneval-eval-only"
DBOS_SYSTEM_DATABASE_URL_ENV = "DBOS_SYSTEM_DATABASE_URL"
GENERATION_QUEUE_NAME = "dr_dspy_humaneval_generation"
SCORING_QUEUE_NAME = "dr_dspy_humaneval_scoring"
EXPERIMENT_QUEUE_HASH_LENGTH = 8
DEFAULT_GENERATION_CONCURRENCY = 200
DEFAULT_SCORING_CONCURRENCY = 32
DEFAULT_WORKER_OPEN_FILE_LIMIT = shared_worker_resources.OPEN_FILE_LIMIT_AUTO
DEFAULT_WORKER_MONITOR_INTERVAL_SECONDS = 5.0
DEFAULT_WORKER_MONITOR_SUMMARY_INTERVAL_SECONDS = 5.0
DEFAULT_SUBMIT_BATCH_SIZE = 512
DEFAULT_OPERATION_BATCH_SIZE = 512
DEFAULT_COST_SIGNIFICANT_DIGITS = 6
PRICE_PER_THOUSAND_SAMPLE_MULTIPLIER = 1000.0
ANALYSIS_TOTAL_LABEL = "Total"
TABLE_ROW_STYLES = ("", "on grey7")
TABLE_TOTAL_ROW_STYLE = "bold black on green3"
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKER_LOG_ROOT = PACKAGE_ROOT / "logs"
DETAILED_WORKER_LOGGER_NAME = "dr_dspy.humaneval_eval_only_worker"
SUBMIT_LOGGER_NAME = "dr_dspy.humaneval_eval_only_submit"
OPERATION_LOGGER_NAME = "dr_dspy.humaneval_eval_only_operations"
CONSOLE = Console(soft_wrap=True)
OPERATOR_TIMESTAMP_FORMAT = "%H:%M:%S"
PREDICTION_TABLE_NAME = "dr_dspy_eval_predictions"
PREDICTION_ID_DIGEST_LENGTH = 32
DIMENSIONS: tuple[Dimension, ...] = (
    Dimension(
        name="model",
        sql_type="TEXT",
        nullable=False,
        report_title="Model",
    ),
    Dimension(
        name="temperature",
        sql_type="DOUBLE PRECISION",
        report_title="Temp",
        report_justify="right",
    ),
    Dimension(
        name="reasoning",
        sql_type="JSONB",
        nullable=False,
        default_sql="'{}'::jsonb",
        in_reporting=False,
        report_title="Reasoning",
    ),
)
REPAIR_DIMENSION_COLUMNS = identity_dimension_names(DIMENSIONS)
REPAIR_ORDER_COLUMNS = (
    "model",
    "temperature",
    "sample_index",
    "repetition_seed",
)


EXPERIMENTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS dr_dspy_eval_experiments (
    experiment_name TEXT PRIMARY KEY,
    script_kind     TEXT        NOT NULL,
    seed            INTEGER     NOT NULL,
    sample_count    INTEGER     NOT NULL,
    instruction     TEXT        NOT NULL,
    metadata        JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""

PREDICTIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS dr_dspy_eval_predictions (
    prediction_id        TEXT PRIMARY KEY,
    experiment_name      TEXT        NOT NULL
        REFERENCES dr_dspy_eval_experiments(experiment_name),
    script_kind          TEXT        NOT NULL,
    submission_id        TEXT        NOT NULL,
    task_id              TEXT        NOT NULL,
    sample_index         INTEGER     NOT NULL,
    model                TEXT        NOT NULL,
    temperature          DOUBLE PRECISION,
    repetition_seed      INTEGER     NOT NULL,
    prompt               TEXT        NOT NULL,
    canonical_solution   TEXT        NOT NULL DEFAULT '',
    ground_truth_code    TEXT        NOT NULL DEFAULT '',
    test                 TEXT        NOT NULL,
    entry_point          TEXT        NOT NULL,
    reasoning            JSONB       NOT NULL DEFAULT '{}'::jsonb,
    generation_status    TEXT        NOT NULL DEFAULT 'pending',
    generation_error     TEXT,
    generation_failure_class TEXT,
    generation_failure_exception_type TEXT,
    generation_underlying_exception_type TEXT,
    generation_exception_message TEXT,
    raw_code             TEXT,
    response_metadata    JSONB       NOT NULL DEFAULT '{}'::jsonb,
    usage_metadata       JSONB       NOT NULL DEFAULT '{}'::jsonb,
    provider_cost        DOUBLE PRECISION,
    scoring_status       TEXT        NOT NULL DEFAULT 'pending',
    score                DOUBLE PRECISION,
    scoring_error        TEXT,
    scoring_failure_class TEXT,
    scoring_failure_exception_type TEXT,
    scoring_underlying_exception_type TEXT,
    scoring_exception_message TEXT,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    generated_at         TIMESTAMPTZ,
    scored_at            TIMESTAMPTZ,
    CONSTRAINT dr_dspy_eval_predictions_identity_key UNIQUE (
        experiment_name,
        task_id,
        model,
        temperature,
        reasoning,
        repetition_seed
    )
)
"""

PREDICTION_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_dr_dspy_eval_predictions_experiment "
    "ON dr_dspy_eval_predictions(experiment_name)",
    "CREATE INDEX IF NOT EXISTS idx_dr_dspy_eval_predictions_generation "
    "ON dr_dspy_eval_predictions(experiment_name, generation_status)",
    "CREATE INDEX IF NOT EXISTS idx_dr_dspy_eval_predictions_scoring "
    "ON dr_dspy_eval_predictions(experiment_name, scoring_status)",
    "CREATE INDEX IF NOT EXISTS idx_dr_dspy_eval_predictions_model "
    "ON dr_dspy_eval_predictions(experiment_name, model, temperature)",
)

PREDICTION_MIGRATION_SQL = (
    "ALTER TABLE dr_dspy_eval_predictions "
    "ADD COLUMN IF NOT EXISTS raw_generation TEXT",
    "ALTER TABLE dr_dspy_eval_predictions "
    "ADD COLUMN IF NOT EXISTS generation_failure_class TEXT",
    "ALTER TABLE dr_dspy_eval_predictions "
    "ADD COLUMN IF NOT EXISTS generation_exception_type TEXT",
    "ALTER TABLE dr_dspy_eval_predictions "
    "ADD COLUMN IF NOT EXISTS generation_exception_message TEXT",
    "ALTER TABLE dr_dspy_eval_predictions "
    "ADD COLUMN IF NOT EXISTS scoring_failure_class TEXT",
    "ALTER TABLE dr_dspy_eval_predictions "
    "ADD COLUMN IF NOT EXISTS scoring_exception_type TEXT",
    "ALTER TABLE dr_dspy_eval_predictions "
    "ADD COLUMN IF NOT EXISTS scoring_exception_message TEXT",
    "ALTER TABLE dr_dspy_eval_predictions "
    "ADD COLUMN IF NOT EXISTS raw_compile_ok BOOLEAN",
    "ALTER TABLE dr_dspy_eval_predictions "
    "ADD COLUMN IF NOT EXISTS raw_compile_error TEXT",
    "ALTER TABLE dr_dspy_eval_predictions "
    "ADD COLUMN IF NOT EXISTS extraction_candidate_count INTEGER",
    "ALTER TABLE dr_dspy_eval_predictions "
    "ADD COLUMN IF NOT EXISTS selected_candidate_index INTEGER",
    "ALTER TABLE dr_dspy_eval_predictions "
    "ADD COLUMN IF NOT EXISTS extracted_compile_ok BOOLEAN",
    "ALTER TABLE dr_dspy_eval_predictions "
    "ADD COLUMN IF NOT EXISTS extracted_compile_error TEXT",
    "ALTER TABLE dr_dspy_eval_predictions "
    "ADD COLUMN IF NOT EXISTS extraction_error TEXT",
    "ALTER TABLE dr_dspy_eval_predictions "
    "ADD COLUMN IF NOT EXISTS canonical_solution TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE dr_dspy_eval_predictions "
    "ADD COLUMN IF NOT EXISTS ground_truth_code TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE dr_dspy_eval_predictions "
    "ADD COLUMN IF NOT EXISTS evaluation_function_names JSONB "
    "NOT NULL DEFAULT '[]'::jsonb",
    "ALTER TABLE dr_dspy_eval_predictions "
    "ADD COLUMN IF NOT EXISTS evaluation_total_cases INTEGER",
    "ALTER TABLE dr_dspy_eval_predictions "
    "ADD COLUMN IF NOT EXISTS evaluation_failure_count INTEGER",
    "ALTER TABLE dr_dspy_eval_predictions "
    "ADD COLUMN IF NOT EXISTS evaluation_status_counts JSONB "
    "NOT NULL DEFAULT '{}'::jsonb",
    "ALTER TABLE dr_dspy_eval_predictions "
    "ADD COLUMN IF NOT EXISTS compression_metrics JSONB "
    "NOT NULL DEFAULT '{}'::jsonb",
    "ALTER TABLE dr_dspy_eval_predictions "
    "ALTER COLUMN compression_metrics SET DEFAULT '{}'::jsonb",
    "ALTER TABLE dr_dspy_eval_predictions "
    "ADD COLUMN IF NOT EXISTS best_compression_ratio DOUBLE PRECISION",
    "ALTER TABLE dr_dspy_eval_predictions "
    "ADD COLUMN IF NOT EXISTS best_compression_percent_reduction "
    "DOUBLE PRECISION",
    "ALTER TABLE dr_dspy_eval_predictions "
    "ADD COLUMN IF NOT EXISTS raw_compression_ratio DOUBLE PRECISION",
    "ALTER TABLE dr_dspy_eval_predictions "
    "ADD COLUMN IF NOT EXISTS generation_failure_exception_type TEXT",
    "ALTER TABLE dr_dspy_eval_predictions "
    "ADD COLUMN IF NOT EXISTS generation_underlying_exception_type TEXT",
    "ALTER TABLE dr_dspy_eval_predictions "
    "ADD COLUMN IF NOT EXISTS scoring_failure_exception_type TEXT",
    "ALTER TABLE dr_dspy_eval_predictions "
    "ADD COLUMN IF NOT EXISTS scoring_underlying_exception_type TEXT",
    "UPDATE dr_dspy_eval_predictions "
    "SET generation_underlying_exception_type = generation_exception_type "
    "WHERE generation_underlying_exception_type IS NULL "
    "AND generation_exception_type IS NOT NULL",
    "UPDATE dr_dspy_eval_predictions "
    "SET scoring_underlying_exception_type = scoring_exception_type "
    "WHERE scoring_underlying_exception_type IS NULL "
    "AND scoring_exception_type IS NOT NULL",
)
PREDICTION_CONSTRAINT_MIGRATION_SQL = (
    """
    DO $$
    DECLARE old_constraint_name text;
    BEGIN
        SELECT c.conname INTO old_constraint_name
        FROM pg_constraint c
        JOIN pg_class t ON t.oid = c.conrelid
        WHERE t.relname = 'dr_dspy_eval_predictions'
          AND c.contype = 'u'
          AND ARRAY(
              SELECT a.attname::text
              FROM unnest(c.conkey) WITH ORDINALITY AS cols(attnum, ord)
              JOIN pg_attribute a
                ON a.attrelid = c.conrelid
               AND a.attnum = cols.attnum
              ORDER BY cols.ord
          ) = ARRAY[
              'experiment_name',
              'task_id',
              'model',
              'temperature',
              'repetition_seed'
          ];

        IF old_constraint_name IS NOT NULL THEN
            EXECUTE format(
                'ALTER TABLE dr_dspy_eval_predictions DROP CONSTRAINT %I',
                old_constraint_name
            );
        END IF;

        IF NOT EXISTS (
            SELECT 1
            FROM pg_constraint c
            JOIN pg_class t ON t.oid = c.conrelid
            WHERE t.relname = 'dr_dspy_eval_predictions'
              AND c.conname = 'dr_dspy_eval_predictions_identity_key'
        ) THEN
            ALTER TABLE dr_dspy_eval_predictions
            ADD CONSTRAINT dr_dspy_eval_predictions_identity_key UNIQUE (
                experiment_name,
                task_id,
                model,
                temperature,
                reasoning,
                repetition_seed
            );
        END IF;
    END $$;
    """,
)


class HumanEvalSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: StrictStr
    sample_index: StrictInt
    prompt: StrictStr
    canonical_solution: StrictStr = ""
    ground_truth_code: StrictStr = ""
    test: StrictStr
    entry_point: StrictStr


class DirectHumanEvalExperimentConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    script_kind: StrictStr
    dataset_name: StrictStr
    dataset_split: StrictStr
    solve_signature: DspySignatureConfig
    default_model_configs: tuple[ModelConfig, ...]
    default_sample_count: StrictInt
    default_seed: StrictInt
    default_temperatures: tuple[float, ...]
    default_repetitions: StrictInt
    default_max_completion_tokens: StrictInt
    default_subprocess_timeout: float


class DirectSubmitSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    script_kind: StrictStr
    experiment_name: StrictStr
    seed: StrictInt
    sample_count: StrictInt
    model_configs: list[ModelConfig]
    temperatures: list[float]
    repetitions: StrictInt
    score_timeout: float

    def jobs_per_sample(self) -> int:
        return (
            len(self.model_configs) * len(self.temperatures) * self.repetitions
        )

    def total_jobs(self) -> int:
        return self.sample_count * self.jobs_per_sample()


class PredictionJob(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_id: StrictStr
    experiment_name: StrictStr
    script_kind: StrictStr = Field(
        default_factory=lambda: experiment_config().script_kind
    )
    submission_id: StrictStr
    task_id: StrictStr
    sample_index: StrictInt
    model: StrictStr
    temperature: float
    repetition_seed: StrictInt
    prompt: StrictStr
    canonical_solution: StrictStr = ""
    ground_truth_code: StrictStr = ""
    test: StrictStr
    entry_point: StrictStr
    reasoning: dict[str, Any] = Field(default_factory=dict)


class GenerationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_id: StrictStr
    raw_generation: StrictStr
    response_metadata: dict[str, Any] = Field(default_factory=dict)
    usage_metadata: dict[str, Any] = Field(default_factory=dict)
    provider_cost: float | None = None


class ScoringTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_id: StrictStr
    task_id: StrictStr
    prompt: StrictStr = ""
    canonical_solution: StrictStr = ""
    ground_truth_code: StrictStr = ""
    raw_generation: StrictStr
    test: StrictStr
    entry_point: StrictStr

    def task(self) -> HumanEvalTask:
        return HumanEvalTask(
            task_id=self.task_id,
            prompt=self.prompt,
            canonical_solution=self.canonical_solution,
            test=self.test,
            entry_point=self.entry_point,
        )


ScoreResult = HumanEvalScoreResult


HumanEvalRow = Mapping[str, Any]

_EXPERIMENT_CONFIG: DirectHumanEvalExperimentConfig | None = None
_SOLVE_SIGNATURE: type[dspy.Signature] | None = None


def build_dspy_signature(config: DspySignatureConfig) -> type[dspy.Signature]:
    return make_signature(
        {field.name: (field.type, field.role) for field in config.fields},
        instructions=config.instructions,
        signature_name=config.name,
    )


def configure_experiment(
    config: DirectHumanEvalExperimentConfig,
) -> None:
    global _EXPERIMENT_CONFIG, _SOLVE_SIGNATURE
    _EXPERIMENT_CONFIG = config
    _SOLVE_SIGNATURE = build_dspy_signature(config.solve_signature)


def experiment_config() -> DirectHumanEvalExperimentConfig:
    if _EXPERIMENT_CONFIG is None:
        raise RuntimeError(
            "HumanEval direct experiment is not configured; call "
            "create_app(config) from the experiment script first."
        )
    return _EXPERIMENT_CONFIG


def solve_signature() -> type[dspy.Signature]:
    if _SOLVE_SIGNATURE is None:
        raise RuntimeError(
            "HumanEval direct DSPy signature is not configured; call "
            "create_app(config) from the experiment script first."
        )
    return _SOLVE_SIGNATURE


def _resolve_worker_log_path(
    *,
    experiment_name: str,
    queue: shared_dbos.QueueSelection,
    log_file: Path | None,
) -> Path:
    return shared_eval_logging.resolve_worker_log_path(
        log_root=DEFAULT_WORKER_LOG_ROOT,
        experiment_name=experiment_name,
        queue=queue,
        log_file=log_file,
        hash_length=EXPERIMENT_QUEUE_HASH_LENGTH,
    )


def _configure_worker_file_logging(log_file: Path) -> logging.Logger:
    return shared_eval_logging.configure_worker_file_logging(
        log_file, logger_name=DETAILED_WORKER_LOGGER_NAME
    )


def _resolve_operation_log_path(
    *,
    experiment_name: str,
    operation_kind: shared_batch.BatchOperationKind,
    log_file: Path | None = None,
) -> Path:
    return shared_batch.resolve_operation_log_path(
        log_root=DEFAULT_WORKER_LOG_ROOT,
        experiment_name=experiment_name,
        operation_kind=operation_kind,
        log_file=log_file,
        hash_length=EXPERIMENT_QUEUE_HASH_LENGTH,
    )


def _configure_operation_file_logging(log_file: Path) -> None:
    shared_batch.configure_operation_file_logging(
        log_file, logger_name=OPERATION_LOGGER_NAME
    )


def _emit_operation_log(event: str, payload: Mapping[str, Any]) -> None:
    shared_batch.emit_operation_log(
        event, payload, logger_name=OPERATION_LOGGER_NAME
    )


def _resolve_submit_log_path(
    *, experiment_name: str, log_file: Path | None = None
) -> Path:
    return _resolve_operation_log_path(
        experiment_name=experiment_name,
        operation_kind=shared_batch.BatchOperationKind.SUBMIT,
        log_file=log_file,
    )


def _configure_submit_file_logging(log_file: Path) -> None:
    shared_batch.configure_operation_file_logging(
        log_file, logger_name=SUBMIT_LOGGER_NAME
    )


def _emit_submit_log(event: str, payload: Mapping[str, Any]) -> None:
    shared_batch.emit_operation_log(
        event, payload, logger_name=SUBMIT_LOGGER_NAME
    )


def _emit_worker_detail_log(event: str, payload: Mapping[str, Any]) -> None:
    shared_eval_logging.emit_worker_detail_log(
        event, payload, logger_name=DETAILED_WORKER_LOGGER_NAME
    )


def _prediction_context_from_job(
    job: PredictionJob,
) -> shared_eval_logging.PredictionLogContext:
    return shared_eval_logging.PredictionLogContext(
        prediction_id=job.prediction_id,
        experiment_name=job.experiment_name,
        task_id=job.task_id,
        sample_index=job.sample_index,
        repetition_seed=job.repetition_seed,
        dimensions={
            "model": job.model,
            "temperature": job.temperature,
            "reasoning": job.reasoning,
        },
    )


def _emit_prediction_log_event(
    event: str,
    context: shared_eval_logging.PredictionLogContext,
    *,
    extra: Mapping[str, Any] | None = None,
) -> None:
    shared_eval_logging.emit_prediction_log_event(
        event,
        context,
        logger_name=DETAILED_WORKER_LOGGER_NAME,
        extra=extra,
    )


@DBOS.workflow(name="humaneval_eval_generate_prediction_v0")
def generate_prediction_workflow(
    database_url: str,
    prediction_id: str,
    experiment_name: str,
    score_timeout: float,
) -> str:
    try:
        result = generate_prediction_step(database_url, prediction_id)
        record_generation_success_step(database_url, result)
    except Exception as error:
        summary = summarize_exception(error)
        record_generation_error_step(database_url, prediction_id, summary)
        return (
            "generation_recoverable_error"
            if summary.is_recoverable
            else "generation_error"
        )
    _enqueue_score_job(
        database_url,
        prediction_id,
        experiment_name=experiment_name,
        timeout=score_timeout,
    )
    mark_scoring_queued_step(database_url, prediction_id)
    return "generated"


@DBOS.workflow(name="humaneval_eval_score_prediction_v0")
def score_prediction_workflow(
    database_url: str, prediction_id: str, timeout: float
) -> str:
    try:
        result = score_prediction_step(database_url, prediction_id, timeout)
        record_score_success_step(database_url, result)
        return "scored"
    except Exception as error:
        summary = summarize_exception(error)
        record_score_error_step(database_url, prediction_id, summary)
        return (
            "score_recoverable_error"
            if summary.is_recoverable
            else "score_error"
        )


@DBOS.workflow(
    name="humaneval_direct_submit_dispatcher_v0",
    max_recovery_attempts=1,
)
def submit_dispatcher_workflow(
    database_url: str, operation_key: str
) -> str:
    completion_modes = shared_batch.BatchDispatcherCompletionMode
    return shared_batch.run_operation_dispatcher(
        database_url,
        operation_kind=shared_batch.BatchOperationKind.SUBMIT,
        operation_key=operation_key,
        configure_logging=_configure_submit_file_logging,
        emit_log=_emit_submit_log,
        started_event="submit_dispatcher_started",
        started_payload=lambda progress: {
            "operation_key": operation_key,
            "workflow_id": progress.workflow_id,
            "attempt": progress.attempt,
            "next_offset": progress.next_offset,
            "total_jobs": progress.total_items,
        },
        failed_event="submit_dispatcher_failed",
        batch_step=submit_batch_step,
        completion_mode=completion_modes.OFFSET_TOTAL,
        completed_event="submit_dispatcher_completed",
        completed_payload=lambda progress: {
            "operation_key": operation_key,
            "total_jobs": progress.total_items,
            "batch_count": progress.batch_count,
        },
    )


def submit_batch_step(
    database_url: str, operation_key: str
) -> shared_batch.BatchOperationResult:
    progress = shared_batch.fetch_operation_progress(
        database_url,
        operation_kind=shared_batch.BatchOperationKind.SUBMIT,
        operation_key=operation_key,
    )
    spec = DirectSubmitSpec(
        **shared_batch.fetch_operation_spec(
            database_url,
            operation_kind=shared_batch.BatchOperationKind.SUBMIT,
            operation_key=operation_key,
        )
    )
    sample_window = shared_batch.operation_item_window(
        start_offset=progress.next_offset,
        limit=int(progress.metadata["batch_size"]),
        total_items=spec.total_jobs(),
        items_per_group=spec.jobs_per_sample(),
    )
    sample_payloads = shared_batch.fetch_operation_items(
        database_url,
        operation_kind=shared_batch.BatchOperationKind.SUBMIT,
        operation_key=operation_key,
        item_kind=shared_batch.BatchOperationItemKind.SAMPLE,
        start_index=sample_window.start_index,
        limit=sample_window.item_count,
    )
    if len(sample_payloads) != sample_window.item_count:
        raise ValueError(
            "submit sample manifest incomplete: "
            f"operation_key={operation_key}, "
            f"start_index={sample_window.start_index}, "
            f"expected={sample_window.item_count}, "
            f"actual={len(sample_payloads)}"
        )
    samples = [HumanEvalSample(**payload) for payload in sample_payloads]
    jobs = build_prediction_jobs_for_offsets(
        spec=spec,
        submission_id=str(progress.metadata["submission_id"]),
        samples=samples,
        start_offset=progress.next_offset,
        limit=int(progress.metadata["batch_size"]),
    )
    jobs = shared_job_ordering.stable_shuffle(
        jobs,
        seed=shared_job_ordering.stable_order_key(
            "submit",
            spec.script_kind,
            spec.experiment_name,
            spec.seed,
            progress.metadata["submission_id"],
        ),
        key=lambda job: job.prediction_id,
    )
    _emit_submit_log(
        "submit_batch_started",
        {
            "operation_key": operation_key,
            "start_offset": progress.next_offset,
            "batch_size": len(jobs),
        },
    )
    try:
        inserted = insert_prediction_jobs(database_url, jobs)
        enqueue_result = _enqueue_generation_jobs(
            database_url, jobs, score_timeout=spec.score_timeout
        )
    except Exception as error:
        _emit_submit_log(
            "submit_batch_failed",
            {
                "operation_key": operation_key,
                "start_offset": progress.next_offset,
                "batch_size": len(jobs),
                "error": repr(error),
            },
        )
        raise
    result = shared_batch.BatchOperationResult(
        start_offset=progress.next_offset,
        next_offset=progress.next_offset + len(jobs),
        batch_size=len(jobs),
        processed=len(jobs),
        inserted=inserted,
        enqueued=enqueue_result.enqueued,
        existing_workflows=enqueue_result.existing,
    )
    shared_batch.record_operation_batch_success(
        database_url,
        operation_kind=shared_batch.BatchOperationKind.SUBMIT,
        operation_key=operation_key,
        result=result,
    )
    _emit_submit_log(
        "submit_batch_succeeded", result.model_dump(mode="json")
    )
    return result


@DBOS.workflow(
    name="humaneval_direct_enqueue_scores_dispatcher_v0",
    max_recovery_attempts=1,
)
def enqueue_scores_dispatcher_workflow(
    database_url: str, operation_key: str
) -> str:
    operation_kind = shared_batch.BatchOperationKind.ENQUEUE_SCORES
    completion_modes = shared_batch.BatchDispatcherCompletionMode
    return shared_batch.run_operation_dispatcher(
        database_url,
        operation_kind=operation_kind,
        operation_key=operation_key,
        configure_logging=_configure_operation_file_logging,
        emit_log=_emit_operation_log,
        started_event="enqueue_scores_dispatcher_started",
        started_payload=lambda progress: {
            "operation_key": operation_key,
            "workflow_id": progress.workflow_id,
            "attempt": progress.attempt,
            "batch_size": progress.metadata["batch_size"],
        },
        failed_event="enqueue_scores_dispatcher_failed",
        batch_step=enqueue_scores_batch_step,
        completion_mode=completion_modes.EMPTY_BATCH,
    )


def enqueue_scores_batch_step(
    database_url: str, operation_key: str
) -> shared_batch.BatchOperationResult:
    operation_kind = shared_batch.BatchOperationKind.ENQUEUE_SCORES
    progress = shared_batch.fetch_operation_progress(
        database_url,
        operation_kind=operation_kind,
        operation_key=operation_key,
    )
    spec = shared_batch.fetch_operation_spec(
        database_url,
        operation_kind=operation_kind,
        operation_key=operation_key,
    )
    prediction_ids = fetch_scoreable_prediction_ids(
        database_url,
        experiment_name=str(spec["experiment_name"]),
        limit=int(spec["batch_size"]),
    )
    _emit_operation_log(
        "enqueue_scores_batch_started",
        {
            "operation_key": operation_key,
            "selected": len(prediction_ids),
            "batch_size": int(spec["batch_size"]),
        },
    )
    enqueue_result = _enqueue_score_jobs(
        database_url,
        prediction_ids,
        experiment_name=str(spec["experiment_name"]),
        timeout=float(spec["timeout"]),
    )
    marked = mark_scoring_queued(database_url, prediction_ids)
    result = shared_batch.BatchOperationResult(
        start_offset=progress.processed_count,
        next_offset=progress.processed_count + len(prediction_ids),
        batch_size=len(prediction_ids),
        processed=len(prediction_ids),
        enqueued=enqueue_result.enqueued,
        existing_workflows=enqueue_result.existing,
        marked=marked,
        counters={"score_selected": len(prediction_ids)},
    )
    shared_batch.record_operation_batch_success(
        database_url,
        operation_kind=operation_kind,
        operation_key=operation_key,
        result=result,
    )
    _emit_operation_log(
        "enqueue_scores_batch_succeeded", result.model_dump(mode="json")
    )
    return result


@DBOS.workflow(
    name="humaneval_direct_repair_dispatcher_v0",
    max_recovery_attempts=1,
)
def repair_dispatcher_workflow(database_url: str, operation_key: str) -> str:
    operation_kind = shared_batch.BatchOperationKind.REPAIR
    completion_modes = shared_batch.BatchDispatcherCompletionMode
    return shared_batch.run_operation_dispatcher(
        database_url,
        operation_kind=operation_kind,
        operation_key=operation_key,
        configure_logging=_configure_operation_file_logging,
        emit_log=_emit_operation_log,
        started_event="repair_dispatcher_started",
        started_payload=lambda progress: {
            "operation_key": operation_key,
            "workflow_id": progress.workflow_id,
            "attempt": progress.attempt,
            "batch_size": progress.metadata["batch_size"],
        },
        failed_event="repair_dispatcher_failed",
        batch_step=repair_batch_step,
        completion_mode=completion_modes.EMPTY_BATCH,
    )


def repair_batch_step(
    database_url: str, operation_key: str
) -> shared_batch.BatchOperationResult:
    operation_kind = shared_batch.BatchOperationKind.REPAIR
    progress = shared_batch.fetch_operation_progress(
        database_url,
        operation_kind=operation_kind,
        operation_key=operation_key,
    )
    spec = shared_batch.fetch_operation_spec(
        database_url,
        operation_kind=operation_kind,
        operation_key=operation_key,
    )
    _emit_operation_log(
        "repair_batch_started",
        {
            "operation_key": operation_key,
            "batch_size": int(spec["batch_size"]),
        },
    )
    config = _build_eval_dbos_config(
        database_url=database_url,
        dbos_system_database_url=str(spec["dbos_system_database_url"]),
        generation_concurrency=int(spec["generation_concurrency"]),
        scoring_concurrency=int(spec["scoring_concurrency"]),
    )
    repair_result = shared_eval_repair.apply_repair_batch(
        config,
        prediction_table=PREDICTION_TABLE_NAME,
        experiment_name=str(spec["experiment_name"]),
        dimension_columns=REPAIR_DIMENSION_COLUMNS,
        order_columns=REPAIR_ORDER_COLUMNS,
        batch_size=int(spec["batch_size"]),
        score_timeout=float(spec["score_timeout"]),
        fetch_generation_jobs=lambda prediction_ids: [
            fetch_prediction_job(config.database_url, prediction_id)
            for prediction_id in prediction_ids
        ],
        reset_generation_errors=lambda prediction_ids: (
            reset_generation_errors_for_retry(
                config.database_url,
                prediction_ids=prediction_ids,
            )
        ),
        enqueue_generation_jobs=lambda jobs, token: _enqueue_generation_jobs(
            config.database_url,
            jobs,
            score_timeout=float(spec["score_timeout"]),
            retry_token=token,
        ),
        enqueue_score_jobs=lambda prediction_ids, timeout, token: (
            _enqueue_score_jobs(
                config.database_url,
                prediction_ids,
                experiment_name=str(spec["experiment_name"]),
                timeout=timeout,
                retry_token=token,
            )
        ),
        repair_token=operation_key,
    )
    result = shared_batch.repair_batch_operation_result(
        progress=progress,
        batch_size=int(spec["batch_size"]),
        repair_result=repair_result,
    )
    shared_batch.record_operation_batch_success(
        database_url,
        operation_kind=operation_kind,
        operation_key=operation_key,
        result=result,
    )
    _emit_operation_log(
        "repair_batch_succeeded", result.model_dump(mode="json")
    )
    return result


def stable_prediction_id(
    *,
    experiment_name: str,
    task_id: str,
    model: str,
    temperature: float,
    reasoning: Mapping[str, Any],
    repetition_seed: int,
) -> str:
    return shared_flow.stable_prediction_id_from_dimensions(
        experiment_name=experiment_name,
        task_id=task_id,
        dimensions={
            "model": model,
            "temperature": temperature,
            "reasoning": dict(reasoning),
        },
        repetition_seed=repetition_seed,
        digest_length=PREDICTION_ID_DIGEST_LENGTH,
    )


def parse_temperatures(raw: str) -> list[float]:
    return shared_flow.parse_float_csv(raw, value_name="temperature")


def load_optional_env_file(env_file: Path | None) -> None:
    if env_file is None:
        load_env_file()
    else:
        load_env_file(env_file)


def default_model_configs() -> list[ModelConfig]:
    return [
        ModelConfig(**config.model_dump(mode="python"))
        for config in experiment_config().default_model_configs
    ]


def build_submit_spec(
    *,
    experiment_name: str,
    seed: int,
    sample_count: int,
    model_configs: Sequence[ModelConfig],
    temperatures: Sequence[float],
    repetitions: int,
    score_timeout: float,
) -> DirectSubmitSpec:
    return DirectSubmitSpec(
        script_kind=experiment_config().script_kind,
        experiment_name=experiment_name,
        seed=seed,
        sample_count=sample_count,
        model_configs=[
            ModelConfig(**config.model_dump(mode="python"))
            for config in model_configs
        ],
        temperatures=list(temperatures),
        repetitions=repetitions,
        score_timeout=score_timeout,
    )


def build_prediction_jobs_for_offsets(
    *,
    spec: DirectSubmitSpec,
    submission_id: str,
    samples: Sequence[HumanEvalSample],
    start_offset: int,
    limit: int,
) -> list[PredictionJob]:
    total_jobs = spec.total_jobs()
    end_offset = min(start_offset + limit, total_jobs)
    samples_by_index = {sample.sample_index: sample for sample in samples}
    jobs: list[PredictionJob] = []
    for offset in range(start_offset, end_offset):
        remaining = offset
        repetition_seed = remaining % spec.repetitions
        remaining //= spec.repetitions
        temperature = spec.temperatures[remaining % len(spec.temperatures)]
        remaining //= len(spec.temperatures)
        model_config = spec.model_configs[
            remaining % len(spec.model_configs)
        ]
        remaining //= len(spec.model_configs)
        sample = samples_by_index.get(remaining)
        if sample is None:
            raise ValueError(
                "missing submit sample manifest item: "
                f"sample_index={remaining}"
            )
        jobs.append(
            PredictionJob(
                prediction_id=stable_prediction_id(
                    experiment_name=spec.experiment_name,
                    task_id=sample.task_id,
                    model=model_config.model,
                    temperature=temperature,
                    reasoning=model_config.reasoning,
                    repetition_seed=repetition_seed,
                ),
                experiment_name=spec.experiment_name,
                script_kind=spec.script_kind,
                submission_id=submission_id,
                task_id=sample.task_id,
                sample_index=sample.sample_index,
                model=model_config.model,
                temperature=temperature,
                repetition_seed=repetition_seed,
                prompt=sample.prompt,
                canonical_solution=sample.canonical_solution,
                ground_truth_code=sample.ground_truth_code,
                test=sample.test,
                entry_point=sample.entry_point,
                reasoning=dict(model_config.reasoning),
            )
        )
    return jobs


def upsert_experiment(
    database_url: str,
    *,
    experiment_name: str,
    seed: int,
    sample_count: int,
    metadata: Mapping[str, Any],
) -> None:
    with shared_dbos.connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO dr_dspy_eval_experiments (
                    experiment_name,
                    script_kind,
                    seed,
                    sample_count,
                    instruction,
                    metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (experiment_name) DO UPDATE SET
                    script_kind = EXCLUDED.script_kind,
                    seed = EXCLUDED.seed,
                    sample_count = EXCLUDED.sample_count,
                    instruction = EXCLUDED.instruction,
                    metadata = EXCLUDED.metadata,
                    updated_at = now()
                """,
                (
                    experiment_name,
                    experiment_config().script_kind,
                    seed,
                    sample_count,
                    experiment_config().solve_signature.instructions,
                    Jsonb(dict(metadata)),
                ),
            )


def insert_prediction_jobs(
    database_url: str, jobs: Sequence[PredictionJob]
) -> int:
    if not jobs:
        return 0
    rows = [
        (
            job.prediction_id,
            job.experiment_name,
            job.script_kind,
            job.submission_id,
            job.task_id,
            job.sample_index,
            job.model,
            job.temperature,
            job.repetition_seed,
            job.prompt,
            job.canonical_solution,
            job.ground_truth_code,
            job.test,
            job.entry_point,
            Jsonb(job.reasoning),
        )
        for job in jobs
    ]
    with shared_dbos.connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO dr_dspy_eval_predictions (
                    prediction_id,
                    experiment_name,
                    script_kind,
                    submission_id,
                    task_id,
                    sample_index,
                    model,
                    temperature,
                    repetition_seed,
                    prompt,
                    canonical_solution,
                    ground_truth_code,
                    test,
                    entry_point,
                    reasoning
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (prediction_id) DO NOTHING
                """,
                rows,
            )
            return cur.rowcount if cur.rowcount is not None else 0


def fetch_prediction_job(
    database_url: str, prediction_id: str
) -> PredictionJob:
    with shared_dbos.connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    prediction_id,
                    experiment_name,
                    script_kind,
                    submission_id,
                    task_id,
                    sample_index,
                    model,
                    temperature,
                    repetition_seed,
                    prompt,
                    canonical_solution,
                    ground_truth_code,
                    test,
                    entry_point,
                    reasoning
                FROM dr_dspy_eval_predictions
                WHERE prediction_id = %s
                """,
                (prediction_id,),
            )
            row = cur.fetchone()
    if row is None:
        raise ValueError(f"prediction_id not found: {prediction_id}")
    return PredictionJob(
        prediction_id=row[0],
        experiment_name=row[1],
        script_kind=row[2],
        submission_id=row[3],
        task_id=row[4],
        sample_index=row[5],
        model=row[6],
        temperature=row[7],
        repetition_seed=row[8],
        prompt=row[9],
        canonical_solution=row[10],
        ground_truth_code=row[11],
        test=row[12],
        entry_point=row[13],
        reasoning=dict(row[14] or {}),
    )


def fetch_prediction_log_context(
    database_url: str, prediction_id: str
) -> shared_eval_logging.PredictionLogContext:
    return _prediction_context_from_job(
        fetch_prediction_job(database_url, prediction_id)
    )


def mark_generation_started(database_url: str, prediction_id: str) -> None:
    with shared_dbos.connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE dr_dspy_eval_predictions
                SET
                    generation_status = 'started',
                    generation_error = NULL,
                    generation_failure_class = NULL,
                    generation_failure_exception_type = NULL,
                    generation_underlying_exception_type = NULL,
                    generation_exception_message = NULL,
                    updated_at = now()
                WHERE prediction_id = %s
                """,
                (prediction_id,),
            )


def record_generation_success(
    database_url: str, result: GenerationResult
) -> None:
    with shared_dbos.connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE dr_dspy_eval_predictions
                SET
                    generation_status = 'generated',
                    generation_error = NULL,
                    generation_failure_class = NULL,
                    generation_failure_exception_type = NULL,
                    generation_underlying_exception_type = NULL,
                    generation_exception_message = NULL,
                    raw_generation = %s,
                    raw_code = NULL,
                    response_metadata = %s,
                    usage_metadata = %s,
                    provider_cost = %s,
                    updated_at = now(),
                    generated_at = now()
                WHERE prediction_id = %s
                """,
                (
                    result.raw_generation,
                    Jsonb(result.response_metadata),
                    Jsonb(result.usage_metadata),
                    result.provider_cost,
                    result.prediction_id,
                ),
            )


def record_generation_error(
    database_url: str,
    prediction_id: str,
    summary: FailureSummary,
) -> None:
    status = (
        "generation_recoverable_error"
        if summary.is_recoverable
        else "generation_error"
    )
    with shared_dbos.connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE dr_dspy_eval_predictions
                SET
                    generation_status = %s,
                    generation_error = %s,
                    generation_failure_class = %s,
                    generation_failure_exception_type = %s,
                    generation_underlying_exception_type = %s,
                    generation_exception_message = %s,
                    raw_generation = NULL,
                    raw_code = NULL,
                    updated_at = now()
                WHERE prediction_id = %s
                """,
                (
                    status,
                    error_text(summary),
                    summary.failure_class.value,
                    summary.failure_exception_type,
                    summary.underlying_exception_type,
                    summary.message,
                    prediction_id,
                ),
            )


def fetch_scoring_target(
    database_url: str, prediction_id: str
) -> ScoringTarget:
    with shared_dbos.connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    prediction_id,
                    task_id,
                    prompt,
                    canonical_solution,
                    ground_truth_code,
                    raw_generation,
                    test,
                    entry_point,
                    generation_status
                FROM dr_dspy_eval_predictions
                WHERE prediction_id = %s
                """,
                (prediction_id,),
            )
            row = cur.fetchone()
    if row is None:
        raise ValueError(f"prediction_id not found: {prediction_id}")
    if row[8] != "generated":
        raise ValueError(f"prediction_id is not generated: {prediction_id}")
    if row[5] is None:
        raise ValueError(
            f"prediction_id has no raw generation: {prediction_id}"
        )
    return ScoringTarget(
        prediction_id=row[0],
        task_id=row[1],
        prompt=row[2],
        canonical_solution=row[3],
        ground_truth_code=row[4] or row[2],
        raw_generation=row[5],
        test=row[6],
        entry_point=row[7],
    )


def fetch_scoreable_prediction_ids(
    database_url: str,
    *,
    experiment_name: str,
    limit: int,
) -> list[str]:
    return shared_eval_repair.fetch_scoreable_prediction_ids(
        database_url,
        prediction_table=PREDICTION_TABLE_NAME,
        experiment_name=experiment_name,
        order_columns=REPAIR_ORDER_COLUMNS,
        limit=limit,
    )


def reset_generation_errors_for_retry(
    database_url: str,
    *,
    prediction_ids: Sequence[str],
) -> int:
    if not prediction_ids:
        return 0
    with shared_dbos.connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE dr_dspy_eval_predictions
                SET
                    generation_status = %s,
                    generation_error = NULL,
                    generation_failure_class = NULL,
                    generation_failure_exception_type = NULL,
                    generation_underlying_exception_type = NULL,
                    generation_exception_message = NULL,
                    raw_generation = NULL,
                    raw_code = NULL,
                    raw_compile_ok = NULL,
                    raw_compile_error = NULL,
                    extraction_candidate_count = NULL,
                    selected_candidate_index = NULL,
                    extracted_compile_ok = NULL,
                    extracted_compile_error = NULL,
                    extraction_error = NULL,
                    response_metadata = '{}'::jsonb,
                    usage_metadata = '{}'::jsonb,
                    provider_cost = NULL,
                    generated_at = NULL,
                    scoring_status = %s,
                    scoring_error = NULL,
                    scoring_failure_class = NULL,
                    scoring_failure_exception_type = NULL,
                    scoring_underlying_exception_type = NULL,
                    scoring_exception_message = NULL,
                    score = NULL,
                    evaluation_function_names = '[]'::jsonb,
                    evaluation_total_cases = NULL,
                    evaluation_failure_count = NULL,
                    evaluation_status_counts = '{}'::jsonb,
                    compression_metrics = '{}'::jsonb,
                    raw_compression_ratio = NULL,
                    best_compression_ratio = NULL,
                    best_compression_percent_reduction = NULL,
                    scored_at = NULL,
                    updated_at = now()
                WHERE
                    prediction_id = ANY(%s)
                    AND generation_status = ANY(%s)
                """,
                (
                    GenerationStatus.PENDING.value,
                    ScoringStatus.PENDING.value,
                    list(prediction_ids),
                    list(GENERATION_RETRY_STATUSES),
                ),
            )
            return cur.rowcount if cur.rowcount is not None else 0


def mark_scoring_queued(
    database_url: str, prediction_ids: Sequence[str]
) -> int:
    return shared_eval_repair.mark_scoring_queued(
        database_url,
        prediction_table=PREDICTION_TABLE_NAME,
        prediction_ids=prediction_ids,
    )


def mark_scoring_started(database_url: str, prediction_id: str) -> None:
    with shared_dbos.connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE dr_dspy_eval_predictions
                SET
                    scoring_status = 'started',
                    scoring_error = NULL,
                    scoring_failure_class = NULL,
                    scoring_failure_exception_type = NULL,
                    scoring_underlying_exception_type = NULL,
                    scoring_exception_message = NULL,
                    updated_at = now()
                WHERE prediction_id = %s
                """,
                (prediction_id,),
            )


def record_score_success(database_url: str, result: ScoreResult) -> None:
    with shared_dbos.connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE dr_dspy_eval_predictions
                SET
                    scoring_status = 'scored',
                    score = %s,
                    scoring_error = %s,
                    scoring_failure_class = NULL,
                    scoring_failure_exception_type = NULL,
                    scoring_underlying_exception_type = NULL,
                    scoring_exception_message = NULL,
                    raw_code = %s,
                    raw_compile_ok = %s,
                    raw_compile_error = %s,
                    extraction_candidate_count = %s,
                    selected_candidate_index = %s,
                    extracted_compile_ok = %s,
                    extracted_compile_error = %s,
                    extraction_error = %s,
                    evaluation_function_names = %s,
                    evaluation_total_cases = %s,
                    evaluation_failure_count = %s,
                    evaluation_status_counts = %s,
                    compression_metrics = %s,
                    raw_compression_ratio = %s,
                    best_compression_ratio = %s,
                    best_compression_percent_reduction = %s,
                    updated_at = now(),
                    scored_at = now()
                WHERE prediction_id = %s
                """,
                (
                    result.score,
                    result.error,
                    result.raw_code,
                    result.raw_compile_ok,
                    result.raw_compile_error,
                    result.extraction_candidate_count,
                    result.selected_candidate_index,
                    result.extracted_compile_ok,
                    result.extracted_compile_error,
                    result.extraction_error,
                    Jsonb(result.evaluation_function_names),
                    result.evaluation_total_cases,
                    result.evaluation_failure_count,
                    Jsonb(result.evaluation_status_counts),
                    Jsonb(
                        {
                            method.value: metric.model_dump(mode="json")
                            for method, metric in (
                                result.compression_metrics.items()
                            )
                        }
                    ),
                    result.raw_compression_ratio,
                    result.best_compression_ratio,
                    result.best_compression_percent_reduction,
                    result.prediction_id,
                ),
            )


def record_score_error(
    database_url: str,
    prediction_id: str,
    summary: FailureSummary,
) -> None:
    status = (
        "score_recoverable_error"
        if summary.is_recoverable
        else "score_error"
    )
    with shared_dbos.connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE dr_dspy_eval_predictions
                SET
                    scoring_status = %s,
                    scoring_error = %s,
                    scoring_failure_class = %s,
                    scoring_failure_exception_type = %s,
                    scoring_underlying_exception_type = %s,
                    scoring_exception_message = %s,
                    updated_at = now()
                WHERE prediction_id = %s
                """,
                (
                    status,
                    error_text(summary),
                    summary.failure_class.value,
                    summary.failure_exception_type,
                    summary.underlying_exception_type,
                    summary.message,
                    prediction_id,
                ),
            )


def score_generated_code(
    target: ScoringTarget, *, timeout: float
) -> ScoreResult:
    return score_humaneval_prediction(
        prediction_id=target.prediction_id,
        raw_generation=target.raw_generation,
        task=target.task(),
        compression_input=target.prompt,
        ground_truth_code=target.ground_truth_code,
        timeout=timeout,
    )


@DBOS.step(name="humaneval_direct_score_prediction_step_v0")
def score_prediction_step(
    database_url: str, prediction_id: str, timeout: float
) -> ScoreResult:
    mark_scoring_started(database_url, prediction_id)
    context = fetch_prediction_log_context(database_url, prediction_id)
    _emit_prediction_log_event(
        "scoring_started", context, extra={"timeout": timeout}
    )
    return score_generated_code(
        fetch_scoring_target(database_url, prediction_id), timeout=timeout
    )


@DBOS.step(name="humaneval_direct_record_score_success_step_v0")
def record_score_success_step(database_url: str, result: ScoreResult) -> None:
    context = fetch_prediction_log_context(database_url, result.prediction_id)
    _emit_prediction_log_event(
        "scoring_succeeded",
        context,
        extra={"score": result.score, "scoring_error": result.error},
    )
    record_score_success(database_url, result)


@DBOS.step(name="humaneval_direct_record_score_error_step_v0")
def record_score_error_step(
    database_url: str, prediction_id: str, summary: FailureSummary
) -> None:
    context = fetch_prediction_log_context(database_url, prediction_id)
    _emit_prediction_log_event(
        "scoring_failed",
        context,
        extra=failure_summary_payload(summary),
    )
    record_score_error(database_url, prediction_id, summary)


@DBOS.step(name="humaneval_direct_mark_scoring_queued_step_v0")
def mark_scoring_queued_step(database_url: str, prediction_id: str) -> None:
    context = fetch_prediction_log_context(database_url, prediction_id)
    _emit_prediction_log_event("scoring_enqueued", context)
    mark_scoring_queued(database_url, [prediction_id])


@DBOS.step(
    name="humaneval_direct_generate_prediction_step_v0",
    retries_allowed=True,
    max_attempts=3,
    interval_seconds=2.0,
    should_retry=should_retry_step,
)
def generate_prediction_step(
    database_url: str, prediction_id: str
) -> GenerationResult:
    mark_generation_started(database_url, prediction_id)
    job = fetch_prediction_job(database_url, prediction_id)
    _emit_prediction_log_event(
        "generation_started", _prediction_context_from_job(job)
    )
    return generate_code_for_job(
        job, client=shared_worker_resources.openrouter_client()
    )


@DBOS.step(name="humaneval_direct_record_generation_success_step_v0")
def record_generation_success_step(
    database_url: str, result: GenerationResult
) -> None:
    context = fetch_prediction_log_context(database_url, result.prediction_id)
    _emit_prediction_log_event(
        "generation_succeeded",
        context,
        extra={
            "provider_cost": result.provider_cost,
            "usage_metadata": result.usage_metadata,
        },
    )
    record_generation_success(database_url, result)


@DBOS.step(name="humaneval_direct_record_generation_error_step_v0")
def record_generation_error_step(
    database_url: str, prediction_id: str, summary: FailureSummary
) -> None:
    context = fetch_prediction_log_context(database_url, prediction_id)
    _emit_prediction_log_event(
        "generation_failed",
        context,
        extra=failure_summary_payload(summary),
    )
    record_generation_error(database_url, prediction_id, summary)


def _build_repair_plan(
    database_url: str,
    *,
    dbos_system_database_url: str,
    experiment_name: str,
) -> shared_eval_repair.RepairPlan:
    return shared_eval_repair.build_repair_plan(
        database_url,
        dbos_system_database_url=dbos_system_database_url,
        prediction_table=PREDICTION_TABLE_NAME,
        experiment_name=experiment_name,
        dimension_columns=REPAIR_DIMENSION_COLUMNS,
        order_columns=REPAIR_ORDER_COLUMNS,
    )


def _operator_log(
    line: str,
    *,
    style: str | None = None,
    now: datetime | None = None,
) -> None:
    shared_eval_logging.operator_log(
        CONSOLE,
        line,
        style=style,
        now=now,
        timestamp_format=OPERATOR_TIMESTAMP_FORMAT,
    )


_APP = typer.Typer(no_args_is_help=True)


def create_app(config: DirectHumanEvalExperimentConfig) -> typer.Typer:
    configure_experiment(config)
    return _APP


QUEUE_NAME_CONFIG = shared_dbos.QueueNameConfig(
    generation_base_name=GENERATION_QUEUE_NAME,
    scoring_base_name=SCORING_QUEUE_NAME,
    hash_length=EXPERIMENT_QUEUE_HASH_LENGTH,
)

def _build_eval_dbos_config(
    *,
    database_url: str | None,
    dbos_system_database_url: str | None,
    generation_concurrency: int,
    scoring_concurrency: int,
) -> shared_dbos.EvalDbosConfig:
    return shared_dbos.build_eval_dbos_config(
        database_url=database_url,
        dbos_system_database_url=dbos_system_database_url,
        generation_concurrency=generation_concurrency,
        scoring_concurrency=scoring_concurrency,
        database_url_env=DATABASE_URL_ENV,
        dbos_system_database_url_env=DBOS_SYSTEM_DATABASE_URL_ENV,
        database_url_error_suffix="for this Postgres-only DBOS harness",
    )


def _create_eval_schema(database_url: str) -> None:
    shared_dbos.create_schema(
        database_url,
        statements=(
            EXPERIMENTS_TABLE_SQL,
            PREDICTIONS_TABLE_SQL,
            shared_batch.operation_table_sql(),
            shared_batch.operation_item_table_sql(),
            *PREDICTION_MIGRATION_SQL,
            *PREDICTION_CONSTRAINT_MIGRATION_SQL,
            *PREDICTION_INDEX_SQL,
            *shared_batch.operation_index_sql(),
            *shared_batch.operation_item_index_sql(),
        ),
    )


def _queue_names_for_selection(
    selection: shared_dbos.QueueSelection, *, experiment_name: str
) -> tuple[str, ...]:
    return shared_dbos.queue_names_for_selection(
        selection,
        experiment_name=experiment_name,
        queue_config=QUEUE_NAME_CONFIG,
    )


def _configure_dbos_runtime(
    config: shared_dbos.EvalDbosConfig,
    *,
    experiment_name: str,
    queue: shared_dbos.QueueSelection | None = None,
    consume_queues: bool = True,
) -> None:
    shared_dbos.configure_dbos_runtime(
        config,
        app_name=DBOS_APP_NAME,
        experiment_name=experiment_name,
        queue=queue,
        queue_config=QUEUE_NAME_CONFIG,
        consume_queues=consume_queues,
        operator_log=_operator_log,
    )


def build_humaneval_samples_from_rows(
    rows: Sequence[HumanEvalRow],
    *,
    seed: int,
    sample_count: int,
) -> list[HumanEvalSample]:
    sampled_tasks = (
        shared_human_eval_sampling.sample_human_eval_tasks_from_rows(
            rows, seed=seed, sample_count=sample_count
        )
    )
    return [
        HumanEvalSample(
            task_id=sample.task.task_id,
            sample_index=sample.sample_index,
            prompt=sample.task.prompt,
            canonical_solution=sample.task.canonical_solution,
            ground_truth_code=(
                sample.task.ground_truth_code_without_comments
                or sample.task.ground_truth_code
            ),
            test=sample.task.test,
            entry_point=sample.task.entry_point,
        )
        for sample in sampled_tasks
    ]


def build_humaneval_samples(
    *,
    seed: int,
    sample_count: int,
) -> list[HumanEvalSample]:
    config = experiment_config()
    return build_humaneval_samples_from_rows(
        shared_human_eval_sampling.load_human_eval_rows(
            dataset_name=config.dataset_name,
            dataset_split=config.dataset_split,
        ),
        seed=seed,
        sample_count=sample_count,
    )


def build_generation_lm(
    job: PredictionJob,
    *,
    event_buffer: LmEventBuffer,
    client: Any = None,
) -> dspy.BaseLM:
    return shared_dspy_runner.build_logged_lm(
        model=job.model,
        reasoning=job.reasoning,
        temperature=job.temperature,
        event_buffer=event_buffer,
        max_completion_tokens=experiment_config().default_max_completion_tokens,
        client=client,
    )


def generate_code_for_job(
    job: PredictionJob,
    *,
    client: Any = None,
) -> GenerationResult:
    event_buffer = LmEventBuffer()
    lm = build_generation_lm(
        job,
        event_buffer=event_buffer,
        client=client,
    )
    raw_generation = shared_dspy_runner.run_predictor(
        signature=solve_signature(),
        input_kwargs={"prompt": job.prompt},
        output_field="code",
        lm=lm,
        event_buffer=event_buffer,
    )
    result = shared_dspy_runner.predictor_run_result(
        raw_generation, event_buffer
    )
    return GenerationResult(
        prediction_id=job.prediction_id,
        raw_generation=result.text,
        response_metadata=result.response_metadata,
        usage_metadata=result.usage_metadata,
        provider_cost=result.provider_cost,
    )


def _enqueue_generation_jobs(
    database_url: str,
    jobs: Sequence[PredictionJob],
    *,
    score_timeout: float,
    retry_token: str | None = None,
) -> shared_dbos.EnqueueWorkflowsResult:
    return shared_dbos.enqueue_generation_workflows(
        database_url,
        jobs,
        queue_config=QUEUE_NAME_CONFIG,
        workflow=generate_prediction_workflow,
        score_timeout=score_timeout,
        retry_token=retry_token,
    )


def _enqueue_score_job(
    database_url: str,
    prediction_id: str,
    *,
    experiment_name: str,
    timeout: float,
    retry_token: str | None = None,
) -> None:
    shared_dbos.enqueue_score_workflow(
        database_url,
        prediction_id,
        experiment_name=experiment_name,
        queue_config=QUEUE_NAME_CONFIG,
        workflow=score_prediction_workflow,
        timeout=timeout,
        retry_token=retry_token,
    )


def _enqueue_score_jobs(
    database_url: str,
    prediction_ids: Sequence[str],
    *,
    experiment_name: str,
    timeout: float,
    retry_token: str | None = None,
) -> shared_dbos.EnqueueWorkflowsResult:
    return shared_dbos.enqueue_score_workflows(
        database_url,
        prediction_ids,
        experiment_name=experiment_name,
        queue_config=QUEUE_NAME_CONFIG,
        workflow=score_prediction_workflow,
        timeout=timeout,
        retry_token=retry_token,
    )


def _start_worker_monitor(
    config: shared_worker_monitor.WorkerMonitorConfig,
    stop_event: threading.Event,
    halt_event: threading.Event,
) -> threading.Thread:
    return shared_worker_monitor.start_worker_monitor(
        config,
        stop_event,
        halt_event,
        operator_log=_operator_log,
        emit_worker_detail_log=_emit_worker_detail_log,
    )


class DirectExperiment:
    prediction_table = PREDICTION_TABLE_NAME

    def create_schema(self, database_url: str) -> None:
        _create_eval_schema(database_url)

    def configure_runtime(
        self,
        config: shared_dbos.EvalDbosConfig,
        experiment_name: str,
        *,
        queue: shared_dbos.QueueSelection | None = None,
        consume_queues: bool = True,
    ) -> None:
        _configure_dbos_runtime(
            config,
            experiment_name=experiment_name,
            queue=queue,
            consume_queues=consume_queues,
        )

    def build_repair_plan(
        self,
        database_url: str,
        *,
        dbos_system_database_url: str,
        experiment_name: str,
    ) -> shared_eval_repair.RepairPlan:
        return _build_repair_plan(
            database_url,
            dbos_system_database_url=dbos_system_database_url,
            experiment_name=experiment_name,
        )

    def configure_pooled_worker_runtime(
        self,
        config: shared_dbos.EvalDbosConfig,
        *,
        experiment_name: str,
        queue: shared_dbos.QueueSelection,
        raw_db_pool_max_size: str,
    ) -> shared_dbos.DbPoolConfig:
        return shared_dbos.configure_worker_db_connection_pools(
            config,
            queue=queue,
            raw_max_size=raw_db_pool_max_size,
        )

    def queue_names_for_selection(
        self,
        selection: shared_dbos.QueueSelection,
        *,
        experiment_name: str,
    ) -> tuple[str, ...]:
        return _queue_names_for_selection(
            selection, experiment_name=experiment_name
        )

    def resolve_worker_log_path(
        self,
        *,
        experiment_name: str,
        queue: shared_dbos.QueueSelection,
        log_file: Path | None,
    ) -> Path:
        return _resolve_worker_log_path(
            experiment_name=experiment_name, queue=queue, log_file=log_file
        )

    def configure_worker_file_logging(self, log_file: Path) -> logging.Logger:
        return _configure_worker_file_logging(log_file)

    def start_worker_monitor(
        self,
        monitor_config: shared_worker_monitor.WorkerMonitorConfig,
        stop_event: threading.Event,
        halt_event: threading.Event,
    ) -> threading.Thread:
        return _start_worker_monitor(monitor_config, stop_event, halt_event)


_BACKEND = DirectExperiment()


def common_config(
    *,
    database_url: str | None,
    dbos_system_database_url: str | None,
    generation_concurrency: int,
    scoring_concurrency: int,
    env_file: Path | None = None,
) -> shared_dbos.EvalDbosConfig:
    load_optional_env_file(env_file)
    return _build_eval_dbos_config(
        database_url=database_url,
        dbos_system_database_url=dbos_system_database_url,
        generation_concurrency=generation_concurrency,
        scoring_concurrency=scoring_concurrency,
    )


@_APP.command()
def init_db(
    database_url: Annotated[
        str | None,
        typer.Option(
            "--database-url",
            help=f"Postgres URL; defaults to {DATABASE_URL_ENV}.",
        ),
    ] = None,
    env_file: Annotated[Path | None, typer.Option()] = None,
) -> None:
    config = common_config(
        database_url=database_url,
        dbos_system_database_url=None,
        generation_concurrency=DEFAULT_GENERATION_CONCURRENCY,
        scoring_concurrency=DEFAULT_SCORING_CONCURRENCY,
        env_file=env_file,
    )
    _create_eval_schema(config.database_url)
    _operator_log("initialized dr-dspy eval tables", style="green")


@_APP.command()
def submit(
    experiment_name: Annotated[
        str,
        typer.Option(
            "--experiment-name",
            help="Human-readable experiment key.",
        ),
    ],
    sample_count: Annotated[
        int | None, typer.Option("--sample-count", min=1)
    ] = None,
    seed: Annotated[int | None, typer.Option("--seed")] = None,
    temperatures: Annotated[
        str | None,
        typer.Option(
            "--temperatures",
            help="Comma-separated temperature values.",
        ),
    ] = None,
    repetitions: Annotated[
        int | None, typer.Option("--repetitions", min=1)
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Plan jobs without writing or enqueueing.",
        ),
    ] = False,
    database_url: Annotated[
        str | None,
        typer.Option(
            "--database-url",
            help=f"Postgres URL; defaults to {DATABASE_URL_ENV}.",
        ),
    ] = None,
    dbos_system_database_url: Annotated[
        str | None,
        typer.Option(
            "--dbos-system-database-url",
            help=(
                "DBOS system database URL; defaults to "
                f"{DBOS_SYSTEM_DATABASE_URL_ENV} or DATABASE_URL."
            ),
        ),
    ] = None,
    generation_concurrency: Annotated[
        int, typer.Option("--generation-concurrency")
    ] = DEFAULT_GENERATION_CONCURRENCY,
    scoring_concurrency: Annotated[
        int, typer.Option("--scoring-concurrency")
    ] = DEFAULT_SCORING_CONCURRENCY,
    score_timeout: Annotated[
        float | None, typer.Option("--score-timeout", min=0.1)
    ] = None,
    batch_size: Annotated[
        int, typer.Option("--batch-size", min=1)
    ] = DEFAULT_SUBMIT_BATCH_SIZE,
    env_file: Annotated[Path | None, typer.Option()] = None,
) -> None:
    experiment = experiment_config()
    sample_count = sample_count or experiment.default_sample_count
    seed = seed if seed is not None else experiment.default_seed
    temperatures = temperatures or ",".join(
        str(value) for value in experiment.default_temperatures
    )
    repetitions = repetitions or experiment.default_repetitions
    score_timeout = (
        score_timeout
        if score_timeout is not None
        else experiment.default_subprocess_timeout
    )
    config = common_config(
        database_url=database_url,
        dbos_system_database_url=dbos_system_database_url,
        generation_concurrency=generation_concurrency,
        scoring_concurrency=scoring_concurrency,
        env_file=env_file,
    )
    model_configs = default_model_configs()
    parsed_temperatures = parse_temperatures(temperatures)
    samples = build_humaneval_samples(seed=seed, sample_count=sample_count)
    submission_id = uuid.uuid4().hex
    submit_spec = build_submit_spec(
        experiment_name=experiment_name,
        model_configs=model_configs,
        seed=seed,
        sample_count=len(samples),
        temperatures=parsed_temperatures,
        repetitions=repetitions,
        score_timeout=score_timeout,
    )
    total_jobs = submit_spec.total_jobs()
    operation_key = shared_batch.operation_key(
        submit_spec.model_dump(mode="json")
    )
    _operator_log(
        f"planned {total_jobs} jobs: samples={len(samples)}, "
        f"models={len(model_configs)}, "
        f"temperatures={len(parsed_temperatures)}, "
        f"repetitions={repetitions}",
        style="cyan",
    )
    if dry_run:
        _operator_log(
            "dry run only; no rows written and no workflows enqueued",
            style="yellow",
        )
        return

    resolved_log_file = _resolve_submit_log_path(
        experiment_name=experiment_name
    )
    metadata = {
        "submission_id": submission_id,
        "operation_key": operation_key,
        "batch_size": batch_size,
        "models": [model.model_dump(mode="json") for model in model_configs],
        "temperatures": parsed_temperatures,
        "repetitions": repetitions,
        "score_timeout": score_timeout,
    }
    _create_eval_schema(config.database_url)
    upsert_experiment(
        config.database_url,
        experiment_name=experiment_name,
        seed=seed,
        sample_count=len(samples),
        metadata=metadata,
    )
    progress = shared_batch.prepare_operation(
        config.database_url,
        operation_kind=shared_batch.BatchOperationKind.SUBMIT,
        operation_key=operation_key,
        experiment_name=experiment_name,
        script_kind=experiment.script_kind,
        spec=submit_spec.model_dump(mode="json"),
        metadata=metadata,
        total_items=total_jobs,
        log_file=resolved_log_file,
    )
    shared_batch.record_operation_items(
        config.database_url,
        operation_kind=shared_batch.BatchOperationKind.SUBMIT,
        operation_key=operation_key,
        item_kind=shared_batch.BatchOperationItemKind.SAMPLE,
        payloads=[sample.model_dump(mode="json") for sample in samples],
    )
    active_log_file = Path(progress.log_file)
    _configure_submit_file_logging(active_log_file)
    _emit_submit_log(
        "submit_planned",
        {
            "operation_key": operation_key,
            "submission_id": progress.metadata["submission_id"],
            "experiment_name": experiment_name,
            "total_jobs": total_jobs,
            "metadata": metadata,
        },
    )
    _configure_dbos_runtime(
        config, experiment_name=experiment_name, consume_queues=False
    )
    launched = shared_batch.ensure_operation_workflow(
        workflow_id=progress.workflow_id,
        workflow=submit_dispatcher_workflow,
        database_url=config.database_url,
        operation_key=operation_key,
    )
    _emit_submit_log(
        "submit_dispatcher_enqueued",
        {
            "operation_key": operation_key,
            "workflow_id": progress.workflow_id,
            "launched": launched,
            "log_file": str(active_log_file),
        },
    )
    _operator_log(f"submit detail log: {active_log_file}", style="cyan")
    final_progress = shared_batch.tail_operation_progress(
        database_url=config.database_url,
        operation_kind=shared_batch.BatchOperationKind.SUBMIT,
        operation_key=operation_key,
        prediction_table=PREDICTION_TABLE_NAME,
        experiment_name=experiment_name,
        operator_log=_operator_log,
    )
    if final_progress.status is shared_batch.BatchOperationStatus.FAILED:
        raise typer.Exit(code=1)


@_APP.command("enqueue-scores")
def enqueue_scores_command(
    experiment_name: Annotated[
        str,
        typer.Option("--experiment-name", help="Experiment to score."),
    ],
    timeout: Annotated[
        float | None, typer.Option("--timeout", min=0.1)
    ] = None,
    database_url: Annotated[
        str | None,
        typer.Option(
            "--database-url",
            help=f"Postgres URL; defaults to {DATABASE_URL_ENV}.",
        ),
    ] = None,
    dbos_system_database_url: Annotated[
        str | None,
        typer.Option(
            "--dbos-system-database-url",
            help=(
                "DBOS system database URL; defaults to "
                f"{DBOS_SYSTEM_DATABASE_URL_ENV} or DATABASE_URL."
            ),
        ),
    ] = None,
    generation_concurrency: Annotated[
        int, typer.Option("--generation-concurrency")
    ] = DEFAULT_GENERATION_CONCURRENCY,
    scoring_concurrency: Annotated[
        int, typer.Option("--scoring-concurrency")
    ] = DEFAULT_SCORING_CONCURRENCY,
    batch_size: Annotated[
        int, typer.Option("--batch-size", min=1)
    ] = DEFAULT_OPERATION_BATCH_SIZE,
    operation_key: Annotated[str | None, typer.Option()] = None,
    env_file: Annotated[Path | None, typer.Option()] = None,
) -> None:
    timeout = (
        timeout
        if timeout is not None
        else experiment_config().default_subprocess_timeout
    )
    config = common_config(
        database_url=database_url,
        dbos_system_database_url=dbos_system_database_url,
        generation_concurrency=generation_concurrency,
        scoring_concurrency=scoring_concurrency,
        env_file=env_file,
    )
    _create_eval_schema(config.database_url)
    resolved_operation_key = operation_key or shared_batch.new_operation_key()
    operation_kind = shared_batch.BatchOperationKind.ENQUEUE_SCORES
    spec = {
        "experiment_name": experiment_name,
        "timeout": timeout,
        "batch_size": batch_size,
    }
    log_file = _resolve_operation_log_path(
        experiment_name=experiment_name,
        operation_kind=operation_kind,
    )
    progress = shared_batch.prepare_operation(
        config.database_url,
        operation_kind=operation_kind,
        operation_key=resolved_operation_key,
        experiment_name=experiment_name,
        script_kind=experiment_config().script_kind,
        spec=spec,
        metadata={"batch_size": batch_size},
        total_items=0,
        log_file=log_file,
    )
    _configure_operation_file_logging(Path(progress.log_file))
    _configure_dbos_runtime(
        config, experiment_name=experiment_name, consume_queues=False
    )
    launched = shared_batch.ensure_operation_workflow(
        workflow_id=progress.workflow_id,
        workflow=enqueue_scores_dispatcher_workflow,
        database_url=config.database_url,
        operation_key=resolved_operation_key,
    )
    _emit_operation_log(
        "enqueue_scores_dispatcher_enqueued",
        {
            "operation_key": resolved_operation_key,
            "workflow_id": progress.workflow_id,
            "launched": launched,
            "log_file": progress.log_file,
        },
    )
    final_progress = shared_batch.tail_operation_progress(
        database_url=config.database_url,
        operation_kind=operation_kind,
        operation_key=resolved_operation_key,
        prediction_table=PREDICTION_TABLE_NAME,
        experiment_name=experiment_name,
        operator_log=_operator_log,
    )
    if final_progress.status is shared_batch.BatchOperationStatus.FAILED:
        raise typer.Exit(code=1)


@_APP.command("repair")
def repair_command(
    experiment_name: Annotated[
        str,
        typer.Option("--experiment-name", help="Experiment to repair."),
    ],
    apply: Annotated[
        bool,
        typer.Option(
            "--apply",
            help=(
                "Apply repair actions. Without this flag, only reports "
                "repairable rows."
            ),
        ),
    ] = False,
    score_timeout: Annotated[
        float | None, typer.Option("--score-timeout", min=0.1)
    ] = None,
    database_url: Annotated[
        str | None,
        typer.Option(
            "--database-url",
            help=f"Postgres URL; defaults to {DATABASE_URL_ENV}.",
        ),
    ] = None,
    dbos_system_database_url: Annotated[
        str | None,
        typer.Option(
            "--dbos-system-database-url",
            help=(
                "DBOS system database URL; defaults to "
                f"{DBOS_SYSTEM_DATABASE_URL_ENV} or DATABASE_URL."
            ),
        ),
    ] = None,
    generation_concurrency: Annotated[
        int, typer.Option("--generation-concurrency")
    ] = DEFAULT_GENERATION_CONCURRENCY,
    scoring_concurrency: Annotated[
        int, typer.Option("--scoring-concurrency")
    ] = DEFAULT_SCORING_CONCURRENCY,
    batch_size: Annotated[
        int, typer.Option("--batch-size", min=1)
    ] = DEFAULT_OPERATION_BATCH_SIZE,
    operation_key: Annotated[
        str | None,
        typer.Option(
            help=(
                "Existing repair operation key to resume or retry; omitted "
                "runs use a fresh operation key."
            )
        ),
    ] = None,
    env_file: Annotated[Path | None, typer.Option()] = None,
) -> None:
    score_timeout = (
        score_timeout
        if score_timeout is not None
        else experiment_config().default_subprocess_timeout
    )
    config = common_config(
        database_url=database_url,
        dbos_system_database_url=dbos_system_database_url,
        generation_concurrency=generation_concurrency,
        scoring_concurrency=scoring_concurrency,
        env_file=env_file,
    )
    if not apply:
        shared_flow.run_repair_command(
            _BACKEND,
            config=config,
            experiment_name=experiment_name,
            score_timeout=score_timeout,
        )
        return

    _create_eval_schema(config.database_url)
    operation_kind = shared_batch.BatchOperationKind.REPAIR
    spec = {
        "experiment_name": experiment_name,
        "score_timeout": score_timeout,
        "batch_size": batch_size,
        "database_url": config.database_url,
        "dbos_system_database_url": config.dbos_system_database_url,
        "generation_concurrency": config.generation_concurrency,
        "scoring_concurrency": config.scoring_concurrency,
    }
    resolved_operation_key = operation_key or shared_batch.new_operation_key()
    log_file = _resolve_operation_log_path(
        experiment_name=experiment_name,
        operation_kind=operation_kind,
    )
    progress = shared_batch.prepare_operation(
        config.database_url,
        operation_kind=operation_kind,
        operation_key=resolved_operation_key,
        experiment_name=experiment_name,
        script_kind=experiment_config().script_kind,
        spec=spec,
        metadata={"batch_size": batch_size},
        total_items=0,
        log_file=log_file,
    )
    _configure_operation_file_logging(Path(progress.log_file))
    _configure_dbos_runtime(
        config, experiment_name=experiment_name, consume_queues=False
    )
    launched = shared_batch.ensure_operation_workflow(
        workflow_id=progress.workflow_id,
        workflow=repair_dispatcher_workflow,
        database_url=config.database_url,
        operation_key=resolved_operation_key,
    )
    _emit_operation_log(
        "repair_dispatcher_enqueued",
        {
            "operation_key": resolved_operation_key,
            "workflow_id": progress.workflow_id,
            "launched": launched,
            "log_file": progress.log_file,
        },
    )
    final_progress = shared_batch.tail_operation_progress(
        database_url=config.database_url,
        operation_kind=operation_kind,
        operation_key=resolved_operation_key,
        prediction_table=PREDICTION_TABLE_NAME,
        experiment_name=experiment_name,
        operator_log=_operator_log,
    )
    if final_progress.status is shared_batch.BatchOperationStatus.FAILED:
        raise typer.Exit(code=1)


@_APP.command()
def worker(
    experiment_name: Annotated[
        str,
        typer.Option(
            "--experiment-name",
            help=(
                "Experiment this worker is serving; used for monitor "
                "counts and detailed log directory naming."
            ),
        ),
    ],
    queue: Annotated[
        shared_dbos.QueueSelection,
        typer.Option("--queue", help="Queue set this worker should consume."),
    ] = shared_dbos.QueueSelection.BOTH,
    database_url: Annotated[
        str | None,
        typer.Option(
            "--database-url",
            help=f"Postgres URL; defaults to {DATABASE_URL_ENV}.",
        ),
    ] = None,
    dbos_system_database_url: Annotated[
        str | None,
        typer.Option(
            "--dbos-system-database-url",
            help=(
                "DBOS system database URL; defaults to "
                f"{DBOS_SYSTEM_DATABASE_URL_ENV} or DATABASE_URL."
            ),
        ),
    ] = None,
    generation_concurrency: Annotated[
        int, typer.Option("--generation-concurrency")
    ] = DEFAULT_GENERATION_CONCURRENCY,
    scoring_concurrency: Annotated[
        int, typer.Option("--scoring-concurrency")
    ] = DEFAULT_SCORING_CONCURRENCY,
    open_file_limit: Annotated[
        str,
        typer.Option(
            "--open-file-limit",
            help=(
                "Requested worker soft open-file limit, or 'auto'. The "
                "process can raise this only up to the OS hard limit."
            ),
        ),
    ] = DEFAULT_WORKER_OPEN_FILE_LIMIT,
    log_file: Annotated[
        Path | None,
        typer.Option(
            "--log-file",
            help="Override the detailed worker log file path.",
        ),
    ] = None,
    monitor: Annotated[
        bool,
        typer.Option(
            "--monitor/--no-monitor",
            help="Print compact queue activity updates to stdout.",
        ),
    ] = True,
    monitor_interval: Annotated[
        float,
        typer.Option("--monitor-interval", min=0.5),
    ] = DEFAULT_WORKER_MONITOR_INTERVAL_SECONDS,
    monitor_summary_interval: Annotated[
        float,
        typer.Option("--monitor-summary-interval", min=1.0),
    ] = DEFAULT_WORKER_MONITOR_SUMMARY_INTERVAL_SECONDS,
    db_pool_max_size: Annotated[
        str,
        typer.Option(
            "--db-pool-max-size",
            help=(
                "Worker Postgres connection pool max size. Use 'auto' to "
                "match active queue capacity plus a small margin."
            ),
        ),
    ] = shared_dbos.DB_POOL_AUTO,
    env_file: Annotated[Path | None, typer.Option()] = None,
) -> None:
    config = common_config(
        database_url=database_url,
        dbos_system_database_url=dbos_system_database_url,
        generation_concurrency=generation_concurrency,
        scoring_concurrency=scoring_concurrency,
        env_file=env_file,
    )
    shared_flow.run_worker_command(
        _BACKEND,
        config=config,
        experiment_name=experiment_name,
        queue=queue,
        open_file_limit=open_file_limit,
        log_file=log_file,
        monitor=monitor,
        monitor_interval=monitor_interval,
        monitor_summary_interval=monitor_summary_interval,
        db_pool_max_size=db_pool_max_size,
    )
