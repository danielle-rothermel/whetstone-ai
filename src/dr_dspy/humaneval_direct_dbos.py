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
from rich.console import Console, Group
from rich.table import Table

import dspy
from dr_dspy import dbos_runtime as shared_dbos
from dr_dspy import dspy_runner as shared_dspy_runner
from dr_dspy import eval_logging as shared_eval_logging
from dr_dspy import eval_repair as shared_eval_repair
from dr_dspy import eval_reporting as shared_eval_reporting
from dr_dspy import human_eval_sampling as shared_human_eval_sampling
from dr_dspy import humaneval_dbos_flow as shared_flow
from dr_dspy import worker_monitor as shared_worker_monitor
from dr_dspy.code_eval import extract_dspy_code
from dr_dspy.human_eval import HumanEvalTask
from dr_dspy.lm_utils import (
    LmEventBuffer,
    ModelConfig,
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
DEFAULT_SCORE_ENQUEUE_LIMIT = 1000
DEFAULT_REPAIR_GENERATION_LIMIT = 1000
DEFAULT_REPAIR_SCORING_LIMIT = 1000
DEFAULT_WORKER_OPEN_FILE_LIMIT = 8192
DEFAULT_WORKER_MONITOR_INTERVAL_SECONDS = 5.0
DEFAULT_WORKER_MONITOR_SUMMARY_INTERVAL_SECONDS = 5.0
DB_POOL_AUTO = shared_dbos.DB_POOL_AUTO
DEFAULT_COST_SIGNIFICANT_DIGITS = 6
PRICE_PER_THOUSAND_SAMPLE_MULTIPLIER = 1000.0
ANALYSIS_TOTAL_LABEL = "Total"
TABLE_ROW_STYLES = ("", "on grey7")
TABLE_TOTAL_ROW_STYLE = "bold black on green3"
MAX_TRACE_SIZE = 10_000
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKER_LOG_ROOT = PACKAGE_ROOT / "logs"
DETAILED_WORKER_LOGGER_NAME = "dr_dspy.humaneval_eval_only_worker"
CONSOLE = Console(soft_wrap=True)
OPERATOR_TIMESTAMP_FORMAT = "%H:%M:%S"
PREDICTION_TABLE_NAME = "dr_dspy_eval_predictions"
PREDICTION_ID_DIGEST_LENGTH = 32
REPAIR_DIMENSION_COLUMNS = ("model", "temperature")
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
    raw_code             TEXT,
    response_metadata    JSONB       NOT NULL DEFAULT '{}'::jsonb,
    usage_metadata       JSONB       NOT NULL DEFAULT '{}'::jsonb,
    provider_cost        DOUBLE PRECISION,
    scoring_status       TEXT        NOT NULL DEFAULT 'pending',
    score                DOUBLE PRECISION,
    scoring_error        TEXT,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    generated_at         TIMESTAMPTZ,
    scored_at            TIMESTAMPTZ,
    UNIQUE (
        experiment_name,
        task_id,
        model,
        temperature,
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
    "NOT NULL DEFAULT '[]'::jsonb",
    "ALTER TABLE dr_dspy_eval_predictions "
    "ADD COLUMN IF NOT EXISTS best_compression_ratio DOUBLE PRECISION",
    "ALTER TABLE dr_dspy_eval_predictions "
    "ADD COLUMN IF NOT EXISTS best_compression_percent_reduction "
    "DOUBLE PRECISION",
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


class AnalysisRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: StrictStr
    temperature: float
    task_id: StrictStr
    repetition_seed: StrictInt
    score: float
    provider_cost: float | None
    raw_compile_ok: bool | None = None
    extracted_compile_ok: bool | None = None
    best_compression_ratio: float | None = None
    best_compression_percent_reduction: float | None = None


class AnalysisSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: StrictStr
    temperature: float
    sample_count: StrictInt
    scored_count: StrictInt
    total_price: float | None
    avg_price_per_sample: float | None
    price_variance: float | None
    avg_performance: float
    performance_variance: float | None
    avg_repetition_variance: float | None
    raw_compile_pass_count: StrictInt
    extracted_compile_pass_count: StrictInt
    extraction_lift: StrictInt
    avg_best_compression_ratio: float | None = None
    avg_best_compression_percent_reduction: float | None = None


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

def resolve_worker_log_path(
    *,
    experiment_name: str,
    queue: QueueSelection,
    log_file: Path | None,
) -> Path:
    return shared_eval_logging.resolve_worker_log_path(
        log_root=DEFAULT_WORKER_LOG_ROOT,
        experiment_name=experiment_name,
        queue=queue,
        log_file=log_file,
        hash_length=EXPERIMENT_QUEUE_HASH_LENGTH,
    )


def configure_worker_file_logging(log_file: Path) -> logging.Logger:
    return shared_eval_logging.configure_worker_file_logging(
        log_file, logger_name=DETAILED_WORKER_LOGGER_NAME
    )


def emit_worker_detail_log(event: str, payload: Mapping[str, Any]) -> None:
    shared_eval_logging.emit_worker_detail_log(
        event, payload, logger_name=DETAILED_WORKER_LOGGER_NAME
    )


def prediction_context_from_job(
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
        },
    )


def emit_prediction_log_event(
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
    return shared_flow.run_generation_workflow(
        database_url=database_url,
        prediction_id=prediction_id,
        experiment_name=experiment_name,
        score_timeout=score_timeout,
        generate_prediction=generate_prediction_step,
        record_generation_success=record_generation_success_step,
        record_generation_error=record_generation_error_step,
        enqueue_score=(
            lambda db_url, pred_id, exp_name, timeout: enqueue_score_job(
                db_url,
                pred_id,
                experiment_name=exp_name,
                timeout=timeout,
            )
        ),
        mark_scoring_queued=mark_scoring_queued_step,
    )


@DBOS.workflow(name="humaneval_eval_score_prediction_v0")
def score_prediction_workflow(
    database_url: str, prediction_id: str, timeout: float
) -> str:
    return shared_flow.run_score_workflow(
        database_url=database_url,
        prediction_id=prediction_id,
        timeout=timeout,
        score_prediction=score_prediction_step,
        record_score_success=record_score_success_step,
        record_score_error=record_score_error_step,
    )


def stable_prediction_id(
    *,
    experiment_name: str,
    task_id: str,
    model: str,
    temperature: float,
    repetition_seed: int,
) -> str:
    return shared_flow.stable_prediction_id_from_dimensions(
        experiment_name=experiment_name,
        task_id=task_id,
        dimensions={"model": model, "temperature": temperature},
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


def build_prediction_jobs(
    *,
    experiment_name: str,
    submission_id: str,
    samples: Sequence[HumanEvalSample],
    model_configs: Sequence[ModelConfig],
    temperatures: Sequence[float],
    repetitions: int,
) -> list[PredictionJob]:
    jobs: list[PredictionJob] = []
    for sample in samples:
        for model_config in model_configs:
            for temperature in temperatures:
                for repetition_seed in range(repetitions):
                    jobs.append(
                        PredictionJob(
                            prediction_id=stable_prediction_id(
                                experiment_name=experiment_name,
                                task_id=sample.task_id,
                                model=model_config.model,
                                temperature=temperature,
                                repetition_seed=repetition_seed,
                            ),
                            experiment_name=experiment_name,
                            script_kind=experiment_config().script_kind,
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
    with connect_db(database_url) as conn:
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
    with connect_db(database_url) as conn:
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
    with connect_db(database_url) as conn:
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
    return prediction_context_from_job(
        fetch_prediction_job(database_url, prediction_id)
    )


def mark_generation_started(database_url: str, prediction_id: str) -> None:
    with connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE dr_dspy_eval_predictions
                SET
                    generation_status = 'started',
                    generation_error = NULL,
                    updated_at = now()
                WHERE prediction_id = %s
                """,
                (prediction_id,),
            )


def record_generation_success(
    database_url: str, result: GenerationResult
) -> None:
    with connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE dr_dspy_eval_predictions
                SET
                    generation_status = 'generated',
                    generation_error = NULL,
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
    database_url: str, prediction_id: str, error: str
) -> None:
    with connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE dr_dspy_eval_predictions
                SET
                    generation_status = 'generation_error',
                    generation_error = %s,
                    raw_generation = NULL,
                    raw_code = NULL,
                    updated_at = now()
                WHERE prediction_id = %s
                """,
                (error, prediction_id),
            )


def fetch_scoring_target(
    database_url: str, prediction_id: str
) -> ScoringTarget:
    with connect_db(database_url) as conn:
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


def fetch_pending_scoring_prediction_ids(
    database_url: str,
    *,
    experiment_name: str,
    limit: int,
) -> list[str]:
    return shared_eval_repair.fetch_pending_scoring_prediction_ids(
        database_url,
        prediction_table=PREDICTION_TABLE_NAME,
        experiment_name=experiment_name,
        order_columns=REPAIR_ORDER_COLUMNS,
        limit=limit,
    )


def fetch_score_error_prediction_ids(
    database_url: str,
    *,
    experiment_name: str,
    limit: int,
) -> list[str]:
    return shared_eval_repair.fetch_score_error_prediction_ids(
        database_url,
        prediction_table=PREDICTION_TABLE_NAME,
        experiment_name=experiment_name,
        order_columns=REPAIR_ORDER_COLUMNS,
        limit=limit,
    )


def fetch_generation_error_prediction_jobs(
    database_url: str,
    *,
    experiment_name: str,
    limit: int,
) -> list[PredictionJob]:
    prediction_ids = shared_eval_repair.fetch_generation_error_prediction_ids(
        database_url,
        prediction_table=PREDICTION_TABLE_NAME,
        experiment_name=experiment_name,
        order_columns=REPAIR_ORDER_COLUMNS,
        limit=limit,
    )
    return [
        fetch_prediction_job(database_url, prediction_id)
        for prediction_id in prediction_ids
    ]


def reset_generation_errors_for_retry(
    database_url: str,
    *,
    prediction_ids: Sequence[str],
) -> int:
    if not prediction_ids:
        return 0
    with connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE dr_dspy_eval_predictions
                SET
                    generation_status = 'pending',
                    generation_error = NULL,
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
                    scoring_status = 'pending',
                    scoring_error = NULL,
                    score = NULL,
                    evaluation_function_names = '[]'::jsonb,
                    evaluation_total_cases = NULL,
                    evaluation_failure_count = NULL,
                    evaluation_status_counts = '{}'::jsonb,
                    compression_metrics = '[]'::jsonb,
                    best_compression_ratio = NULL,
                    best_compression_percent_reduction = NULL,
                    scored_at = NULL,
                    updated_at = now()
                WHERE
                    prediction_id = ANY(%s)
                    AND generation_status = 'generation_error'
                """,
                (list(prediction_ids),),
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
    with connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE dr_dspy_eval_predictions
                SET
                    scoring_status = 'started',
                    scoring_error = NULL,
                    updated_at = now()
                WHERE prediction_id = %s
                """,
                (prediction_id,),
            )


def record_score_success(database_url: str, result: ScoreResult) -> None:
    with connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE dr_dspy_eval_predictions
                SET
                    scoring_status = 'scored',
                    score = %s,
                    scoring_error = %s,
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
                        [
                            metric.model_dump(mode="json")
                            for metric in result.compression_metrics
                        ]
                    ),
                    result.best_compression_ratio,
                    result.best_compression_percent_reduction,
                    result.prediction_id,
                ),
            )


def record_score_error(
    database_url: str, prediction_id: str, error: str
) -> None:
    with connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE dr_dspy_eval_predictions
                SET
                    scoring_status = 'score_error',
                    scoring_error = %s,
                    updated_at = now()
                WHERE prediction_id = %s
                """,
                (error, prediction_id),
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
    return shared_flow.run_score_prediction_step(
        database_url=database_url,
        prediction_id=prediction_id,
        timeout=timeout,
        mark_scoring_started=mark_scoring_started,
        fetch_prediction_log_context=fetch_prediction_log_context,
        emit_prediction_log_event=emit_prediction_log_event,
        score_generated_prediction=(
            lambda db_url, pred_id, score_timeout: score_generated_code(
                fetch_scoring_target(db_url, pred_id),
                timeout=score_timeout,
            )
        ),
    )


@DBOS.step(name="humaneval_direct_record_score_success_step_v0")
def record_score_success_step(
    database_url: str, result: ScoreResult
) -> None:
    shared_flow.run_record_score_success_step(
        database_url=database_url,
        result=result,
        prediction_id=result.prediction_id,
        score=result.score,
        error=result.error,
        fetch_prediction_log_context=fetch_prediction_log_context,
        emit_prediction_log_event=emit_prediction_log_event,
        record_score_success=record_score_success,
    )


@DBOS.step(name="humaneval_direct_record_score_error_step_v0")
def record_score_error_step(
    database_url: str, prediction_id: str, error: str
) -> None:
    shared_flow.run_record_score_error_step(
        database_url=database_url,
        prediction_id=prediction_id,
        error=error,
        fetch_prediction_log_context=fetch_prediction_log_context,
        emit_prediction_log_event=emit_prediction_log_event,
        record_score_error=record_score_error,
    )


@DBOS.step(name="humaneval_direct_mark_scoring_queued_step_v0")
def mark_scoring_queued_step(database_url: str, prediction_id: str) -> None:
    shared_flow.run_mark_scoring_queued_step(
        database_url=database_url,
        prediction_id=prediction_id,
        fetch_prediction_log_context=fetch_prediction_log_context,
        emit_prediction_log_event=emit_prediction_log_event,
        mark_scoring_queued=mark_scoring_queued,
    )


@DBOS.step(
    name="humaneval_direct_generate_prediction_step_v0",
    retries_allowed=True,
    max_attempts=3,
    interval_seconds=2.0,
)
def generate_prediction_step(
    database_url: str, prediction_id: str
) -> GenerationResult:
    return shared_flow.run_generate_prediction_step(
        database_url=database_url,
        prediction_id=prediction_id,
        mark_generation_started=mark_generation_started,
        fetch_prediction_job=fetch_prediction_job,
        prediction_context_from_job=prediction_context_from_job,
        emit_prediction_log_event=emit_prediction_log_event,
        generate_code_for_job=generate_code_for_job,
    )


@DBOS.step(name="humaneval_direct_record_generation_success_step_v0")
def record_generation_success_step(
    database_url: str, result: GenerationResult
) -> None:
    shared_flow.run_record_generation_success_step(
        database_url=database_url,
        result=result,
        prediction_id=result.prediction_id,
        fetch_prediction_log_context=fetch_prediction_log_context,
        emit_prediction_log_event=emit_prediction_log_event,
        success_extra=lambda generation_result: {
            "provider_cost": generation_result.provider_cost,
            "usage_metadata": generation_result.usage_metadata,
        },
        record_generation_success=record_generation_success,
    )


@DBOS.step(name="humaneval_direct_record_generation_error_step_v0")
def record_generation_error_step(
    database_url: str, prediction_id: str, error: str
) -> None:
    shared_flow.run_record_generation_error_step(
        database_url=database_url,
        prediction_id=prediction_id,
        error=error,
        fetch_prediction_log_context=fetch_prediction_log_context,
        emit_prediction_log_event=emit_prediction_log_event,
        record_generation_error=record_generation_error,
    )


def build_repair_plan(
    database_url: str,
    *,
    dbos_system_database_url: str,
    experiment_name: str,
    generation_limit: int,
    scoring_limit: int,
) -> shared_eval_repair.RepairPlan:
    return shared_eval_repair.build_repair_plan(
        database_url,
        dbos_system_database_url=dbos_system_database_url,
        prediction_table=PREDICTION_TABLE_NAME,
        experiment_name=experiment_name,
        dimension_columns=REPAIR_DIMENSION_COLUMNS,
        order_columns=REPAIR_ORDER_COLUMNS,
        generation_limit=generation_limit,
        scoring_limit=scoring_limit,
    )


def apply_repair(
    config: EvalDbosConfig,
    *,
    experiment_name: str,
    generation_limit: int,
    scoring_limit: int,
    score_timeout: float,
    repair_token: str | None = None,
) -> shared_eval_repair.RepairApplyResult:
    return shared_eval_repair.apply_repair(
        config,
        prediction_table=PREDICTION_TABLE_NAME,
        experiment_name=experiment_name,
        dimension_columns=REPAIR_DIMENSION_COLUMNS,
        order_columns=REPAIR_ORDER_COLUMNS,
        generation_limit=generation_limit,
        scoring_limit=scoring_limit,
        score_timeout=score_timeout,
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
        configure_runtime=lambda: configure_dbos_runtime(
            config,
            experiment_name=experiment_name,
            consume_queues=False,
        ),
        enqueue_generation_jobs=lambda jobs, token: enqueue_generation_jobs(
            config.database_url,
            jobs,
            score_timeout=score_timeout,
            retry_token=token,
        ),
        enqueue_score_jobs=lambda prediction_ids, timeout, token: (
            enqueue_score_jobs(
                config.database_url,
                prediction_ids,
                experiment_name=experiment_name,
                timeout=timeout,
                retry_token=token,
            )
        ),
        repair_token=repair_token,
    )


def fetch_analysis_records(
    database_url: str, *, experiment_name: str
) -> list[AnalysisRecord]:
    with connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    model,
                    temperature,
                    task_id,
                    repetition_seed,
                    score,
                    provider_cost,
                    raw_compile_ok,
                    extracted_compile_ok,
                    best_compression_ratio,
                    best_compression_percent_reduction
                FROM dr_dspy_eval_predictions
                WHERE
                    experiment_name = %s
                    AND scoring_status = 'scored'
                    AND score IS NOT NULL
                ORDER BY model, temperature, task_id, repetition_seed
                """,
                (experiment_name,),
            )
            rows = cur.fetchall()
    return [
        AnalysisRecord(
            model=row[0],
            temperature=row[1],
            task_id=row[2],
            repetition_seed=row[3],
            score=row[4],
            provider_cost=row[5],
            raw_compile_ok=row[6],
            extracted_compile_ok=row[7],
            best_compression_ratio=row[8],
            best_compression_percent_reduction=row[9],
        )
        for row in rows
    ]


def summarize_analysis_records(
    records: Sequence[AnalysisRecord],
) -> list[AnalysisSummary]:
    return shared_flow.summarize_analysis_records(
        records,
        group_key=lambda record: (record.model, record.temperature),
        model_label=lambda record: record.model,
        temperature=lambda record: record.temperature,
        task_id=lambda record: record.task_id,
        score=lambda record: record.score,
        provider_cost=lambda record: record.provider_cost,
        raw_compile_ok=lambda record: record.raw_compile_ok,
        extracted_compile_ok=lambda record: record.extracted_compile_ok,
        best_compression_ratio=lambda record: record.best_compression_ratio,
        best_compression_percent_reduction=(
            lambda record: record.best_compression_percent_reduction
        ),
        summary_factory=AnalysisSummary,
    )


def operator_log(
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


def analysis_markdown(
    *, experiment_name: str, summaries: Sequence[AnalysisSummary]
) -> str:
    return shared_eval_reporting.analysis_markdown(
        experiment_name=experiment_name, summaries=summaries
    )


def analysis_table(
    *, experiment_name: str, summaries: Sequence[AnalysisSummary]
) -> Group:
    return shared_eval_reporting.analysis_table(
        experiment_name=experiment_name, summaries=summaries
    )


def write_analysis_csv(
    summaries: Sequence[AnalysisSummary], *, csv_path: Path
) -> None:
    shared_eval_reporting.write_analysis_csv(
        summaries,
        csv_path=csv_path,
        fieldnames=list(AnalysisSummary.model_fields),
    )


def enqueue_scores_line(
    *,
    experiment_name: str,
    selected_count: int,
    limit: int,
    timeout: float,
) -> str:
    return shared_eval_reporting.enqueue_scores_line(
        experiment_name=experiment_name,
        selected_count=selected_count,
        limit=limit,
        timeout=timeout,
    )


def enqueue_scores_style(selected_count: int) -> str:
    return shared_eval_reporting.enqueue_scores_style(selected_count)


def repair_plan_line(
    *,
    experiment_name: str,
    plan: shared_eval_repair.RepairPlan,
    apply: bool,
) -> str:
    return shared_eval_reporting.repair_plan_line(
        experiment_name=experiment_name,
        gen_stranded=len(plan.stranded_generations),
        gen_errors=len(plan.generation_retry_prediction_ids),
        score_pending=len(plan.pending_scoring_prediction_ids),
        score_stranded=len(plan.stranded_scoring),
        score_errors=len(plan.scoring_retry_prediction_ids),
        apply=apply,
    )


def repair_apply_line(
    *, experiment_name: str, result: shared_eval_repair.RepairApplyResult
) -> str:
    return shared_eval_reporting.repair_apply_line(
        experiment_name=experiment_name,
        stranded_generations_marked=result.stranded_generations_marked,
        generation_retries_enqueued=result.generation_retries_enqueued,
        stranded_scoring_marked=result.stranded_scoring_marked,
        pending_scoring_enqueued=result.pending_scoring_enqueued,
        scoring_retries_enqueued=result.scoring_retries_enqueued,
        repair_token=result.repair_token,
    )


def repair_plan_style(
    plan: shared_eval_repair.RepairPlan, *, apply: bool
) -> str:
    return shared_eval_reporting.repair_plan_style(
        apply=apply,
        gen_stranded=len(plan.stranded_generations),
        gen_errors=len(plan.generation_retry_prediction_ids),
        score_pending=len(plan.pending_scoring_prediction_ids),
        score_stranded=len(plan.stranded_scoring),
        score_errors=len(plan.scoring_retry_prediction_ids),
    )


def fetch_status_counts(
    database_url: str, *, experiment_name: str | None
) -> list[dict[str, Any]]:
    return shared_eval_reporting.fetch_status_counts(
        database_url,
        prediction_table=PREDICTION_TABLE_NAME,
        dimension_columns=REPAIR_DIMENSION_COLUMNS,
        experiment_name=experiment_name,
    )


def status_counts_table(
    rows: Sequence[Mapping[str, Any]],
    *,
    experiment_name: str | None,
) -> Table:
    return shared_eval_reporting.status_counts_table(
        rows,
        title="Eval Status",
        dimensions=(
            shared_eval_reporting.StatusDimension(
                key="model", title="Model"
            ),
            shared_eval_reporting.StatusDimension(
                key="temperature", title="Temp", justify="right"
            ),
        ),
        experiment_name=experiment_name,
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

QueueSelection = shared_dbos.QueueSelection
EvalDbosConfig = shared_dbos.EvalDbosConfig
DbPoolConfig = shared_dbos.DbPoolConfig
OpenFileLimitResult = shared_dbos.OpenFileLimitResult
DB_POOLS = shared_dbos.DB_POOLS
WorkerQueueSnapshot = shared_worker_monitor.WorkerQueueSnapshot
WorkerMonitorConfig = shared_worker_monitor.WorkerMonitorConfig

open_file_limit_line = shared_dbos.open_file_limit_line
open_file_limit_style = shared_dbos.open_file_limit_style
close_db_connection_pools = shared_dbos.close_db_connection_pools
connect_db = shared_dbos.connect_db


def raise_open_file_limit(requested: int) -> OpenFileLimitResult:
    return shared_dbos.raise_open_file_limit(requested)


def build_eval_dbos_config(
    *,
    database_url: str | None,
    dbos_system_database_url: str | None,
    generation_concurrency: int,
    scoring_concurrency: int,
) -> EvalDbosConfig:
    return shared_dbos.build_eval_dbos_config(
        database_url=database_url,
        dbos_system_database_url=dbos_system_database_url,
        generation_concurrency=generation_concurrency,
        scoring_concurrency=scoring_concurrency,
        database_url_env=DATABASE_URL_ENV,
        dbos_system_database_url_env=DBOS_SYSTEM_DATABASE_URL_ENV,
        database_url_error_suffix="for this Postgres-only DBOS harness",
    )


def configure_worker_db_connection_pools(
    config: EvalDbosConfig,
    *,
    queue: QueueSelection,
    raw_max_size: str,
) -> DbPoolConfig:
    return shared_dbos.configure_worker_db_connection_pools(
        config, queue=queue, raw_max_size=raw_max_size
    )


def create_eval_schema(database_url: str) -> None:
    shared_dbos.create_schema(
        database_url,
        statements=(
            EXPERIMENTS_TABLE_SQL,
            PREDICTIONS_TABLE_SQL,
            *PREDICTION_MIGRATION_SQL,
            *PREDICTION_INDEX_SQL,
        ),
    )


def queue_names_for_selection(
    selection: QueueSelection, *, experiment_name: str
) -> tuple[str, ...]:
    return shared_dbos.queue_names_for_selection(
        selection,
        experiment_name=experiment_name,
        queue_config=QUEUE_NAME_CONFIG,
    )


def configure_dbos_runtime(
    config: EvalDbosConfig,
    *,
    experiment_name: str,
    queue: QueueSelection | None = None,
    consume_queues: bool = True,
) -> None:
    shared_dbos.configure_dbos_runtime(
        config,
        app_name=DBOS_APP_NAME,
        experiment_name=experiment_name,
        queue=queue,
        queue_config=QUEUE_NAME_CONFIG,
        consume_queues=consume_queues,
        operator_log=operator_log,
    )


def configure_pooled_worker_runtime(
    config: EvalDbosConfig,
    *,
    experiment_name: str,
    queue: QueueSelection,
    raw_db_pool_max_size: str,
) -> DbPoolConfig:
    pool_config = configure_worker_db_connection_pools(
        config,
        queue=queue,
        raw_max_size=raw_db_pool_max_size,
    )
    configure_dbos_runtime(
        config,
        experiment_name=experiment_name,
        queue=queue,
    )
    return pool_config


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
        max_trace_size=MAX_TRACE_SIZE,
        after_prediction=extract_dspy_code,
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


def enqueue_generation_jobs(
    database_url: str,
    jobs: Sequence[PredictionJob],
    *,
    score_timeout: float,
    retry_token: str | None = None,
) -> None:
    shared_dbos.enqueue_generation_workflows(
        database_url,
        jobs,
        queue_config=QUEUE_NAME_CONFIG,
        workflow=generate_prediction_workflow,
        score_timeout=score_timeout,
        retry_token=retry_token,
    )


def enqueue_score_job(
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


def enqueue_score_jobs(
    database_url: str,
    prediction_ids: Sequence[str],
    *,
    experiment_name: str,
    timeout: float,
    retry_token: str | None = None,
) -> None:
    shared_dbos.enqueue_score_workflows(
        database_url,
        prediction_ids,
        experiment_name=experiment_name,
        queue_config=QUEUE_NAME_CONFIG,
        workflow=score_prediction_workflow,
        timeout=timeout,
        retry_token=retry_token,
    )


def start_worker_monitor(
    config: WorkerMonitorConfig, stop_event: threading.Event
) -> threading.Thread:
    return shared_worker_monitor.start_worker_monitor(
        config,
        stop_event,
        operator_log=operator_log,
        emit_worker_detail_log=emit_worker_detail_log,
    )


def common_config(
    *,
    database_url: str | None,
    dbos_system_database_url: str | None,
    generation_concurrency: int,
    scoring_concurrency: int,
    env_file: Path | None = None,
) -> EvalDbosConfig:
    load_optional_env_file(env_file)
    return build_eval_dbos_config(
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
    create_eval_schema(config.database_url)
    operator_log("initialized dr-dspy eval tables", style="green")


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
    jobs = build_prediction_jobs(
        experiment_name=experiment_name,
        submission_id=submission_id,
        samples=samples,
        model_configs=model_configs,
        temperatures=parsed_temperatures,
        repetitions=repetitions,
    )
    operator_log(
        f"planned {len(jobs)} jobs: samples={len(samples)}, "
        f"models={len(model_configs)}, "
        f"temperatures={len(parsed_temperatures)}, "
        f"repetitions={repetitions}",
        style="cyan",
    )
    if dry_run:
        operator_log(
            "dry run only; no rows written and no workflows enqueued",
            style="yellow",
        )
        return

    shared_flow.run_submit_jobs(
        config=config,
        experiment_name=experiment_name,
        seed=seed,
        sample_count=sample_count,
        metadata={
            "submission_id": submission_id,
            "models": [
                model.model_dump(mode="json") for model in model_configs
            ],
            "temperatures": parsed_temperatures,
            "repetitions": repetitions,
            "score_timeout": score_timeout,
        },
        jobs=jobs,
        score_timeout=score_timeout,
        create_schema=create_eval_schema,
        upsert_experiment=upsert_experiment,
        insert_prediction_jobs=insert_prediction_jobs,
        configure_runtime=(
            lambda dbos_config, exp_name: configure_dbos_runtime(
                dbos_config,
                experiment_name=exp_name,
                consume_queues=False,
            )
        ),
        enqueue_generation_jobs=(
            lambda db_url, generation_jobs, timeout: enqueue_generation_jobs(
                db_url,
                generation_jobs,
                score_timeout=timeout,
            )
        ),
        operator_log=operator_log,
    )


@_APP.command()
def status(
    experiment_name: Annotated[
        str | None,
        typer.Option(
            "--experiment-name",
            help="Limit status to one experiment.",
        ),
    ] = None,
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
    shared_flow.run_status_command(
        database_url=config.database_url,
        experiment_name=experiment_name,
        fetch_status_counts=fetch_status_counts,
        status_counts_table=status_counts_table,
        console=CONSOLE,
        operator_log=operator_log,
    )


@_APP.command("enqueue-scores")
def enqueue_scores_command(
    experiment_name: Annotated[
        str,
        typer.Option("--experiment-name", help="Experiment to score."),
    ],
    limit: Annotated[
        int, typer.Option("--limit", min=1)
    ] = DEFAULT_SCORE_ENQUEUE_LIMIT,
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
    shared_flow.run_enqueue_scores_command(
        config=config,
        experiment_name=experiment_name,
        limit=limit,
        timeout=timeout,
        create_schema=create_eval_schema,
        fetch_scoreable_prediction_ids=fetch_scoreable_prediction_ids,
        configure_runtime=(
            lambda dbos_config, exp_name: configure_dbos_runtime(
                dbos_config,
                experiment_name=exp_name,
                consume_queues=False,
            )
        ),
        enqueue_score_jobs=(
            lambda db_url, prediction_ids, exp_name, score_timeout: (
                enqueue_score_jobs(
                    db_url,
                    prediction_ids,
                    experiment_name=exp_name,
                    timeout=score_timeout,
                )
            )
        ),
        mark_scoring_queued=mark_scoring_queued,
        enqueue_scores_line=enqueue_scores_line,
        enqueue_scores_style=enqueue_scores_style,
        operator_log=operator_log,
    )


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
    generation_limit: Annotated[
        int, typer.Option("--generation-limit", min=1)
    ] = DEFAULT_REPAIR_GENERATION_LIMIT,
    scoring_limit: Annotated[
        int, typer.Option("--scoring-limit", min=1)
    ] = DEFAULT_REPAIR_SCORING_LIMIT,
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
    shared_flow.run_repair_command(
        config=config,
        experiment_name=experiment_name,
        generation_limit=generation_limit,
        scoring_limit=scoring_limit,
        score_timeout=score_timeout,
        apply=apply,
        build_repair_plan=build_repair_plan,
        apply_repair=apply_repair,
        repair_plan_line=repair_plan_line,
        repair_plan_style=repair_plan_style,
        repair_apply_line=repair_apply_line,
        plan_has_work=lambda plan: bool(
            plan.stranded_generations
            or plan.generation_retry_prediction_ids
            or plan.pending_scoring_prediction_ids
            or plan.stranded_scoring
            or plan.scoring_retry_prediction_ids
        ),
        operator_log=operator_log,
    )


@_APP.command()
def analyze(
    experiment_name: Annotated[
        str,
        typer.Option("--experiment-name", help="Experiment to analyze."),
    ],
    csv_path: Annotated[
        Path | None,
        typer.Option("--csv-path", help="Optional CSV output path."),
    ] = None,
    database_url: Annotated[
        str | None,
        typer.Option(
            "--database-url",
            help=f"Postgres URL; defaults to {DATABASE_URL_ENV}.",
        ),
    ] = None,
    markdown: Annotated[
        bool,
        typer.Option(
            "--markdown",
            help="Print the analysis table as Markdown.",
        ),
    ] = False,
    env_file: Annotated[Path | None, typer.Option()] = None,
) -> None:
    config = common_config(
        database_url=database_url,
        dbos_system_database_url=None,
        generation_concurrency=DEFAULT_GENERATION_CONCURRENCY,
        scoring_concurrency=DEFAULT_SCORING_CONCURRENCY,
        env_file=env_file,
    )
    shared_flow.run_analyze_command(
        database_url=config.database_url,
        experiment_name=experiment_name,
        csv_path=csv_path,
        markdown=markdown,
        fetch_analysis_records=fetch_analysis_records,
        summarize_analysis_records=summarize_analysis_records,
        analysis_markdown=analysis_markdown,
        analysis_table=analysis_table,
        write_analysis_csv=lambda summaries, path: write_analysis_csv(
            summaries, csv_path=path
        ),
        console=CONSOLE,
        operator_log=operator_log,
    )


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
        QueueSelection,
        typer.Option("--queue", help="Queue set this worker should consume."),
    ] = QueueSelection.BOTH,
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
        int,
        typer.Option(
            "--open-file-limit",
            min=1,
            help=(
                "Requested worker soft open-file limit. The process can "
                "raise this only up to the OS hard limit."
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
    ] = DB_POOL_AUTO,
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
        config=config,
        experiment_name=experiment_name,
        queue=queue,
        open_file_limit=open_file_limit,
        log_file=log_file,
        monitor=monitor,
        monitor_interval=monitor_interval,
        monitor_summary_interval=monitor_summary_interval,
        db_pool_max_size=db_pool_max_size,
        prediction_table=PREDICTION_TABLE_NAME,
        db_pools=DB_POOLS,
        raise_open_file_limit=raise_open_file_limit,
        open_file_limit_line=open_file_limit_line,
        open_file_limit_style=open_file_limit_style,
        create_schema=create_eval_schema,
        configure_pooled_worker_runtime=configure_pooled_worker_runtime,
        resolve_worker_log_path=resolve_worker_log_path,
        configure_worker_file_logging=configure_worker_file_logging,
        queue_names_for_selection=queue_names_for_selection,
        start_worker_monitor=start_worker_monitor,
        close_db_connection_pools=close_db_connection_pools,
        operator_log=operator_log,
    )
