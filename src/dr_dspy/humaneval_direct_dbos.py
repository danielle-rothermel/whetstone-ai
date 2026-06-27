from __future__ import annotations

import logging
import resource
import threading
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Protocol

import typer
from dbos import DBOS, DBOSConfig, SetWorkflowID
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool
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
from dr_dspy import analysis as shared_analysis
from dr_dspy import dbos_runtime as shared_dbos
from dr_dspy import dspy_runner as shared_dspy_runner
from dr_dspy import eval_logging as shared_eval_logging
from dr_dspy import eval_repair as shared_eval_repair
from dr_dspy import eval_reporting as shared_eval_reporting
from dr_dspy import human_eval_sampling as shared_human_eval_sampling
from dr_dspy import humaneval_dbos_flow as shared_flow
from dr_dspy import worker_monitor as shared_worker_monitor
from dr_dspy.code_eval import (
    DEFAULT_CAPTURE_LIMIT_BYTES,
    extract_dspy_code,
)
from dr_dspy.human_eval import HumanEvalTask
from dr_dspy.lm_utils import (
    LmEventBuffer,
    ModelConfig,
)
from dr_dspy.runtime import configure_multiprocessing, load_env_file
from dr_dspy.scoring import score_generated_code_for_humaneval
from dr_dspy.signatures import FieldSignature
from dspy.signatures.signature import make_signature

psycopg = shared_dbos.psycopg

# Configuration

DATABASE_URL_ENV = "DATABASE_URL"
SCRIPT_KIND = "humaneval_eval_only_dbos_v0"
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
DEFAULT_DB_POOL_MARGIN = 8
DB_POOL_AUTO = "auto"
DEFAULT_SEED = 0
DEFAULT_SAMPLE_COUNT = 10
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TEMPERATURES = (DEFAULT_TEMPERATURE,)
DEFAULT_MAX_COMPLETION_TOKENS = 1000
DEFAULT_SUBPROCESS_TIMEOUT = 15.0
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
DATASET_NAME = "evalplus/humanevalplus"
DATASET_SPLIT = "test"
PREDICTION_TABLE_NAME = "dr_dspy_eval_predictions"
REPAIR_DIMENSION_COLUMNS = ("model", "temperature")
REPAIR_ORDER_COLUMNS = (
    "model",
    "temperature",
    "sample_index",
    "repetition_seed",
)
SOLVE_NAME = "Solve"
SOLVE_FIELDS = [
    FieldSignature(name="prompt", type=str, role=dspy.InputField()),
    FieldSignature(name="code", type=dspy.Code, role=dspy.OutputField()),
]
SOLVE_INSTRUCTIONS = (
    "Write functional code in Python according to the prompt. "
    "Output only the code solution."
)

DEFAULT_MODEL_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "model": "openai/gpt-5.1-codex-mini",
        "reasoning": {},
    },
    {
        "model": "moonshotai/kimi-k2-0905",
        "reasoning": {},
    },
    {
        "model": "qwen/qwen3-coder-next",
        "reasoning": {},
    },
    {
        "model": "deepseek/deepseek-v3.1-terminus",
        "reasoning": {"enabled": False},
    },
    {
        "model": "moonshotai/kimi-k2",
        "reasoning": {},
    },
    {
        "model": "z-ai/glm-4.7",
        "reasoning": {"enabled": False},
    },
    {
        "model": "z-ai/glm-5",
        "reasoning": {"enabled": False},
    },
    {
        "model": "deepseek/deepseek-v4-pro",
        "reasoning": {"enabled": False},
    },
    {
        "model": "deepseek/deepseek-v4-flash",
        "reasoning": {"enabled": False},
    },
    {
        "model": "mistralai/mistral-large-2512",
        "reasoning": {},
    },
    {
        "model": "openai/gpt-oss-120b",
        "reasoning": {"effort": "low"},
    },
    {
        "model": "mistralai/codestral-2508",
        "reasoning": {},
    },
    {
        "model": "qwen/qwen3-coder-flash",
        "reasoning": {},
    },
    {
        "model": "openai/gpt-5-nano",
        "reasoning": {"effort": "low"},
    },
    {
        "model": "deepseek/deepseek-chat-v3.1",
        "reasoning": {"enabled": False},
    },
    {
        "model": "openai/gpt-5.4-nano",
        "reasoning": {"effort": "none"},
    },
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
    "ADD COLUMN IF NOT EXISTS extracted_compile_ok BOOLEAN",
    "ALTER TABLE dr_dspy_eval_predictions "
    "ADD COLUMN IF NOT EXISTS extracted_compile_error TEXT",
    "ALTER TABLE dr_dspy_eval_predictions "
    "ADD COLUMN IF NOT EXISTS extraction_error TEXT",
    "ALTER TABLE dr_dspy_eval_predictions "
    "ADD COLUMN IF NOT EXISTS score_stdout TEXT",
    "ALTER TABLE dr_dspy_eval_predictions "
    "ADD COLUMN IF NOT EXISTS score_stderr TEXT",
    "ALTER TABLE dr_dspy_eval_predictions "
    "ADD COLUMN IF NOT EXISTS score_stdout_truncated BOOLEAN",
    "ALTER TABLE dr_dspy_eval_predictions "
    "ADD COLUMN IF NOT EXISTS score_stderr_truncated BOOLEAN",
)


GENERATION_REPAIR_ERROR = shared_eval_repair.GENERATION_REPAIR_ERROR
SCORING_REPAIR_ERROR = shared_eval_repair.SCORING_REPAIR_ERROR


class HumanEvalSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: StrictStr
    sample_index: StrictInt
    prompt: StrictStr
    test: StrictStr
    entry_point: StrictStr


class PredictionLogContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_id: StrictStr
    experiment_name: StrictStr
    task_id: StrictStr
    sample_index: StrictInt
    model: StrictStr
    temperature: float | None
    repetition_seed: StrictInt


class PredictionJob(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_id: StrictStr
    experiment_name: StrictStr
    script_kind: StrictStr = SCRIPT_KIND
    submission_id: StrictStr
    task_id: StrictStr
    sample_index: StrictInt
    model: StrictStr
    temperature: float
    repetition_seed: StrictInt
    prompt: StrictStr
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


class GenerationRepairCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_id: StrictStr
    model: StrictStr
    temperature: float
    task_id: StrictStr
    sample_index: StrictInt
    repetition_seed: StrictInt
    dbos_status: StrictStr


class ScoringRepairCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_id: StrictStr
    model: StrictStr
    temperature: float
    task_id: StrictStr
    sample_index: StrictInt
    repetition_seed: StrictInt
    scoring_status: StrictStr
    dbos_status: StrictStr


class RepairPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stranded_generations: list[GenerationRepairCandidate] = Field(
        default_factory=list
    )
    generation_retry_jobs: list[PredictionJob] = Field(default_factory=list)
    pending_scoring_prediction_ids: list[StrictStr] = Field(
        default_factory=list
    )
    stranded_scoring: list[ScoringRepairCandidate] = Field(
        default_factory=list
    )
    scoring_retry_prediction_ids: list[StrictStr] = Field(
        default_factory=list
    )


class RepairApplyResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repair_token: StrictStr
    stranded_generations_marked: StrictInt
    generation_retries_enqueued: StrictInt
    generation_retries_reset: StrictInt
    stranded_scoring_marked: StrictInt
    pending_scoring_enqueued: StrictInt
    scoring_retries_enqueued: StrictInt
    scoring_retries_marked_queued: StrictInt


class ScoringTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_id: StrictStr
    task_id: StrictStr
    raw_generation: StrictStr
    test: StrictStr
    entry_point: StrictStr


class ScoreResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_id: StrictStr
    score: float
    error: str | None
    raw_code: str | None = None
    raw_compile_ok: bool
    raw_compile_error: str | None = None
    extraction_candidate_count: int
    extracted_compile_ok: bool
    extracted_compile_error: str | None = None
    extraction_error: str | None = None
    stdout: str = ""
    stderr: str = ""
    stdout_truncated: bool = False
    stderr_truncated: bool = False


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


HumanEvalRow = Mapping[str, Any]


class HumanEvalDataset(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> HumanEvalRow: ...


Solve = make_signature(
    {field.name: (field.type, field.role) for field in SOLVE_FIELDS},
    instructions=SOLVE_INSTRUCTIONS,
    signature_name=SOLVE_NAME,
)


DB_POOLS: dict[str, ConnectionPool] = {}


def sanitize_log_name(name: str) -> str:
    return shared_eval_logging.sanitize_log_name(name)


def hashed_experiment_log_name(experiment_name: str) -> str:
    return shared_eval_logging.hashed_experiment_log_name(
        experiment_name, hash_length=EXPERIMENT_QUEUE_HASH_LENGTH
    )


def default_worker_log_path(
    *,
    experiment_name: str,
    queue: QueueSelection,
    now: datetime | None = None,
    pid: int | None = None,
) -> Path:
    return shared_eval_logging.default_worker_log_path(
        log_root=DEFAULT_WORKER_LOG_ROOT,
        experiment_name=experiment_name,
        queue=queue,
        hash_length=EXPERIMENT_QUEUE_HASH_LENGTH,
        now=now,
        pid=pid,
    )


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


def prediction_context_from_job(job: PredictionJob) -> PredictionLogContext:
    return PredictionLogContext(
        prediction_id=job.prediction_id,
        experiment_name=job.experiment_name,
        task_id=job.task_id,
        sample_index=job.sample_index,
        model=job.model,
        temperature=job.temperature,
        repetition_seed=job.repetition_seed,
    )


def emit_prediction_log_event(
    event: str,
    context: PredictionLogContext,
    *,
    extra: Mapping[str, Any] | None = None,
) -> None:
    payload = context.model_dump(mode="json")
    if extra is not None:
        payload.update(extra)
    emit_worker_detail_log(event, payload)


@DBOS.workflow(name="humaneval_eval_generate_prediction_v0")
def generate_prediction_workflow(
    database_url: str,
    prediction_id: str,
    experiment_name: str,
    score_timeout: float = DEFAULT_SUBPROCESS_TIMEOUT,
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
        digest_length=32,
    )


def parse_temperatures(raw: str) -> list[float]:
    return shared_flow.parse_float_csv(raw, value_name="temperature")


def default_model_configs() -> list[ModelConfig]:
    return [ModelConfig(**config) for config in DEFAULT_MODEL_CONFIGS]


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
                            submission_id=submission_id,
                            task_id=sample.task_id,
                            sample_index=sample.sample_index,
                            model=model_config.model,
                            temperature=temperature,
                            repetition_seed=repetition_seed,
                            prompt=sample.prompt,
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
                    SCRIPT_KIND,
                    seed,
                    sample_count,
                    SOLVE_INSTRUCTIONS,
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
                    test,
                    entry_point,
                    reasoning
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s
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
        test=row[10],
        entry_point=row[11],
        reasoning=dict(row[12] or {}),
    )


def fetch_prediction_log_context(
    database_url: str, prediction_id: str
) -> PredictionLogContext:
    with connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    prediction_id,
                    experiment_name,
                    task_id,
                    sample_index,
                    model,
                    temperature,
                    repetition_seed
                FROM dr_dspy_eval_predictions
                WHERE prediction_id = %s
                """,
                (prediction_id,),
            )
            row = cur.fetchone()
    if row is None:
        raise ValueError(f"prediction_id not found: {prediction_id}")
    return PredictionLogContext(
        prediction_id=row[0],
        experiment_name=row[1],
        task_id=row[2],
        sample_index=row[3],
        model=row[4],
        temperature=row[5],
        repetition_seed=row[6],
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
                    raw_generation,
                    test,
                    entry_point
                FROM dr_dspy_eval_predictions
                WHERE prediction_id = %s
                """,
                (prediction_id,),
            )
            row = cur.fetchone()
    if row is None:
        raise ValueError(f"prediction_id not found: {prediction_id}")
    if row[2] is None:
        raise ValueError(
            f"prediction_id has no raw generation: {prediction_id}"
        )
    return ScoringTarget(
        prediction_id=row[0],
        task_id=row[1],
        raw_generation=row[2],
        test=row[3],
        entry_point=row[4],
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
                    score_stdout = NULL,
                    score_stderr = NULL,
                    score_stdout_truncated = NULL,
                    score_stderr_truncated = NULL,
                    scored_at = NULL,
                    updated_at = now()
                WHERE
                    prediction_id = ANY(%s)
                    AND generation_status = 'generation_error'
                """,
                (list(prediction_ids),),
            )
            return cur.rowcount if cur.rowcount is not None else 0


def fetch_started_generation_repair_candidates(
    database_url: str,
    *,
    dbos_system_database_url: str,
    experiment_name: str,
) -> list[GenerationRepairCandidate]:
    candidates = (
        shared_eval_repair.fetch_started_generation_repair_candidates(
            database_url,
            dbos_system_database_url=dbos_system_database_url,
            prediction_table=PREDICTION_TABLE_NAME,
            experiment_name=experiment_name,
            dimension_columns=REPAIR_DIMENSION_COLUMNS,
            order_columns=REPAIR_ORDER_COLUMNS,
        )
    )
    return [
        GenerationRepairCandidate(
            prediction_id=candidate.prediction_id,
            model=candidate.dimensions["model"],
            temperature=candidate.dimensions["temperature"],
            task_id=candidate.task_id,
            sample_index=candidate.sample_index,
            repetition_seed=candidate.repetition_seed,
            dbos_status=candidate.dbos_status,
        )
        for candidate in candidates
    ]


def mark_started_generations_as_repaired_errors(
    database_url: str,
    *,
    prediction_ids: Sequence[str],
) -> int:
    return shared_eval_repair.mark_started_generations_as_repaired_errors(
        database_url,
        prediction_table=PREDICTION_TABLE_NAME,
        prediction_ids=prediction_ids,
    )


def fetch_stranded_scoring_repair_candidates(
    database_url: str,
    *,
    dbos_system_database_url: str,
    experiment_name: str,
    limit: int,
) -> list[ScoringRepairCandidate]:
    candidates = (
        shared_eval_repair.fetch_stranded_scoring_repair_candidates(
            database_url,
            dbos_system_database_url=dbos_system_database_url,
            prediction_table=PREDICTION_TABLE_NAME,
            experiment_name=experiment_name,
            dimension_columns=REPAIR_DIMENSION_COLUMNS,
            order_columns=REPAIR_ORDER_COLUMNS,
            limit=limit,
        )
    )
    return [
        ScoringRepairCandidate(
            prediction_id=candidate.prediction_id,
            model=candidate.dimensions["model"],
            temperature=candidate.dimensions["temperature"],
            task_id=candidate.task_id,
            sample_index=candidate.sample_index,
            repetition_seed=candidate.repetition_seed,
            scoring_status=candidate.scoring_status or "",
            dbos_status=candidate.dbos_status,
        )
        for candidate in candidates
    ]


def mark_stranded_scoring_as_errors(
    database_url: str,
    *,
    prediction_ids: Sequence[str],
) -> int:
    return shared_eval_repair.mark_stranded_scoring_as_errors(
        database_url,
        prediction_table=PREDICTION_TABLE_NAME,
        prediction_ids=prediction_ids,
    )


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
                    extracted_compile_ok = %s,
                    extracted_compile_error = %s,
                    extraction_error = %s,
                    score_stdout = %s,
                    score_stderr = %s,
                    score_stdout_truncated = %s,
                    score_stderr_truncated = %s,
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
                    result.extracted_compile_ok,
                    result.extracted_compile_error,
                    result.extraction_error,
                    result.stdout,
                    result.stderr,
                    result.stdout_truncated,
                    result.stderr_truncated,
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
    task = HumanEvalTask(
        task_id=target.task_id,
        prompt="",
        canonical_solution="",
        test=target.test,
        entry_point=target.entry_point,
    )
    result = score_generated_code_for_humaneval(
        raw_generation=target.raw_generation,
        task=task,
        timeout=timeout,
        capture_limit_bytes=DEFAULT_CAPTURE_LIMIT_BYTES,
    )
    return ScoreResult(
        prediction_id=target.prediction_id,
        score=result.score,
        error=result.error,
        raw_code=result.raw_code,
        raw_compile_ok=result.raw_compile_ok,
        raw_compile_error=result.raw_compile_error,
        extraction_candidate_count=result.extraction_candidate_count,
        extracted_compile_ok=result.extracted_compile_ok,
        extracted_compile_error=result.extracted_compile_error,
        extraction_error=result.extraction_error,
        stdout=result.stdout,
        stderr=result.stderr,
        stdout_truncated=result.stdout_truncated,
        stderr_truncated=result.stderr_truncated,
    )


@DBOS.step()
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


@DBOS.step()
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


@DBOS.step()
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


@DBOS.step()
def mark_scoring_queued_step(database_url: str, prediction_id: str) -> None:
    shared_flow.run_mark_scoring_queued_step(
        database_url=database_url,
        prediction_id=prediction_id,
        fetch_prediction_log_context=fetch_prediction_log_context,
        emit_prediction_log_event=emit_prediction_log_event,
        mark_scoring_queued=mark_scoring_queued,
    )


@DBOS.step(retries_allowed=True, max_attempts=3, interval_seconds=2.0)
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


@DBOS.step()
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


@DBOS.step()
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
) -> RepairPlan:
    return RepairPlan(
        stranded_generations=fetch_started_generation_repair_candidates(
            database_url,
            dbos_system_database_url=dbos_system_database_url,
            experiment_name=experiment_name,
        ),
        generation_retry_jobs=fetch_generation_error_prediction_jobs(
            database_url,
            experiment_name=experiment_name,
            limit=generation_limit,
        ),
        pending_scoring_prediction_ids=fetch_pending_scoring_prediction_ids(
            database_url,
            experiment_name=experiment_name,
            limit=scoring_limit,
        ),
        stranded_scoring=fetch_stranded_scoring_repair_candidates(
            database_url,
            dbos_system_database_url=dbos_system_database_url,
            experiment_name=experiment_name,
            limit=scoring_limit,
        ),
        scoring_retry_prediction_ids=fetch_score_error_prediction_ids(
            database_url,
            experiment_name=experiment_name,
            limit=scoring_limit,
        ),
    )


def apply_repair(
    config: EvalDbosConfig,
    *,
    experiment_name: str,
    generation_limit: int,
    scoring_limit: int,
    score_timeout: float,
    repair_token: str | None = None,
) -> RepairApplyResult:
    result = shared_eval_repair.apply_repair(
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
    return RepairApplyResult(
        **result.model_dump(mode="json"),
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
                    extracted_compile_ok
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
        summary_factory=AnalysisSummary,
    )


def operator_timestamp(now: datetime | None = None) -> str:
    return shared_eval_logging.operator_timestamp(
        now, timestamp_format=OPERATOR_TIMESTAMP_FORMAT
    )


def timestamped_line(line: str, *, now: datetime | None = None) -> str:
    return shared_eval_logging.timestamped_line(
        line, now=now, timestamp_format=OPERATOR_TIMESTAMP_FORMAT
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


def repair_generation_started_line(
    *,
    experiment_name: str,
    selected_count: int,
    applied_count: int | None,
) -> str:
    applied = "-" if applied_count is None else str(applied_count)
    return (
        f"{'Repair Gen':<14} | "
        f"selected={selected_count:>5} | "
        f"applied={applied:>5} | "
        f"experiment={experiment_name}"
    )


def repair_generation_started_style(
    *, selected_count: int, applied_count: int | None
) -> str:
    if selected_count == 0:
        return "yellow"
    if applied_count is None:
        return "cyan"
    return "green"


def retry_generation_errors_line(
    *,
    experiment_name: str,
    selected_count: int,
    reset_count: int | None,
    limit: int,
    retry_token: str | None,
) -> str:
    reset = "-" if reset_count is None else str(reset_count)
    token = "-" if retry_token is None else retry_token
    return (
        f"{'Retry Gen':<14} | "
        f"selected={selected_count:>5} | "
        f"reset={reset:>5} | "
        f"limit={limit:>5} | "
        f"token={token} | "
        f"experiment={experiment_name}"
    )


def retry_generation_errors_style(
    *, selected_count: int, reset_count: int | None
) -> str:
    if selected_count == 0:
        return "yellow"
    if reset_count is None:
        return "cyan"
    return "green"


def repair_plan_line(
    *,
    experiment_name: str,
    plan: RepairPlan,
    apply: bool,
) -> str:
    return shared_eval_reporting.repair_plan_line(
        experiment_name=experiment_name,
        gen_stranded=len(plan.stranded_generations),
        gen_errors=len(plan.generation_retry_jobs),
        score_pending=len(plan.pending_scoring_prediction_ids),
        score_stranded=len(plan.stranded_scoring),
        score_errors=len(plan.scoring_retry_prediction_ids),
        apply=apply,
    )


def repair_apply_line(
    *, experiment_name: str, result: RepairApplyResult
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


def repair_plan_style(plan: RepairPlan, *, apply: bool) -> str:
    return shared_eval_reporting.repair_plan_style(
        apply=apply,
        gen_stranded=len(plan.stranded_generations),
        gen_errors=len(plan.generation_retry_jobs),
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
        dimension_columns=("model", "temperature"),
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


app = typer.Typer(no_args_is_help=True)


QUEUE_NAME_CONFIG = shared_dbos.QueueNameConfig(
    generation_base_name=GENERATION_QUEUE_NAME,
    scoring_base_name=SCORING_QUEUE_NAME,
    hash_length=EXPERIMENT_QUEUE_HASH_LENGTH,
)

QueueSelection = shared_dbos.QueueSelection
DbosWorkflowStatus = shared_dbos.DbosWorkflowStatus
DBOS_ACTIVE_WORKFLOW_STATUSES = shared_dbos.DBOS_ACTIVE_WORKFLOW_STATUSES
DBOS_FAILED_WORKFLOW_STATUSES = shared_dbos.DBOS_FAILED_WORKFLOW_STATUSES
MISSING_DBOS_WORKFLOW_STATUS = shared_dbos.MISSING_DBOS_WORKFLOW_STATUS
EvalDbosConfig = shared_dbos.EvalDbosConfig
DbPoolConfig = shared_dbos.DbPoolConfig
OpenFileLimitResult = shared_dbos.OpenFileLimitResult
EvalQueueNames = shared_dbos.EvalQueueNames
DB_POOLS = shared_dbos.DB_POOLS
WorkerQueueSnapshot = shared_worker_monitor.WorkerQueueSnapshot
WorkerMonitorConfig = shared_worker_monitor.WorkerMonitorConfig

variance_or_none = shared_analysis.variance_or_none
average_or_none = shared_analysis.average_or_none
format_float = shared_analysis.format_float
format_cost = shared_analysis.format_cost
align_decimal_column = shared_analysis.align_decimal_column
format_float_column = shared_analysis.format_float_column
format_cost_column = shared_analysis.format_cost_column
price_per_thousand_samples = shared_analysis.price_per_thousand_samples
sum_present_float = shared_analysis.sum_present_float

format_resource_limit = shared_dbos.format_resource_limit
open_file_limit_line = shared_dbos.open_file_limit_line
open_file_limit_style = shared_dbos.open_file_limit_style
close_db_connection_pools = shared_dbos.close_db_connection_pools
connect_db = shared_dbos.connect_db
generation_workflow_id = shared_dbos.generation_workflow_id
score_workflow_id = shared_dbos.score_workflow_id
worker_monitor_line = shared_worker_monitor.worker_monitor_line
worker_monitor_style = shared_worker_monitor.worker_monitor_style


def raise_open_file_limit(requested: int) -> OpenFileLimitResult:
    shared_dbos.resource = resource
    return shared_dbos.raise_open_file_limit(requested)


def resolve_database_url(database_url: str | None) -> str:
    return shared_dbos.resolve_database_url(
        database_url,
        database_url_env=DATABASE_URL_ENV,
        error_suffix="for this Postgres-only DBOS harness",
    )


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


def build_dbos_config(config: EvalDbosConfig) -> DBOSConfig:
    return shared_dbos.build_dbos_config(config, app_name=DBOS_APP_NAME)


def configure_db_connection_pools(
    database_urls: Sequence[str], *, max_size: int
) -> None:
    shared_dbos.ConnectionPool = ConnectionPool
    shared_dbos.configure_db_connection_pools(
        database_urls, max_size=max_size
    )


def auto_db_pool_max_size(
    *,
    queue: QueueSelection,
    generation_concurrency: int,
    scoring_concurrency: int,
) -> int:
    return shared_dbos.auto_db_pool_max_size(
        queue=queue,
        generation_concurrency=generation_concurrency,
        scoring_concurrency=scoring_concurrency,
        margin=DEFAULT_DB_POOL_MARGIN,
    )


def resolve_db_pool_config(
    *,
    raw_max_size: str,
    queue: QueueSelection,
    generation_concurrency: int,
    scoring_concurrency: int,
) -> DbPoolConfig:
    return shared_dbos.resolve_db_pool_config(
        raw_max_size=raw_max_size,
        queue=queue,
        generation_concurrency=generation_concurrency,
        scoring_concurrency=scoring_concurrency,
    )


def configure_worker_db_connection_pools(
    config: EvalDbosConfig,
    *,
    queue: QueueSelection,
    raw_max_size: str,
) -> DbPoolConfig:
    shared_dbos.ConnectionPool = ConnectionPool
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


def experiment_hash(experiment_name: str) -> str:
    return shared_dbos.experiment_hash(
        experiment_name, hash_length=EXPERIMENT_QUEUE_HASH_LENGTH
    )


def eval_queue_names(experiment_name: str) -> EvalQueueNames:
    return shared_dbos.eval_queue_names(experiment_name, QUEUE_NAME_CONFIG)


def register_eval_queues(
    config: EvalDbosConfig, *, experiment_name: str
) -> None:
    shared_dbos.DBOS = DBOS
    shared_dbos.register_eval_queues(
        config, experiment_name=experiment_name, queue_config=QUEUE_NAME_CONFIG
    )


def eval_queue_concurrency_by_name(
    config: EvalDbosConfig, *, experiment_name: str
) -> dict[str, int]:
    return shared_dbos.eval_queue_concurrency_by_name(
        config, experiment_name=experiment_name, queue_config=QUEUE_NAME_CONFIG
    )


def sync_existing_dbos_queue_concurrency(
    config: EvalDbosConfig, *, experiment_name: str
) -> int:
    return shared_dbos.sync_existing_dbos_queue_concurrency(
        config, experiment_name=experiment_name, queue_config=QUEUE_NAME_CONFIG
    )


def queue_config_line(config: EvalDbosConfig, *, experiment_name: str) -> str:
    return shared_dbos.queue_config_line(
        config, experiment_name=experiment_name
    )


def queue_names_for_selection(
    selection: QueueSelection, *, experiment_name: str
) -> tuple[str, ...]:
    return shared_dbos.queue_names_for_selection(
        selection,
        experiment_name=experiment_name,
        queue_config=QUEUE_NAME_CONFIG,
    )


def listen_to_selected_queue(
    selection: QueueSelection, *, experiment_name: str
) -> None:
    shared_dbos.DBOS = DBOS
    shared_dbos.listen_to_selected_queue(
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
    DBOS(config=build_dbos_config(config))
    sync_existing_dbos_queue_concurrency(
        config, experiment_name=experiment_name
    )
    if queue is not None:
        listen_to_selected_queue(queue, experiment_name=experiment_name)
    elif not consume_queues:
        DBOS.listen_queues([])
    DBOS.launch()
    register_eval_queues(config, experiment_name=experiment_name)
    operator_log(queue_config_line(config, experiment_name=experiment_name))


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
    return build_humaneval_samples_from_rows(
        shared_human_eval_sampling.load_human_eval_rows(
            dataset_name=DATASET_NAME,
            dataset_split=DATASET_SPLIT,
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
        max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
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
        signature=Solve,
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
    shared_dbos.DBOS = DBOS
    shared_dbos.SetWorkflowID = SetWorkflowID
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
    shared_dbos.DBOS = DBOS
    shared_dbos.SetWorkflowID = SetWorkflowID
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
    shared_dbos.DBOS = DBOS
    shared_dbos.SetWorkflowID = SetWorkflowID
    shared_dbos.enqueue_score_workflows(
        database_url,
        prediction_ids,
        experiment_name=experiment_name,
        queue_config=QUEUE_NAME_CONFIG,
        workflow=score_prediction_workflow,
        timeout=timeout,
        retry_token=retry_token,
    )


def fetch_prediction_phase_counts(
    database_url: str,
    *,
    status_column: str,
    experiment_name: str,
) -> dict[str, int]:
    return shared_worker_monitor.fetch_prediction_phase_counts(
        database_url,
        prediction_table=PREDICTION_TABLE_NAME,
        status_column=status_column,
        experiment_name=experiment_name,
    )


def fetch_dbos_status_counts(
    dbos_system_database_url: str, queue_names: Sequence[str]
) -> dict[str, int]:
    return shared_worker_monitor.fetch_dbos_status_counts(
        dbos_system_database_url, queue_names
    )


def fetch_worker_queue_snapshot(
    config: WorkerMonitorConfig,
) -> WorkerQueueSnapshot:
    return shared_worker_monitor.fetch_worker_queue_snapshot(config)


def run_worker_monitor(
    config: WorkerMonitorConfig, stop_event: threading.Event
) -> None:
    shared_worker_monitor.run_worker_monitor(
        config,
        stop_event,
        operator_log=operator_log,
        emit_worker_detail_log=emit_worker_detail_log,
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
) -> EvalDbosConfig:
    load_env_file()
    return build_eval_dbos_config(
        database_url=database_url,
        dbos_system_database_url=dbos_system_database_url,
        generation_concurrency=generation_concurrency,
        scoring_concurrency=scoring_concurrency,
    )


@app.command()
def init_db(
    database_url: Annotated[
        str | None,
        typer.Option(
            "--database-url",
            help=f"Postgres URL; defaults to {DATABASE_URL_ENV}.",
        ),
    ] = None,
) -> None:
    config = common_config(
        database_url=database_url,
        dbos_system_database_url=None,
        generation_concurrency=DEFAULT_GENERATION_CONCURRENCY,
        scoring_concurrency=DEFAULT_SCORING_CONCURRENCY,
    )
    create_eval_schema(config.database_url)
    operator_log("initialized dr-dspy eval tables", style="green")


@app.command()
def submit(
    experiment_name: Annotated[
        str,
        typer.Option(
            "--experiment-name",
            help="Human-readable experiment key.",
        ),
    ],
    sample_count: Annotated[
        int, typer.Option("--sample-count", min=1)
    ] = DEFAULT_SAMPLE_COUNT,
    seed: Annotated[int, typer.Option("--seed")] = DEFAULT_SEED,
    temperatures: Annotated[
        str,
        typer.Option(
            "--temperatures",
            help="Comma-separated temperature values.",
        ),
    ] = ",".join(str(value) for value in DEFAULT_TEMPERATURES),
    repetitions: Annotated[int, typer.Option("--repetitions", min=1)] = 1,
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
        float, typer.Option("--score-timeout", min=0.1)
    ] = DEFAULT_SUBPROCESS_TIMEOUT,
) -> None:
    config = common_config(
        database_url=database_url,
        dbos_system_database_url=dbos_system_database_url,
        generation_concurrency=generation_concurrency,
        scoring_concurrency=scoring_concurrency,
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


@app.command()
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
) -> None:
    config = common_config(
        database_url=database_url,
        dbos_system_database_url=None,
        generation_concurrency=DEFAULT_GENERATION_CONCURRENCY,
        scoring_concurrency=DEFAULT_SCORING_CONCURRENCY,
    )
    shared_flow.run_status_command(
        database_url=config.database_url,
        experiment_name=experiment_name,
        fetch_status_counts=fetch_status_counts,
        status_counts_table=status_counts_table,
        console=CONSOLE,
        operator_log=operator_log,
    )


@app.command("enqueue-scores")
def enqueue_scores_command(
    experiment_name: Annotated[
        str,
        typer.Option("--experiment-name", help="Experiment to score."),
    ],
    limit: Annotated[
        int, typer.Option("--limit", min=1)
    ] = DEFAULT_SCORE_ENQUEUE_LIMIT,
    timeout: Annotated[
        float, typer.Option("--timeout", min=0.1)
    ] = DEFAULT_SUBPROCESS_TIMEOUT,
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
) -> None:
    config = common_config(
        database_url=database_url,
        dbos_system_database_url=dbos_system_database_url,
        generation_concurrency=generation_concurrency,
        scoring_concurrency=scoring_concurrency,
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


@app.command("repair")
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
        float, typer.Option("--score-timeout", min=0.1)
    ] = DEFAULT_SUBPROCESS_TIMEOUT,
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
) -> None:
    config = common_config(
        database_url=database_url,
        dbos_system_database_url=dbos_system_database_url,
        generation_concurrency=generation_concurrency,
        scoring_concurrency=scoring_concurrency,
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
            or plan.generation_retry_jobs
            or plan.pending_scoring_prediction_ids
            or plan.stranded_scoring
            or plan.scoring_retry_prediction_ids
        ),
        operator_log=operator_log,
    )


@app.command()
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
) -> None:
    config = common_config(
        database_url=database_url,
        dbos_system_database_url=None,
        generation_concurrency=DEFAULT_GENERATION_CONCURRENCY,
        scoring_concurrency=DEFAULT_SCORING_CONCURRENCY,
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


@app.command()
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
) -> None:
    config = common_config(
        database_url=database_url,
        dbos_system_database_url=dbos_system_database_url,
        generation_concurrency=generation_concurrency,
        scoring_concurrency=scoring_concurrency,
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


if __name__ == "__main__":
    configure_multiprocessing()
    logging.getLogger("dspy").setLevel(logging.WARNING)
    app()
