from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import re
import resource
import statistics
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
from rich import box
from rich.console import Console, Group
from rich.table import Table

import dspy
from dr_dspy import analysis as shared_analysis
from dr_dspy import dbos_runtime as shared_dbos
from dr_dspy import dspy_runner as shared_dspy_runner
from dr_dspy import human_eval_sampling as shared_human_eval_sampling
from dr_dspy import worker_monitor as shared_worker_monitor
from dr_dspy.code_eval import (
    DEFAULT_CAPTURE_LIMIT_BYTES,
    extract_dspy_code,
)
from dr_dspy.human_eval import HumanEvalTask
from dr_dspy.lm_utils import (
    LmEventBuffer,
    ModelConfig,
    stable_json,
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


GENERATION_REPAIR_ERROR = "Reconciled from DBOS failed generation workflow."
SCORING_REPAIR_ERROR = (
    "Reconciled from missing or failed DBOS scoring workflow."
)


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


class TemperatureProbeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: StrictStr
    temperature: float
    accepted: bool
    error: str | None = None
    response_metadata: dict[str, Any] = Field(default_factory=dict)


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
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "-", name.strip()).strip("-.")
    return sanitized or "experiment"


def hashed_experiment_log_name(experiment_name: str) -> str:
    return (
        f"{sanitize_log_name(experiment_name)}-"
        f"{experiment_hash(experiment_name)}"
    )


def default_worker_log_path(
    *,
    experiment_name: str,
    queue: QueueSelection,
    now: datetime | None = None,
    pid: int | None = None,
) -> Path:
    resolved_now = now or datetime.now()
    resolved_pid = pid if pid is not None else os.getpid()
    filename = (
        f"{resolved_now:%Y%m%d-%H%M%S}-{queue.value}-pid"
        f"{resolved_pid}.log"
    )
    return (
        DEFAULT_WORKER_LOG_ROOT
        / hashed_experiment_log_name(experiment_name)
        / filename
    )


def resolve_worker_log_path(
    *,
    experiment_name: str,
    queue: QueueSelection,
    log_file: Path | None,
) -> Path:
    if log_file is not None:
        return log_file
    return default_worker_log_path(
        experiment_name=experiment_name, queue=queue
    )


def configure_worker_file_logging(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(DETAILED_WORKER_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    logger.addHandler(handler)
    return logger


def emit_worker_detail_log(event: str, payload: Mapping[str, Any]) -> None:
    logger = logging.getLogger(DETAILED_WORKER_LOGGER_NAME)
    if not logger.handlers:
        return
    logger.info(stable_json({"event": event, **payload}))


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
    use_mock_lm: bool = False,
    score_timeout: float = DEFAULT_SUBPROCESS_TIMEOUT,
) -> str:
    try:
        result = generate_prediction_step(
            database_url, prediction_id, use_mock_lm
        )
        record_generation_success_step(database_url, result)
    except Exception as e:
        record_generation_error_step(database_url, prediction_id, repr(e))
        return "generation_error"
    enqueue_score_job(
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
    except Exception as e:
        record_score_error_step(database_url, prediction_id, repr(e))
        return "score_error"


def stable_prediction_id(
    *,
    experiment_name: str,
    task_id: str,
    model: str,
    temperature: float,
    repetition_seed: int,
) -> str:
    digest = hashlib.sha256(
        stable_json(
            {
                "experiment_name": experiment_name,
                "task_id": task_id,
                "model": model,
                "temperature": temperature,
                "repetition_seed": repetition_seed,
            }
        ).encode("utf-8")
    ).hexdigest()
    return digest[:32]


def parse_temperatures(raw: str) -> list[float]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("at least one temperature is required")
    return [float(value) for value in values]


def parse_reasoning_json(raw: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("reasoning JSON must be an object")
    return value


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
    with connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT prediction_id
                FROM dr_dspy_eval_predictions
                WHERE
                    experiment_name = %s
                    AND generation_status = 'generated'
                    AND scoring_status IN ('pending', 'score_error')
                ORDER BY model, temperature, sample_index, repetition_seed
                LIMIT %s
                """,
                (experiment_name, limit),
            )
            rows = cur.fetchall()
    return [row[0] for row in rows]


def fetch_pending_scoring_prediction_ids(
    database_url: str,
    *,
    experiment_name: str,
    limit: int,
) -> list[str]:
    with connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT prediction_id
                FROM dr_dspy_eval_predictions
                WHERE
                    experiment_name = %s
                    AND generation_status = 'generated'
                    AND scoring_status = 'pending'
                ORDER BY model, temperature, sample_index, repetition_seed
                LIMIT %s
                """,
                (experiment_name, limit),
            )
            rows = cur.fetchall()
    return [row[0] for row in rows]


def fetch_score_error_prediction_ids(
    database_url: str,
    *,
    experiment_name: str,
    limit: int,
) -> list[str]:
    with connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT prediction_id
                FROM dr_dspy_eval_predictions
                WHERE
                    experiment_name = %s
                    AND generation_status = 'generated'
                    AND scoring_status = 'score_error'
                ORDER BY model, temperature, sample_index, repetition_seed
                LIMIT %s
                """,
                (experiment_name, limit),
            )
            rows = cur.fetchall()
    return [row[0] for row in rows]


def fetch_generation_error_prediction_jobs(
    database_url: str,
    *,
    experiment_name: str,
    limit: int,
) -> list[PredictionJob]:
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
                WHERE
                    experiment_name = %s
                    AND generation_status = 'generation_error'
                ORDER BY model, temperature, sample_index, repetition_seed
                LIMIT %s
                """,
                (experiment_name, limit),
            )
            rows = cur.fetchall()
    return [
        PredictionJob(
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
        for row in rows
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
    with connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    prediction_id,
                    model,
                    temperature,
                    task_id,
                    sample_index,
                    repetition_seed
                FROM dr_dspy_eval_predictions
                WHERE
                    experiment_name = %s
                    AND generation_status = 'started'
                ORDER BY model, temperature, sample_index, repetition_seed
                """,
                (experiment_name,),
            )
            app_rows = cur.fetchall()

    if not app_rows:
        return []

    prediction_ids = [row[0] for row in app_rows]
    workflow_ids = [
        generation_workflow_id(prediction_id)
        for prediction_id in prediction_ids
    ]
    with connect_db(dbos_system_database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT workflow_uuid, status
                FROM dbos.workflow_status
                WHERE
                    workflow_uuid = ANY(%s)
                    AND status = ANY(%s)
                """,
                (workflow_ids, list(DBOS_FAILED_WORKFLOW_STATUSES)),
            )
            dbos_rows = cur.fetchall()

    dbos_status_by_prediction_id = {
        workflow_uuid.removeprefix("generate:"): status
        for workflow_uuid, status in dbos_rows
    }
    return [
        GenerationRepairCandidate(
            prediction_id=row[0],
            model=row[1],
            temperature=row[2],
            task_id=row[3],
            sample_index=row[4],
            repetition_seed=row[5],
            dbos_status=dbos_status_by_prediction_id[row[0]],
        )
        for row in app_rows
        if row[0] in dbos_status_by_prediction_id
    ]


def mark_started_generations_as_repaired_errors(
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
                    generation_status = 'generation_error',
                    generation_error = %s,
                    updated_at = now()
                WHERE
                    prediction_id = ANY(%s)
                    AND generation_status = 'started'
                """,
                (
                    GENERATION_REPAIR_ERROR,
                    list(prediction_ids),
                ),
            )
            return cur.rowcount if cur.rowcount is not None else 0


def fetch_stranded_scoring_repair_candidates(
    database_url: str,
    *,
    dbos_system_database_url: str,
    experiment_name: str,
    limit: int,
) -> list[ScoringRepairCandidate]:
    with connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    prediction_id,
                    model,
                    temperature,
                    task_id,
                    sample_index,
                    repetition_seed,
                    scoring_status
                FROM dr_dspy_eval_predictions
                WHERE
                    experiment_name = %s
                    AND generation_status = 'generated'
                    AND scoring_status IN ('started', 'queued')
                ORDER BY model, temperature, sample_index, repetition_seed
                LIMIT %s
                """,
                (experiment_name, limit),
            )
            app_rows = cur.fetchall()

    if not app_rows:
        return []

    prediction_ids = [row[0] for row in app_rows]
    stable_workflow_ids = [
        score_workflow_id(prediction_id) for prediction_id in prediction_ids
    ]
    retry_workflow_suffixes = [
        f":{prediction_id}" for prediction_id in prediction_ids
    ]
    with connect_db(dbos_system_database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT workflow_uuid, status
                FROM dbos.workflow_status
                WHERE
                    workflow_uuid = ANY(%s)
                    OR workflow_uuid LIKE ANY(%s)
                """,
                (
                    stable_workflow_ids,
                    [
                        f"score-retry:%{suffix}"
                        for suffix in retry_workflow_suffixes
                    ],
                ),
            )
            dbos_rows = cur.fetchall()

    active_prediction_ids: set[str] = set()
    failed_status_by_prediction_id: dict[str, str] = {}
    seen_prediction_ids: set[str] = set()
    for workflow_uuid, status in dbos_rows:
        prediction_id = workflow_uuid.rsplit(":", 1)[-1]
        seen_prediction_ids.add(prediction_id)
        if status in DBOS_ACTIVE_WORKFLOW_STATUSES:
            active_prediction_ids.add(prediction_id)
        if status in DBOS_FAILED_WORKFLOW_STATUSES:
            failed_status_by_prediction_id[prediction_id] = status

    candidates: list[ScoringRepairCandidate] = []
    for row in app_rows:
        prediction_id = row[0]
        if prediction_id in active_prediction_ids:
            continue
        dbos_status = failed_status_by_prediction_id.get(prediction_id)
        if dbos_status is None and prediction_id in seen_prediction_ids:
            continue
        candidates.append(
            ScoringRepairCandidate(
                prediction_id=prediction_id,
                model=row[1],
                temperature=row[2],
                task_id=row[3],
                sample_index=row[4],
                repetition_seed=row[5],
                scoring_status=row[6],
                dbos_status=dbos_status or MISSING_DBOS_WORKFLOW_STATUS,
            )
        )
    return candidates


def mark_stranded_scoring_as_errors(
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
                    scoring_status = 'score_error',
                    scoring_error = %s,
                    updated_at = now()
                WHERE
                    prediction_id = ANY(%s)
                    AND scoring_status IN ('started', 'queued')
                """,
                (SCORING_REPAIR_ERROR, list(prediction_ids)),
            )
            return cur.rowcount if cur.rowcount is not None else 0


def mark_scoring_queued(
    database_url: str, prediction_ids: Sequence[str]
) -> int:
    if not prediction_ids:
        return 0
    with connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE dr_dspy_eval_predictions
                SET
                    scoring_status = 'queued',
                    scoring_error = NULL,
                    updated_at = now()
                WHERE prediction_id = ANY(%s)
                    AND scoring_status IN ('pending', 'score_error')
                """,
                (list(prediction_ids),),
            )
            return cur.rowcount if cur.rowcount is not None else 0


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
    mark_scoring_started(database_url, prediction_id)
    context = fetch_prediction_log_context(database_url, prediction_id)
    emit_prediction_log_event(
        "scoring_started",
        context,
        extra={"timeout": timeout},
    )
    target = fetch_scoring_target(database_url, prediction_id)
    return score_generated_code(target, timeout=timeout)


@DBOS.step()
def record_score_success_step(
    database_url: str, result: ScoreResult
) -> None:
    context = fetch_prediction_log_context(database_url, result.prediction_id)
    emit_prediction_log_event(
        "scoring_succeeded",
        context,
        extra={"score": result.score, "scoring_error": result.error},
    )
    record_score_success(database_url, result)


@DBOS.step()
def record_score_error_step(
    database_url: str, prediction_id: str, error: str
) -> None:
    context = fetch_prediction_log_context(database_url, prediction_id)
    emit_prediction_log_event(
        "scoring_failed",
        context,
        extra={"error": error},
    )
    record_score_error(database_url, prediction_id, error)


@DBOS.step()
def mark_scoring_queued_step(database_url: str, prediction_id: str) -> None:
    context = fetch_prediction_log_context(database_url, prediction_id)
    emit_prediction_log_event("scoring_enqueued", context)
    mark_scoring_queued(database_url, [prediction_id])


def mock_solver(
    messages: list[dict[str, Any]], _kwargs: dict[str, Any]
) -> dict[str, str]:
    text = "\n".join(str(message.get("content", "")) for message in messages)
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("def ") and "(" in stripped:
            signature = stripped.split(":", 1)[0]
            return {"code": f"{signature}:\n    return None\n"}
    return {"code": "def solution():\n    return None\n"}


def run_temperature_probe(
    *,
    model: str,
    reasoning: Mapping[str, Any],
    temperatures: Sequence[float],
    client: Any = None,
) -> list[TemperatureProbeResult]:
    results: list[TemperatureProbeResult] = []
    for temperature in temperatures:
        job = PredictionJob(
            prediction_id=stable_prediction_id(
                experiment_name="temperature-probe",
                task_id="probe",
                model=model,
                temperature=temperature,
                repetition_seed=0,
            ),
            experiment_name="temperature-probe",
            submission_id="probe",
            task_id="probe",
            sample_index=0,
            model=model,
            temperature=temperature,
            repetition_seed=0,
            prompt="Return exactly the word ok.",
            test="",
            entry_point="",
            reasoning=dict(reasoning),
        )
        event_buffer = LmEventBuffer()
        try:
            lm = build_generation_lm(
                job,
                use_mock_lm=False,
                event_buffer=event_buffer,
                client=client,
            )
            lm.forward(
                messages=[
                    {
                        "role": "user",
                        "content": "Return exactly the word ok.",
                    }
                ]
            )
        except Exception as e:
            results.append(
                TemperatureProbeResult(
                    model=model,
                    temperature=temperature,
                    accepted=False,
                    error=repr(e),
                )
            )
            continue
        results.append(
            TemperatureProbeResult(
                model=model,
                temperature=temperature,
                accepted=True,
                response_metadata=event_buffer.latest_response_metadata(),
            )
        )
    return results


@DBOS.step(retries_allowed=True, max_attempts=3, interval_seconds=2.0)
def generate_prediction_step(
    database_url: str, prediction_id: str, use_mock_lm: bool
) -> GenerationResult:
    mark_generation_started(database_url, prediction_id)
    job = fetch_prediction_job(database_url, prediction_id)
    emit_prediction_log_event(
        "generation_started",
        prediction_context_from_job(job),
        extra={"use_mock_lm": use_mock_lm},
    )
    return generate_code_for_job(job, use_mock_lm=use_mock_lm)


@DBOS.step()
def record_generation_success_step(
    database_url: str, result: GenerationResult
) -> None:
    context = fetch_prediction_log_context(database_url, result.prediction_id)
    emit_prediction_log_event(
        "generation_succeeded",
        context,
        extra={
            "provider_cost": result.provider_cost,
            "usage_metadata": result.usage_metadata,
        },
    )
    record_generation_success(database_url, result)


@DBOS.step()
def record_generation_error_step(
    database_url: str, prediction_id: str, error: str
) -> None:
    context = fetch_prediction_log_context(database_url, prediction_id)
    emit_prediction_log_event(
        "generation_failed",
        context,
        extra={"error": error},
    )
    record_generation_error(database_url, prediction_id, error)


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
    mock_generation: bool,
    repair_token: str | None = None,
) -> RepairApplyResult:
    resolved_repair_token = repair_token or uuid.uuid4().hex

    stranded_generations = fetch_started_generation_repair_candidates(
        config.database_url,
        dbos_system_database_url=config.dbos_system_database_url,
        experiment_name=experiment_name,
    )
    stranded_generations_marked = (
        mark_started_generations_as_repaired_errors(
            config.database_url,
            prediction_ids=[
                candidate.prediction_id
                for candidate in stranded_generations
            ],
        )
    )

    generation_retry_jobs = fetch_generation_error_prediction_jobs(
        config.database_url,
        experiment_name=experiment_name,
        limit=generation_limit,
    )
    configure_dbos_runtime(
        config, experiment_name=experiment_name, consume_queues=False
    )
    enqueue_generation_jobs(
        config.database_url,
        generation_retry_jobs,
        use_mock_lm=mock_generation,
        score_timeout=score_timeout,
        retry_token=resolved_repair_token,
    )
    generation_retries_reset = reset_generation_errors_for_retry(
        config.database_url,
        prediction_ids=[job.prediction_id for job in generation_retry_jobs],
    )

    stranded_scoring = fetch_stranded_scoring_repair_candidates(
        config.database_url,
        dbos_system_database_url=config.dbos_system_database_url,
        experiment_name=experiment_name,
        limit=scoring_limit,
    )
    stranded_scoring_marked = mark_stranded_scoring_as_errors(
        config.database_url,
        prediction_ids=[
            candidate.prediction_id for candidate in stranded_scoring
        ],
    )

    pending_scoring_prediction_ids = fetch_pending_scoring_prediction_ids(
        config.database_url,
        experiment_name=experiment_name,
        limit=scoring_limit,
    )
    enqueue_score_jobs(
        config.database_url,
        pending_scoring_prediction_ids,
        experiment_name=experiment_name,
        timeout=score_timeout,
    )
    pending_scoring_enqueued = mark_scoring_queued(
        config.database_url, pending_scoring_prediction_ids
    )

    scoring_retry_prediction_ids = fetch_score_error_prediction_ids(
        config.database_url,
        experiment_name=experiment_name,
        limit=scoring_limit,
    )
    enqueue_score_jobs(
        config.database_url,
        scoring_retry_prediction_ids,
        experiment_name=experiment_name,
        timeout=score_timeout,
        retry_token=resolved_repair_token,
    )
    scoring_retries_marked_queued = mark_scoring_queued(
        config.database_url, scoring_retry_prediction_ids
    )

    return RepairApplyResult(
        repair_token=resolved_repair_token,
        stranded_generations_marked=stranded_generations_marked,
        generation_retries_enqueued=len(generation_retry_jobs),
        generation_retries_reset=generation_retries_reset,
        stranded_scoring_marked=stranded_scoring_marked,
        pending_scoring_enqueued=pending_scoring_enqueued,
        scoring_retries_enqueued=len(scoring_retry_prediction_ids),
        scoring_retries_marked_queued=scoring_retries_marked_queued,
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
    grouped: dict[tuple[str, float], list[AnalysisRecord]] = {}
    for record in records:
        key = (record.model, record.temperature)
        grouped.setdefault(key, []).append(record)

    summaries: list[AnalysisSummary] = []
    for (model, temperature), group in sorted(grouped.items()):
        scores = [record.score for record in group]
        raw_compile_pass_count = sum(
            1 for record in group if record.raw_compile_ok is True
        )
        extracted_compile_pass_count = sum(
            1 for record in group if record.extracted_compile_ok is True
        )
        costs = [
            record.provider_cost
            for record in group
            if record.provider_cost is not None
        ]
        task_ids = {record.task_id for record in group}
        repetition_variances: list[float] = []
        by_task: dict[str, list[float]] = {}
        for record in group:
            by_task.setdefault(record.task_id, []).append(record.score)
        for task_scores in by_task.values():
            task_variance = variance_or_none(task_scores)
            if task_variance is not None:
                repetition_variances.append(task_variance)

        total_price = sum(costs) if costs else None
        avg_price_per_sample = (
            total_price / len(group) if total_price is not None else None
        )
        summaries.append(
            AnalysisSummary(
                model=model,
                temperature=temperature,
                sample_count=len(task_ids),
                scored_count=len(group),
                total_price=total_price,
                avg_price_per_sample=avg_price_per_sample,
                price_variance=variance_or_none(costs),
                avg_performance=statistics.fmean(scores),
                performance_variance=variance_or_none(scores),
                avg_repetition_variance=average_or_none(
                    repetition_variances
                ),
                raw_compile_pass_count=raw_compile_pass_count,
                extracted_compile_pass_count=extracted_compile_pass_count,
                extraction_lift=(
                    extracted_compile_pass_count - raw_compile_pass_count
                ),
            )
        )
    return summaries


def operator_timestamp(now: datetime | None = None) -> str:
    resolved_now = now or datetime.now()
    return resolved_now.strftime(OPERATOR_TIMESTAMP_FORMAT)


def timestamped_line(line: str, *, now: datetime | None = None) -> str:
    return f"{operator_timestamp(now)} | {line}"


def operator_log(
    line: str,
    *,
    style: str | None = None,
    now: datetime | None = None,
) -> None:
    CONSOLE.print(timestamped_line(line, now=now), style=style)


def analysis_markdown(
    *, experiment_name: str, summaries: Sequence[AnalysisSummary]
) -> str:
    lines = [
        f"# Eval Analysis: {experiment_name}",
        "",
        "| Model | Temp | Samples | Scored | Total Price | "
        "Avg Price/1k Samples | Avg Perf | Raw Compile | "
        "Extracted Compile | Extraction Lift | Price Var | Perf Var | "
        "Rep Var |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    total_price_values = [summary.total_price for summary in summaries]
    total_price_sum = sum_present_float(total_price_values)
    total_prices = format_cost_column(
        [*total_price_values, total_price_sum]
        if summaries
        else total_price_values
    )
    row_total_prices = total_prices[: len(summaries)]
    prices_per_thousand_samples = format_cost_column(
        [
            price_per_thousand_samples(summary.avg_price_per_sample)
            for summary in summaries
        ]
    )
    for summary, total_price, price_per_thousand in zip(
        summaries,
        row_total_prices,
        prices_per_thousand_samples,
        strict=True,
    ):
        lines.append(
            "| {model} | {temperature} | {sample_count} | {scored_count} | "
            "{total_price} | {avg_price_per_sample} | {avg_performance} | "
            "{raw_compile_pass_count} | {extracted_compile_pass_count} | "
            "{extraction_lift} | {price_variance} | "
            "{performance_variance} | {avg_repetition_variance} |".format(
                model=summary.model,
                temperature=format_float(summary.temperature),
                sample_count=summary.sample_count,
                scored_count=summary.scored_count,
                total_price=total_price,
                avg_price_per_sample=price_per_thousand,
                avg_performance=format_float(summary.avg_performance),
                raw_compile_pass_count=summary.raw_compile_pass_count,
                extracted_compile_pass_count=(
                    summary.extracted_compile_pass_count
                ),
                extraction_lift=summary.extraction_lift,
                price_variance=format_float(summary.price_variance),
                performance_variance=format_float(
                    summary.performance_variance
                ),
                avg_repetition_variance=format_float(
                    summary.avg_repetition_variance
                ),
            )
        )
    if summaries:
        lines.append(
            (
                "| {model} |  | {sample_count} | {scored_count} | "
                "{total_price} |  |  | {raw_compile_pass_count} | "
                "{extracted_compile_pass_count} | {extraction_lift} | "
                " |  |  |"
            ).format(
                model=ANALYSIS_TOTAL_LABEL,
                sample_count=sum(
                    summary.sample_count for summary in summaries
                ),
                scored_count=sum(
                    summary.scored_count for summary in summaries
                ),
                total_price=total_prices[-1],
                raw_compile_pass_count=sum(
                    summary.raw_compile_pass_count for summary in summaries
                ),
                extracted_compile_pass_count=sum(
                    summary.extracted_compile_pass_count
                    for summary in summaries
                ),
                extraction_lift=sum(
                    summary.extraction_lift for summary in summaries
                ),
            )
        )
    return "\n".join(lines) + "\n"


def analysis_table(
    *, experiment_name: str, summaries: Sequence[AnalysisSummary]
) -> Group:
    performance_table = Table(
        title=f"Eval Analysis: {experiment_name}",
        box=box.SIMPLE_HEAVY,
        show_lines=False,
        row_styles=TABLE_ROW_STYLES,
    )
    performance_table.add_column("Model", min_width=28, overflow="fold")
    performance_table.add_column("Temp", justify="right")
    performance_table.add_column("Samples", justify="right")
    performance_table.add_column("Scored", justify="right")
    performance_table.add_column("Avg Perf", justify="right")
    performance_table.add_column("Raw Compile", justify="right")
    performance_table.add_column("Extracted Compile", justify="right")
    performance_table.add_column("Lift", justify="right")

    cost_table = Table(
        title="Cost",
        box=box.SIMPLE_HEAVY,
        show_lines=False,
        row_styles=TABLE_ROW_STYLES,
    )
    cost_table.add_column("Model", min_width=28, overflow="fold")
    cost_table.add_column("Temp", justify="right")
    cost_table.add_column("Total $", justify="right")
    cost_table.add_column("Avg $/1k Samples", justify="right")

    variance_table = Table(
        title="Variance",
        box=box.SIMPLE_HEAVY,
        show_lines=False,
        row_styles=TABLE_ROW_STYLES,
    )
    variance_table.add_column("Model", min_width=28, overflow="fold")
    variance_table.add_column("Temp", justify="right")
    variance_table.add_column("Price Var", justify="right")
    variance_table.add_column("Perf Var", justify="right")
    variance_table.add_column("Rep Var", justify="right")

    total_price_values = [summary.total_price for summary in summaries]
    total_price_sum = sum_present_float(total_price_values)
    temperatures = format_float_column(
        [summary.temperature for summary in summaries]
    )
    avg_performances = format_float_column(
        [summary.avg_performance for summary in summaries]
    )
    total_prices = format_cost_column(
        [*total_price_values, total_price_sum]
        if summaries
        else total_price_values
    )
    row_total_prices = total_prices[: len(summaries)]
    prices_per_thousand_samples = format_cost_column(
        [
            price_per_thousand_samples(summary.avg_price_per_sample)
            for summary in summaries
        ]
    )
    price_variances = format_float_column(
        [summary.price_variance for summary in summaries]
    )
    performance_variances = format_float_column(
        [summary.performance_variance for summary in summaries]
    )
    repetition_variances = format_float_column(
        [summary.avg_repetition_variance for summary in summaries]
    )

    for (
        summary,
        temperature,
        avg_performance,
        total_price,
        price_per_thousand,
        price_variance,
        performance_variance,
        repetition_variance,
    ) in zip(
        summaries,
        temperatures,
        avg_performances,
        row_total_prices,
        prices_per_thousand_samples,
        price_variances,
        performance_variances,
        repetition_variances,
        strict=True,
    ):
        performance_table.add_row(
            summary.model,
            temperature,
            str(summary.sample_count),
            str(summary.scored_count),
            avg_performance,
            str(summary.raw_compile_pass_count),
            str(summary.extracted_compile_pass_count),
            str(summary.extraction_lift),
        )
        cost_table.add_row(
            summary.model,
            temperature,
            total_price,
            price_per_thousand,
        )
        variance_table.add_row(
            summary.model,
            temperature,
            price_variance,
            performance_variance,
            repetition_variance,
        )
    if summaries:
        performance_table.add_row(
            ANALYSIS_TOTAL_LABEL,
            "",
            str(sum(summary.sample_count for summary in summaries)),
            str(sum(summary.scored_count for summary in summaries)),
            "",
            str(sum(summary.raw_compile_pass_count for summary in summaries)),
            str(
                sum(
                    summary.extracted_compile_pass_count
                    for summary in summaries
                )
            ),
            str(sum(summary.extraction_lift for summary in summaries)),
            style=TABLE_TOTAL_ROW_STYLE,
        )
        cost_table.add_row(
            ANALYSIS_TOTAL_LABEL,
            "",
            total_prices[-1],
            "",
            style=TABLE_TOTAL_ROW_STYLE,
        )
    return Group(performance_table, cost_table, variance_table)


def write_analysis_csv(
    summaries: Sequence[AnalysisSummary], *, csv_path: Path
) -> None:
    fieldnames = list(AnalysisSummary.model_fields)
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary.model_dump(mode="json"))


def enqueue_scores_line(
    *,
    experiment_name: str,
    selected_count: int,
    limit: int,
    timeout: float,
) -> str:
    return (
        f"{'Enqueue Scores':<14} | "
        f"selected={selected_count:>5} | "
        f"limit={limit:>5} | "
        f"timeout={timeout:>6.1f}s | "
        f"experiment={experiment_name}"
    )


def enqueue_scores_style(selected_count: int) -> str:
    if selected_count == 0:
        return "yellow"
    return "green"


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
    mode = "apply" if apply else "dry-run"
    return (
        f"{'Repair Plan':<14} | "
        f"gen_stranded={len(plan.stranded_generations):>5} | "
        f"gen_errors={len(plan.generation_retry_jobs):>5} | "
        f"score_pending={len(plan.pending_scoring_prediction_ids):>5} | "
        f"score_stranded={len(plan.stranded_scoring):>5} | "
        f"score_errors={len(plan.scoring_retry_prediction_ids):>5} | "
        f"mode={mode} | "
        f"experiment={experiment_name}"
    )


def repair_apply_line(
    *, experiment_name: str, result: RepairApplyResult
) -> str:
    return (
        f"{'Repair Apply':<14} | "
        f"gen_marked={result.stranded_generations_marked:>5} | "
        f"gen_retry={result.generation_retries_enqueued:>5} | "
        f"score_marked={result.stranded_scoring_marked:>5} | "
        f"score_pending={result.pending_scoring_enqueued:>5} | "
        f"score_retry={result.scoring_retries_enqueued:>5} | "
        f"token={result.repair_token} | "
        f"experiment={experiment_name}"
    )


def repair_plan_style(plan: RepairPlan, *, apply: bool) -> str:
    if apply:
        return "green"
    if (
        plan.stranded_generations
        or plan.generation_retry_jobs
        or plan.pending_scoring_prediction_ids
        or plan.stranded_scoring
        or plan.scoring_retry_prediction_ids
    ):
        return "cyan"
    return "yellow"


def fetch_status_counts(
    database_url: str, *, experiment_name: str | None
) -> list[dict[str, Any]]:
    where_clause = ""
    params: tuple[str, ...] = ()
    if experiment_name is not None:
        where_clause = "WHERE experiment_name = %s"
        params = (experiment_name,)

    query = f"""
        SELECT
            experiment_name,
            model,
            temperature,
            generation_status,
            scoring_status,
            COUNT(*) AS count
        FROM dr_dspy_eval_predictions
        {where_clause}
        GROUP BY
            experiment_name,
            model,
            temperature,
            generation_status,
            scoring_status
        ORDER BY
            experiment_name,
            model,
            temperature,
            generation_status,
            scoring_status
    """
    with connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
    return [
        {
            "experiment_name": row[0],
            "model": row[1],
            "temperature": row[2],
            "generation_status": row[3],
            "scoring_status": row[4],
            "count": row[5],
        }
        for row in rows
    ]


def status_counts_table(
    rows: Sequence[Mapping[str, Any]],
    *,
    experiment_name: str | None,
) -> Table:
    title = "Eval Status"
    if experiment_name is not None:
        title = f"Eval Status: {experiment_name}"
    table = Table(
        title=title,
        box=box.SIMPLE_HEAVY,
        show_lines=False,
        row_styles=TABLE_ROW_STYLES,
    )
    if experiment_name is None:
        table.add_column("Experiment", overflow="fold")
    table.add_column("Model", overflow="fold")
    table.add_column("Temp", justify="right")
    table.add_column("Generation", justify="center")
    table.add_column("Scoring", justify="center")
    table.add_column("Count", justify="right")
    for row in rows:
        values = []
        if experiment_name is None:
            values.append(str(row["experiment_name"]))
        values.extend(
            [
                str(row["model"]),
                format_float(row["temperature"]),
                str(row["generation_status"]),
                str(row["scoring_status"]),
                str(row["count"]),
            ]
        )
        table.add_row(*values)
    return table


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
    use_mock_lm: bool,
    event_buffer: LmEventBuffer,
    client: Any = None,
) -> dspy.BaseLM:
    return shared_dspy_runner.build_logged_lm(
        model=job.model,
        reasoning=job.reasoning,
        temperature=job.temperature,
        use_mock_lm=use_mock_lm,
        event_buffer=event_buffer,
        mock_solver=mock_solver,
        max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
        client=client,
    )


def generate_code_for_job(
    job: PredictionJob,
    *,
    use_mock_lm: bool,
    client: Any = None,
) -> GenerationResult:
    event_buffer = LmEventBuffer()
    lm = build_generation_lm(
        job,
        use_mock_lm=use_mock_lm,
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
    use_mock_lm: bool,
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
        use_mock_lm=use_mock_lm,
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
    mock_generation: Annotated[
        bool,
        typer.Option(
            "--mock-generation",
            help="Enqueue generation jobs that use the deterministic mock LM.",
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

    create_eval_schema(config.database_url)
    upsert_experiment(
        config.database_url,
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
    )
    inserted = insert_prediction_jobs(config.database_url, jobs)
    configure_dbos_runtime(
        config, experiment_name=experiment_name, consume_queues=False
    )
    enqueue_generation_jobs(
        config.database_url,
        jobs,
        use_mock_lm=mock_generation,
        score_timeout=score_timeout,
    )
    operator_log(
        f"inserted {inserted} new prediction rows",
        style="green" if inserted else "yellow",
    )
    operator_log(
        f"enqueued {len(jobs)} generation workflows",
        style="green" if jobs else "yellow",
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
    rows = fetch_status_counts(
        config.database_url, experiment_name=experiment_name
    )
    if not rows:
        operator_log("no prediction rows found", style="yellow")
        return
    CONSOLE.print(
        status_counts_table(rows, experiment_name=experiment_name)
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
    create_eval_schema(config.database_url)
    prediction_ids = fetch_scoreable_prediction_ids(
        config.database_url, experiment_name=experiment_name, limit=limit
    )
    configure_dbos_runtime(
        config, experiment_name=experiment_name, consume_queues=False
    )
    enqueue_score_jobs(
        config.database_url,
        prediction_ids,
        experiment_name=experiment_name,
        timeout=timeout,
    )
    mark_scoring_queued(config.database_url, prediction_ids)
    operator_log(
        enqueue_scores_line(
            experiment_name=experiment_name,
            selected_count=len(prediction_ids),
            limit=limit,
            timeout=timeout,
        ),
        style=enqueue_scores_style(len(prediction_ids)),
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
    mock_generation: Annotated[
        bool,
        typer.Option(
            "--mock-generation",
            help="Retry generation with the deterministic mock LM.",
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
) -> None:
    config = common_config(
        database_url=database_url,
        dbos_system_database_url=dbos_system_database_url,
        generation_concurrency=generation_concurrency,
        scoring_concurrency=scoring_concurrency,
    )
    plan = build_repair_plan(
        config.database_url,
        dbos_system_database_url=config.dbos_system_database_url,
        experiment_name=experiment_name,
        generation_limit=generation_limit,
        scoring_limit=scoring_limit,
    )
    operator_log(
        repair_plan_line(
            experiment_name=experiment_name,
            plan=plan,
            apply=apply,
        ),
        style=repair_plan_style(plan, apply=apply),
    )
    if apply:
        result = apply_repair(
            config,
            experiment_name=experiment_name,
            generation_limit=generation_limit,
            scoring_limit=scoring_limit,
            score_timeout=score_timeout,
            mock_generation=mock_generation,
        )
        operator_log(
            repair_apply_line(experiment_name=experiment_name, result=result),
            style="green",
        ),
    elif (
        plan.stranded_generations
        or plan.generation_retry_jobs
        or plan.pending_scoring_prediction_ids
        or plan.stranded_scoring
        or plan.scoring_retry_prediction_ids
    ):
        operator_log(
            "dry run only; rerun with --apply to reconcile statuses and "
            "enqueue fresh retry workflows",
            style="yellow",
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
    records = fetch_analysis_records(
        config.database_url, experiment_name=experiment_name
    )
    summaries = summarize_analysis_records(records)
    if markdown:
        typer.echo(
            analysis_markdown(
                experiment_name=experiment_name, summaries=summaries
            ),
            nl=False,
        )
    else:
        CONSOLE.print(
            analysis_table(
                experiment_name=experiment_name, summaries=summaries
            )
        )
    if csv_path is not None:
        write_analysis_csv(summaries, csv_path=csv_path)
        operator_log(f"wrote {csv_path}", style="green")


@app.command("temperature-probe")
def temperature_probe(
    model: Annotated[
        str,
        typer.Option("--model", help="OpenRouter model ID to probe."),
    ] = "openai/gpt-5.4-nano",
    temperatures: Annotated[
        str,
        typer.Option(
            "--temperatures",
            help="Comma-separated temperature values.",
        ),
    ] = "0.0,0.2,1.0",
    reasoning_json: Annotated[
        str,
        typer.Option(
            "--reasoning-json",
            help="OpenRouter reasoning JSON object.",
        ),
    ] = '{"enabled": false}',
    confirm_live: Annotated[
        bool,
        typer.Option(
            "--confirm-live",
            help="Required because this command calls OpenRouter.",
        ),
    ] = False,
) -> None:
    load_env_file()
    if not confirm_live:
        typer.echo(
            "temperature-probe calls OpenRouter; rerun with --confirm-live",
            err=True,
        )
        raise typer.Exit(2)
    results = run_temperature_probe(
        model=model,
        reasoning=parse_reasoning_json(reasoning_json),
        temperatures=parse_temperatures(temperatures),
    )
    for result in results:
        status_text = "accepted" if result.accepted else "rejected"
        typer.echo(
            f"{result.model} temp={result.temperature}: {status_text}"
        )
        if result.error:
            typer.echo(f"  error={result.error}")


@app.command("temperature-sweep")
def temperature_sweep(
    experiment_name: Annotated[
        str,
        typer.Option(
            "--experiment-name",
            help="Human-readable experiment key.",
        ),
    ],
    sample_count: Annotated[
        int, typer.Option("--sample-count", min=1)
    ] = 20,
    seed: Annotated[int, typer.Option("--seed")] = DEFAULT_SEED,
    temperatures: Annotated[
        str,
        typer.Option(
            "--temperatures",
            help="Comma-separated temperature values.",
        ),
    ] = "0.0,0.1,0.2,0.4",
    enqueue: Annotated[
        bool,
        typer.Option(
            "--enqueue",
            help="Actually write rows and enqueue generation workflows.",
        ),
    ] = False,
    mock_generation: Annotated[
        bool,
        typer.Option(
            "--mock-generation",
            help="Enqueue generation jobs that use the deterministic mock LM.",
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
) -> None:
    submit(
        experiment_name=experiment_name,
        sample_count=sample_count,
        seed=seed,
        temperatures=temperatures,
        repetitions=1,
        dry_run=not enqueue,
        mock_generation=mock_generation,
        database_url=database_url,
        dbos_system_database_url=dbos_system_database_url,
        generation_concurrency=generation_concurrency,
        scoring_concurrency=scoring_concurrency,
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
    file_limit = raise_open_file_limit(open_file_limit)
    operator_log(
        open_file_limit_line(file_limit),
        style=open_file_limit_style(file_limit),
    )
    config = common_config(
        database_url=database_url,
        dbos_system_database_url=dbos_system_database_url,
        generation_concurrency=generation_concurrency,
        scoring_concurrency=scoring_concurrency,
    )
    create_eval_schema(config.database_url)
    pool_config = configure_pooled_worker_runtime(
        config,
        experiment_name=experiment_name,
        queue=queue,
        raw_db_pool_max_size=db_pool_max_size,
    )
    operator_log(
        f"{'DB Pool':<14} | max_size={pool_config.max_size:>5} | "
        f"urls={len(DB_POOLS):>2}",
        style="cyan",
    )
    resolved_log_file = resolve_worker_log_path(
        experiment_name=experiment_name,
        queue=queue,
        log_file=log_file,
    )
    configure_worker_file_logging(resolved_log_file)
    selected_queue_names = queue_names_for_selection(
        queue, experiment_name=experiment_name
    )
    operator_log(
        f"worker listening on {queue.value} queue(s): "
        f"{', '.join(selected_queue_names)}",
        style="cyan",
    )
    operator_log(f"detailed worker log: {resolved_log_file}", style="cyan")

    stop_event = threading.Event()
    monitor_thread: threading.Thread | None = None
    if monitor:
        monitor_config = WorkerMonitorConfig(
            database_url=config.database_url,
            dbos_system_database_url=config.dbos_system_database_url,
            experiment_name=experiment_name,
            prediction_table=PREDICTION_TABLE_NAME,
            queue_selection=queue,
            queue_names=selected_queue_names,
            interval_seconds=monitor_interval,
            summary_interval_seconds=monitor_summary_interval,
        )
        monitor_thread = start_worker_monitor(monitor_config, stop_event)
        operator_log(
            "worker monitor enabled: "
            f"interval={monitor_interval}s, "
            f"summary_interval={monitor_summary_interval}s",
            style="cyan",
        )
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        operator_log("worker stopping", style="cyan")
    finally:
        stop_event.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=1.0)
        close_db_connection_pools()


if __name__ == "__main__":
    configure_multiprocessing()
    logging.getLogger("dspy").setLevel(logging.WARNING)
    app()
