from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import random
import re
import statistics
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Protocol, cast

import psycopg
import typer
from datasets import load_dataset  # type: ignore[import-not-found]
from dbos import DBOS, DBOSConfig, SetWorkflowID
from psycopg.types.json import Jsonb
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
)
from rich import box
from rich.console import Console, Group
from rich.table import Table

import dspy
from dr_dspy.code_eval import extract_dspy_code, run_python_check
from dr_dspy.event_log import DATABASE_URL_ENV
from dr_dspy.lm_logging import LoggingCallableLM
from dr_dspy.openrouter_lm import OPENROUTER_API_KEY_ENV, LoggingOpenRouterLM
from dr_dspy.run_metadata import FieldSignature
from dr_dspy.runtime import configure_multiprocessing, load_env_file
from dspy.signatures.signature import make_signature

# Configuration

SCRIPT_KIND = "humaneval_eval_only_dbos_v0"
DBOS_APP_NAME = "dr-dspy-humaneval-eval-only"
DBOS_SYSTEM_DATABASE_URL_ENV = "DBOS_SYSTEM_DATABASE_URL"
GENERATION_QUEUE_NAME = "dr_dspy_humaneval_generation"
SCORING_QUEUE_NAME = "dr_dspy_humaneval_scoring"
DEFAULT_GENERATION_CONCURRENCY = 200
DEFAULT_SCORING_CONCURRENCY = 32
DEFAULT_SCORE_ENQUEUE_LIMIT = 1000
DEFAULT_WORKER_MONITOR_INTERVAL_SECONDS = 5.0
DEFAULT_WORKER_MONITOR_SUMMARY_INTERVAL_SECONDS = 60.0
DEFAULT_SEED = 0
DEFAULT_SAMPLE_COUNT = 10
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TEMPERATURES = (DEFAULT_TEMPERATURE,)
DEFAULT_MAX_COMPLETION_TOKENS = 1000
DEFAULT_SUBPROCESS_TIMEOUT = 15.0
DEFAULT_COST_SIGNIFICANT_DIGITS = 6
PRICE_PER_THOUSAND_SAMPLE_MULTIPLIER = 1000.0
MAX_TRACE_SIZE = 10_000
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKER_LOG_ROOT = PACKAGE_ROOT / "logs"
DETAILED_WORKER_LOGGER_NAME = "dr_dspy.humaneval_eval_only_worker"
CONSOLE = Console(soft_wrap=True)
OPERATOR_TIMESTAMP_FORMAT = "%H:%M:%S"
DATASET_NAME = "evalplus/humanevalplus"
DATASET_SPLIT = "test"
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
        "model": "google/gemini-2.5-flash",
        "reasoning": {"enabled": False},
    },
    {
        "model": "openai/gpt-oss-20b",
        "reasoning": {"effort": "low"},
    },
    {
        "model": "openai/gpt-oss-120b",
        "reasoning": {"effort": "low"},
    },
    {
        "model": "openai/gpt-5-nano",
        "reasoning": {"effort": "minimal", "exclude": False},
    },
    {
        "model": "deepseek/deepseek-v4-flash",
        "reasoning": {"enabled": False},
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


class QueueSelection(StrEnum):
    GENERATION = "generation"
    SCORING = "scoring"
    BOTH = "both"


class DbosWorkflowStatus(StrEnum):
    PENDING = "PENDING"
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"
    MAX_RECOVERY_ATTEMPTS_EXCEEDED = "MAX_RECOVERY_ATTEMPTS_EXCEEDED"
    CANCELLED = "CANCELLED"
    ENQUEUED = "ENQUEUED"
    DELAYED = "DELAYED"


DBOS_ACTIVE_WORKFLOW_STATUSES = (
    DbosWorkflowStatus.ENQUEUED.value,
    DbosWorkflowStatus.PENDING.value,
    DbosWorkflowStatus.DELAYED.value,
)
DBOS_FAILED_WORKFLOW_STATUSES = (
    DbosWorkflowStatus.ERROR.value,
    DbosWorkflowStatus.CANCELLED.value,
    DbosWorkflowStatus.MAX_RECOVERY_ATTEMPTS_EXCEEDED.value,
)


class EvalDbosConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    database_url: StrictStr
    dbos_system_database_url: StrictStr
    generation_concurrency: StrictInt = DEFAULT_GENERATION_CONCURRENCY
    scoring_concurrency: StrictInt = DEFAULT_SCORING_CONCURRENCY


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: StrictStr
    reasoning: dict[str, Any] = Field(default_factory=dict)


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


class WorkerMonitorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    database_url: StrictStr
    dbos_system_database_url: StrictStr
    queue_selection: QueueSelection
    queue_names: tuple[StrictStr, ...]
    interval_seconds: StrictFloat
    summary_interval_seconds: StrictFloat


class WorkerQueueSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dbos_status_counts: dict[StrictStr, StrictInt] = Field(
        default_factory=dict
    )
    generation_status_counts: dict[StrictStr, StrictInt] = Field(
        default_factory=dict
    )
    scoring_status_counts: dict[StrictStr, StrictInt] = Field(
        default_factory=dict
    )

    @property
    def active_total(self) -> int:
        return sum(
            self.dbos_status_counts.get(status, 0)
            for status in DBOS_ACTIVE_WORKFLOW_STATUSES
        )

    @property
    def success_total(self) -> int:
        return self.dbos_status_counts.get(
            DbosWorkflowStatus.SUCCESS.value, 0
        )

    @property
    def failure_total(self) -> int:
        return sum(
            self.dbos_status_counts.get(status, 0)
            for status in DBOS_FAILED_WORKFLOW_STATUSES
        )


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
    code: StrictStr
    response_metadata: dict[str, Any] = Field(default_factory=dict)
    usage_metadata: dict[str, Any] = Field(default_factory=dict)
    provider_cost: float | None = None


class ScoringTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_id: StrictStr
    task_id: StrictStr
    code: StrictStr
    test: StrictStr
    entry_point: StrictStr


class ScoreResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_id: StrictStr
    score: float
    error: str | None


class AnalysisRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: StrictStr
    temperature: float
    task_id: StrictStr
    repetition_seed: StrictInt
    score: float
    provider_cost: float | None


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


class LmEventBuffer:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def put_event(self, event_type: str, **kwargs: Any) -> None:
        self.events.append({"event_type": event_type, **kwargs})

    def latest_response_metadata(self) -> dict[str, Any]:
        for event in reversed(self.events):
            if event["event_type"] == "lm.response":
                payload = event.get("payload")
                if isinstance(payload, dict):
                    response = payload.get("response")
                    if isinstance(response, dict):
                        return response
        return {}


def resolve_database_url(database_url: str | None) -> str:
    resolved = database_url or os.environ.get(DATABASE_URL_ENV)
    if not resolved:
        raise ValueError(
            f"--database-url or {DATABASE_URL_ENV} is required for this "
            "Postgres-only DBOS harness"
        )
    return resolved


def build_eval_dbos_config(
    *,
    database_url: str | None,
    dbos_system_database_url: str | None,
    generation_concurrency: int,
    scoring_concurrency: int,
) -> EvalDbosConfig:
    resolved_database_url = resolve_database_url(database_url)
    return EvalDbosConfig(
        database_url=resolved_database_url,
        dbos_system_database_url=(
            dbos_system_database_url
            or os.environ.get(DBOS_SYSTEM_DATABASE_URL_ENV)
            or resolved_database_url
        ),
        generation_concurrency=generation_concurrency,
        scoring_concurrency=scoring_concurrency,
    )


def build_dbos_config(config: EvalDbosConfig) -> DBOSConfig:
    return {
        "name": DBOS_APP_NAME,
        "system_database_url": config.dbos_system_database_url,
    }


def create_eval_schema(database_url: str) -> None:
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(EXPERIMENTS_TABLE_SQL)
            cur.execute(PREDICTIONS_TABLE_SQL)
            for statement in PREDICTION_INDEX_SQL:
                cur.execute(statement)


def register_eval_queues(config: EvalDbosConfig) -> None:
    DBOS.register_queue(
        GENERATION_QUEUE_NAME,
        worker_concurrency=config.generation_concurrency,
    )
    DBOS.register_queue(
        SCORING_QUEUE_NAME,
        worker_concurrency=config.scoring_concurrency,
    )


def queue_names_for_selection(selection: QueueSelection) -> tuple[str, ...]:
    if selection is QueueSelection.GENERATION:
        return (GENERATION_QUEUE_NAME,)
    if selection is QueueSelection.SCORING:
        return (SCORING_QUEUE_NAME,)
    return (GENERATION_QUEUE_NAME, SCORING_QUEUE_NAME)


def sanitize_log_run_name(run_name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "-", run_name.strip()).strip(
        "-."
    )
    return sanitized or "worker"


def default_worker_log_path(
    *,
    run_name: str,
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
        DEFAULT_WORKER_LOG_ROOT / sanitize_log_run_name(run_name) / filename
    )


def resolve_worker_log_path(
    *,
    run_name: str | None,
    queue: QueueSelection,
    log_file: Path | None,
) -> Path:
    if log_file is not None:
        return log_file
    resolved_run_name = run_name or f"worker-{queue.value}"
    return default_worker_log_path(run_name=resolved_run_name, queue=queue)


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
    database_url: str, prediction_id: str, use_mock_lm: bool = False
) -> str:
    try:
        result = generate_prediction_step(
            database_url, prediction_id, use_mock_lm
        )
        record_generation_success_step(database_url, result)
        return "generated"
    except Exception as e:
        record_generation_error_step(database_url, prediction_id, repr(e))
        return "generation_error"


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


def listen_to_selected_queue(selection: QueueSelection) -> None:
    DBOS.listen_queues(list(queue_names_for_selection(selection)))


def configure_dbos_runtime(
    config: EvalDbosConfig,
    *,
    queue: QueueSelection | None = None,
    consume_queues: bool = True,
) -> None:
    DBOS(config=build_dbos_config(config))
    if queue is not None:
        listen_to_selected_queue(queue)
    elif not consume_queues:
        DBOS.listen_queues([])
    DBOS.launch()
    register_eval_queues(config)


def stable_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


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


def build_humaneval_samples(
    *,
    seed: int,
    sample_count: int,
) -> list[HumanEvalSample]:
    dataset = cast(
        HumanEvalDataset,
        load_dataset(DATASET_NAME, split=DATASET_SPLIT),
    )
    rows: list[HumanEvalRow] = [
        dataset[index] for index in range(len(dataset))
    ]
    return build_humaneval_samples_from_rows(
        rows, seed=seed, sample_count=sample_count
    )


def build_humaneval_samples_from_rows(
    rows: Sequence[HumanEvalRow],
    *,
    seed: int,
    sample_count: int,
) -> list[HumanEvalSample]:
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    samples: list[HumanEvalSample] = []
    for sample_index, row_index in enumerate(indices[:sample_count]):
        row = rows[row_index]
        samples.append(
            HumanEvalSample(
                task_id=str(row["task_id"]),
                sample_index=sample_index,
                prompt=str(row["prompt"]),
                test=str(row["test"]),
                entry_point=str(row["entry_point"]),
            )
        )
    return samples


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
    with psycopg.connect(database_url) as conn:
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
    with psycopg.connect(database_url) as conn:
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
    with psycopg.connect(database_url) as conn:
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
    with psycopg.connect(database_url) as conn:
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
    with psycopg.connect(database_url) as conn:
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
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE dr_dspy_eval_predictions
                SET
                    generation_status = 'generated',
                    generation_error = NULL,
                    raw_code = %s,
                    response_metadata = %s,
                    usage_metadata = %s,
                    provider_cost = %s,
                    updated_at = now(),
                    generated_at = now()
                WHERE prediction_id = %s
                """,
                (
                    result.code,
                    Jsonb(result.response_metadata),
                    Jsonb(result.usage_metadata),
                    result.provider_cost,
                    result.prediction_id,
                ),
            )


def record_generation_error(
    database_url: str, prediction_id: str, error: str
) -> None:
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE dr_dspy_eval_predictions
                SET
                    generation_status = 'generation_error',
                    generation_error = %s,
                    updated_at = now()
                WHERE prediction_id = %s
                """,
                (error, prediction_id),
            )


def fetch_scoring_target(
    database_url: str, prediction_id: str
) -> ScoringTarget:
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    prediction_id,
                    task_id,
                    raw_code,
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
            f"prediction_id has no generated code: {prediction_id}"
        )
    return ScoringTarget(
        prediction_id=row[0],
        task_id=row[1],
        code=row[2],
        test=row[3],
        entry_point=row[4],
    )


def fetch_scoreable_prediction_ids(
    database_url: str,
    *,
    experiment_name: str,
    limit: int,
) -> list[str]:
    with psycopg.connect(database_url) as conn:
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


def mark_scoring_queued(
    database_url: str, prediction_ids: Sequence[str]
) -> None:
    if not prediction_ids:
        return
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE dr_dspy_eval_predictions
                SET
                    scoring_status = 'queued',
                    scoring_error = NULL,
                    updated_at = now()
                WHERE prediction_id = ANY(%s)
                """,
                (list(prediction_ids),),
            )


def mark_scoring_started(database_url: str, prediction_id: str) -> None:
    with psycopg.connect(database_url) as conn:
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
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE dr_dspy_eval_predictions
                SET
                    scoring_status = 'scored',
                    score = %s,
                    scoring_error = %s,
                    updated_at = now(),
                    scored_at = now()
                WHERE prediction_id = %s
                """,
                (result.score, result.error, result.prediction_id),
            )


def record_score_error(
    database_url: str, prediction_id: str, error: str
) -> None:
    with psycopg.connect(database_url) as conn:
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
    result = run_python_check(
        code=target.code,
        test=target.test,
        entry_point=target.entry_point,
        timeout=timeout,
    )
    return ScoreResult(
        prediction_id=target.prediction_id,
        score=result.score,
        error=result.error,
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


def build_generation_lm(
    job: PredictionJob,
    *,
    use_mock_lm: bool,
    event_buffer: LmEventBuffer,
    client: Any = None,
) -> dspy.BaseLM:
    if use_mock_lm:
        return LoggingCallableLM(
            mock_solver,
            log=event_buffer.put_event,
            model="callable/mock",
        )
    if not os.environ.get(OPENROUTER_API_KEY_ENV) and client is None:
        raise ValueError(f"{OPENROUTER_API_KEY_ENV} is not set")
    return LoggingOpenRouterLM(
        job.model,
        log=event_buffer.put_event,
        client=client,
        cache=False,
        reasoning=job.reasoning,
        max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
        temperature=job.temperature,
    )


def usage_metadata_from_response(
    response_metadata: Mapping[str, Any]
) -> dict[str, Any]:
    usage = response_metadata.get("usage")
    return dict(usage) if isinstance(usage, Mapping) else {}


def provider_cost_from_response(
    response_metadata: Mapping[str, Any]
) -> float | None:
    for key in ("cost", "total_cost"):
        value = response_metadata.get(key)
        if isinstance(value, int | float):
            return float(value)
    usage = response_metadata.get("usage")
    if isinstance(usage, Mapping):
        value = usage.get("cost")
        if isinstance(value, int | float):
            return float(value)
    return None


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
    with dspy.context(
        lm=lm,
        callbacks=[],
        track_usage=True,
        max_trace_size=MAX_TRACE_SIZE,
    ):
        pred = dspy.Predict(Solve)(prompt=job.prompt)
    response_metadata = event_buffer.latest_response_metadata()
    return GenerationResult(
        prediction_id=job.prediction_id,
        code=extract_dspy_code(pred),
        response_metadata=response_metadata,
        usage_metadata=usage_metadata_from_response(response_metadata),
        provider_cost=provider_cost_from_response(response_metadata),
    )


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


def enqueue_generation_jobs(
    database_url: str, jobs: Sequence[PredictionJob], *, use_mock_lm: bool
) -> None:
    for job in jobs:
        workflow_id = f"generate:{job.prediction_id}"
        with SetWorkflowID(workflow_id):
            DBOS.enqueue_workflow(
                GENERATION_QUEUE_NAME,
                generate_prediction_workflow,
                database_url,
                job.prediction_id,
                use_mock_lm,
            )


def enqueue_score_jobs(
    database_url: str, prediction_ids: Sequence[str], *, timeout: float
) -> None:
    for prediction_id in prediction_ids:
        workflow_id = f"score:{prediction_id}"
        with SetWorkflowID(workflow_id):
            DBOS.enqueue_workflow(
                SCORING_QUEUE_NAME,
                score_prediction_workflow,
                database_url,
                prediction_id,
                timeout,
            )


def fetch_analysis_records(
    database_url: str, *, experiment_name: str
) -> list[AnalysisRecord]:
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    model,
                    temperature,
                    task_id,
                    repetition_seed,
                    score,
                    provider_cost
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
        )
        for row in rows
    ]


def variance_or_none(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    return statistics.variance(values)


def average_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return statistics.fmean(values)


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
            )
        )
    return summaries


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"


def format_cost(value: float | None) -> str:
    if value is None:
        return ""
    if value == 0:
        return "0"
    decimals = max(
        DEFAULT_COST_SIGNIFICANT_DIGITS
        - math.floor(math.log10(abs(value)))
        - 1,
        0,
    )
    return f"{value:.{decimals}f}".rstrip("0").rstrip(".")


def price_per_thousand_samples(value: float | None) -> float | None:
    if value is None:
        return None
    return value * PRICE_PER_THOUSAND_SAMPLE_MULTIPLIER


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
        "Avg Price/1k Samples | Avg Perf | Price Var | Perf Var | Rep Var |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            "| {model} | {temperature} | {sample_count} | {scored_count} | "
            "{total_price} | {avg_price_per_sample} | {avg_performance} | "
            "{price_variance} | {performance_variance} | "
            "{avg_repetition_variance} |".format(
                model=summary.model,
                temperature=format_float(summary.temperature),
                sample_count=summary.sample_count,
                scored_count=summary.scored_count,
                total_price=format_cost(summary.total_price),
                avg_price_per_sample=format_cost(
                    price_per_thousand_samples(
                        summary.avg_price_per_sample
                    )
                ),
                avg_performance=format_float(summary.avg_performance),
                price_variance=format_float(summary.price_variance),
                performance_variance=format_float(
                    summary.performance_variance
                ),
                avg_repetition_variance=format_float(
                    summary.avg_repetition_variance
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
    )
    performance_table.add_column("Model", min_width=28, overflow="fold")
    performance_table.add_column("Temp", justify="right")
    performance_table.add_column("Samples", justify="right")
    performance_table.add_column("Scored", justify="right")
    performance_table.add_column("Avg Perf", justify="right")

    cost_table = Table(
        title="Cost",
        box=box.SIMPLE_HEAVY,
        show_lines=False,
    )
    cost_table.add_column("Model", min_width=28, overflow="fold")
    cost_table.add_column("Temp", justify="right")
    cost_table.add_column("Total $", justify="right")
    cost_table.add_column("Avg $/1k Samples", justify="right")

    variance_table = Table(
        title="Variance",
        box=box.SIMPLE_HEAVY,
        show_lines=False,
    )
    variance_table.add_column("Model", min_width=28, overflow="fold")
    variance_table.add_column("Temp", justify="right")
    variance_table.add_column("Price Var", justify="right")
    variance_table.add_column("Perf Var", justify="right")
    variance_table.add_column("Rep Var", justify="right")

    for summary in summaries:
        performance_table.add_row(
            summary.model,
            format_float(summary.temperature),
            str(summary.sample_count),
            str(summary.scored_count),
            format_float(summary.avg_performance),
        )
        cost_table.add_row(
            summary.model,
            format_float(summary.temperature),
            format_cost(summary.total_price),
            format_cost(
                price_per_thousand_samples(summary.avg_price_per_sample)
            ),
        )
        variance_table.add_row(
            summary.model,
            format_float(summary.temperature),
            format_float(summary.price_variance),
            format_float(summary.performance_variance),
            format_float(summary.avg_repetition_variance),
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


def fetch_dbos_status_counts(
    dbos_system_database_url: str, queue_names: Sequence[str]
) -> dict[str, int]:
    with psycopg.connect(dbos_system_database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT status, COUNT(*)
                FROM dbos.workflow_status
                WHERE queue_name = ANY(%s)
                GROUP BY status
                """,
                (list(queue_names),),
            )
            rows = cur.fetchall()
    return {row[0]: row[1] for row in rows}


def fetch_prediction_phase_counts(
    database_url: str, status_column: str
) -> dict[str, int]:
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            if status_column == "generation_status":
                cur.execute(
                    """
                    SELECT generation_status, COUNT(*)
                    FROM dr_dspy_eval_predictions
                    GROUP BY generation_status
                    """
                )
            elif status_column == "scoring_status":
                cur.execute(
                    """
                    SELECT scoring_status, COUNT(*)
                    FROM dr_dspy_eval_predictions
                    GROUP BY scoring_status
                    """
                )
            else:
                raise ValueError(f"unsupported status column: {status_column}")
            rows = cur.fetchall()
    return {row[0]: row[1] for row in rows}


def fetch_worker_queue_snapshot(
    config: WorkerMonitorConfig,
) -> WorkerQueueSnapshot:
    generation_counts: dict[str, int] = {}
    scoring_counts: dict[str, int] = {}
    if config.queue_selection in (
        QueueSelection.GENERATION,
        QueueSelection.BOTH,
    ):
        generation_counts = fetch_prediction_phase_counts(
            config.database_url, "generation_status"
        )
    if config.queue_selection in (
        QueueSelection.SCORING,
        QueueSelection.BOTH,
    ):
        scoring_counts = fetch_prediction_phase_counts(
            config.database_url, "scoring_status"
        )
    return WorkerQueueSnapshot(
        dbos_status_counts=fetch_dbos_status_counts(
            config.dbos_system_database_url,
            config.queue_names,
        ),
        generation_status_counts=generation_counts,
        scoring_status_counts=scoring_counts,
    )


def count_for_status(counts: Mapping[str, int], status: str) -> int:
    return int(counts.get(status, 0))


def worker_monitor_counts(
    snapshot: WorkerQueueSnapshot,
    *,
    completed_since_start: int,
    failures_since_start: int,
) -> dict[str, int]:
    counts = {
        "active": snapshot.active_total,
        "enqueued": count_for_status(
            snapshot.dbos_status_counts,
            DbosWorkflowStatus.ENQUEUED.value,
        ),
        "pending": count_for_status(
            snapshot.dbos_status_counts,
            DbosWorkflowStatus.PENDING.value,
        ),
        "delayed": count_for_status(
            snapshot.dbos_status_counts,
            DbosWorkflowStatus.DELAYED.value,
        ),
        "completed": completed_since_start,
        "errors": failures_since_start,
        "gen_pending": count_for_phase_status(
            snapshot.generation_status_counts, "pending"
        ),
        "gen_started": count_for_phase_status(
            snapshot.generation_status_counts, "started"
        ),
        "gen_done": count_for_phase_status(
            snapshot.generation_status_counts, "generated"
        ),
        "gen_errors": count_for_phase_status(
            snapshot.generation_status_counts, "generation_error"
        ),
        "score_pending": count_for_phase_status(
            snapshot.scoring_status_counts, "pending"
        ),
        "score_queued": count_for_phase_status(
            snapshot.scoring_status_counts, "queued"
        ),
        "score_started": count_for_phase_status(
            snapshot.scoring_status_counts, "started"
        ),
        "score_done": count_for_phase_status(
            snapshot.scoring_status_counts, "scored"
        ),
        "score_errors": count_for_phase_status(
            snapshot.scoring_status_counts, "score_error"
        ),
    }
    return counts


def count_for_phase_status(counts: Mapping[str, int], status: str) -> int:
    if not counts:
        return -1
    return int(counts.get(status, 0))


def format_worker_count(value: int, *, width: int) -> str:
    if value < 0:
        return f"{'-':>{width}}"
    return f"{value:>{width}}"


def worker_monitor_line(
    snapshot: WorkerQueueSnapshot,
    *,
    was_active: bool | None,
    initial_success_total: int,
    initial_failure_total: int,
    force_summary: bool,
) -> str | None:
    is_active = snapshot.active_total > 0
    changed_state = was_active is None or is_active != was_active
    if not changed_state and not (force_summary and is_active):
        return None

    completed_since_start = max(
        snapshot.success_total - initial_success_total,
        0,
    )
    failures_since_start = max(
        snapshot.failure_total - initial_failure_total,
        0,
    )
    counts = worker_monitor_counts(
        snapshot,
        completed_since_start=completed_since_start,
        failures_since_start=failures_since_start,
    )
    state = "Queue Active" if is_active else "Queue Empty"
    return (
        f"{state:<12} | "
        f"active={format_worker_count(counts['active'], width=4)} | "
        f"enqueued={format_worker_count(counts['enqueued'], width=4)} | "
        f"pending={format_worker_count(counts['pending'], width=4)} | "
        f"delayed={format_worker_count(counts['delayed'], width=4)} | "
        f"completed={format_worker_count(counts['completed'], width=4)} | "
        f"errors={format_worker_count(counts['errors'], width=4)} | "
        "gen "
        f"pend={format_worker_count(counts['gen_pending'], width=4)} "
        f"start={format_worker_count(counts['gen_started'], width=4)} "
        f"done={format_worker_count(counts['gen_done'], width=4)} "
        f"err={format_worker_count(counts['gen_errors'], width=4)} | "
        "score "
        f"pend={format_worker_count(counts['score_pending'], width=4)} "
        f"queue={format_worker_count(counts['score_queued'], width=4)} "
        f"start={format_worker_count(counts['score_started'], width=4)} "
        f"done={format_worker_count(counts['score_done'], width=4)} "
        f"err={format_worker_count(counts['score_errors'], width=4)}"
    )


def worker_monitor_style(snapshot: WorkerQueueSnapshot) -> str:
    if snapshot.active_total > 0:
        return "green"
    return "yellow"


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


def run_worker_monitor(
    config: WorkerMonitorConfig, stop_event: threading.Event
) -> None:
    was_active: bool | None = None
    initial_success_total: int | None = None
    initial_failure_total: int | None = None
    last_summary_at = 0.0
    last_error: str | None = None
    while not stop_event.is_set():
        try:
            snapshot = fetch_worker_queue_snapshot(config)
            if initial_success_total is None:
                initial_success_total = snapshot.success_total
                initial_failure_total = snapshot.failure_total
            force_summary = (
                snapshot.active_total > 0
                and time.monotonic() - last_summary_at
                >= config.summary_interval_seconds
            )
            line = worker_monitor_line(
                snapshot,
                was_active=was_active,
                initial_success_total=initial_success_total,
                initial_failure_total=initial_failure_total or 0,
                force_summary=force_summary,
            )
            if line is not None:
                operator_log(line, style=worker_monitor_style(snapshot))
                last_summary_at = time.monotonic()
            was_active = snapshot.active_total > 0
            last_error = None
        except Exception as e:
            error = repr(e)
            emit_worker_detail_log(
                "worker_monitor_error",
                {"error": error},
            )
            if error != last_error:
                operator_log(
                    f"worker monitor error: {error}; retrying",
                    style="red",
                )
                last_error = error
        stop_event.wait(config.interval_seconds)


def start_worker_monitor(
    config: WorkerMonitorConfig, stop_event: threading.Event
) -> threading.Thread:
    thread = threading.Thread(
        target=run_worker_monitor,
        args=(config, stop_event),
        name=f"worker-monitor-{config.queue_selection.value}",
        daemon=True,
    )
    thread.start()
    return thread


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
    with psycopg.connect(database_url) as conn:
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
    table = Table(title=title, box=box.SIMPLE_HEAVY, show_lines=False)
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
        },
    )
    inserted = insert_prediction_jobs(config.database_url, jobs)
    configure_dbos_runtime(config, consume_queues=False)
    enqueue_generation_jobs(
        config.database_url, jobs, use_mock_lm=mock_generation
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
    mark_scoring_queued(config.database_url, prediction_ids)
    configure_dbos_runtime(config, consume_queues=False)
    enqueue_score_jobs(
        config.database_url,
        prediction_ids,
        timeout=timeout,
    )
    operator_log(
        enqueue_scores_line(
            experiment_name=experiment_name,
            selected_count=len(prediction_ids),
            limit=limit,
            timeout=timeout,
        ),
        style=enqueue_scores_style(len(prediction_ids)),
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
    run_name: Annotated[
        str | None,
        typer.Option(
            "--run-name",
            help=(
                "Name for the default detailed log directory under logs/."
            ),
        ),
    ] = None,
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
) -> None:
    config = common_config(
        database_url=database_url,
        dbos_system_database_url=dbos_system_database_url,
        generation_concurrency=generation_concurrency,
        scoring_concurrency=scoring_concurrency,
    )
    create_eval_schema(config.database_url)
    resolved_log_file = resolve_worker_log_path(
        run_name=run_name,
        queue=queue,
        log_file=log_file,
    )
    configure_worker_file_logging(resolved_log_file)
    configure_dbos_runtime(config, queue=queue)
    selected_queue_names = queue_names_for_selection(queue)
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


if __name__ == "__main__":
    configure_multiprocessing()
    logging.getLogger("dspy").setLevel(logging.WARNING)
    app()
