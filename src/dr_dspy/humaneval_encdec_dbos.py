from __future__ import annotations

import hashlib
import json
import logging
import statistics
import threading
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Protocol, cast

import typer
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
from rich.console import Console
from rich.table import Table

import dspy
from dr_dspy import analysis as shared_analysis
from dr_dspy import dbos_runtime as shared_dbos
from dr_dspy import dspy_runner as shared_dspy_runner
from dr_dspy import eval_logging as shared_eval_logging
from dr_dspy import eval_repair as shared_eval_repair
from dr_dspy import eval_reporting as shared_eval_reporting
from dr_dspy import human_eval_sampling as shared_human_eval_sampling
from dr_dspy import worker_monitor as shared_worker_monitor
from dr_dspy.compression import CompressionMetric, compression_metrics
from dr_dspy.human_eval import HumanEvalTask
from dr_dspy.lm_utils import (
    LmEventBuffer,
    ModelConfig,
    provider_cost_from_response,
    stable_json,
    usage_metadata_from_response,
)
from dr_dspy.runtime import configure_multiprocessing, load_env_file
from dr_dspy.scoring import (
    GeneratedCodeScore,
    score_generated_code_for_humaneval,
)
from dr_dspy.signatures import FieldSignature
from dspy.signatures.signature import make_signature

DATABASE_URL_ENV = "DATABASE_URL"
SCRIPT_KIND = "humaneval_eval_only_encdec_dbos_v0"
DBOS_APP_NAME = "dr-dspy-humaneval-encdec-eval"
DBOS_SYSTEM_DATABASE_URL_ENV = "DBOS_SYSTEM_DATABASE_URL"
GENERATION_QUEUE_NAME = "dr_dspy_humaneval_encdec_generation"
SCORING_QUEUE_NAME = "dr_dspy_humaneval_encdec_scoring"
DEFAULT_GENERATION_CONCURRENCY = 64
DEFAULT_SCORING_CONCURRENCY = 32
DEFAULT_SAMPLE_COUNT = 10
DEFAULT_SEED = 0
DEFAULT_TEMPERATURE = 0.0
DEFAULT_REPETITIONS = 1
DEFAULT_MAX_COMPLETION_TOKENS = 2000
DEFAULT_SUBPROCESS_TIMEOUT = 15.0
DEFAULT_WORKER_OPEN_FILE_LIMIT = 8192
DEFAULT_WORKER_MONITOR_INTERVAL_SECONDS = 5.0
DEFAULT_WORKER_MONITOR_SUMMARY_INTERVAL_SECONDS = 5.0
DEFAULT_SCORE_ENQUEUE_LIMIT = 1000
DEFAULT_REPAIR_GENERATION_LIMIT = 1000
DEFAULT_REPAIR_SCORING_LIMIT = 1000
DATASET_NAME = "evalplus/humanevalplus"
DATASET_SPLIT = "test"
PREDICTION_TABLE_NAME = "dr_dspy_encdec_eval_predictions"
REPAIR_DIMENSION_COLUMNS = (
    "encoder_model",
    "decoder_model",
    "encoder_temperature",
    "decoder_temperature",
)
REPAIR_ORDER_COLUMNS = (
    "encoder_model",
    "decoder_model",
    "encoder_temperature",
    "decoder_temperature",
    "sample_index",
    "repetition_seed",
)
EXPERIMENT_QUEUE_HASH_LENGTH = 8
DEFAULT_WORKER_LOG_ROOT = Path(__file__).resolve().parents[1] / "logs"
DETAILED_WORKER_LOGGER_NAME = "dr_dspy.humaneval_encdec_worker"
OPERATOR_TIMESTAMP_FORMAT = "%H:%M:%S"
CONSOLE = Console(soft_wrap=True)

ENCODER_FIELDS = [
    FieldSignature(name="code", type=str, role=dspy.InputField()),
    FieldSignature(name="description", type=str, role=dspy.OutputField()),
]
DECODER_FIELDS = [
    FieldSignature(name="description", type=str, role=dspy.InputField()),
    FieldSignature(name="code", type=dspy.Code, role=dspy.OutputField()),
]
ENCODER_INSTRUCTIONS = (
    "Encode this Python function implementation into a complete lossless "
    "description. Preserve all behavior needed to reconstruct the code, but "
    "do not output Python code."
)
DECODER_INSTRUCTIONS = (
    "Decode the description into functional Python code. "
    "Output only Python code."
)

DEFAULT_MODEL_PAIRS: tuple[dict[str, Any], ...] = (
    {
        "encoder": {"model": "openai/gpt-5.1-codex-mini", "reasoning": {}},
        "decoder": {"model": "openai/gpt-5.1-codex-mini", "reasoning": {}},
    },
)

EXPERIMENTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS dr_dspy_encdec_eval_experiments (
    experiment_name TEXT PRIMARY KEY,
    script_kind     TEXT        NOT NULL,
    seed            INTEGER     NOT NULL,
    sample_count    INTEGER     NOT NULL,
    encoder_instruction TEXT    NOT NULL,
    decoder_instruction TEXT    NOT NULL,
    metadata        JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""

PREDICTIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS dr_dspy_encdec_eval_predictions (
    prediction_id        TEXT PRIMARY KEY,
    experiment_name      TEXT        NOT NULL
        REFERENCES dr_dspy_encdec_eval_experiments(experiment_name),
    script_kind          TEXT        NOT NULL,
    submission_id        TEXT        NOT NULL,
    task_id              TEXT        NOT NULL,
    sample_index         INTEGER     NOT NULL,
    encoder_model        TEXT        NOT NULL,
    decoder_model        TEXT        NOT NULL,
    encoder_temperature  DOUBLE PRECISION,
    decoder_temperature  DOUBLE PRECISION,
    repetition_seed      INTEGER     NOT NULL,
    prompt               TEXT        NOT NULL,
    canonical_solution   TEXT        NOT NULL,
    ground_truth_code    TEXT        NOT NULL,
    test                 TEXT        NOT NULL,
    entry_point          TEXT        NOT NULL,
    encoder_reasoning    JSONB       NOT NULL DEFAULT '{}'::jsonb,
    decoder_reasoning    JSONB       NOT NULL DEFAULT '{}'::jsonb,
    generation_status    TEXT        NOT NULL DEFAULT 'pending',
    generation_error     TEXT,
    encoded_description  TEXT,
    decoded_generation   TEXT,
    raw_generation       TEXT,
    encoder_response_metadata JSONB  NOT NULL DEFAULT '{}'::jsonb,
    decoder_response_metadata JSONB  NOT NULL DEFAULT '{}'::jsonb,
    encoder_usage_metadata    JSONB  NOT NULL DEFAULT '{}'::jsonb,
    decoder_usage_metadata    JSONB  NOT NULL DEFAULT '{}'::jsonb,
    encoder_provider_cost DOUBLE PRECISION,
    decoder_provider_cost DOUBLE PRECISION,
    provider_cost        DOUBLE PRECISION,
    scoring_status       TEXT        NOT NULL DEFAULT 'pending',
    score                DOUBLE PRECISION,
    scoring_error        TEXT,
    raw_code             TEXT,
    raw_compile_ok       BOOLEAN,
    raw_compile_error    TEXT,
    extraction_candidate_count INTEGER,
    selected_candidate_index INTEGER,
    extracted_compile_ok BOOLEAN,
    extracted_compile_error TEXT,
    extraction_error     TEXT,
    evaluation_function_names JSONB NOT NULL DEFAULT '[]'::jsonb,
    evaluation_total_cases INTEGER,
    evaluation_failure_count INTEGER,
    evaluation_status_counts JSONB NOT NULL DEFAULT '{}'::jsonb,
    compression_metrics  JSONB       NOT NULL DEFAULT '[]'::jsonb,
    best_compression_ratio DOUBLE PRECISION,
    best_compression_percent_reduction DOUBLE PRECISION,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    generated_at         TIMESTAMPTZ,
    scored_at            TIMESTAMPTZ,
    UNIQUE (
        experiment_name,
        task_id,
        encoder_model,
        decoder_model,
        encoder_temperature,
        decoder_temperature,
        repetition_seed
    )
)
"""

PREDICTION_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS "
    "idx_dr_dspy_encdec_eval_predictions_experiment "
    "ON dr_dspy_encdec_eval_predictions(experiment_name)",
    "CREATE INDEX IF NOT EXISTS "
    "idx_dr_dspy_encdec_eval_predictions_generation "
    "ON dr_dspy_encdec_eval_predictions(experiment_name, generation_status)",
    "CREATE INDEX IF NOT EXISTS idx_dr_dspy_encdec_eval_predictions_scoring "
    "ON dr_dspy_encdec_eval_predictions(experiment_name, scoring_status)",
    "CREATE INDEX IF NOT EXISTS idx_dr_dspy_encdec_eval_predictions_models "
    "ON dr_dspy_encdec_eval_predictions("
    "experiment_name, encoder_model, decoder_model)",
)


class EncDecPair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    encoder: ModelConfig
    decoder: ModelConfig


class HumanEvalDataset(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Mapping[str, Any]: ...


class EncDecSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: StrictStr
    sample_index: StrictInt
    prompt: StrictStr
    canonical_solution: StrictStr
    ground_truth_code: StrictStr
    test: StrictStr
    entry_point: StrictStr


class EncDecJob(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_id: StrictStr
    experiment_name: StrictStr
    submission_id: StrictStr
    task_id: StrictStr
    sample_index: StrictInt
    encoder_model: StrictStr
    decoder_model: StrictStr
    encoder_temperature: StrictFloat | None
    decoder_temperature: StrictFloat | None
    repetition_seed: StrictInt
    prompt: StrictStr
    canonical_solution: StrictStr
    ground_truth_code: StrictStr
    test: StrictStr
    entry_point: StrictStr
    encoder_reasoning: dict[str, Any] = Field(default_factory=dict)
    decoder_reasoning: dict[str, Any] = Field(default_factory=dict)

    def task(self) -> HumanEvalTask:
        return HumanEvalTask(
            task_id=self.task_id,
            prompt=self.prompt,
            canonical_solution=self.canonical_solution,
            test=self.test,
            entry_point=self.entry_point,
        )


class GenerationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_id: StrictStr
    encoded_description: StrictStr
    decoded_generation: StrictStr
    encoder_response_metadata: dict[str, Any] = Field(default_factory=dict)
    decoder_response_metadata: dict[str, Any] = Field(default_factory=dict)
    encoder_usage_metadata: dict[str, Any] = Field(default_factory=dict)
    decoder_usage_metadata: dict[str, Any] = Field(default_factory=dict)
    encoder_provider_cost: float | None = None
    decoder_provider_cost: float | None = None

    @property
    def provider_cost(self) -> float | None:
        costs = [
            cost
            for cost in (
                self.encoder_provider_cost,
                self.decoder_provider_cost,
            )
            if cost is not None
        ]
        return sum(costs) if costs else None


class ScoringTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_id: StrictStr
    task_id: StrictStr
    prompt: StrictStr
    canonical_solution: StrictStr
    ground_truth_code: StrictStr
    test: StrictStr
    entry_point: StrictStr
    encoded_description: StrictStr
    raw_generation: StrictStr

    def task(self) -> HumanEvalTask:
        return HumanEvalTask(
            task_id=self.task_id,
            prompt=self.prompt,
            canonical_solution=self.canonical_solution,
            test=self.test,
            entry_point=self.entry_point,
        )


class ScoreResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_id: StrictStr
    score: float
    error: str | None
    raw_code: str | None = None
    raw_compile_ok: bool
    raw_compile_error: str | None = None
    extraction_candidate_count: int
    selected_candidate_index: int | None = None
    extracted_compile_ok: bool
    extracted_compile_error: str | None = None
    extraction_error: str | None = None
    evaluation_function_names: list[str] = Field(default_factory=list)
    evaluation_total_cases: int | None = None
    evaluation_failure_count: int | None = None
    evaluation_status_counts: dict[str, int] = Field(default_factory=dict)
    compression_metrics: list[CompressionMetric] = Field(default_factory=list)
    best_compression_ratio: float | None = None
    best_compression_percent_reduction: float | None = None


class AnalysisRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    encoder_model: StrictStr
    decoder_model: StrictStr
    encoder_temperature: float | None
    decoder_temperature: float | None
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


EncodeCode = make_signature(
    {field.name: (field.type, field.role) for field in ENCODER_FIELDS},
    instructions=ENCODER_INSTRUCTIONS,
    signature_name="EncodeCode",
)
DecodeCode = make_signature(
    {field.name: (field.type, field.role) for field in DECODER_FIELDS},
    instructions=DECODER_INSTRUCTIONS,
    signature_name="DecodeCode",
)


def operator_log(message: str, *, style: str | None = None) -> None:
    shared_eval_logging.operator_log(
        CONSOLE,
        message,
        style=style,
        timestamp_format=OPERATOR_TIMESTAMP_FORMAT,
    )


def resolve_worker_log_path(
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


def configure_worker_file_logging(log_file: Path) -> logging.Logger:
    return shared_eval_logging.configure_worker_file_logging(
        log_file, logger_name=DETAILED_WORKER_LOGGER_NAME
    )


def emit_worker_detail_log(event: str, payload: Mapping[str, Any]) -> None:
    shared_eval_logging.emit_worker_detail_log(
        event, payload, logger_name=DETAILED_WORKER_LOGGER_NAME
    )


def prediction_context_from_job(
    job: EncDecJob,
) -> shared_eval_logging.PredictionLogContext:
    return shared_eval_logging.PredictionLogContext(
        prediction_id=job.prediction_id,
        experiment_name=job.experiment_name,
        task_id=job.task_id,
        sample_index=job.sample_index,
        repetition_seed=job.repetition_seed,
        dimensions={
            "encoder_model": job.encoder_model,
            "decoder_model": job.decoder_model,
            "encoder_temperature": job.encoder_temperature,
            "decoder_temperature": job.decoder_temperature,
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


def load_optional_env_file(env_file: Path | None) -> None:
    if env_file is None:
        load_env_file()
    else:
        load_env_file(env_file)


def parse_model_pairs(raw: str | None) -> list[EncDecPair]:
    if raw is None:
        return [EncDecPair(**pair) for pair in DEFAULT_MODEL_PAIRS]
    if raw.startswith("@"):
        raw = Path(raw[1:]).read_text(encoding="utf-8")
    value = json.loads(raw)
    if not isinstance(value, list):
        raise ValueError("--model-pairs-json must be a JSON list")
    return [EncDecPair(**item) for item in value]


def stable_prediction_id(
    *,
    experiment_name: str,
    task_id: str,
    encoder_model: str,
    decoder_model: str,
    encoder_temperature: float,
    decoder_temperature: float,
    repetition_seed: int,
) -> str:
    raw = stable_json(
        {
            "experiment_name": experiment_name,
            "task_id": task_id,
            "encoder_model": encoder_model,
            "decoder_model": decoder_model,
            "encoder_temperature": encoder_temperature,
            "decoder_temperature": decoder_temperature,
            "repetition_seed": repetition_seed,
        }
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_prediction_jobs(
    *,
    experiment_name: str,
    submission_id: str,
    samples: Sequence[EncDecSample],
    model_pairs: Sequence[EncDecPair],
    encoder_temperatures: Sequence[float],
    decoder_temperatures: Sequence[float],
    repetitions: int,
) -> list[EncDecJob]:
    jobs: list[EncDecJob] = []
    for sample in samples:
        for pair in model_pairs:
            for encoder_temperature in encoder_temperatures:
                for decoder_temperature in decoder_temperatures:
                    for repetition_seed in range(repetitions):
                        jobs.append(
                            EncDecJob(
                                prediction_id=stable_prediction_id(
                                    experiment_name=experiment_name,
                                    task_id=sample.task_id,
                                    encoder_model=pair.encoder.model,
                                    decoder_model=pair.decoder.model,
                                    encoder_temperature=encoder_temperature,
                                    decoder_temperature=decoder_temperature,
                                    repetition_seed=repetition_seed,
                                ),
                                experiment_name=experiment_name,
                                submission_id=submission_id,
                                task_id=sample.task_id,
                                sample_index=sample.sample_index,
                                encoder_model=pair.encoder.model,
                                decoder_model=pair.decoder.model,
                                encoder_temperature=encoder_temperature,
                                decoder_temperature=decoder_temperature,
                                repetition_seed=repetition_seed,
                                prompt=sample.prompt,
                                canonical_solution=sample.canonical_solution,
                                ground_truth_code=sample.ground_truth_code,
                                test=sample.test,
                                entry_point=sample.entry_point,
                                encoder_reasoning=dict(pair.encoder.reasoning),
                                decoder_reasoning=dict(pair.decoder.reasoning),
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
                INSERT INTO dr_dspy_encdec_eval_experiments (
                    experiment_name,
                    script_kind,
                    seed,
                    sample_count,
                    encoder_instruction,
                    decoder_instruction,
                    metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (experiment_name) DO UPDATE SET
                    script_kind = EXCLUDED.script_kind,
                    seed = EXCLUDED.seed,
                    sample_count = EXCLUDED.sample_count,
                    encoder_instruction = EXCLUDED.encoder_instruction,
                    decoder_instruction = EXCLUDED.decoder_instruction,
                    metadata = EXCLUDED.metadata,
                    updated_at = now()
                """,
                (
                    experiment_name,
                    SCRIPT_KIND,
                    seed,
                    sample_count,
                    ENCODER_INSTRUCTIONS,
                    DECODER_INSTRUCTIONS,
                    Jsonb(dict(metadata)),
                ),
            )


def insert_prediction_jobs(
    database_url: str, jobs: Sequence[EncDecJob]
) -> int:
    if not jobs:
        return 0
    rows = [
        (
            job.prediction_id,
            job.experiment_name,
            SCRIPT_KIND,
            job.submission_id,
            job.task_id,
            job.sample_index,
            job.encoder_model,
            job.decoder_model,
            job.encoder_temperature,
            job.decoder_temperature,
            job.repetition_seed,
            job.prompt,
            job.canonical_solution,
            job.ground_truth_code,
            job.test,
            job.entry_point,
            Jsonb(job.encoder_reasoning),
            Jsonb(job.decoder_reasoning),
        )
        for job in jobs
    ]
    with connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO dr_dspy_encdec_eval_predictions (
                    prediction_id,
                    experiment_name,
                    script_kind,
                    submission_id,
                    task_id,
                    sample_index,
                    encoder_model,
                    decoder_model,
                    encoder_temperature,
                    decoder_temperature,
                    repetition_seed,
                    prompt,
                    canonical_solution,
                    ground_truth_code,
                    test,
                    entry_point,
                    encoder_reasoning,
                    decoder_reasoning
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT DO NOTHING
                """,
                rows,
            )
            return cur.rowcount


def generate_code_for_job(job: EncDecJob) -> GenerationResult:
    encoder_events = LmEventBuffer()
    decoder_events = LmEventBuffer()

    encoder_lm = build_lm(
        model=job.encoder_model,
        reasoning=job.encoder_reasoning,
        temperature=job.encoder_temperature,
        event_buffer=encoder_events,
    )
    encoded_description = run_predictor(
        signature=EncodeCode,
        input_kwargs={"code": job.ground_truth_code},
        output_field="description",
        lm=encoder_lm,
        event_buffer=encoder_events,
    )
    decoder_lm = build_lm(
        model=job.decoder_model,
        reasoning=job.decoder_reasoning,
        temperature=job.decoder_temperature,
        event_buffer=decoder_events,
    )
    decoded_generation = run_predictor(
        signature=DecodeCode,
        input_kwargs={"description": encoded_description},
        output_field="code",
        lm=decoder_lm,
        event_buffer=decoder_events,
    )
    encoder_metadata = encoder_events.latest_response_metadata()
    decoder_metadata = decoder_events.latest_response_metadata()
    return GenerationResult(
        prediction_id=job.prediction_id,
        encoded_description=encoded_description,
        decoded_generation=decoded_generation,
        encoder_response_metadata=encoder_metadata,
        decoder_response_metadata=decoder_metadata,
        encoder_usage_metadata=usage_metadata_from_response(encoder_metadata),
        decoder_usage_metadata=usage_metadata_from_response(decoder_metadata),
        encoder_provider_cost=provider_cost_from_response(encoder_metadata),
        decoder_provider_cost=provider_cost_from_response(decoder_metadata),
    )


def fetch_prediction_job(database_url: str, prediction_id: str) -> EncDecJob:
    with connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    prediction_id,
                    experiment_name,
                    submission_id,
                    task_id,
                    sample_index,
                    encoder_model,
                    decoder_model,
                    encoder_temperature,
                    decoder_temperature,
                    repetition_seed,
                    prompt,
                    canonical_solution,
                    ground_truth_code,
                    test,
                    entry_point,
                    encoder_reasoning,
                    decoder_reasoning
                FROM dr_dspy_encdec_eval_predictions
                WHERE prediction_id = %s
                """,
                (prediction_id,),
            )
            row = cur.fetchone()
    if row is None:
        raise ValueError(f"prediction_id not found: {prediction_id}")
    return EncDecJob(
        prediction_id=row[0],
        experiment_name=row[1],
        submission_id=row[2],
        task_id=row[3],
        sample_index=row[4],
        encoder_model=row[5],
        decoder_model=row[6],
        encoder_temperature=row[7],
        decoder_temperature=row[8],
        repetition_seed=row[9],
        prompt=row[10],
        canonical_solution=row[11],
        ground_truth_code=row[12],
        test=row[13],
        entry_point=row[14],
        encoder_reasoning=dict(row[15]),
        decoder_reasoning=dict(row[16]),
    )


def fetch_prediction_log_context(
    database_url: str, prediction_id: str
) -> shared_eval_logging.PredictionLogContext:
    job = fetch_prediction_job(database_url, prediction_id)
    return prediction_context_from_job(job)


def mark_generation_started(database_url: str, prediction_id: str) -> None:
    with connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE dr_dspy_encdec_eval_predictions
                SET generation_status = 'started',
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
                UPDATE dr_dspy_encdec_eval_predictions
                SET generation_status = 'generated',
                    generation_error = NULL,
                    encoded_description = %s,
                    decoded_generation = %s,
                    raw_generation = %s,
                    encoder_response_metadata = %s,
                    decoder_response_metadata = %s,
                    encoder_usage_metadata = %s,
                    decoder_usage_metadata = %s,
                    encoder_provider_cost = %s,
                    decoder_provider_cost = %s,
                    provider_cost = %s,
                    generated_at = now(),
                    updated_at = now()
                WHERE prediction_id = %s
                """,
                (
                    result.encoded_description,
                    result.decoded_generation,
                    result.decoded_generation,
                    Jsonb(result.encoder_response_metadata),
                    Jsonb(result.decoder_response_metadata),
                    Jsonb(result.encoder_usage_metadata),
                    Jsonb(result.decoder_usage_metadata),
                    result.encoder_provider_cost,
                    result.decoder_provider_cost,
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
                UPDATE dr_dspy_encdec_eval_predictions
                SET generation_status = 'generation_error',
                    generation_error = %s,
                    updated_at = now()
                WHERE prediction_id = %s
                """,
                (error, prediction_id),
            )


def mark_scoring_started(database_url: str, prediction_id: str) -> None:
    with connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE dr_dspy_encdec_eval_predictions
                SET scoring_status = 'started',
                    scoring_error = NULL,
                    updated_at = now()
                WHERE prediction_id = %s
                """,
                (prediction_id,),
            )


def mark_scoring_queued(
    database_url: str, prediction_ids: Sequence[str]
) -> int:
    return shared_eval_repair.mark_scoring_queued(
        database_url,
        prediction_table=PREDICTION_TABLE_NAME,
        prediction_ids=prediction_ids,
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
                    test,
                    entry_point,
                    encoded_description,
                    raw_generation
                FROM dr_dspy_encdec_eval_predictions
                WHERE prediction_id = %s
                """,
                (prediction_id,),
            )
            row = cur.fetchone()
    if row is None:
        raise ValueError(f"prediction_id not found: {prediction_id}")
    if row[7] is None or row[8] is None:
        raise ValueError(f"prediction_id is not generated: {prediction_id}")
    return ScoringTarget(
        prediction_id=row[0],
        task_id=row[1],
        prompt=row[2],
        canonical_solution=row[3],
        ground_truth_code=row[4],
        test=row[5],
        entry_point=row[6],
        encoded_description=row[7],
        raw_generation=row[8],
    )


def best_metric(
    metrics: Sequence[CompressionMetric],
) -> CompressionMetric | None:
    comparable = [
        metric
        for metric in metrics
        if metric.ratio_to_ground_truth is not None
    ]
    if not comparable:
        return None
    return min(
        comparable,
        key=lambda metric: cast(float, metric.ratio_to_ground_truth),
    )


def score_generated_code(
    target: ScoringTarget, *, timeout: float
) -> ScoreResult:
    generated_score: GeneratedCodeScore = score_generated_code_for_humaneval(
        raw_generation=target.raw_generation,
        task=target.task(),
        timeout=timeout,
    )
    metrics = compression_metrics(
        ground_truth_code=target.ground_truth_code,
        encoded_description=target.encoded_description,
    )
    best = best_metric(metrics)
    evaluation = generated_score.evaluation
    return ScoreResult(
        prediction_id=target.prediction_id,
        score=generated_score.score,
        error=generated_score.error,
        raw_code=generated_score.raw_code,
        raw_compile_ok=generated_score.raw_compile_ok,
        raw_compile_error=generated_score.raw_compile_error,
        extraction_candidate_count=generated_score.extraction_candidate_count,
        selected_candidate_index=generated_score.selected_candidate_index,
        extracted_compile_ok=generated_score.extracted_compile_ok,
        extracted_compile_error=generated_score.extracted_compile_error,
        extraction_error=generated_score.extraction_error,
        evaluation_function_names=evaluation.function_names
        if evaluation
        else [],
        evaluation_total_cases=evaluation.total_cases if evaluation else None,
        evaluation_failure_count=len(evaluation.failures)
        if evaluation
        else None,
        evaluation_status_counts=evaluation.status_counts
        if evaluation
        else {},
        compression_metrics=metrics,
        best_compression_ratio=best.ratio_to_ground_truth if best else None,
        best_compression_percent_reduction=(
            best.percent_reduction_vs_ground_truth if best else None
        ),
    )


def record_score_success(database_url: str, result: ScoreResult) -> None:
    with connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE dr_dspy_encdec_eval_predictions
                SET scoring_status = 'scored',
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
                    scored_at = now(),
                    updated_at = now()
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
                UPDATE dr_dspy_encdec_eval_predictions
                SET scoring_status = 'score_error',
                    scoring_error = %s,
                    updated_at = now()
                WHERE prediction_id = %s
                """,
                (error, prediction_id),
            )


@DBOS.step(
    name="humaneval_encdec_generate_prediction_step_v0",
    retries_allowed=True,
    max_attempts=3,
    interval_seconds=2.0,
)
def generate_prediction_step(
    database_url: str, prediction_id: str
) -> GenerationResult:
    mark_generation_started(database_url, prediction_id)
    job = fetch_prediction_job(database_url, prediction_id)
    emit_prediction_log_event(
        "generation_started",
        prediction_context_from_job(job),
    )
    return generate_code_for_job(job)


@DBOS.step(name="humaneval_encdec_record_generation_success_step_v0")
def record_generation_success_step(
    database_url: str, result: GenerationResult
) -> None:
    context = fetch_prediction_log_context(database_url, result.prediction_id)
    emit_prediction_log_event(
        "generation_succeeded",
        context,
        extra={
            "provider_cost": result.provider_cost,
            "encoder_usage_metadata": result.encoder_usage_metadata,
            "decoder_usage_metadata": result.decoder_usage_metadata,
        },
    )
    record_generation_success(database_url, result)


@DBOS.step(name="humaneval_encdec_record_generation_error_step_v0")
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


@DBOS.step(name="humaneval_encdec_mark_scoring_queued_step_v0")
def mark_scoring_queued_step(database_url: str, prediction_id: str) -> None:
    context = fetch_prediction_log_context(database_url, prediction_id)
    emit_prediction_log_event("scoring_enqueued", context)
    mark_scoring_queued(database_url, [prediction_id])


@DBOS.step(name="humaneval_encdec_score_prediction_step_v0")
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
    return score_generated_code(
        fetch_scoring_target(database_url, prediction_id),
        timeout=timeout,
    )


@DBOS.step(name="humaneval_encdec_record_score_success_step_v0")
def record_score_success_step(database_url: str, result: ScoreResult) -> None:
    context = fetch_prediction_log_context(database_url, result.prediction_id)
    emit_prediction_log_event(
        "scoring_succeeded",
        context,
        extra={"score": result.score, "scoring_error": result.error},
    )
    record_score_success(database_url, result)


@DBOS.step(name="humaneval_encdec_record_score_error_step_v0")
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


@DBOS.workflow(name="humaneval_encdec_generate_prediction_v0")
def generate_prediction_workflow(
    database_url: str,
    prediction_id: str,
    experiment_name: str,
    score_timeout: float = DEFAULT_SUBPROCESS_TIMEOUT,
) -> str:
    try:
        result = generate_prediction_step(database_url, prediction_id)
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


@DBOS.workflow(name="humaneval_encdec_score_prediction_v0")
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


def fetch_generation_error_prediction_jobs(
    database_url: str,
    *,
    experiment_name: str,
    limit: int,
) -> list[EncDecJob]:
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
                UPDATE dr_dspy_encdec_eval_predictions
                SET
                    generation_status = 'pending',
                    generation_error = NULL,
                    encoded_description = NULL,
                    decoded_generation = NULL,
                    raw_generation = NULL,
                    encoder_response_metadata = '{}'::jsonb,
                    decoder_response_metadata = '{}'::jsonb,
                    encoder_usage_metadata = '{}'::jsonb,
                    decoder_usage_metadata = '{}'::jsonb,
                    encoder_provider_cost = NULL,
                    decoder_provider_cost = NULL,
                    provider_cost = NULL,
                    generated_at = NULL,
                    scoring_status = 'pending',
                    scoring_error = NULL,
                    score = NULL,
                    raw_code = NULL,
                    raw_compile_ok = NULL,
                    raw_compile_error = NULL,
                    extraction_candidate_count = NULL,
                    selected_candidate_index = NULL,
                    extracted_compile_ok = NULL,
                    extracted_compile_error = NULL,
                    extraction_error = NULL,
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
            jobs,
            database_url=config.database_url,
            experiment_name=experiment_name,
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


QUEUE_NAME_CONFIG = shared_dbos.QueueNameConfig(
    generation_base_name=GENERATION_QUEUE_NAME,
    scoring_base_name=SCORING_QUEUE_NAME,
    hash_length=EXPERIMENT_QUEUE_HASH_LENGTH,
)

QueueSelection = shared_dbos.QueueSelection
EvalDbosConfig = shared_dbos.EvalDbosConfig
EvalQueueNames = shared_dbos.EvalQueueNames
DbPoolConfig = shared_dbos.DbPoolConfig
OpenFileLimitResult = shared_dbos.OpenFileLimitResult
DB_POOL_AUTO = shared_dbos.DB_POOL_AUTO
DB_POOLS = shared_dbos.DB_POOLS
connect_db = shared_dbos.connect_db
close_db_connection_pools = shared_dbos.close_db_connection_pools
generation_workflow_id = shared_dbos.generation_workflow_id
score_workflow_id = shared_dbos.score_workflow_id
WorkerMonitorConfig = shared_worker_monitor.WorkerMonitorConfig
WorkerQueueSnapshot = shared_worker_monitor.WorkerQueueSnapshot
open_file_limit_line = shared_dbos.open_file_limit_line
open_file_limit_style = shared_dbos.open_file_limit_style


def resolve_database_url(database_url: str | None) -> str:
    return shared_dbos.resolve_database_url(
        database_url,
        database_url_env=DATABASE_URL_ENV,
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
    )


def build_dbos_config(config: EvalDbosConfig) -> DBOSConfig:
    return shared_dbos.build_dbos_config(config, app_name=DBOS_APP_NAME)


def raise_open_file_limit(requested: int) -> OpenFileLimitResult:
    return shared_dbos.raise_open_file_limit(requested)


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
            *PREDICTION_INDEX_SQL,
        ),
    )


def parse_temperatures(raw: str) -> list[float]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("at least one temperature is required")
    return [float(value) for value in values]


def build_humaneval_samples(
    *, seed: int, sample_count: int
) -> list[EncDecSample]:
    return [
        EncDecSample(
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
        for sample in shared_human_eval_sampling.sample_human_eval_tasks(
            seed=seed,
            sample_count=sample_count,
            dataset_name=DATASET_NAME,
            dataset_split=DATASET_SPLIT,
        )
    ]


def build_lm(
    *,
    model: str,
    reasoning: Mapping[str, Any],
    temperature: float | None,
    event_buffer: LmEventBuffer,
) -> dspy.BaseLM:
    return shared_dspy_runner.build_logged_lm(
        model=model,
        reasoning=reasoning,
        temperature=temperature,
        event_buffer=event_buffer,
        max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
    )


prediction_field_text = shared_dspy_runner.prediction_field_text


def run_predictor(
    *,
    signature: type[dspy.Signature],
    input_kwargs: Mapping[str, Any],
    output_field: str,
    lm: dspy.BaseLM,
    event_buffer: LmEventBuffer,
) -> str:
    return shared_dspy_runner.run_predictor(
        signature=signature,
        input_kwargs=input_kwargs,
        output_field=output_field,
        lm=lm,
        event_buffer=event_buffer,
    )


def experiment_name_hash(experiment_name: str) -> str:
    return shared_dbos.experiment_hash(experiment_name)


def generation_queue(experiment_name: str) -> str:
    return shared_dbos.eval_queue_names(
        experiment_name, QUEUE_NAME_CONFIG
    ).generation


def scoring_queue(experiment_name: str) -> str:
    return shared_dbos.eval_queue_names(
        experiment_name, QUEUE_NAME_CONFIG
    ).scoring


def configure_dbos_runtime(
    config: EvalDbosConfig,
    *,
    experiment_name: str,
    queue: QueueSelection | None = None,
    consume_queues: bool = True,
) -> None:
    shared_dbos.DBOS = DBOS
    shared_dbos.configure_dbos_runtime(
        config,
        app_name=DBOS_APP_NAME,
        experiment_name=experiment_name,
        queue=queue,
        queue_config=QUEUE_NAME_CONFIG,
        consume_queues=consume_queues,
        operator_log=operator_log,
    )


def queue_names_for_selection(
    selection: QueueSelection, *, experiment_name: str
) -> tuple[str, ...]:
    return shared_dbos.queue_names_for_selection(
        selection,
        experiment_name=experiment_name,
        queue_config=QUEUE_NAME_CONFIG,
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
        config, experiment_name=experiment_name, queue=queue
    )
    return pool_config


def enqueue_generation_jobs(
    jobs: Sequence[EncDecJob],
    *,
    database_url: str,
    experiment_name: str,
    score_timeout: float,
    retry_token: str | None = None,
) -> None:
    _ = experiment_name
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
        title="Enc-Dec Eval Status",
        dimensions=(
            shared_eval_reporting.StatusDimension(
                key="encoder_model", title="Encoder"
            ),
            shared_eval_reporting.StatusDimension(
                key="decoder_model", title="Decoder"
            ),
            shared_eval_reporting.StatusDimension(
                key="encoder_temperature", title="Enc Temp", justify="right"
            ),
            shared_eval_reporting.StatusDimension(
                key="decoder_temperature", title="Dec Temp", justify="right"
            ),
        ),
        experiment_name=experiment_name,
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


def analysis_model_label(record: AnalysisRecord) -> str:
    encoder_temperature = (
        "None"
        if record.encoder_temperature is None
        else f"{record.encoder_temperature:g}"
    )
    return (
        f"{record.encoder_model} @ {encoder_temperature} -> "
        f"{record.decoder_model}"
    )


def fetch_analysis_records(
    database_url: str, *, experiment_name: str
) -> list[AnalysisRecord]:
    with connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    encoder_model,
                    decoder_model,
                    encoder_temperature,
                    decoder_temperature,
                    task_id,
                    repetition_seed,
                    score,
                    provider_cost,
                    raw_compile_ok,
                    extracted_compile_ok,
                    best_compression_ratio,
                    best_compression_percent_reduction
                FROM dr_dspy_encdec_eval_predictions
                WHERE
                    experiment_name = %s
                    AND scoring_status = 'scored'
                    AND score IS NOT NULL
                ORDER BY
                    encoder_model,
                    decoder_model,
                    encoder_temperature,
                    decoder_temperature,
                    task_id,
                    repetition_seed
                """,
                (experiment_name,),
            )
            rows = cur.fetchall()
    return [
        AnalysisRecord(
            encoder_model=row[0],
            decoder_model=row[1],
            encoder_temperature=row[2],
            decoder_temperature=row[3],
            task_id=row[4],
            repetition_seed=row[5],
            score=row[6],
            provider_cost=row[7],
            raw_compile_ok=row[8],
            extracted_compile_ok=row[9],
            best_compression_ratio=row[10],
            best_compression_percent_reduction=row[11],
        )
        for row in rows
    ]


def summarize_analysis_records(
    records: Sequence[AnalysisRecord],
) -> list[AnalysisSummary]:
    grouped: dict[
        tuple[str, str, float | None, float | None], list[AnalysisRecord]
    ] = {}
    for record in records:
        key = (
            record.encoder_model,
            record.decoder_model,
            record.encoder_temperature,
            record.decoder_temperature,
        )
        grouped.setdefault(key, []).append(record)

    summaries: list[AnalysisSummary] = []
    for (_encoder, _decoder, _enc_temp, dec_temp), group in sorted(
        grouped.items()
    ):
        scores = [record.score for record in group]
        costs = [
            record.provider_cost
            for record in group
            if record.provider_cost is not None
        ]
        task_ids = {record.task_id for record in group}
        by_task: dict[str, list[float]] = {}
        for record in group:
            by_task.setdefault(record.task_id, []).append(record.score)
        repetition_variances = [
            variance
            for variance in (
                shared_analysis.variance_or_none(task_scores)
                for task_scores in by_task.values()
            )
            if variance is not None
        ]
        raw_compile_pass_count = sum(
            1 for record in group if record.raw_compile_ok is True
        )
        extracted_compile_pass_count = sum(
            1 for record in group if record.extracted_compile_ok is True
        )
        total_price = sum(costs) if costs else None
        summaries.append(
            AnalysisSummary(
                model=analysis_model_label(group[0]),
                temperature=dec_temp if dec_temp is not None else 0.0,
                sample_count=len(task_ids),
                scored_count=len(group),
                total_price=total_price,
                avg_price_per_sample=(
                    total_price / len(group)
                    if total_price is not None
                    else None
                ),
                price_variance=shared_analysis.variance_or_none(costs),
                avg_performance=statistics.fmean(scores),
                performance_variance=shared_analysis.variance_or_none(scores),
                avg_repetition_variance=shared_analysis.average_or_none(
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


def analysis_markdown(
    *, experiment_name: str, summaries: Sequence[AnalysisSummary]
) -> str:
    return shared_eval_reporting.analysis_markdown(
        experiment_name=experiment_name, summaries=summaries
    )


def analysis_table(
    *, experiment_name: str, summaries: Sequence[AnalysisSummary]
) -> object:
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


worker_monitor_line = shared_worker_monitor.worker_monitor_line
worker_monitor_style = shared_worker_monitor.worker_monitor_style


def start_worker_monitor(
    config: WorkerMonitorConfig, stop_event: threading.Event
) -> threading.Thread:
    return shared_worker_monitor.start_worker_monitor(
        config,
        stop_event,
        operator_log=operator_log,
        emit_worker_detail_log=emit_worker_detail_log,
    )


app = typer.Typer(no_args_is_help=True)


@app.command("init-db")
def init_db(
    database_url: Annotated[str | None, typer.Option()] = None,
    env_file: Annotated[Path | None, typer.Option()] = None,
) -> None:
    load_optional_env_file(env_file)
    create_eval_schema(resolve_database_url(database_url))
    operator_log("initialized enc-dec eval schema")


@app.command()
def submit(
    experiment_name: Annotated[str, typer.Option()],
    database_url: Annotated[str | None, typer.Option()] = None,
    dbos_system_database_url: Annotated[str | None, typer.Option()] = None,
    model_pairs_json: Annotated[str | None, typer.Option()] = None,
    sample_count: Annotated[int, typer.Option()] = DEFAULT_SAMPLE_COUNT,
    seed: Annotated[int, typer.Option()] = DEFAULT_SEED,
    encoder_temperatures: Annotated[str, typer.Option()] = str(
        DEFAULT_TEMPERATURE
    ),
    decoder_temperatures: Annotated[str, typer.Option()] = str(
        DEFAULT_TEMPERATURE
    ),
    repetitions: Annotated[int, typer.Option()] = DEFAULT_REPETITIONS,
    generation_concurrency: Annotated[
        int, typer.Option()
    ] = DEFAULT_GENERATION_CONCURRENCY,
    scoring_concurrency: Annotated[
        int, typer.Option()
    ] = DEFAULT_SCORING_CONCURRENCY,
    score_timeout: Annotated[
        float, typer.Option()
    ] = DEFAULT_SUBPROCESS_TIMEOUT,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Plan jobs without writing or enqueueing.",
        ),
    ] = False,
    env_file: Annotated[Path | None, typer.Option()] = None,
) -> None:
    load_optional_env_file(env_file)
    config = build_eval_dbos_config(
        database_url=database_url,
        dbos_system_database_url=dbos_system_database_url,
        generation_concurrency=generation_concurrency,
        scoring_concurrency=scoring_concurrency,
    )
    model_pairs = parse_model_pairs(model_pairs_json)
    samples = build_humaneval_samples(seed=seed, sample_count=sample_count)
    jobs = build_prediction_jobs(
        experiment_name=experiment_name,
        submission_id=uuid.uuid4().hex,
        samples=samples,
        model_pairs=model_pairs,
        encoder_temperatures=parse_temperatures(encoder_temperatures),
        decoder_temperatures=parse_temperatures(decoder_temperatures),
        repetitions=repetitions,
    )
    metadata = {
        "model_pairs": [pair.model_dump(mode="json") for pair in model_pairs],
        "encoder_temperatures": parse_temperatures(encoder_temperatures),
        "decoder_temperatures": parse_temperatures(decoder_temperatures),
        "repetitions": repetitions,
    }
    operator_log(
        f"planned {len(jobs)} jobs: samples={len(samples)}, "
        f"model_pairs={len(model_pairs)}, "
        f"encoder_temperatures={len(metadata['encoder_temperatures'])}, "
        f"decoder_temperatures={len(metadata['decoder_temperatures'])}, "
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
        metadata=metadata,
    )
    inserted = insert_prediction_jobs(config.database_url, jobs)
    configure_dbos_runtime(
        config, experiment_name=experiment_name, consume_queues=False
    )
    enqueue_generation_jobs(
        jobs,
        database_url=config.database_url,
        experiment_name=experiment_name,
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
def worker(
    experiment_name: Annotated[str, typer.Option()],
    database_url: Annotated[str | None, typer.Option()] = None,
    dbos_system_database_url: Annotated[str | None, typer.Option()] = None,
    queue: Annotated[QueueSelection, typer.Option()] = QueueSelection.BOTH,
    generation_concurrency: Annotated[
        int, typer.Option()
    ] = DEFAULT_GENERATION_CONCURRENCY,
    scoring_concurrency: Annotated[
        int, typer.Option()
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
    configure_multiprocessing()
    load_optional_env_file(env_file)
    file_limit = raise_open_file_limit(open_file_limit)
    operator_log(
        open_file_limit_line(file_limit),
        style=open_file_limit_style(file_limit),
    )
    config = build_eval_dbos_config(
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


@app.command()
def status(
    experiment_name: Annotated[str | None, typer.Option()] = None,
    database_url: Annotated[str | None, typer.Option()] = None,
    env_file: Annotated[Path | None, typer.Option()] = None,
) -> None:
    load_optional_env_file(env_file)
    rows = fetch_status_counts(
        resolve_database_url(database_url), experiment_name=experiment_name
    )
    if not rows:
        operator_log("no prediction rows found", style="yellow")
        return
    CONSOLE.print(
        status_counts_table(rows, experiment_name=experiment_name)
    )


@app.command("enqueue-scores")
def enqueue_scores_command(
    experiment_name: Annotated[str, typer.Option()],
    limit: Annotated[
        int, typer.Option("--limit", min=1)
    ] = DEFAULT_SCORE_ENQUEUE_LIMIT,
    timeout: Annotated[
        float, typer.Option("--timeout", min=0.1)
    ] = DEFAULT_SUBPROCESS_TIMEOUT,
    database_url: Annotated[str | None, typer.Option()] = None,
    dbos_system_database_url: Annotated[str | None, typer.Option()] = None,
    generation_concurrency: Annotated[
        int, typer.Option()
    ] = DEFAULT_GENERATION_CONCURRENCY,
    scoring_concurrency: Annotated[
        int, typer.Option()
    ] = DEFAULT_SCORING_CONCURRENCY,
    env_file: Annotated[Path | None, typer.Option()] = None,
) -> None:
    load_optional_env_file(env_file)
    config = build_eval_dbos_config(
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


@app.command()
def analyze(
    experiment_name: Annotated[str, typer.Option()],
    csv_path: Annotated[
        Path | None,
        typer.Option("--csv-path", help="Optional CSV output path."),
    ] = None,
    database_url: Annotated[str | None, typer.Option()] = None,
    markdown: Annotated[
        bool,
        typer.Option(
            "--markdown",
            help="Print the analysis table as Markdown.",
        ),
    ] = False,
    env_file: Annotated[Path | None, typer.Option()] = None,
) -> None:
    load_optional_env_file(env_file)
    records = fetch_analysis_records(
        resolve_database_url(database_url), experiment_name=experiment_name
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


@app.command()
def repair(
    experiment_name: Annotated[str, typer.Option()],
    database_url: Annotated[str | None, typer.Option()] = None,
    dbos_system_database_url: Annotated[str | None, typer.Option()] = None,
    generation_limit: Annotated[
        int, typer.Option("--generation-limit", min=1)
    ] = DEFAULT_REPAIR_GENERATION_LIMIT,
    scoring_limit: Annotated[
        int, typer.Option("--scoring-limit", min=1)
    ] = DEFAULT_REPAIR_SCORING_LIMIT,
    generation_concurrency: Annotated[
        int, typer.Option()
    ] = DEFAULT_GENERATION_CONCURRENCY,
    scoring_concurrency: Annotated[
        int, typer.Option()
    ] = DEFAULT_SCORING_CONCURRENCY,
    score_timeout: Annotated[
        float, typer.Option()
    ] = DEFAULT_SUBPROCESS_TIMEOUT,
    apply: Annotated[bool, typer.Option("--apply")] = False,
    env_file: Annotated[Path | None, typer.Option()] = None,
) -> None:
    load_optional_env_file(env_file)
    config = build_eval_dbos_config(
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
    if not apply:
        if (
            plan.stranded_generations
            or plan.generation_retry_prediction_ids
            or plan.pending_scoring_prediction_ids
            or plan.stranded_scoring
            or plan.scoring_retry_prediction_ids
        ):
            operator_log(
                "dry run only; rerun with --apply to reconcile statuses and "
                "enqueue fresh retry workflows",
                style="yellow",
            )
        return
    result = apply_repair(
        config,
        experiment_name=experiment_name,
        generation_limit=generation_limit,
        scoring_limit=scoring_limit,
        score_timeout=score_timeout,
    )
    operator_log(
        repair_apply_line(experiment_name=experiment_name, result=result),
        style="green",
    )


if __name__ == "__main__":
    app()
