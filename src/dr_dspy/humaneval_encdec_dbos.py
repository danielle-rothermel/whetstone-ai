from __future__ import annotations

import json
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
    StrictFloat,
    StrictInt,
    StrictStr,
)
from rich.console import Console
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
from dr_dspy.experiment_dimensions import (
    Dimension,
    dimension_columns_ddl,
    identity_constraint_columns,
    identity_dimension_names,
    reporting_dimension_names,
    status_dimensions,
)
from dr_dspy.human_eval import HumanEvalTask
from dr_dspy.lm_utils import (
    LmEventBuffer,
    ModelConfig,
    provider_cost_from_response,
    usage_metadata_from_response,
)
from dr_dspy.runtime import (
    load_env_file,
)
from dr_dspy.scoring import (
    HumanEvalScoreResult,
    score_humaneval_prediction,
)
from dr_dspy.signatures import DspySignatureConfig
from dspy.signatures.signature import make_signature

DATABASE_URL_ENV = "DATABASE_URL"
DBOS_APP_NAME = "dr-dspy-humaneval-encdec-eval"
DBOS_SYSTEM_DATABASE_URL_ENV = "DBOS_SYSTEM_DATABASE_URL"
GENERATION_QUEUE_NAME = "dr_dspy_humaneval_encdec_generation"
SCORING_QUEUE_NAME = "dr_dspy_humaneval_encdec_scoring"
MIN_ENCODER_CHAR_BUDGET = 50
DEFAULT_GENERATION_CONCURRENCY = 64
DEFAULT_SCORING_CONCURRENCY = 32
DEFAULT_WORKER_OPEN_FILE_LIMIT = 8192
DEFAULT_WORKER_MONITOR_INTERVAL_SECONDS = 5.0
DEFAULT_WORKER_MONITOR_SUMMARY_INTERVAL_SECONDS = 5.0
DEFAULT_SCORE_ENQUEUE_LIMIT = 1000
DEFAULT_REPAIR_GENERATION_LIMIT = 1000
DEFAULT_REPAIR_SCORING_LIMIT = 1000
PREDICTION_TABLE_NAME = "dr_dspy_encdec_eval_predictions"
PREDICTION_ID_DIGEST_LENGTH = 32
DIMENSIONS: tuple[Dimension, ...] = (
    Dimension(
        name="encoder_model",
        sql_type="TEXT",
        nullable=False,
        report_title="Encoder",
    ),
    Dimension(
        name="decoder_model",
        sql_type="TEXT",
        nullable=False,
        report_title="Decoder",
    ),
    Dimension(
        name="encoder_temperature",
        sql_type="DOUBLE PRECISION",
        report_title="Enc Temp",
        report_justify="right",
    ),
    Dimension(
        name="decoder_temperature",
        sql_type="DOUBLE PRECISION",
        report_title="Dec Temp",
        report_justify="right",
    ),
    Dimension(
        name="budget_ratio",
        sql_type="DOUBLE PRECISION",
        report_title="Budget",
        report_justify="right",
    ),
    Dimension(
        name="encoder_reasoning",
        sql_type="JSONB",
        nullable=False,
        default_sql="'{}'::jsonb",
        in_reporting=False,
        report_title="Encoder Reasoning",
    ),
    Dimension(
        name="decoder_reasoning",
        sql_type="JSONB",
        nullable=False,
        default_sql="'{}'::jsonb",
        in_reporting=False,
        report_title="Decoder Reasoning",
    ),
)
REPAIR_DIMENSION_COLUMNS = identity_dimension_names(DIMENSIONS)
REPORTING_DIMENSION_COLUMNS = reporting_dimension_names(DIMENSIONS)
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
    repetition_seed      INTEGER     NOT NULL,
__DIMENSION_COLUMNS__
    prompt               TEXT        NOT NULL,
    canonical_solution   TEXT        NOT NULL,
    ground_truth_code    TEXT        NOT NULL,
    test                 TEXT        NOT NULL,
    entry_point          TEXT        NOT NULL,
    encoder_char_budget  INTEGER,
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
    compression_metrics  JSONB       NOT NULL DEFAULT '{}'::jsonb,
    raw_compression_ratio DOUBLE PRECISION,
    best_compression_ratio DOUBLE PRECISION,
    best_compression_percent_reduction DOUBLE PRECISION,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    generated_at         TIMESTAMPTZ,
    scored_at            TIMESTAMPTZ,
    CONSTRAINT dr_dspy_encdec_eval_predictions_identity_key UNIQUE (
__IDENTITY_CONSTRAINT__
    )
)
""".replace(
    "__DIMENSION_COLUMNS__", dimension_columns_ddl(DIMENSIONS).rstrip()
).replace(
    "__IDENTITY_CONSTRAINT__",
    "        " + ",\n        ".join(identity_constraint_columns(DIMENSIONS)),
)

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


class EncDecHumanEvalExperimentConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    script_kind: StrictStr
    dataset_name: StrictStr
    dataset_split: StrictStr
    encoder_signature: DspySignatureConfig
    budgeted_encoder_signature: DspySignatureConfig
    decoder_signature: DspySignatureConfig
    default_model_pairs: tuple[EncDecPair, ...]
    default_sample_count: StrictInt
    default_seed: StrictInt
    default_encoder_temperatures: tuple[float, ...]
    default_decoder_temperatures: tuple[float, ...]
    default_budget_ratios: tuple[float | None, ...]
    default_repetitions: StrictInt
    default_max_completion_tokens: StrictInt
    default_subprocess_timeout: float


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
    budget_ratio: StrictFloat | None = None
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
    encoder_char_budget: int | None = None

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


ScoreResult = HumanEvalScoreResult


class AnalysisRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    encoder_model: StrictStr
    decoder_model: StrictStr
    encoder_temperature: float | None
    decoder_temperature: float | None
    budget_ratio: float | None
    task_id: StrictStr
    repetition_seed: StrictInt
    score: float
    provider_cost: float | None
    raw_compile_ok: bool | None = None
    extracted_compile_ok: bool | None = None
    raw_compression_ratio: float | None = None
    best_compression_ratio: float | None = None
    best_compression_percent_reduction: float | None = None


class AnalysisSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dimensions: dict[str, Any]
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
    avg_raw_compression_ratio: float | None = None
    avg_best_compression_ratio: float | None = None
    avg_best_compression_percent_reduction: float | None = None


_EXPERIMENT_CONFIG: EncDecHumanEvalExperimentConfig | None = None
_ENCODER_SIGNATURE: type[dspy.Signature] | None = None
_BUDGETED_ENCODER_SIGNATURE: type[dspy.Signature] | None = None
_DECODER_SIGNATURE: type[dspy.Signature] | None = None


def build_dspy_signature(config: DspySignatureConfig) -> type[dspy.Signature]:
    return make_signature(
        {field.name: (field.type, field.role) for field in config.fields},
        instructions=config.instructions,
        signature_name=config.name,
    )


def configure_experiment(config: EncDecHumanEvalExperimentConfig) -> None:
    global _EXPERIMENT_CONFIG, _ENCODER_SIGNATURE
    global _BUDGETED_ENCODER_SIGNATURE, _DECODER_SIGNATURE
    _EXPERIMENT_CONFIG = config
    _ENCODER_SIGNATURE = build_dspy_signature(config.encoder_signature)
    _BUDGETED_ENCODER_SIGNATURE = build_dspy_signature(
        config.budgeted_encoder_signature
    )
    _DECODER_SIGNATURE = build_dspy_signature(config.decoder_signature)


def experiment_config() -> EncDecHumanEvalExperimentConfig:
    if _EXPERIMENT_CONFIG is None:
        raise RuntimeError(
            "HumanEval enc-dec experiment is not configured; call "
            "create_app(config) from the experiment script first."
        )
    return _EXPERIMENT_CONFIG


def encoder_signature() -> type[dspy.Signature]:
    if _ENCODER_SIGNATURE is None:
        raise RuntimeError(
            "HumanEval enc-dec encoder signature is not configured; call "
            "create_app(config) from the experiment script first."
        )
    return _ENCODER_SIGNATURE


def budgeted_encoder_signature() -> type[dspy.Signature]:
    if _BUDGETED_ENCODER_SIGNATURE is None:
        raise RuntimeError(
            "HumanEval enc-dec budgeted encoder signature is not configured; "
            "call create_app(config) from the experiment script first."
        )
    return _BUDGETED_ENCODER_SIGNATURE


def decoder_signature() -> type[dspy.Signature]:
    if _DECODER_SIGNATURE is None:
        raise RuntimeError(
            "HumanEval enc-dec decoder signature is not configured; call "
            "create_app(config) from the experiment script first."
        )
    return _DECODER_SIGNATURE


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
            "budget_ratio": job.budget_ratio,
            "encoder_reasoning": job.encoder_reasoning,
            "decoder_reasoning": job.decoder_reasoning,
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


def parse_budget_ratios(raw: str) -> list[float | None]:
    values: list[float | None] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        if token.lower() == "none":
            values.append(None)
        else:
            values.append(float(token))
    if not values:
        raise ValueError("at least one budget ratio is required")
    return values


def parse_model_pairs(raw: str | None) -> list[EncDecPair]:
    if raw is None:
        return [
            EncDecPair(**pair.model_dump(mode="python"))
            for pair in experiment_config().default_model_pairs
        ]
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
    encoder_temperature: float | None,
    decoder_temperature: float | None,
    budget_ratio: float | None,
    encoder_reasoning: Mapping[str, Any],
    decoder_reasoning: Mapping[str, Any],
    repetition_seed: int,
) -> str:
    return shared_flow.stable_prediction_id_from_dimensions(
        experiment_name=experiment_name,
        task_id=task_id,
        dimensions={
            "encoder_model": encoder_model,
            "decoder_model": decoder_model,
            "encoder_temperature": encoder_temperature,
            "decoder_temperature": decoder_temperature,
            "budget_ratio": budget_ratio,
            "encoder_reasoning": dict(encoder_reasoning),
            "decoder_reasoning": dict(decoder_reasoning),
        },
        repetition_seed=repetition_seed,
        digest_length=PREDICTION_ID_DIGEST_LENGTH,
    )


def build_prediction_jobs(
    *,
    experiment_name: str,
    submission_id: str,
    samples: Sequence[EncDecSample],
    model_pairs: Sequence[EncDecPair],
    encoder_temperatures: Sequence[float],
    decoder_temperatures: Sequence[float],
    budget_ratios: Sequence[float | None],
    repetitions: int,
) -> list[EncDecJob]:
    jobs: list[EncDecJob] = []
    for sample in samples:
        for pair in model_pairs:
            for encoder_temperature in encoder_temperatures:
                for decoder_temperature in decoder_temperatures:
                    for budget_ratio in budget_ratios:
                        for repetition_seed in range(repetitions):
                            jobs.append(
                                EncDecJob(
                                    prediction_id=stable_prediction_id(
                                        experiment_name=experiment_name,
                                        task_id=sample.task_id,
                                        encoder_model=pair.encoder.model,
                                        decoder_model=pair.decoder.model,
                                        encoder_temperature=(
                                            encoder_temperature
                                        ),
                                        decoder_temperature=(
                                            decoder_temperature
                                        ),
                                        budget_ratio=budget_ratio,
                                        encoder_reasoning=(
                                            pair.encoder.reasoning
                                        ),
                                        decoder_reasoning=(
                                            pair.decoder.reasoning
                                        ),
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
                                    budget_ratio=budget_ratio,
                                    repetition_seed=repetition_seed,
                                    prompt=sample.prompt,
                                    canonical_solution=(
                                        sample.canonical_solution
                                    ),
                                    ground_truth_code=sample.ground_truth_code,
                                    test=sample.test,
                                    entry_point=sample.entry_point,
                                    encoder_reasoning=dict(
                                        pair.encoder.reasoning
                                    ),
                                    decoder_reasoning=dict(
                                        pair.decoder.reasoning
                                    ),
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
                    experiment_config().script_kind,
                    seed,
                    sample_count,
                    experiment_config().encoder_signature.instructions,
                    experiment_config().decoder_signature.instructions,
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
            experiment_config().script_kind,
            job.submission_id,
            job.task_id,
            job.sample_index,
            job.encoder_model,
            job.decoder_model,
            job.encoder_temperature,
            job.decoder_temperature,
            job.budget_ratio,
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
                    budget_ratio,
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
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
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
    if job.budget_ratio is None:
        encoder_char_budget: int | None = None
        encoder_program = encoder_signature()
        encoder_inputs: dict[str, Any] = {"code": job.ground_truth_code}
    else:
        # Budget is a character-length ratio against the ground-truth code
        # the encoder is shown (already stripped of comments/docstrings),
        # floored so tiny functions stay describable.
        encoder_char_budget = max(
            MIN_ENCODER_CHAR_BUDGET,
            round(job.budget_ratio * len(job.ground_truth_code)),
        )
        encoder_program = budgeted_encoder_signature()
        encoder_inputs = {
            "code": job.ground_truth_code,
            "max_characters": encoder_char_budget,
        }
    encoded_description = run_predictor(
        signature=encoder_program,
        input_kwargs=encoder_inputs,
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
        signature=decoder_signature(),
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
        encoder_char_budget=encoder_char_budget,
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
                    budget_ratio,
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
        budget_ratio=row[9],
        repetition_seed=row[10],
        prompt=row[11],
        canonical_solution=row[12],
        ground_truth_code=row[13],
        test=row[14],
        entry_point=row[15],
        encoder_reasoning=dict(row[16]),
        decoder_reasoning=dict(row[17]),
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
                    encoder_char_budget = %s,
                    raw_code = NULL,
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
                    result.encoder_char_budget,
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
                    encoded_description = NULL,
                    decoded_generation = NULL,
                    raw_generation = NULL,
                    raw_code = NULL,
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
                    raw_generation,
                    generation_status
                FROM dr_dspy_encdec_eval_predictions
                WHERE prediction_id = %s
                """,
                (prediction_id,),
            )
            row = cur.fetchone()
    if row is None:
        raise ValueError(f"prediction_id not found: {prediction_id}")
    if row[9] != "generated":
        raise ValueError(f"prediction_id is not generated: {prediction_id}")
    if row[7] is None:
        raise ValueError(
            f"prediction_id has no encoded description: {prediction_id}"
        )
    if row[8] is None:
        raise ValueError(
            f"prediction_id has no raw generation: {prediction_id}"
        )
    return ScoringTarget(
        prediction_id=row[0],
        task_id=row[1],
        prompt=row[2],
        canonical_solution=row[3],
        ground_truth_code=row[4] or row[2],
        test=row[5],
        entry_point=row[6],
        encoded_description=row[7],
        raw_generation=row[8],
    )


def score_generated_code(
    target: ScoringTarget, *, timeout: float
) -> ScoreResult:
    return score_humaneval_prediction(
        prediction_id=target.prediction_id,
        raw_generation=target.raw_generation,
        task=target.task(),
        compression_input=target.encoded_description,
        ground_truth_code=target.ground_truth_code,
        timeout=timeout,
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
                    raw_compression_ratio = %s,
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
        "generation_started", prediction_context_from_job(job)
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
        "generation_failed", context, extra={"error": error}
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
        "scoring_started", context, extra={"timeout": timeout}
    )
    return score_generated_code(
        fetch_scoring_target(database_url, prediction_id), timeout=timeout
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
        "scoring_failed", context, extra={"error": error}
    )
    record_score_error(database_url, prediction_id, error)


@DBOS.workflow(name="humaneval_encdec_generate_prediction_v0")
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
        record_generation_error_step(database_url, prediction_id, repr(error))
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
    except Exception as error:
        record_score_error_step(database_url, prediction_id, repr(error))
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
                    compression_metrics = '{}'::jsonb,
                    raw_compression_ratio = NULL,
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


QUEUE_NAME_CONFIG = shared_dbos.QueueNameConfig(
    generation_base_name=GENERATION_QUEUE_NAME,
    scoring_base_name=SCORING_QUEUE_NAME,
    hash_length=EXPERIMENT_QUEUE_HASH_LENGTH,
)

QueueSelection = shared_dbos.QueueSelection
EvalDbosConfig = shared_dbos.EvalDbosConfig
DbPoolConfig = shared_dbos.DbPoolConfig
OpenFileLimitResult = shared_dbos.OpenFileLimitResult
DB_POOL_AUTO = shared_dbos.DB_POOL_AUTO
DB_POOLS = shared_dbos.DB_POOLS
connect_db = shared_dbos.connect_db
close_db_connection_pools = shared_dbos.close_db_connection_pools
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
    return shared_flow.parse_float_csv(raw, value_name="temperature")


def build_humaneval_samples(
    *, seed: int, sample_count: int
) -> list[EncDecSample]:
    config = experiment_config()
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
            dataset_name=config.dataset_name,
            dataset_split=config.dataset_split,
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
        max_completion_tokens=experiment_config().default_max_completion_tokens,
    )


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
    database_url: str,
    jobs: Sequence[EncDecJob],
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


def fetch_status_counts(
    database_url: str, *, experiment_name: str | None
) -> list[dict[str, Any]]:
    return shared_eval_reporting.fetch_status_counts(
        database_url,
        prediction_table=PREDICTION_TABLE_NAME,
        dimension_columns=REPORTING_DIMENSION_COLUMNS,
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
        dimensions=status_dimensions(DIMENSIONS),
        experiment_name=experiment_name,
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
                    budget_ratio,
                    task_id,
                    repetition_seed,
                    score,
                    provider_cost,
                    raw_compile_ok,
                    extracted_compile_ok,
                    raw_compression_ratio,
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
                    budget_ratio,
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
            budget_ratio=row[4],
            task_id=row[5],
            repetition_seed=row[6],
            score=row[7],
            provider_cost=row[8],
            raw_compile_ok=row[9],
            extracted_compile_ok=row[10],
            raw_compression_ratio=row[11],
            best_compression_ratio=row[12],
            best_compression_percent_reduction=row[13],
        )
        for row in rows
    ]


def summarize_analysis_records(
    records: Sequence[AnalysisRecord],
) -> list[AnalysisSummary]:
    return shared_flow.summarize_analysis_records(
        records,
        group_key=lambda record: (
            record.encoder_model,
            record.decoder_model,
            str(record.encoder_temperature),
            str(record.decoder_temperature),
            str(record.budget_ratio),
        ),
        dimension_values=lambda record: {
            "encoder_model": record.encoder_model,
            "decoder_model": record.decoder_model,
            "encoder_temperature": record.encoder_temperature,
            "decoder_temperature": record.decoder_temperature,
            "budget_ratio": record.budget_ratio,
        },
        task_id=lambda record: record.task_id,
        score=lambda record: record.score,
        provider_cost=lambda record: record.provider_cost,
        raw_compile_ok=lambda record: record.raw_compile_ok,
        extracted_compile_ok=lambda record: record.extracted_compile_ok,
        raw_compression_ratio=lambda record: record.raw_compression_ratio,
        best_compression_ratio=lambda record: record.best_compression_ratio,
        best_compression_percent_reduction=(
            lambda record: record.best_compression_percent_reduction
        ),
        summary_factory=AnalysisSummary,
    )


def analysis_markdown(
    *, experiment_name: str, summaries: Sequence[AnalysisSummary]
) -> str:
    return shared_eval_reporting.analysis_markdown(
        experiment_name=experiment_name,
        summaries=summaries,
        dimensions=status_dimensions(DIMENSIONS),
    )


def analysis_table(
    *, experiment_name: str, summaries: Sequence[AnalysisSummary]
) -> object:
    return shared_eval_reporting.analysis_table(
        experiment_name=experiment_name,
        summaries=summaries,
        dimensions=status_dimensions(DIMENSIONS),
    )


def write_analysis_csv(
    summaries: Sequence[AnalysisSummary], csv_path: Path
) -> None:
    shared_eval_reporting.write_analysis_csv(
        summaries,
        csv_path=csv_path,
        fieldnames=[
            *REPORTING_DIMENSION_COLUMNS,
            *(
                name
                for name in AnalysisSummary.model_fields
                if name != "dimensions"
            ),
        ],
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


class EncDecExperiment:
    prediction_table = PREDICTION_TABLE_NAME
    dimensions = DIMENSIONS

    def create_schema(self, database_url: str) -> None:
        create_eval_schema(database_url)

    def upsert_experiment(
        self,
        database_url: str,
        *,
        experiment_name: str,
        seed: int,
        sample_count: int,
        metadata: Mapping[str, Any],
    ) -> None:
        upsert_experiment(
            database_url,
            experiment_name=experiment_name,
            seed=seed,
            sample_count=sample_count,
            metadata=metadata,
        )

    def insert_prediction_jobs(
        self, database_url: str, jobs: Sequence[EncDecJob]
    ) -> int:
        return insert_prediction_jobs(database_url, jobs)

    def configure_runtime(
        self,
        config: EvalDbosConfig,
        experiment_name: str,
        *,
        consume_queues: bool = True,
    ) -> None:
        configure_dbos_runtime(
            config,
            experiment_name=experiment_name,
            consume_queues=consume_queues,
        )

    def enqueue_generation_jobs(
        self,
        database_url: str,
        jobs: Sequence[EncDecJob],
        *,
        score_timeout: float,
        retry_token: str | None = None,
    ) -> None:
        enqueue_generation_jobs(
            database_url,
            jobs,
            score_timeout=score_timeout,
            retry_token=retry_token,
        )

    def mark_generation_started(
        self, database_url: str, prediction_id: str
    ) -> None:
        mark_generation_started(database_url, prediction_id)

    def fetch_prediction_job(
        self, database_url: str, prediction_id: str
    ) -> EncDecJob:
        return fetch_prediction_job(database_url, prediction_id)

    def generate_code_for_job(self, job: EncDecJob) -> GenerationResult:
        return generate_code_for_job(job)

    def record_generation_success(
        self, database_url: str, result: GenerationResult
    ) -> None:
        record_generation_success(database_url, result)

    def record_generation_error(
        self, database_url: str, prediction_id: str, error: str
    ) -> None:
        record_generation_error(database_url, prediction_id, error)

    def generation_success_log_extra(
        self, result: GenerationResult
    ) -> Mapping[str, Any]:
        return {
            "provider_cost": result.provider_cost,
            "encoder_usage_metadata": result.encoder_usage_metadata,
            "decoder_usage_metadata": result.decoder_usage_metadata,
        }

    def prediction_context_from_job(
        self, job: EncDecJob
    ) -> shared_eval_logging.PredictionLogContext:
        return prediction_context_from_job(job)

    def fetch_prediction_log_context(
        self, database_url: str, prediction_id: str
    ) -> shared_eval_logging.PredictionLogContext:
        return fetch_prediction_log_context(database_url, prediction_id)

    def emit_prediction_log_event(
        self,
        event: str,
        context: shared_eval_logging.PredictionLogContext,
        *,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        emit_prediction_log_event(event, context, extra=extra)

    def mark_scoring_started(
        self, database_url: str, prediction_id: str
    ) -> None:
        mark_scoring_started(database_url, prediction_id)

    def mark_scoring_queued(
        self, database_url: str, prediction_ids: Sequence[str]
    ) -> int:
        return mark_scoring_queued(database_url, prediction_ids)

    def score_prediction(
        self, database_url: str, prediction_id: str, timeout: float
    ) -> ScoreResult:
        return score_generated_code(
            fetch_scoring_target(database_url, prediction_id),
            timeout=timeout,
        )

    def record_score_success(
        self, database_url: str, result: ScoreResult
    ) -> None:
        record_score_success(database_url, result)

    def record_score_error(
        self, database_url: str, prediction_id: str, error: str
    ) -> None:
        record_score_error(database_url, prediction_id, error)

    def enqueue_score(
        self,
        database_url: str,
        prediction_id: str,
        *,
        experiment_name: str,
        timeout: float,
    ) -> None:
        enqueue_score_job(
            database_url,
            prediction_id,
            experiment_name=experiment_name,
            timeout=timeout,
        )

    def enqueue_score_jobs(
        self,
        database_url: str,
        prediction_ids: Sequence[str],
        *,
        experiment_name: str,
        timeout: float,
        retry_token: str | None = None,
    ) -> None:
        enqueue_score_jobs(
            database_url,
            prediction_ids,
            experiment_name=experiment_name,
            timeout=timeout,
            retry_token=retry_token,
        )

    def fetch_scoreable_prediction_ids(
        self, database_url: str, *, experiment_name: str, limit: int
    ) -> list[str]:
        return fetch_scoreable_prediction_ids(
            database_url, experiment_name=experiment_name, limit=limit
        )

    def build_repair_plan(
        self,
        database_url: str,
        *,
        dbos_system_database_url: str,
        experiment_name: str,
        generation_limit: int,
        scoring_limit: int,
    ) -> shared_eval_repair.RepairPlan:
        return build_repair_plan(
            database_url,
            dbos_system_database_url=dbos_system_database_url,
            experiment_name=experiment_name,
            generation_limit=generation_limit,
            scoring_limit=scoring_limit,
        )

    def apply_repair(
        self,
        config: EvalDbosConfig,
        *,
        experiment_name: str,
        generation_limit: int,
        scoring_limit: int,
        score_timeout: float,
        repair_token: str | None = None,
    ) -> shared_eval_repair.RepairApplyResult:
        return apply_repair(
            config,
            experiment_name=experiment_name,
            generation_limit=generation_limit,
            scoring_limit=scoring_limit,
            score_timeout=score_timeout,
            repair_token=repair_token,
        )

    def fetch_status_counts(
        self, database_url: str, *, experiment_name: str | None
    ) -> list[dict[str, Any]]:
        return fetch_status_counts(
            database_url, experiment_name=experiment_name
        )

    def status_counts_table(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        experiment_name: str | None,
    ) -> Table:
        return status_counts_table(rows, experiment_name=experiment_name)

    def fetch_analysis_records(
        self, database_url: str, *, experiment_name: str
    ) -> list[AnalysisRecord]:
        return fetch_analysis_records(
            database_url, experiment_name=experiment_name
        )

    def summarize_analysis_records(
        self, records: Sequence[AnalysisRecord]
    ) -> list[AnalysisSummary]:
        return summarize_analysis_records(records)

    def analysis_table(
        self, *, experiment_name: str, summaries: Sequence[AnalysisSummary]
    ) -> object:
        return analysis_table(
            experiment_name=experiment_name, summaries=summaries
        )

    def analysis_markdown(
        self, *, experiment_name: str, summaries: Sequence[AnalysisSummary]
    ) -> str:
        return analysis_markdown(
            experiment_name=experiment_name, summaries=summaries
        )

    def write_analysis_csv(
        self, summaries: Sequence[AnalysisSummary], csv_path: Path
    ) -> None:
        write_analysis_csv(summaries, csv_path)

    def configure_pooled_worker_runtime(
        self,
        config: EvalDbosConfig,
        *,
        experiment_name: str,
        queue: QueueSelection,
        raw_db_pool_max_size: str,
    ) -> DbPoolConfig:
        return configure_pooled_worker_runtime(
            config,
            experiment_name=experiment_name,
            queue=queue,
            raw_db_pool_max_size=raw_db_pool_max_size,
        )

    def queue_names_for_selection(
        self, selection: QueueSelection, *, experiment_name: str
    ) -> tuple[str, ...]:
        return queue_names_for_selection(
            selection, experiment_name=experiment_name
        )

    def resolve_worker_log_path(
        self,
        *,
        experiment_name: str,
        queue: QueueSelection,
        log_file: Path | None,
    ) -> Path:
        return resolve_worker_log_path(
            experiment_name=experiment_name, queue=queue, log_file=log_file
        )

    def configure_worker_file_logging(self, log_file: Path) -> logging.Logger:
        return configure_worker_file_logging(log_file)

    def start_worker_monitor(
        self,
        monitor_config: WorkerMonitorConfig,
        stop_event: threading.Event,
    ) -> threading.Thread:
        return start_worker_monitor(monitor_config, stop_event)


_BACKEND = EncDecExperiment()

_APP = typer.Typer(no_args_is_help=True)


def create_app(config: EncDecHumanEvalExperimentConfig) -> typer.Typer:
    configure_experiment(config)
    return _APP


@_APP.command("init-db")
def init_db(
    database_url: Annotated[str | None, typer.Option()] = None,
    env_file: Annotated[Path | None, typer.Option()] = None,
) -> None:
    load_optional_env_file(env_file)
    create_eval_schema(resolve_database_url(database_url))
    operator_log("initialized enc-dec eval schema")


@_APP.command()
def submit(
    experiment_name: Annotated[str, typer.Option()],
    database_url: Annotated[str | None, typer.Option()] = None,
    dbos_system_database_url: Annotated[str | None, typer.Option()] = None,
    model_pairs_json: Annotated[str | None, typer.Option()] = None,
    sample_count: Annotated[int | None, typer.Option()] = None,
    seed: Annotated[int | None, typer.Option()] = None,
    encoder_temperatures: Annotated[str | None, typer.Option()] = None,
    decoder_temperatures: Annotated[str | None, typer.Option()] = None,
    budget_ratios: Annotated[
        str | None,
        typer.Option(
            "--budget-ratios",
            help=(
                "Comma-separated encoder char-budget ratios; use 'none' "
                "for no budget."
            ),
        ),
    ] = None,
    repetitions: Annotated[int | None, typer.Option()] = None,
    generation_concurrency: Annotated[
        int, typer.Option()
    ] = DEFAULT_GENERATION_CONCURRENCY,
    scoring_concurrency: Annotated[
        int, typer.Option()
    ] = DEFAULT_SCORING_CONCURRENCY,
    score_timeout: Annotated[float | None, typer.Option()] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Plan jobs without writing or enqueueing.",
        ),
    ] = False,
    env_file: Annotated[Path | None, typer.Option()] = None,
) -> None:
    experiment = experiment_config()
    sample_count = sample_count or experiment.default_sample_count
    seed = seed if seed is not None else experiment.default_seed
    encoder_temperatures = encoder_temperatures or ",".join(
        str(value) for value in experiment.default_encoder_temperatures
    )
    decoder_temperatures = decoder_temperatures or ",".join(
        str(value) for value in experiment.default_decoder_temperatures
    )
    budget_ratios = budget_ratios or ",".join(
        "none" if value is None else str(value)
        for value in experiment.default_budget_ratios
    )
    repetitions = repetitions or experiment.default_repetitions
    score_timeout = (
        score_timeout
        if score_timeout is not None
        else experiment.default_subprocess_timeout
    )
    load_optional_env_file(env_file)
    config = build_eval_dbos_config(
        database_url=database_url,
        dbos_system_database_url=dbos_system_database_url,
        generation_concurrency=generation_concurrency,
        scoring_concurrency=scoring_concurrency,
    )
    model_pairs = parse_model_pairs(model_pairs_json)
    samples = build_humaneval_samples(seed=seed, sample_count=sample_count)
    parsed_encoder_temperatures = parse_temperatures(encoder_temperatures)
    parsed_decoder_temperatures = parse_temperatures(decoder_temperatures)
    parsed_budget_ratios = parse_budget_ratios(budget_ratios)
    jobs = build_prediction_jobs(
        experiment_name=experiment_name,
        submission_id=uuid.uuid4().hex,
        samples=samples,
        model_pairs=model_pairs,
        encoder_temperatures=parsed_encoder_temperatures,
        decoder_temperatures=parsed_decoder_temperatures,
        budget_ratios=parsed_budget_ratios,
        repetitions=repetitions,
    )
    metadata = {
        "model_pairs": [pair.model_dump(mode="json") for pair in model_pairs],
        "encoder_temperatures": parsed_encoder_temperatures,
        "decoder_temperatures": parsed_decoder_temperatures,
        "budget_ratios": parsed_budget_ratios,
        "repetitions": repetitions,
    }
    operator_log(
        f"planned {len(jobs)} jobs: samples={len(samples)}, "
        f"model_pairs={len(model_pairs)}, "
        f"encoder_temperatures={len(metadata['encoder_temperatures'])}, "
        f"decoder_temperatures={len(metadata['decoder_temperatures'])}, "
        f"budget_ratios={len(parsed_budget_ratios)}, "
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
        _BACKEND,
        config=config,
        experiment_name=experiment_name,
        seed=seed,
        sample_count=sample_count,
        metadata=metadata,
        jobs=jobs,
        score_timeout=score_timeout,
    )


@_APP.command()
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
    load_optional_env_file(env_file)
    config = build_eval_dbos_config(
        database_url=database_url,
        dbos_system_database_url=dbos_system_database_url,
        generation_concurrency=generation_concurrency,
        scoring_concurrency=scoring_concurrency,
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


@_APP.command()
def status(
    experiment_name: Annotated[str | None, typer.Option()] = None,
    database_url: Annotated[str | None, typer.Option()] = None,
    env_file: Annotated[Path | None, typer.Option()] = None,
) -> None:
    load_optional_env_file(env_file)
    shared_flow.run_status_command(
        _BACKEND,
        database_url=resolve_database_url(database_url),
        experiment_name=experiment_name,
    )


@_APP.command("enqueue-scores")
def enqueue_scores_command(
    experiment_name: Annotated[str, typer.Option()],
    limit: Annotated[
        int, typer.Option("--limit", min=1)
    ] = DEFAULT_SCORE_ENQUEUE_LIMIT,
    timeout: Annotated[
        float | None, typer.Option("--timeout", min=0.1)
    ] = None,
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
    timeout = (
        timeout
        if timeout is not None
        else experiment_config().default_subprocess_timeout
    )
    load_optional_env_file(env_file)
    config = build_eval_dbos_config(
        database_url=database_url,
        dbos_system_database_url=dbos_system_database_url,
        generation_concurrency=generation_concurrency,
        scoring_concurrency=scoring_concurrency,
    )
    shared_flow.run_enqueue_scores_command(
        _BACKEND,
        config=config,
        experiment_name=experiment_name,
        limit=limit,
        timeout=timeout,
    )


@_APP.command()
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
    shared_flow.run_analyze_command(
        _BACKEND,
        database_url=resolve_database_url(database_url),
        experiment_name=experiment_name,
        csv_path=csv_path,
        markdown=markdown,
    )


@_APP.command()
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
    score_timeout: Annotated[float | None, typer.Option()] = None,
    apply: Annotated[bool, typer.Option("--apply")] = False,
    env_file: Annotated[Path | None, typer.Option()] = None,
) -> None:
    score_timeout = (
        score_timeout
        if score_timeout is not None
        else experiment_config().default_subprocess_timeout
    )
    load_optional_env_file(env_file)
    config = build_eval_dbos_config(
        database_url=database_url,
        dbos_system_database_url=dbos_system_database_url,
        generation_concurrency=generation_concurrency,
        scoring_concurrency=scoring_concurrency,
    )
    shared_flow.run_repair_command(
        _BACKEND,
        config=config,
        experiment_name=experiment_name,
        generation_limit=generation_limit,
        scoring_limit=scoring_limit,
        score_timeout=score_timeout,
        apply=apply,
    )
