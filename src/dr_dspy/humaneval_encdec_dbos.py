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

import dspy
from dr_dspy import batch_operation as shared_batch
from dr_dspy import dbos_runtime as shared_dbos
from dr_dspy import dspy_runner as shared_dspy_runner
from dr_dspy import eval_logging as shared_eval_logging
from dr_dspy import eval_repair as shared_eval_repair
from dr_dspy import human_eval_sampling as shared_human_eval_sampling
from dr_dspy import humaneval_dbos_flow as shared_flow
from dr_dspy import worker_monitor as shared_worker_monitor
from dr_dspy import worker_resources as shared_worker_resources
from dr_dspy.experiment_dimensions import (
    Dimension,
    dimension_columns_ddl,
    identity_constraint_columns,
    identity_dimension_names,
)
from dr_dspy.failure_policy import (
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
    provider_cost_from_response,
    usage_metadata_from_response,
)
from dr_dspy.prediction_status import (
    GENERATION_RETRY_STATUSES,
    GenerationStatus,
    ScoringStatus,
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
DEFAULT_WORKER_OPEN_FILE_LIMIT = shared_worker_resources.OPEN_FILE_LIMIT_AUTO
DEFAULT_WORKER_MONITOR_INTERVAL_SECONDS = 5.0
DEFAULT_WORKER_MONITOR_SUMMARY_INTERVAL_SECONDS = 5.0
DEFAULT_SUBMIT_BATCH_SIZE = 512
DEFAULT_OPERATION_BATCH_SIZE = 512
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
SUBMIT_LOGGER_NAME = "dr_dspy.humaneval_encdec_submit"
OPERATION_LOGGER_NAME = "dr_dspy.humaneval_encdec_operations"
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
    generation_failure_class TEXT,
    generation_exception_type TEXT,
    generation_exception_message TEXT,
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
    scoring_failure_class TEXT,
    scoring_exception_type TEXT,
    scoring_exception_message TEXT,
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

PREDICTION_MIGRATION_SQL = (
    "ALTER TABLE dr_dspy_encdec_eval_predictions "
    "ADD COLUMN IF NOT EXISTS generation_failure_class TEXT",
    "ALTER TABLE dr_dspy_encdec_eval_predictions "
    "ADD COLUMN IF NOT EXISTS generation_exception_type TEXT",
    "ALTER TABLE dr_dspy_encdec_eval_predictions "
    "ADD COLUMN IF NOT EXISTS generation_exception_message TEXT",
    "ALTER TABLE dr_dspy_encdec_eval_predictions "
    "ADD COLUMN IF NOT EXISTS scoring_failure_class TEXT",
    "ALTER TABLE dr_dspy_encdec_eval_predictions "
    "ADD COLUMN IF NOT EXISTS scoring_exception_type TEXT",
    "ALTER TABLE dr_dspy_encdec_eval_predictions "
    "ADD COLUMN IF NOT EXISTS scoring_exception_message TEXT",
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


class EncDecSubmitSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    script_kind: StrictStr
    experiment_name: StrictStr
    seed: StrictInt
    sample_count: StrictInt
    model_pairs: list[EncDecPair]
    encoder_temperatures: list[float]
    decoder_temperatures: list[float]
    budget_ratios: list[float | None]
    repetitions: StrictInt
    score_timeout: float

    def jobs_per_sample(self) -> int:
        return (
            len(self.model_pairs)
            * len(self.encoder_temperatures)
            * len(self.decoder_temperatures)
            * len(self.budget_ratios)
            * self.repetitions
        )

    def total_jobs(self) -> int:
        return self.sample_count * self.jobs_per_sample()


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


def build_submit_spec(
    *,
    experiment_name: str,
    seed: int,
    sample_count: int,
    model_pairs: Sequence[EncDecPair],
    encoder_temperatures: Sequence[float],
    decoder_temperatures: Sequence[float],
    budget_ratios: Sequence[float | None],
    repetitions: int,
    score_timeout: float,
) -> EncDecSubmitSpec:
    return EncDecSubmitSpec(
        script_kind=experiment_config().script_kind,
        experiment_name=experiment_name,
        seed=seed,
        sample_count=sample_count,
        model_pairs=[
            EncDecPair(**pair.model_dump(mode="python"))
            for pair in model_pairs
        ],
        encoder_temperatures=list(encoder_temperatures),
        decoder_temperatures=list(decoder_temperatures),
        budget_ratios=list(budget_ratios),
        repetitions=repetitions,
        score_timeout=score_timeout,
    )


def build_prediction_jobs_for_offsets(
    *,
    spec: EncDecSubmitSpec,
    submission_id: str,
    samples: Sequence[EncDecSample],
    start_offset: int,
    limit: int,
) -> list[EncDecJob]:
    total_jobs = spec.total_jobs()
    end_offset = min(start_offset + limit, total_jobs)
    samples_by_index = {sample.sample_index: sample for sample in samples}
    jobs: list[EncDecJob] = []
    for offset in range(start_offset, end_offset):
        remaining = offset
        repetition_seed = remaining % spec.repetitions
        remaining //= spec.repetitions
        budget_ratio = spec.budget_ratios[
            remaining % len(spec.budget_ratios)
        ]
        remaining //= len(spec.budget_ratios)
        decoder_temperature = spec.decoder_temperatures[
            remaining % len(spec.decoder_temperatures)
        ]
        remaining //= len(spec.decoder_temperatures)
        encoder_temperature = spec.encoder_temperatures[
            remaining % len(spec.encoder_temperatures)
        ]
        remaining //= len(spec.encoder_temperatures)
        pair = spec.model_pairs[remaining % len(spec.model_pairs)]
        remaining //= len(spec.model_pairs)
        sample = samples_by_index.get(remaining)
        if sample is None:
            raise ValueError(
                "missing submit sample manifest item: "
                f"sample_index={remaining}"
            )
        jobs.append(
            EncDecJob(
                prediction_id=stable_prediction_id(
                    experiment_name=spec.experiment_name,
                    task_id=sample.task_id,
                    encoder_model=pair.encoder.model,
                    decoder_model=pair.decoder.model,
                    encoder_temperature=encoder_temperature,
                    decoder_temperature=decoder_temperature,
                    budget_ratio=budget_ratio,
                    encoder_reasoning=pair.encoder.reasoning,
                    decoder_reasoning=pair.decoder.reasoning,
                    repetition_seed=repetition_seed,
                ),
                experiment_name=spec.experiment_name,
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
    with shared_dbos.connect_db(database_url) as conn:
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
    with shared_dbos.connect_db(database_url) as conn:
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
    return generate_code_for_job_with_client(
        job, client=shared_worker_resources.openrouter_client()
    )


def generate_code_for_job_with_client(
    job: EncDecJob,
    *,
    client: Any = None,
) -> GenerationResult:
    encoder_events = LmEventBuffer()
    decoder_events = LmEventBuffer()

    encoder_lm = build_lm(
        model=job.encoder_model,
        reasoning=job.encoder_reasoning,
        temperature=job.encoder_temperature,
        event_buffer=encoder_events,
        client=client,
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
        client=client,
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
    with shared_dbos.connect_db(database_url) as conn:
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
    return _prediction_context_from_job(job)


def mark_generation_started(database_url: str, prediction_id: str) -> None:
    with shared_dbos.connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE dr_dspy_encdec_eval_predictions
                SET generation_status = 'started',
                    generation_error = NULL,
                    generation_failure_class = NULL,
                    generation_exception_type = NULL,
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
                UPDATE dr_dspy_encdec_eval_predictions
                SET generation_status = 'generated',
                    generation_error = NULL,
                    generation_failure_class = NULL,
                    generation_exception_type = NULL,
                    generation_exception_message = NULL,
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
                UPDATE dr_dspy_encdec_eval_predictions
                SET generation_status = %s,
                    generation_error = %s,
                    generation_failure_class = %s,
                    generation_exception_type = %s,
                    generation_exception_message = %s,
                    encoded_description = NULL,
                    decoded_generation = NULL,
                    raw_generation = NULL,
                    raw_code = NULL,
                    updated_at = now()
                WHERE prediction_id = %s
                """,
                (
                    status,
                    error_text(summary),
                    summary.failure_class.value,
                    summary.exception_type,
                    summary.message,
                    prediction_id,
                ),
            )


def mark_scoring_started(database_url: str, prediction_id: str) -> None:
    with shared_dbos.connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE dr_dspy_encdec_eval_predictions
                SET scoring_status = 'started',
                    scoring_error = NULL,
                    scoring_failure_class = NULL,
                    scoring_exception_type = NULL,
                    scoring_exception_message = NULL,
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
    with shared_dbos.connect_db(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE dr_dspy_encdec_eval_predictions
                SET scoring_status = 'scored',
                    score = %s,
                    scoring_error = %s,
                    scoring_failure_class = NULL,
                    scoring_exception_type = NULL,
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
                UPDATE dr_dspy_encdec_eval_predictions
                SET scoring_status = %s,
                    scoring_error = %s,
                    scoring_failure_class = %s,
                    scoring_exception_type = %s,
                    scoring_exception_message = %s,
                    updated_at = now()
                WHERE prediction_id = %s
                """,
                (
                    status,
                    error_text(summary),
                    summary.failure_class.value,
                    summary.exception_type,
                    summary.message,
                    prediction_id,
                ),
            )


@DBOS.step(
    name="humaneval_encdec_generate_prediction_step_v0",
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
    return generate_code_for_job(job)


@DBOS.step(name="humaneval_encdec_record_generation_success_step_v0")
def record_generation_success_step(
    database_url: str, result: GenerationResult
) -> None:
    context = fetch_prediction_log_context(database_url, result.prediction_id)
    _emit_prediction_log_event(
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
    database_url: str, prediction_id: str, summary: FailureSummary
) -> None:
    context = fetch_prediction_log_context(database_url, prediction_id)
    _emit_prediction_log_event(
        "generation_failed",
        context,
        extra=failure_summary_payload(summary),
    )
    record_generation_error(database_url, prediction_id, summary)


@DBOS.step(name="humaneval_encdec_mark_scoring_queued_step_v0")
def mark_scoring_queued_step(database_url: str, prediction_id: str) -> None:
    context = fetch_prediction_log_context(database_url, prediction_id)
    _emit_prediction_log_event("scoring_enqueued", context)
    mark_scoring_queued(database_url, [prediction_id])


@DBOS.step(name="humaneval_encdec_score_prediction_step_v0")
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


@DBOS.step(name="humaneval_encdec_record_score_success_step_v0")
def record_score_success_step(database_url: str, result: ScoreResult) -> None:
    context = fetch_prediction_log_context(database_url, result.prediction_id)
    _emit_prediction_log_event(
        "scoring_succeeded",
        context,
        extra={"score": result.score, "scoring_error": result.error},
    )
    record_score_success(database_url, result)


@DBOS.step(name="humaneval_encdec_record_score_error_step_v0")
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


@DBOS.workflow(name="humaneval_encdec_score_prediction_v0")
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
    name="humaneval_encdec_submit_dispatcher_v0",
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
    spec = EncDecSubmitSpec(
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
    samples = [EncDecSample(**payload) for payload in sample_payloads]
    jobs = build_prediction_jobs_for_offsets(
        spec=spec,
        submission_id=str(progress.metadata["submission_id"]),
        samples=samples,
        start_offset=progress.next_offset,
        limit=int(progress.metadata["batch_size"]),
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
    name="humaneval_encdec_enqueue_scores_dispatcher_v0",
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
            "limit": progress.total_items,
        },
        failed_event="enqueue_scores_dispatcher_failed",
        batch_step=enqueue_scores_batch_step,
        completion_mode=completion_modes.PROCESSED_TOTAL_OR_EMPTY_BATCH,
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
    remaining = max(progress.total_items - progress.processed_count, 0)
    limit = min(int(spec["batch_size"]), remaining)
    prediction_ids = fetch_scoreable_prediction_ids(
        database_url,
        experiment_name=str(spec["experiment_name"]),
        limit=limit,
    )
    _emit_operation_log(
        "enqueue_scores_batch_started",
        {
            "operation_key": operation_key,
            "selected": len(prediction_ids),
            "remaining": remaining,
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
    name="humaneval_encdec_repair_dispatcher_v0",
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
            "limit": progress.total_items,
        },
        failed_event="repair_dispatcher_failed",
        batch_step=repair_batch_step,
        completion_mode=completion_modes.PROCESSED_TOTAL_OR_EMPTY_BATCH,
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
    generation_limit = int(spec["generation_limit"])
    scoring_limit = int(spec["scoring_limit"])
    generation_processed = int(
        progress.counters.get("generation_processed", 0)
    )
    scoring_processed = int(progress.counters.get("scoring_processed", 0))
    remaining_generation = max(generation_limit - generation_processed, 0)
    remaining_scoring = max(scoring_limit - scoring_processed, 0)
    _emit_operation_log(
        "repair_batch_started",
        {
            "operation_key": operation_key,
            "remaining_generation": remaining_generation,
            "remaining_scoring": remaining_scoring,
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
        generation_limit=remaining_generation,
        scoring_limit=remaining_scoring,
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
                UPDATE dr_dspy_encdec_eval_predictions
                SET
                    generation_status = %s,
                    generation_error = NULL,
                    generation_failure_class = NULL,
                    generation_exception_type = NULL,
                    generation_exception_message = NULL,
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
                    scoring_status = %s,
                    scoring_error = NULL,
                    scoring_failure_class = NULL,
                    scoring_exception_type = NULL,
                    scoring_exception_message = NULL,
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


def _build_repair_plan(
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


QUEUE_NAME_CONFIG = shared_dbos.QueueNameConfig(
    generation_base_name=GENERATION_QUEUE_NAME,
    scoring_base_name=SCORING_QUEUE_NAME,
    hash_length=EXPERIMENT_QUEUE_HASH_LENGTH,
)

def _resolve_database_url(database_url: str | None) -> str:
    return shared_dbos.resolve_database_url(
        database_url,
        database_url_env=DATABASE_URL_ENV,
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
            *PREDICTION_INDEX_SQL,
            *shared_batch.operation_index_sql(),
            *shared_batch.operation_item_index_sql(),
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
    client: Any = None,
) -> dspy.BaseLM:
    return shared_dspy_runner.build_logged_lm(
        model=model,
        reasoning=reasoning,
        temperature=temperature,
        event_buffer=event_buffer,
        max_completion_tokens=experiment_config().default_max_completion_tokens,
        client=client,
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


def _queue_names_for_selection(
    selection: shared_dbos.QueueSelection, *, experiment_name: str
) -> tuple[str, ...]:
    return shared_dbos.queue_names_for_selection(
        selection,
        experiment_name=experiment_name,
        queue_config=QUEUE_NAME_CONFIG,
    )


def _enqueue_generation_jobs(
    database_url: str,
    jobs: Sequence[EncDecJob],
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


class EncDecExperiment:
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
        generation_limit: int,
        scoring_limit: int,
    ) -> shared_eval_repair.RepairPlan:
        return _build_repair_plan(
            database_url,
            dbos_system_database_url=dbos_system_database_url,
            experiment_name=experiment_name,
            generation_limit=generation_limit,
            scoring_limit=scoring_limit,
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
    _create_eval_schema(_resolve_database_url(database_url))
    _operator_log("initialized enc-dec eval schema")


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
    batch_size: Annotated[
        int, typer.Option("--batch-size", min=1)
    ] = DEFAULT_SUBMIT_BATCH_SIZE,
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
    config = _build_eval_dbos_config(
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
    submission_id = uuid.uuid4().hex
    submit_spec = build_submit_spec(
        experiment_name=experiment_name,
        seed=seed,
        sample_count=len(samples),
        model_pairs=model_pairs,
        encoder_temperatures=parsed_encoder_temperatures,
        decoder_temperatures=parsed_decoder_temperatures,
        budget_ratios=parsed_budget_ratios,
        repetitions=repetitions,
        score_timeout=score_timeout,
    )
    total_jobs = submit_spec.total_jobs()
    operation_key = shared_batch.operation_key(
        submit_spec.model_dump(mode="json")
    )
    metadata = {
        "submission_id": submission_id,
        "operation_key": operation_key,
        "batch_size": batch_size,
        "model_pairs": [pair.model_dump(mode="json") for pair in model_pairs],
        "encoder_temperatures": parsed_encoder_temperatures,
        "decoder_temperatures": parsed_decoder_temperatures,
        "budget_ratios": parsed_budget_ratios,
        "repetitions": repetitions,
        "score_timeout": score_timeout,
    }
    _operator_log(
        f"planned {total_jobs} jobs: samples={len(samples)}, "
        f"model_pairs={len(model_pairs)}, "
        f"encoder_temperatures={len(metadata['encoder_temperatures'])}, "
        f"decoder_temperatures={len(metadata['decoder_temperatures'])}, "
        f"budget_ratios={len(parsed_budget_ratios)}, "
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


@_APP.command()
def worker(
    experiment_name: Annotated[str, typer.Option()],
    database_url: Annotated[str | None, typer.Option()] = None,
    dbos_system_database_url: Annotated[str | None, typer.Option()] = None,
    queue: Annotated[
        shared_dbos.QueueSelection, typer.Option()
    ] = shared_dbos.QueueSelection.BOTH,
    generation_concurrency: Annotated[
        int, typer.Option()
    ] = DEFAULT_GENERATION_CONCURRENCY,
    scoring_concurrency: Annotated[
        int, typer.Option()
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
    load_optional_env_file(env_file)
    config = _build_eval_dbos_config(
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
    load_optional_env_file(env_file)
    config = _build_eval_dbos_config(
        database_url=database_url,
        dbos_system_database_url=dbos_system_database_url,
        generation_concurrency=generation_concurrency,
        scoring_concurrency=scoring_concurrency,
    )
    _create_eval_schema(config.database_url)
    resolved_operation_key = operation_key or shared_batch.new_operation_key()
    operation_kind = shared_batch.BatchOperationKind.ENQUEUE_SCORES
    spec = {
        "experiment_name": experiment_name,
        "limit": limit,
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
        total_items=limit,
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
    apply: Annotated[bool, typer.Option("--apply")] = False,
    env_file: Annotated[Path | None, typer.Option()] = None,
) -> None:
    score_timeout = (
        score_timeout
        if score_timeout is not None
        else experiment_config().default_subprocess_timeout
    )
    load_optional_env_file(env_file)
    config = _build_eval_dbos_config(
        database_url=database_url,
        dbos_system_database_url=dbos_system_database_url,
        generation_concurrency=generation_concurrency,
        scoring_concurrency=scoring_concurrency,
    )
    if not apply:
        shared_flow.run_repair_command(
            _BACKEND,
            config=config,
            experiment_name=experiment_name,
            generation_limit=generation_limit,
            scoring_limit=scoring_limit,
            score_timeout=score_timeout,
        )
        return

    _create_eval_schema(config.database_url)
    operation_kind = shared_batch.BatchOperationKind.REPAIR
    spec = {
        "experiment_name": experiment_name,
        "generation_limit": generation_limit,
        "scoring_limit": scoring_limit,
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
        total_items=generation_limit + scoring_limit,
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
