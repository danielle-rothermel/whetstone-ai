from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import threading
import uuid
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Annotated, Any, Protocol, cast

import psycopg
import typer
from datasets import load_dataset  # type: ignore[import-not-found]
from dbos import DBOS, DBOSConfig, SetWorkflowID
from psycopg.types.json import Jsonb
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

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
DEFAULT_SEED = 0
DEFAULT_SAMPLE_COUNT = 10
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TEMPERATURES = (DEFAULT_TEMPERATURE,)
DEFAULT_MAX_COMPLETION_TOKENS = 1000
DEFAULT_SUBPROCESS_TIMEOUT = 15.0
MAX_TRACE_SIZE = 10_000
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


class QueueSelection(str, Enum):
    GENERATION = "generation"
    SCORING = "scoring"
    BOTH = "both"


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
    if selection is QueueSelection.GENERATION:
        DBOS.listen_queues([GENERATION_QUEUE_NAME])
        return
    if selection is QueueSelection.SCORING:
        DBOS.listen_queues([SCORING_QUEUE_NAME])
        return
    DBOS.listen_queues([GENERATION_QUEUE_NAME, SCORING_QUEUE_NAME])


def configure_dbos_runtime(
    config: EvalDbosConfig,
    *,
    queue: QueueSelection | None = None,
    consume_queues: bool = True,
) -> None:
    DBOS(config=build_dbos_config(config))
    register_eval_queues(config)
    if queue is not None:
        listen_to_selected_queue(queue)
    elif not consume_queues:
        DBOS.listen_queues([])
    DBOS.launch()


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
    target = fetch_scoring_target(database_url, prediction_id)
    return score_generated_code(target, timeout=timeout)


@DBOS.step()
def record_score_success_step(
    database_url: str, result: ScoreResult
) -> None:
    record_score_success(database_url, result)


@DBOS.step()
def record_score_error_step(
    database_url: str, prediction_id: str, error: str
) -> None:
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
    return generate_code_for_job(job, use_mock_lm=use_mock_lm)


@DBOS.step()
def record_generation_success_step(
    database_url: str, result: GenerationResult
) -> None:
    record_generation_success(database_url, result)


@DBOS.step()
def record_generation_error_step(
    database_url: str, prediction_id: str, error: str
) -> None:
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
    typer.echo("initialized dr-dspy eval tables")


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
    typer.echo(
        f"planned {len(jobs)} jobs: samples={len(samples)}, "
        f"models={len(model_configs)}, "
        f"temperatures={len(parsed_temperatures)}, "
        f"repetitions={repetitions}"
    )
    if dry_run:
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
    typer.echo(f"inserted {inserted} new prediction rows")
    typer.echo(f"enqueued {len(jobs)} generation workflows")


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
        typer.echo("no prediction rows found")
        return
    for row in rows:
        typer.echo(
            "{experiment_name} | {model} | temp={temperature} | "
            "generation={generation_status} | scoring={scoring_status} | "
            "count={count}".format(**row)
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
    typer.echo(f"enqueued {len(prediction_ids)} scoring workflows")


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
) -> None:
    config = common_config(
        database_url=database_url,
        dbos_system_database_url=dbos_system_database_url,
        generation_concurrency=generation_concurrency,
        scoring_concurrency=scoring_concurrency,
    )
    create_eval_schema(config.database_url)
    configure_dbos_runtime(config, queue=queue)
    typer.echo(f"worker listening on {queue.value} queue(s)")
    threading.Event().wait()


if __name__ == "__main__":
    configure_multiprocessing()
    logging.getLogger("dspy").setLevel(logging.WARNING)
    app()
