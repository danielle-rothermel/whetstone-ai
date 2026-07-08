"""Single-run enc-dec inspection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from dr_code.humaneval import (
    HUMANEVAL_SCORING_PROFILE_ID,
    HUMANEVAL_SCORING_PROFILE_VERSION,
    EvaluationCaseStatus,
    SubmissionOutcome,
)
from dr_graph import NodeOutcome, NodeOutput, resolve_node_inputs
from dr_providers import PromptMessage
from sqlalchemy import Select, and_, select
from sqlalchemy.engine import Connection, Engine

from whetstone.analysis.frames import (
    _dedupe_score_attempts,
    _dimension_values,
    extract_encoder_decoder_models,
    normalize_compression_target,
)
from whetstone.db import io, schema
from whetstone.migration.v0_reshape import V0_SOURCE_METADATA_KEY
from whetstone.platform.persistence import (
    load_generation_run,
    load_node_attempts_for_generation_run,
    load_prediction_spec,
)
from whetstone.platform.prompts import build_node_messages
from whetstone.records import (
    GenerationRunRecord,
    NodeAttemptRecord,
    NodeAttemptStatus,
    PredictionSpecRecord,
    ScoreAttemptRecord,
    ScoreAttemptStatus,
)

SCRIPT_NAME = "sample_run_inspector"
PROMPTS_SOURCE = "reconstructed_from_graph_snapshot"
PER_TEST_DISPLAY_LIMIT = 20
ENCODER_NODE_ID = "encoder"
DECODER_NODE_ID = "decoder"


@dataclass(frozen=True)
class RunIndexRow:
    prediction_id: str
    generation_run_id: str
    score_attempt_id: str | None
    fair_order_key: str
    task_id: str
    generation_status: str
    score_status: str | None


@dataclass(frozen=True)
class RunBundle:
    spec: PredictionSpecRecord
    generation_run: GenerationRunRecord
    node_attempts: tuple[NodeAttemptRecord, ...]
    score_attempt: ScoreAttemptRecord | None
    sample_index: int
    sample_count: int


class SampleIndexError(ValueError):
    """Raised when --sample-index is out of range for the experiment index."""


def select_encdec_run_index_rows(
    experiment_names: tuple[str, ...],
    *,
    scoring_profile_id: str = HUMANEVAL_SCORING_PROFILE_ID,
    scoring_profile_version: str = HUMANEVAL_SCORING_PROFILE_VERSION,
    limit: int | None = None,
) -> Select[tuple[Any, ...]]:
    score_match = and_(
        schema.score_attempts.c.generation_run_id
        == schema.generation_runs.c.generation_run_id,
        schema.score_attempts.c.prediction_id
        == schema.generation_runs.c.prediction_id,
        schema.score_attempts.c.scoring_profile_id == scoring_profile_id,
        schema.score_attempts.c.scoring_profile_version
        == scoring_profile_version,
    )
    statement = (
        select(
            schema.prediction_specs.c.prediction_id,
            schema.generation_runs.c.generation_run_id,
            schema.score_attempts.c.score_attempt_id,
            schema.prediction_specs.c.fair_order_key,
            schema.prediction_specs.c.task_id,
            schema.generation_runs.c.status.label("generation_status"),
            schema.score_attempts.c.status.label("score_status"),
            schema.score_attempts.c.attempt_index.label("score_attempt_index"),
        )
        .select_from(
            schema.prediction_specs.join(
                schema.generation_runs,
                schema.generation_runs.c.prediction_id
                == schema.prediction_specs.c.prediction_id,
            ).outerjoin(schema.score_attempts, score_match)
        )
        .where(schema.prediction_specs.c.experiment_name.in_(experiment_names))
        .where(schema.prediction_specs.c.graph_layout == "encdec")
        .order_by(
            schema.prediction_specs.c.fair_order_key,
            schema.prediction_specs.c.prediction_id,
            schema.generation_runs.c.generation_run_id,
            schema.score_attempts.c.attempt_index,
        )
    )
    if limit is not None:
        statement = statement.limit(limit)
    return statement


def _frame_to_index_rows(frame: pd.DataFrame) -> list[RunIndexRow]:
    rows: list[RunIndexRow] = []
    for record in frame.to_dict(orient="records"):
        score_attempt_id = record.get("score_attempt_id")
        rows.append(
            RunIndexRow(
                prediction_id=str(record["prediction_id"]),
                generation_run_id=str(record["generation_run_id"]),
                score_attempt_id=(
                    str(score_attempt_id)
                    if score_attempt_id is not None
                    and not pd.isna(score_attempt_id)
                    else None
                ),
                fair_order_key=str(record["fair_order_key"]),
                task_id=str(record["task_id"]),
                generation_status=str(record["generation_status"]),
                score_status=(
                    str(record["score_status"])
                    if record.get("score_status") is not None
                    and not pd.isna(record["score_status"])
                    else None
                ),
            )
        )
    return rows


def list_encdec_run_index(
    engine: Engine,
    experiment_name: str,
    *,
    require_score: bool = True,
    limit: int | None = None,
    scoring_profile_id: str = HUMANEVAL_SCORING_PROFILE_ID,
    scoring_profile_version: str = HUMANEVAL_SCORING_PROFILE_VERSION,
) -> list[RunIndexRow]:
    statement = select_encdec_run_index_rows(
        (experiment_name,),
        scoring_profile_id=scoring_profile_id,
        scoring_profile_version=scoring_profile_version,
        limit=limit,
    )
    with engine.connect() as connection:
        frame = pd.read_sql(statement, connection)
    frame = _dedupe_score_attempts(frame)
    rows = _frame_to_index_rows(frame)
    if require_score:
        rows = [
            row
            for row in rows
            if row.score_attempt_id is not None
            and row.score_status == ScoreAttemptStatus.SUCCESS.value
        ]
    return rows


def resolve_sample_index(
    index_rows: list[RunIndexRow],
    sample_index: int,
) -> RunIndexRow:
    count = len(index_rows)
    if sample_index < 0 or sample_index >= count:
        hint = f"Use --sample-index 0..{max(count - 1, 0)}"
        raise SampleIndexError(
            f"sample_index {sample_index} out of range: "
            f"{count} enc-dec runs available ({hint})"
        )
    return index_rows[sample_index]


def load_score_attempt(
    connection: Connection,
    *,
    score_attempt_id: str,
) -> ScoreAttemptRecord:
    row = (
        connection.execute(
            select(schema.score_attempts).where(
                schema.score_attempts.c.score_attempt_id == score_attempt_id
            )
        )
        .mappings()
        .one()
    )
    return io.score_attempt_record_from_row(dict(row))


def load_run_bundle(
    engine: Engine,
    index_row: RunIndexRow,
    *,
    sample_index: int,
    sample_count: int,
) -> RunBundle:
    with engine.connect() as connection:
        spec = load_prediction_spec(
            connection,
            prediction_id=index_row.prediction_id,
        )
        generation_run = load_generation_run(
            connection,
            generation_run_id=index_row.generation_run_id,
        )
        node_attempts = load_node_attempts_for_generation_run(
            connection,
            generation_run_id=index_row.generation_run_id,
        )
        score_attempt = None
        if index_row.score_attempt_id is not None:
            score_attempt = load_score_attempt(
                connection,
                score_attempt_id=index_row.score_attempt_id,
            )
    return RunBundle(
        spec=spec,
        generation_run=generation_run,
        node_attempts=node_attempts,
        score_attempt=score_attempt,
        sample_index=sample_index,
        sample_count=sample_count,
    )


def node_attempt_by_id(
    bundle: RunBundle,
    node_id: str,
) -> NodeAttemptRecord | None:
    for attempt in bundle.node_attempts:
        if attempt.node_id == node_id:
            return attempt
    return None


def _messages_to_json(
    messages: tuple[PromptMessage, ...],
) -> list[dict[str, str]]:
    return [
        {"role": message.role.value, "content": message.content}
        for message in messages
    ]


def reconstruct_prompts(
    bundle: RunBundle,
) -> tuple[dict[str, tuple[PromptMessage, ...]], list[str]]:
    errors: list[str] = []
    prompts: dict[str, tuple[PromptMessage, ...]] = {}
    graph = bundle.spec.graph.graph
    task_inputs = bundle.spec.task.inputs.values

    try:
        encoder_node = graph.node(ENCODER_NODE_ID)
        encoder_inputs = resolve_node_inputs(
            node=encoder_node,
            inputs=task_inputs,
            outcomes={},
            graph=graph,
        )
        prompts[ENCODER_NODE_ID] = build_node_messages(
            node=encoder_node,
            node_inputs=encoder_inputs,
        )
    except Exception as error:
        errors.append(f"encoder prompt: {error}")

    try:
        decoder_node = graph.node(DECODER_NODE_ID)
        encoder_attempt = node_attempt_by_id(bundle, ENCODER_NODE_ID)
        if (
            encoder_attempt is None
            or encoder_attempt.status is not NodeAttemptStatus.SUCCESS
            or encoder_attempt.output is None
        ):
            errors.append("decoder prompt: encoder output unavailable")
        else:
            upstream = NodeOutcome.success(
                node_id=ENCODER_NODE_ID,
                output=NodeOutput(values=dict(encoder_attempt.output.values)),
            )
            decoder_inputs = resolve_node_inputs(
                node=decoder_node,
                inputs=task_inputs,
                outcomes={ENCODER_NODE_ID: upstream},
                graph=graph,
            )
            prompts[DECODER_NODE_ID] = build_node_messages(
                node=decoder_node,
                node_inputs=decoder_inputs,
            )
    except Exception as error:
        errors.append(f"decoder prompt: {error}")

    return prompts, errors


def summarize_test_results(
    score_attempt: ScoreAttemptRecord | None,
) -> dict[str, Any]:
    if score_attempt is None:
        return {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "error": 0,
            "truncated": False,
        }
    results = score_attempt.per_test_results
    passed = sum(
        1 for result in results if result.status is EvaluationCaseStatus.PASSED
    )
    failed = sum(
        1 for result in results if result.status is EvaluationCaseStatus.FAILED
    )
    error = sum(
        1
        for result in results
        if result.status
        in {EvaluationCaseStatus.ERROR, EvaluationCaseStatus.TIMEOUT}
    )
    if score_attempt.metrics is not None:
        metrics = score_attempt.metrics.model_dump(mode="json")
    else:
        metrics = {}
    custom = metrics.get("custom") or {}
    evaluation = custom.get("evaluation") or {}
    truncated = bool(evaluation.get("per_test_results_truncated"))
    return {
        "total": len(results),
        "passed": passed,
        "failed": failed,
        "error": error,
        "truncated": truncated,
    }


def _serialize_optional(model: Any) -> Any:
    if model is None:
        return None
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    return model


def _extract_v0_source(bundle: RunBundle) -> Any:
    metadata = bundle.generation_run.summary.metadata
    if V0_SOURCE_METADATA_KEY in metadata:
        return metadata[V0_SOURCE_METADATA_KEY]
    return None


def build_debug_metadata(
    bundle: RunBundle,
    *,
    reconstructed_prompts: dict[str, tuple[PromptMessage, ...]],
    reconstruction_errors: list[str],
    output_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    dim_values = _dimension_values(
        bundle.spec.dimensions.model_dump(mode="json")
    )
    provider_configs = [
        config.model_dump(mode="json")
        for config in bundle.spec.provider_configs
    ]
    encoder_model, decoder_model = extract_encoder_decoder_models(
        dim_values,
        provider_configs,
    )
    prompts_json = {
        node_id: _messages_to_json(messages)
        for node_id, messages in reconstructed_prompts.items()
    }
    score_attempt = bundle.score_attempt
    test_summary = summarize_test_results(score_attempt)
    per_test_json = None
    if score_attempt is not None:
        per_test_json = [
            result.model_dump(mode="json")
            for result in score_attempt.per_test_results
        ]

    return {
        "inspector": {
            "script": SCRIPT_NAME,
            "generated_at": datetime.now(UTC).isoformat(),
            "sample_index": bundle.sample_index,
            "sample_count": bundle.sample_count,
            "prompts_source": PROMPTS_SOURCE,
            "output_paths": output_paths or {},
        },
        "experiment": {
            "experiment_name": bundle.spec.experiment_name,
            "fair_order_key": bundle.spec.fair_order_key,
            "fair_order_seed": bundle.spec.fair_order_seed,
        },
        "spec": {
            "prediction_id": bundle.spec.prediction_id,
            "task_id": bundle.spec.task_id,
            "repetition_seed": bundle.spec.repetition_seed,
            "dimensions": dim_values,
            "dimensions_digest": bundle.spec.dimensions_digest,
            "graph_layout": bundle.spec.graph.layout,
            "graph_digest": bundle.spec.graph.graph_digest,
            "provider_axis": bundle.spec.provider_axis.model_dump(mode="json"),
            "provider_configs": provider_configs,
            "compression_target": normalize_compression_target(dim_values),
            "encoder_model": encoder_model,
            "decoder_model": decoder_model,
            "model": bundle.spec.provider_axis.model,
        },
        "task_snapshot": bundle.spec.task.model_dump(mode="json"),
        "generation_run": {
            "generation_run_id": bundle.generation_run.generation_run_id,
            "prediction_id": bundle.generation_run.prediction_id,
            "attempt_index": bundle.generation_run.attempt_index,
            "status": bundle.generation_run.status.value,
            "terminal_node_id": bundle.generation_run.terminal_node_id,
            "terminal_output_node_id": (
                bundle.generation_run.terminal_output_node_id
            ),
            "started_at": bundle.generation_run.started_at.isoformat(),
            "completed_at": bundle.generation_run.completed_at.isoformat(),
            "summary": bundle.generation_run.summary.model_dump(mode="json"),
            "v0_source": _extract_v0_source(bundle),
        },
        "node_attempts": [
            {
                "node_id": attempt.node_id,
                "node_attempt_id": attempt.node_attempt_id,
                "attempt_index": attempt.attempt_index,
                "status": attempt.status.value,
                "provider_config": _serialize_optional(
                    attempt.provider_config
                ),
                "output": _serialize_optional(attempt.output),
                "usage_cost": attempt.usage_cost.model_dump(mode="json"),
                "response_metadata": attempt.response_metadata.model_dump(
                    mode="json"
                ),
                "failure": _serialize_optional(attempt.failure),
                "started_at": attempt.started_at.isoformat(),
                "completed_at": attempt.completed_at.isoformat(),
            }
            for attempt in bundle.node_attempts
        ],
        "score_attempt": (
            None
            if score_attempt is None
            else {
                "score_attempt_id": score_attempt.score_attempt_id,
                "generation_run_id": score_attempt.generation_run_id,
                "prediction_id": score_attempt.prediction_id,
                "attempt_index": score_attempt.attempt_index,
                "scoring_profile_id": score_attempt.scoring_profile_id,
                "scoring_profile_version": (
                    score_attempt.scoring_profile_version
                ),
                "parser_profile_id": score_attempt.parser_profile_id,
                "parser_version": score_attempt.parser_version,
                "dataset_name": score_attempt.dataset_name,
                "dataset_split": score_attempt.dataset_split,
                "status": score_attempt.status.value,
                "score": score_attempt.score,
                "submission_outcome": score_attempt.submission_outcome.value,
                "extracted_submission": _serialize_optional(
                    score_attempt.extracted_submission
                ),
                "metrics": _serialize_optional(score_attempt.metrics),
                "test_summary": test_summary,
                "per_test_results": per_test_json,
                "started_at": score_attempt.started_at.isoformat(),
                "completed_at": score_attempt.completed_at.isoformat(),
            }
        ),
        "reconstructed_prompts": prompts_json,
        "reconstruction_errors": reconstruction_errors,
    }


def is_passing_run(bundle: RunBundle) -> bool:
    score_attempt = bundle.score_attempt
    if score_attempt is None:
        return False
    if score_attempt.status is not ScoreAttemptStatus.SUCCESS:
        return False
    if score_attempt.submission_outcome is SubmissionOutcome.PASSED:
        return True
    return score_attempt.score is not None and score_attempt.score == 1.0
