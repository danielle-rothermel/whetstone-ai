from __future__ import annotations

from enum import StrEnum
from typing import Any

from dr_code.humaneval import SubmissionOutcome
from dr_graph import GraphRunStatus, NodeError, NodeOutput
from dr_providers import FailureClass
from pydantic import BaseModel
from sqlalchemy import Select, select
from sqlalchemy.sql.dml import Insert

from whetstone.db import schema
from whetstone.eval_failures.recording import ensure_recordable
from whetstone.records import (
    DatasetSnapshotIdentityPayload,
    DimensionsPayload,
    ExperimentRecord,
    ExtractedSubmissionPayload,
    FailureMetadataPayload,
    GenerationRunRecord,
    GenerationRunStatus,
    GenerationRunSummaryPayload,
    GraphSnapshotPayload,
    MetricsPayload,
    NodeAttemptRecord,
    NodeAttemptStatus,
    NodeOutputPayload,
    PerTestResultPayload,
    PredictionSpecRecord,
    ProviderConfigRef,
    ResponseMetadataPayload,
    ScoreAttemptRecord,
    ScoreAttemptStatus,
    ScoreHarnessFailureRecord,
    TaskSnapshotPayload,
    UsageCostPayload,
)
from whetstone.records.providers import (
    find_provider_config_ref,
    provider_snapshot_matches_axis,
)

type Row = dict[str, Any]

EXPERIMENT_JSONB_FIELDS = ("config_metadata",)
PREDICTION_SPEC_JSONB_FIELDS = (
    "task_snapshot",
    "graph_snapshot",
    "dimensions",
    "provider_configs",
)
GENERATION_RUN_JSONB_FIELDS = ("summary",)
NODE_ATTEMPT_JSONB_FIELDS = (
    "provider_config",
    "output",
    "usage_cost",
    "response_metadata",
    "failure",
)
SCORE_ATTEMPT_JSONB_FIELDS = (
    "dataset_snapshot",
    "extracted_submission",
    "metrics",
    "per_test_results",
)
SCORE_HARNESS_FAILURE_JSONB_FIELDS = (
    "dataset_snapshot",
    "extracted_submission",
    "cause",
)


def node_output_payload_from_graph_output(
    output: NodeOutput,
) -> NodeOutputPayload:
    return NodeOutputPayload(values=output.values, metadata=output.metadata)


def failure_payload_from_node_error(
    error: NodeError,
) -> FailureMetadataPayload:
    failure_class = (
        FailureClass(error.failure_class)
        if error.failure_class is not None
        else None
    )
    metadata = dict(error.metadata)
    underlying = metadata.pop("underlying_exception_type", None)
    return FailureMetadataPayload(
        failure_class=failure_class,
        error_type=error.error_type,
        underlying_exception_type=(
            underlying if isinstance(underlying, str) else None
        ),
        message=error.message,
        metadata=metadata,
    )


def generation_status_from_graph_status(
    status: GraphRunStatus,
) -> GenerationRunStatus:
    """Map graph-run terminal status to persisted generation-run status.

    Value sets are kept in parity today; richer summary construction belongs
    in the workflow layer when graph runs are persisted.
    """
    return GenerationRunStatus(status.value)


def experiment_row(record: ExperimentRecord) -> Row:
    row = {
        "experiment_name": record.experiment_name,
        "description": record.description,
        "config_metadata": record.config_metadata,
        "created_at": record.created_at,
    }
    _validate_jsonb_fields(row, *EXPERIMENT_JSONB_FIELDS)
    return row


def prediction_spec_row(record: PredictionSpecRecord) -> Row:
    provider_axis = record.provider_axis
    row = {
        "prediction_id": record.prediction_id,
        "experiment_name": record.experiment_name,
        "task_id": record.task_id,
        "repetition_seed": record.repetition_seed,
        "graph_digest": record.graph.graph_digest,
        "dimensions_digest": record.dimensions_digest,
        "graph_layout": record.graph.layout,
        "provider_kind": provider_axis.provider_kind.value,
        "endpoint_kind": provider_axis.endpoint_kind.value,
        "model": provider_axis.model,
        "throttle_key": provider_axis.throttle_key,
        "provider_axis_config_id": provider_axis.config_id,
        "task_snapshot": _dump(record.task),
        "graph_snapshot": _dump(record.graph),
        "dimensions": _dump(record.dimensions),
        "provider_configs": _dump_many(record.provider_configs),
        "created_at": record.created_at,
    }
    _validate_prediction_spec_provider_row(row)
    _validate_jsonb_fields(row, *PREDICTION_SPEC_JSONB_FIELDS)
    return row


def generation_run_row(record: GenerationRunRecord) -> Row:
    row = {
        "generation_run_id": record.generation_run_id,
        "prediction_id": record.prediction_id,
        "attempt_index": record.attempt_index,
        "status": record.status.value,
        "terminal_node_id": record.terminal_node_id,
        "terminal_output_node_id": record.terminal_output_node_id,
        "summary": _dump(record.summary),
        "started_at": record.started_at,
        "completed_at": record.completed_at,
    }
    _validate_jsonb_fields(row, *GENERATION_RUN_JSONB_FIELDS)
    return row


def node_attempt_row(record: NodeAttemptRecord) -> Row:
    provider_config = record.provider_config
    row = {
        "node_attempt_id": record.node_attempt_id,
        "generation_run_id": record.generation_run_id,
        "prediction_id": record.prediction_id,
        "node_id": record.node_id,
        "attempt_index": record.attempt_index,
        "status": record.status.value,
        "provider_kind": _enum_value(provider_config.provider_kind)
        if provider_config
        else None,
        "endpoint_kind": _enum_value(provider_config.endpoint_kind)
        if provider_config
        else None,
        "model": provider_config.model if provider_config else None,
        "throttle_key": provider_config.throttle_key
        if provider_config
        else None,
        "config_id": provider_config.config_id if provider_config else None,
        "provider_config": _dump_optional(provider_config),
        "output": _dump_optional(record.output),
        "usage_cost": _dump(record.usage_cost),
        "response_metadata": _dump(record.response_metadata),
        "failure": _dump_optional(record.failure),
        "started_at": record.started_at,
        "completed_at": record.completed_at,
    }
    _validate_node_attempt_provider_row(row)
    _validate_jsonb_fields(row, *NODE_ATTEMPT_JSONB_FIELDS)
    return row


def score_attempt_row(record: ScoreAttemptRecord) -> Row:
    row = {
        "score_attempt_id": record.score_attempt_id,
        "prediction_id": record.prediction_id,
        "generation_run_id": record.generation_run_id,
        "attempt_index": record.attempt_index,
        "execution_recipe_digest": record.execution_recipe_digest,
        "platform_item_id": record.platform_item_id,
        "platform_attempt": record.platform_attempt,
        "scoring_profile_id": record.scoring_profile_id,
        "scoring_profile_version": record.scoring_profile_version,
        "parser_profile_id": record.parser_profile_id,
        "parser_version": record.parser_version,
        "dataset_name": record.dataset_name,
        "dataset_split": record.dataset_split,
        "dataset_snapshot": _dump(record.dataset_snapshot),
        "status": record.status.value,
        "submission_outcome": _enum_value(record.submission_outcome),
        "score": record.score,
        "extracted_submission": _dump(record.extracted_submission),
        "metrics": _dump_optional(record.metrics),
        "per_test_results": _dump_many(record.per_test_results),
        "started_at": record.started_at,
        "completed_at": record.completed_at,
    }
    _validate_jsonb_fields(row, *SCORE_ATTEMPT_JSONB_FIELDS)
    return row


def score_harness_failure_row(record: ScoreHarnessFailureRecord) -> Row:
    row = {
        "score_attempt_id": record.score_attempt_id,
        "prediction_id": record.prediction_id,
        "generation_run_id": record.generation_run_id,
        "attempt_index": record.attempt_index,
        "scoring_profile_id": record.scoring_profile_id,
        "scoring_profile_version": record.scoring_profile_version,
        "parser_profile_id": record.parser_profile_id,
        "parser_version": record.parser_version,
        "dataset_name": record.dataset_name,
        "dataset_split": record.dataset_split,
        "dataset_snapshot": _dump(record.dataset_snapshot),
        "failure": _dump(record),
        "started_at": record.started_at,
        "completed_at": record.completed_at,
    }
    _validate_jsonb_fields(row, *SCORE_HARNESS_FAILURE_JSONB_FIELDS)
    return row


def experiment_record_from_row(row: Row) -> ExperimentRecord:
    return ExperimentRecord(
        experiment_name=row["experiment_name"],
        description=row["description"],
        config_metadata=row["config_metadata"],
        created_at=row["created_at"],
    )


def prediction_spec_record_from_row(row: Row) -> PredictionSpecRecord:
    provider_configs = _load_many(ProviderConfigRef, row["provider_configs"])
    return PredictionSpecRecord(
        prediction_id=row["prediction_id"],
        experiment_name=row["experiment_name"],
        task_id=row["task_id"],
        repetition_seed=row["repetition_seed"],
        graph=_load(GraphSnapshotPayload, row["graph_snapshot"]),
        dimensions=_load(DimensionsPayload, row["dimensions"]),
        dimensions_digest=row["dimensions_digest"],
        task=_load(TaskSnapshotPayload, row["task_snapshot"]),
        provider_configs=provider_configs,
        provider_axis=find_provider_config_ref(
            provider_configs,
            provider_kind=row["provider_kind"],
            endpoint_kind=row["endpoint_kind"],
            model=row["model"],
            throttle_key=row["throttle_key"],
            config_id=row.get("provider_axis_config_id"),
        ),
        created_at=row["created_at"],
    )


def generation_run_record_from_row(row: Row) -> GenerationRunRecord:
    return GenerationRunRecord(
        generation_run_id=row["generation_run_id"],
        prediction_id=row["prediction_id"],
        attempt_index=row["attempt_index"],
        execution_recipe_digest=row["execution_recipe_digest"],
        platform_item_id=row["platform_item_id"],
        platform_attempt=row["platform_attempt"],
        status=GenerationRunStatus(row["status"]),
        terminal_node_id=row["terminal_node_id"],
        terminal_output_node_id=row["terminal_output_node_id"],
        summary=_load(GenerationRunSummaryPayload, row["summary"]),
        started_at=row["started_at"],
        completed_at=row["completed_at"],
    )


def node_attempt_record_from_row(row: Row) -> NodeAttemptRecord:
    return NodeAttemptRecord(
        node_attempt_id=row["node_attempt_id"],
        generation_run_id=row["generation_run_id"],
        prediction_id=row["prediction_id"],
        node_id=row["node_id"],
        attempt_index=row["attempt_index"],
        status=NodeAttemptStatus(row["status"]),
        provider_config=_load_optional(
            ProviderConfigRef,
            row["provider_config"],
        ),
        output=_load_optional(NodeOutputPayload, row["output"]),
        usage_cost=_load(UsageCostPayload, row["usage_cost"]),
        response_metadata=_load(
            ResponseMetadataPayload,
            row["response_metadata"],
        ),
        failure=_load_optional(FailureMetadataPayload, row["failure"]),
        started_at=row["started_at"],
        completed_at=row["completed_at"],
    )


def score_attempt_record_from_row(row: Row) -> ScoreAttemptRecord:
    submission_outcome = row["submission_outcome"]
    return ScoreAttemptRecord(
        score_attempt_id=row["score_attempt_id"],
        prediction_id=row["prediction_id"],
        generation_run_id=row["generation_run_id"],
        attempt_index=row["attempt_index"],
        execution_recipe_digest=row["execution_recipe_digest"],
        platform_item_id=row["platform_item_id"],
        platform_attempt=row["platform_attempt"],
        scoring_profile_id=row["scoring_profile_id"],
        scoring_profile_version=row["scoring_profile_version"],
        parser_profile_id=row["parser_profile_id"],
        parser_version=row["parser_version"],
        dataset_name=row["dataset_name"],
        dataset_split=row["dataset_split"],
        dataset_snapshot=_load(
            DatasetSnapshotIdentityPayload,
            row["dataset_snapshot"],
        ),
        status=ScoreAttemptStatus(row["status"]),
        submission_outcome=SubmissionOutcome(submission_outcome),
        score=row["score"],
        extracted_submission=_load(
            ExtractedSubmissionPayload,
            row["extracted_submission"],
        ),
        metrics=_load_optional(MetricsPayload, row["metrics"]),
        per_test_results=_load_many(
            PerTestResultPayload,
            row["per_test_results"],
        ),
        started_at=row["started_at"],
        completed_at=row["completed_at"],
    )


def score_harness_failure_record_from_row(
    row: Row,
) -> ScoreHarnessFailureRecord:
    return ScoreHarnessFailureRecord.model_validate(row["failure"])


def insert_experiment(record: ExperimentRecord) -> Insert:
    return schema.experiments.insert().values(experiment_row(record))


def insert_prediction_spec(record: PredictionSpecRecord) -> Insert:
    return schema.prediction_specs.insert().values(
        prediction_spec_row(record)
    )


def insert_generation_run(record: GenerationRunRecord) -> Insert:
    return schema.generation_runs.insert().values(generation_run_row(record))


def insert_node_attempt(record: NodeAttemptRecord) -> Insert:
    return schema.node_attempts.insert().values(node_attempt_row(record))


def insert_score_attempt(record: ScoreAttemptRecord) -> Insert:
    return schema.score_attempts.insert().values(score_attempt_row(record))


def insert_score_harness_failure(
    record: ScoreHarnessFailureRecord,
) -> Insert:
    return schema.score_harness_failures.insert().values(
        score_harness_failure_row(record)
    )


def select_prediction_spec(prediction_id: str) -> Select[tuple[Any, ...]]:
    return select(schema.prediction_specs).where(
        schema.prediction_specs.c.prediction_id == prediction_id
    )


def _validate_jsonb_fields(row: Row, *fields: str) -> None:
    for field in fields:
        value = row.get(field)
        if value is not None:
            ensure_recordable(value)


def _load[ModelT: BaseModel](model_type: type[ModelT], value: Any) -> ModelT:
    return model_type.model_validate(value)


def _load_optional[ModelT: BaseModel](
    model_type: type[ModelT],
    value: Any | None,
) -> ModelT | None:
    if value is None:
        return None
    return _load(model_type, value)


def _load_many[ModelT: BaseModel](
    model_type: type[ModelT],
    values: Any,
) -> tuple[ModelT, ...]:
    return tuple(_load(model_type, value) for value in values)


def _validate_prediction_spec_provider_row(row: Row) -> None:
    provider_configs = row["provider_configs"]
    if not any(
        provider_snapshot_matches_axis(
            config,
            provider_kind=row["provider_kind"],
            endpoint_kind=row["endpoint_kind"],
            model=row["model"],
            throttle_key=row["throttle_key"],
            config_id=row.get("provider_axis_config_id"),
        )
        for config in provider_configs
    ):
        raise ValueError(
            "denormalized provider columns must match "
            "provider_configs snapshot"
        )


def _validate_node_attempt_provider_row(row: Row) -> None:
    provider_config = row["provider_config"]
    indexed = {
        "provider_kind": row["provider_kind"],
        "endpoint_kind": row["endpoint_kind"],
        "model": row["model"],
        "throttle_key": row["throttle_key"],
        "config_id": row.get("config_id"),
    }
    if provider_config is None:
        if any(value is not None for value in indexed.values()):
            raise ValueError(
                "provider index columns must be null when "
                "provider_config is null"
            )
        return
    if not provider_snapshot_matches_axis(
        provider_config,
        provider_kind=indexed["provider_kind"],
        endpoint_kind=indexed["endpoint_kind"],
        model=indexed["model"],
        throttle_key=indexed["throttle_key"],
        config_id=indexed["config_id"],
    ):
        raise ValueError(
            "denormalized provider columns must match provider_config snapshot"
        )


def select_generation_run(
    generation_run_id: str,
) -> Select[tuple[Any, ...]]:
    return select(schema.generation_runs).where(
        schema.generation_runs.c.generation_run_id == generation_run_id
    )


def select_node_attempts_by_generation_run(
    generation_run_id: str,
) -> Select[tuple[Any, ...]]:
    return (
        select(schema.node_attempts)
        .where(schema.node_attempts.c.generation_run_id == generation_run_id)
        .order_by(
            schema.node_attempts.c.node_id,
            schema.node_attempts.c.attempt_index,
        )
    )


def _dump(value: BaseModel) -> dict[str, Any]:
    return value.model_dump(mode="json")


def _dump_optional(value: BaseModel | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return _dump(value)


def _dump_many(values: tuple[BaseModel, ...]) -> list[dict[str, Any]]:
    return [_dump(value) for value in values]


def _enum_value(value: StrEnum | None) -> str | None:
    if value is None:
        return None
    return value.value
