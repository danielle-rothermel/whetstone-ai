from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel
from sqlalchemy import Select, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.sql.dml import Insert, Update

from dr_dspy.db import schema
from dr_dspy.eval_failures.recording import ensure_recordable
from dr_dspy.eval_failures.types import FailureClass
from dr_dspy.graph import GraphRunStatus, NodeError, NodeOutput
from dr_dspy.humaneval.scoring import GeneratedCodeOutcome
from dr_dspy.records import (
    BatchSubmitItemRecord,
    BatchSubmitItemStatus,
    BatchSubmitOperationRecord,
    BatchSubmitOperationStatus,
    DimensionsPayload,
    ExperimentRecord,
    ExtractedCodePayload,
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
    PredictionProjectionRecord,
    PredictionSpecRecord,
    ProviderConfigRef,
    ResponseMetadataPayload,
    ScoreAttemptRecord,
    ScoreAttemptStatus,
    TaskSnapshotPayload,
    UsageCostPayload,
)
from dr_dspy.records.providers import (
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
    "extracted_code",
    "metrics",
    "per_test_results",
    "failure",
)
BATCH_SUBMIT_OPERATION_JSONB_FIELDS = ("spec", "metadata")
BATCH_SUBMIT_ITEM_JSONB_FIELDS = ("enqueue_metadata", "failure")


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
    return FailureMetadataPayload(
        failure_class=failure_class,
        error_type=error.error_type,
        message=error.message,
        metadata=error.metadata,
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
        "fair_order_seed": record.fair_order_seed,
        "fair_order_key": record.fair_order_key,
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
        "scoring_profile_id": record.scoring_profile_id,
        "scoring_profile_version": record.scoring_profile_version,
        "parser_profile_id": record.parser_profile_id,
        "parser_version": record.parser_version,
        "status": record.status.value,
        "generated_code_outcome": _enum_value(record.generated_code_outcome),
        "score": record.score,
        "extracted_code": _dump_optional(record.extracted_code),
        "metrics": _dump_optional(record.metrics),
        "per_test_results": _dump_many(record.per_test_results),
        "failure": _dump_optional(record.failure),
        "started_at": record.started_at,
        "completed_at": record.completed_at,
    }
    _validate_jsonb_fields(row, *SCORE_ATTEMPT_JSONB_FIELDS)
    return row


def prediction_projection_row(record: PredictionProjectionRecord) -> Row:
    return {
        "prediction_id": record.prediction_id,
        "generation_run_id": record.generation_run_id,
        "score_attempt_id": record.score_attempt_id,
        "projection_profile_id": record.projection_profile_id,
        "projection_version": record.projection_version,
        "selected_at": record.selected_at,
        "selection_reason": record.selection_reason,
    }


def batch_submit_operation_row(record: BatchSubmitOperationRecord) -> Row:
    row = {
        "operation_key": record.operation_key,
        "experiment_name": record.experiment_name,
        "status": record.status.value,
        "requested_count": record.requested_count,
        "inserted_count": record.inserted_count,
        "already_present_count": record.already_present_count,
        "enqueued_count": record.enqueued_count,
        "failed_count": record.failed_count,
        "spec": record.spec,
        "metadata": record.metadata,
        "created_at": record.created_at,
        "completed_at": record.completed_at,
    }
    _validate_jsonb_fields(row, *BATCH_SUBMIT_OPERATION_JSONB_FIELDS)
    return row


def batch_submit_item_row(record: BatchSubmitItemRecord) -> Row:
    row = {
        "batch_submit_item_id": record.batch_submit_item_id,
        "operation_key": record.operation_key,
        "item_index": record.item_index,
        "prediction_id": record.prediction_id,
        "fair_order_key": record.fair_order_key,
        "status": record.status.value,
        "enqueue_metadata": record.enqueue_metadata,
        "failure": _dump_optional(record.failure),
        "created_at": record.created_at,
    }
    _validate_jsonb_fields(row, *BATCH_SUBMIT_ITEM_JSONB_FIELDS)
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
        fair_order_seed=row["fair_order_seed"],
        fair_order_key=row["fair_order_key"],
        created_at=row["created_at"],
    )


def generation_run_record_from_row(row: Row) -> GenerationRunRecord:
    return GenerationRunRecord(
        generation_run_id=row["generation_run_id"],
        prediction_id=row["prediction_id"],
        attempt_index=row["attempt_index"],
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
    generated_code_outcome = row["generated_code_outcome"]
    return ScoreAttemptRecord(
        score_attempt_id=row["score_attempt_id"],
        prediction_id=row["prediction_id"],
        generation_run_id=row["generation_run_id"],
        attempt_index=row["attempt_index"],
        scoring_profile_id=row["scoring_profile_id"],
        scoring_profile_version=row["scoring_profile_version"],
        parser_profile_id=row["parser_profile_id"],
        parser_version=row["parser_version"],
        status=ScoreAttemptStatus(row["status"]),
        generated_code_outcome=(
            GeneratedCodeOutcome(generated_code_outcome)
            if generated_code_outcome is not None
            else None
        ),
        score=row["score"],
        extracted_code=_load_optional(
            ExtractedCodePayload,
            row["extracted_code"],
        ),
        metrics=_load_optional(MetricsPayload, row["metrics"]),
        per_test_results=_load_many(
            PerTestResultPayload,
            row["per_test_results"],
        ),
        failure=_load_optional(FailureMetadataPayload, row["failure"]),
        started_at=row["started_at"],
        completed_at=row["completed_at"],
    )


def prediction_projection_record_from_row(
    row: Row,
) -> PredictionProjectionRecord:
    return PredictionProjectionRecord(
        prediction_id=row["prediction_id"],
        generation_run_id=row["generation_run_id"],
        score_attempt_id=row["score_attempt_id"],
        projection_profile_id=row["projection_profile_id"],
        projection_version=row["projection_version"],
        selected_at=row["selected_at"],
        selection_reason=row["selection_reason"],
    )


def batch_submit_operation_record_from_row(
    row: Row,
) -> BatchSubmitOperationRecord:
    return BatchSubmitOperationRecord(
        operation_key=row["operation_key"],
        experiment_name=row["experiment_name"],
        status=BatchSubmitOperationStatus(row["status"]),
        requested_count=row["requested_count"],
        inserted_count=row["inserted_count"],
        already_present_count=row["already_present_count"],
        enqueued_count=row["enqueued_count"],
        failed_count=row["failed_count"],
        spec=row["spec"],
        metadata=row["metadata"],
        created_at=row["created_at"],
        completed_at=row["completed_at"],
    )


def batch_submit_item_record_from_row(row: Row) -> BatchSubmitItemRecord:
    return BatchSubmitItemRecord(
        batch_submit_item_id=row["batch_submit_item_id"],
        operation_key=row["operation_key"],
        item_index=row["item_index"],
        prediction_id=row["prediction_id"],
        fair_order_key=row["fair_order_key"],
        status=BatchSubmitItemStatus(row["status"]),
        enqueue_metadata=row["enqueue_metadata"],
        failure=_load_optional(FailureMetadataPayload, row["failure"]),
        created_at=row["created_at"],
    )


def insert_experiment(record: ExperimentRecord) -> Insert:
    return schema.experiments.insert().values(experiment_row(record))


def insert_prediction_spec(record: PredictionSpecRecord) -> Insert:
    return schema.prediction_specs.insert().values(
        prediction_spec_row(record)
    )


def insert_prediction_spec_on_conflict_do_nothing(
    record: PredictionSpecRecord,
) -> Insert:
    return (
        pg_insert(schema.prediction_specs)
        .values(prediction_spec_row(record))
        .on_conflict_do_nothing(index_elements=["prediction_id"])
    )


def insert_generation_run(record: GenerationRunRecord) -> Insert:
    return schema.generation_runs.insert().values(generation_run_row(record))


def insert_node_attempt(record: NodeAttemptRecord) -> Insert:
    return schema.node_attempts.insert().values(node_attempt_row(record))


def insert_score_attempt(record: ScoreAttemptRecord) -> Insert:
    return schema.score_attempts.insert().values(score_attempt_row(record))


def insert_prediction_projection(
    record: PredictionProjectionRecord,
) -> Insert:
    return schema.prediction_projection.insert().values(
        prediction_projection_row(record)
    )


def insert_batch_submit_operation(
    record: BatchSubmitOperationRecord,
) -> Insert:
    return schema.batch_submit_operations.insert().values(
        batch_submit_operation_row(record)
    )


def insert_batch_submit_operation_on_conflict_do_nothing(
    record: BatchSubmitOperationRecord,
) -> Insert:
    return (
        pg_insert(schema.batch_submit_operations)
        .values(batch_submit_operation_row(record))
        .on_conflict_do_nothing(index_elements=["operation_key"])
    )


def update_batch_submit_operation(
    record: BatchSubmitOperationRecord,
) -> Update:
    row = batch_submit_operation_row(record)
    operation_key = row.pop("operation_key")
    return (
        update(schema.batch_submit_operations)
        .where(schema.batch_submit_operations.c.operation_key == operation_key)
        .values(row)
    )


def insert_batch_submit_item(record: BatchSubmitItemRecord) -> Insert:
    return schema.batch_submit_items.insert().values(
        batch_submit_item_row(record)
    )


def insert_batch_submit_item_on_conflict_do_nothing(
    record: BatchSubmitItemRecord,
) -> Insert:
    return (
        pg_insert(schema.batch_submit_items)
        .values(batch_submit_item_row(record))
        .on_conflict_do_nothing(
            constraint="uq_dr_dspy_batch_items_operation_prediction"
        )
    )


def select_batch_submit_operation(
    operation_key: str,
) -> Select[tuple[Any, ...]]:
    return select(schema.batch_submit_operations).where(
        schema.batch_submit_operations.c.operation_key == operation_key
    )


def select_batch_submit_items(
    operation_key: str,
) -> Select[tuple[Any, ...]]:
    return select(schema.batch_submit_items).where(
        schema.batch_submit_items.c.operation_key == operation_key
    ).order_by(schema.batch_submit_items.c.item_index)


def select_prediction_spec(prediction_id: str) -> Select[tuple[Any, ...]]:
    return select(schema.prediction_specs).where(
        schema.prediction_specs.c.prediction_id == prediction_id
    )


def select_prediction_projections(
    prediction_id: str,
) -> Select[tuple[Any, ...]]:
    return select(schema.prediction_projection).where(
        schema.prediction_projection.c.prediction_id == prediction_id
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
