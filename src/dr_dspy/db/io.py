from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel
from sqlalchemy import Select, select
from sqlalchemy.sql.dml import Insert

from dr_dspy.db import schema
from dr_dspy.eval_failures.types import FailureClass
from dr_dspy.graph import GraphRunStatus, NodeError, NodeOutput
from dr_dspy.records import (
    BatchSubmitItemRecord,
    BatchSubmitOperationRecord,
    ExperimentRecord,
    FailureMetadataPayload,
    GenerationRunRecord,
    GenerationRunStatus,
    NodeAttemptRecord,
    NodeOutputPayload,
    PredictionProjectionRecord,
    PredictionSpecRecord,
    ScoreAttemptRecord,
)

type Row = dict[str, Any]


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
    return GenerationRunStatus(status.value)


def experiment_row(record: ExperimentRecord) -> Row:
    return {
        "experiment_name": record.experiment_name,
        "description": record.description,
        "config_metadata": record.config_metadata,
        "created_at": record.created_at,
    }


def prediction_spec_row(record: PredictionSpecRecord) -> Row:
    provider_axis = record.provider_axis
    return {
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
        "fair_order_seed": record.fair_order_seed,
        "fair_order_key": record.fair_order_key,
        "task_snapshot": _dump(record.task),
        "graph_snapshot": _dump(record.graph),
        "dimensions": _dump(record.dimensions),
        "provider_configs": _dump_many(record.provider_configs),
        "created_at": record.created_at,
    }


def generation_run_row(record: GenerationRunRecord) -> Row:
    return {
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


def node_attempt_row(record: NodeAttemptRecord) -> Row:
    provider_config = record.provider_config
    return {
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
        "provider_config": _dump_optional(provider_config),
        "output": _dump_optional(record.output),
        "usage_cost": _dump(record.usage_cost),
        "response_metadata": _dump(record.response_metadata),
        "failure": _dump_optional(record.failure),
        "started_at": record.started_at,
        "completed_at": record.completed_at,
    }


def score_attempt_row(record: ScoreAttemptRecord) -> Row:
    return {
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
    return {
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


def batch_submit_item_row(record: BatchSubmitItemRecord) -> Row:
    return {
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


def insert_batch_submit_item(record: BatchSubmitItemRecord) -> Insert:
    return schema.batch_submit_items.insert().values(
        batch_submit_item_row(record)
    )


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
