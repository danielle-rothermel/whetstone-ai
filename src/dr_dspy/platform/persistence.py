from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any

from sqlalchemy import null
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Connection

from dr_dspy.db import io, schema
from dr_dspy.graph import (
    GraphRunResult,
    NodeOutcomeStatus,
    TerminalError,
)
from dr_dspy.platform.node_execution import (
    NodeStepResult,
    NodeStepStatus,
)
from dr_dspy.records import (
    GenerationRunRecord,
    GenerationRunStatus,
    GenerationRunSummaryPayload,
    GenerationTerminalErrorPayload,
    NodeAttemptRecord,
    NodeAttemptStatus,
    PredictionSpecRecord,
    ProviderConfigRef,
    stable_node_attempt_id,
)
from dr_dspy.records.providers import find_provider_config_ref


def load_prediction_spec(
    connection: Connection,
    *,
    prediction_id: str,
) -> PredictionSpecRecord:
    row = connection.execute(
        io.select_prediction_spec(prediction_id)
    ).mappings().one()
    return prediction_spec_from_row(dict(row))


def prediction_spec_from_row(row: Mapping[str, Any]) -> PredictionSpecRecord:
    provider_configs = tuple(
        ProviderConfigRef.model_validate(provider_config)
        for provider_config in row["provider_configs"]
    )
    provider_axis = find_provider_config_ref(
        provider_configs,
        provider_kind=row["provider_kind"],
        endpoint_kind=row["endpoint_kind"],
        model=row["model"],
        throttle_key=row["throttle_key"],
        config_id=row.get("provider_axis_config_id"),
    )
    return PredictionSpecRecord(
        prediction_id=row["prediction_id"],
        experiment_name=row["experiment_name"],
        task_id=row["task_id"],
        repetition_seed=row["repetition_seed"],
        graph=row["graph_snapshot"],
        dimensions=row["dimensions"],
        dimensions_digest=row["dimensions_digest"],
        task=row["task_snapshot"],
        provider_configs=provider_configs,
        provider_axis=provider_axis,
        fair_order_seed=row["fair_order_seed"],
        fair_order_key=row["fair_order_key"],
        created_at=row["created_at"],
    )


def generation_run_record_from_result(
    *,
    spec: PredictionSpecRecord,
    generation_run_id: str,
    attempt_index: int,
    result: GraphRunResult,
    started_at: datetime,
    completed_at: datetime,
) -> GenerationRunRecord:
    status = GenerationRunStatus(result.status.value)
    return GenerationRunRecord(
        generation_run_id=generation_run_id,
        prediction_id=spec.prediction_id,
        attempt_index=attempt_index,
        status=status,
        terminal_node_id=result.terminal_node_id,
        terminal_output_node_id=(
            result.terminal_node_id
            if result.terminal_output is not None
            else None
        ),
        summary=GenerationRunSummaryPayload(
            execution_order=result.execution_order,
            terminal_node_id=result.terminal_node_id,
            terminal_output=result.terminal_output,
            terminal_error=_terminal_error_payload(result.terminal_error),
        ),
        started_at=started_at,
        completed_at=completed_at,
    )


def node_attempt_records_from_steps(
    *,
    spec: PredictionSpecRecord,
    generation_run_id: str,
    step_results: Iterable[NodeStepResult],
) -> tuple[NodeAttemptRecord, ...]:
    """Build one terminal node-attempt row for each invoked graph node.

    DBOS retries happen inside the node execution step and do not create
    separate append-only node attempt rows. Until the platform adds explicit
    node reattempt workflows, every invoked node in a generation run is stored
    as attempt index 0.
    """

    return tuple(
        node_attempt_record_from_step(
            spec=spec,
            generation_run_id=generation_run_id,
            step_result=step_result,
            attempt_index=0,
        )
        for step_result in step_results
    )


def node_attempt_record_from_step(
    *,
    spec: PredictionSpecRecord,
    generation_run_id: str,
    step_result: NodeStepResult,
    attempt_index: int,
) -> NodeAttemptRecord:
    return NodeAttemptRecord(
        node_attempt_id=stable_node_attempt_id(
            generation_run_id=generation_run_id,
            node_id=step_result.node_id,
            attempt_index=attempt_index,
        ),
        generation_run_id=generation_run_id,
        prediction_id=spec.prediction_id,
        node_id=step_result.node_id,
        attempt_index=attempt_index,
        status=(
            NodeAttemptStatus.SUCCESS
            if step_result.status is NodeStepStatus.SUCCESS
            else NodeAttemptStatus.ERROR
        ),
        provider_config=step_result.provider_config,
        output=step_result.output,
        usage_cost=step_result.usage_cost,
        response_metadata=step_result.response_metadata,
        failure=step_result.failure,
        started_at=step_result.started_at,
        completed_at=step_result.completed_at,
    )


def persist_generation_result(
    connection: Connection,
    *,
    generation_run: GenerationRunRecord,
    node_attempts: Iterable[NodeAttemptRecord],
) -> None:
    connection.execute(idempotent_insert_generation_run(generation_run))
    for node_attempt in node_attempts:
        connection.execute(idempotent_insert_node_attempt(node_attempt))


_NODE_ATTEMPT_NULLABLE_JSONB_COLUMNS = frozenset(
    {"provider_config", "output", "failure"}
)


def _postgres_insert_values(
    row: Mapping[str, Any],
    *,
    nullable_jsonb_columns: frozenset[str],
) -> dict[str, Any]:
    return {
        key: (
            null()
            if value is None and key in nullable_jsonb_columns
            else value
        )
        for key, value in row.items()
    }


def idempotent_insert_generation_run(record: GenerationRunRecord) -> Any:
    return (
        insert(schema.generation_runs)
        .values(io.generation_run_row(record))
        .on_conflict_do_nothing(index_elements=["generation_run_id"])
    )


def idempotent_insert_node_attempt(record: NodeAttemptRecord) -> Any:
    return (
        insert(schema.node_attempts)
        .values(
            _postgres_insert_values(
                io.node_attempt_row(record),
                nullable_jsonb_columns=_NODE_ATTEMPT_NULLABLE_JSONB_COLUMNS,
            )
        )
        .on_conflict_do_nothing(index_elements=["node_attempt_id"])
    )


def _terminal_error_payload(
    terminal_error: TerminalError | None,
) -> GenerationTerminalErrorPayload | None:
    if terminal_error is None:
        return None

    status = (
        GenerationRunStatus.BLOCKED
        if terminal_error.status is NodeOutcomeStatus.BLOCKED
        else GenerationRunStatus.ERROR
    )
    return GenerationTerminalErrorPayload(
        node_id=terminal_error.node_id,
        status=status,
        failure=(
            io.failure_payload_from_node_error(terminal_error.error)
            if terminal_error.error is not None
            else None
        ),
        blocked_by=terminal_error.blocked_by,
    )
