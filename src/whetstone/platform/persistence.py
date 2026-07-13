from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from enum import StrEnum
from typing import Any

from dr_graph import (
    GraphRunResult,
    NodeOutcomeStatus,
    TerminalError,
)
from pydantic import BaseModel, ConfigDict
from sqlalchemy import null, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Connection

from whetstone.db import io, schema
from whetstone.platform.node_execution import NodeStepResult
from whetstone.records import (
    GenerationRunRecord,
    GenerationRunStatus,
    GenerationRunSummaryPayload,
    GenerationTerminalErrorPayload,
    NodeAttemptRecord,
    PredictionSpecRecord,
    ProviderConfigRef,
    ResponseMetadataPayload,
    ScoreAttemptRecord,
    ScoreHarnessFailureRecord,
    UsageCostPayload,
    stable_node_attempt_id,
)
from whetstone.records.providers import find_provider_config_ref

# Node-attempt rows reuse the column name ``attempt_index``, but the meaning
# differs from ``generation_runs.attempt_index``: generation runs index whole
# workflow reruns for a prediction, while node attempts index retries of an
# individual node inside one generation run. DBOS step retries do not create
# new node-attempt rows; until explicit node reattempt workflows exist, every
# invoked node is stored at this initial index.
INITIAL_NODE_ATTEMPT_INDEX = 0


class ScoreAttemptInsertStatus(StrEnum):
    INSERTED = "inserted"
    ALREADY_PRESENT = "already_present"


class ScoreAttemptInsertResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score_attempt_id: str
    status: ScoreAttemptInsertStatus


def load_prediction_spec(
    connection: Connection,
    *,
    prediction_id: str,
) -> PredictionSpecRecord:
    row = (
        connection.execute(io.select_prediction_spec(prediction_id))
        .mappings()
        .one()
    )
    return prediction_spec_from_row(dict(row))


def load_generation_run(
    connection: Connection,
    *,
    generation_run_id: str,
) -> GenerationRunRecord:
    row = (
        connection.execute(io.select_generation_run(generation_run_id))
        .mappings()
        .one()
    )
    return generation_run_from_row(dict(row))


def load_node_attempts_for_generation_run(
    connection: Connection,
    *,
    generation_run_id: str,
) -> tuple[NodeAttemptRecord, ...]:
    rows = connection.execute(
        io.select_node_attempts_by_generation_run(generation_run_id)
    ).mappings()
    return tuple(node_attempt_from_row(dict(row)) for row in rows)


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
        created_at=row["created_at"],
    )


def generation_run_from_row(row: Mapping[str, Any]) -> GenerationRunRecord:
    return GenerationRunRecord(
        generation_run_id=row["generation_run_id"],
        prediction_id=row["prediction_id"],
        attempt_index=row["attempt_index"],
        execution_recipe_digest=row["execution_recipe_digest"],
        platform_item_id=row["platform_item_id"],
        platform_attempt=row["platform_attempt"],
        status=row["status"],
        terminal_node_id=row["terminal_node_id"],
        terminal_output_node_id=row["terminal_output_node_id"],
        summary=row["summary"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
    )


def node_attempt_from_row(row: Mapping[str, Any]) -> NodeAttemptRecord:
    return NodeAttemptRecord(
        node_attempt_id=row["node_attempt_id"],
        generation_run_id=row["generation_run_id"],
        prediction_id=row["prediction_id"],
        node_id=row["node_id"],
        attempt_index=row["attempt_index"],
        status=row["status"],
        provider_config=(
            ProviderConfigRef.model_validate(row["provider_config"])
            if row["provider_config"] is not None
            else None
        ),
        output=row["output"],
        usage_cost=UsageCostPayload.model_validate(row["usage_cost"]),
        response_metadata=ResponseMetadataPayload.model_validate(
            row["response_metadata"]
        ),
        failure=row["failure"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
    )


def generation_run_record_from_result(
    *,
    spec: PredictionSpecRecord,
    generation_run_id: str,
    attempt_index: int,
    execution_recipe_digest: str,
    platform_item_id: str,
    result: GraphRunResult,
    started_at: datetime,
    completed_at: datetime,
) -> GenerationRunRecord:
    status = GenerationRunStatus(result.status.value)
    return GenerationRunRecord(
        generation_run_id=generation_run_id,
        prediction_id=spec.prediction_id,
        attempt_index=attempt_index,
        execution_recipe_digest=execution_recipe_digest,
        platform_item_id=platform_item_id,
        platform_attempt=attempt_index,
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
            terminal_submission_text=terminal_submission_text(
                result.terminal_output,
                status=status,
            ),
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
    """Build one terminal node-attempt row for each invoked graph node."""

    return tuple(
        node_attempt_record_from_step(
            spec=spec,
            generation_run_id=generation_run_id,
            step_result=step_result,
            attempt_index=INITIAL_NODE_ATTEMPT_INDEX,
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
        status=step_result.status,
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
    """Append rows and reject a deterministic identity with changed truth."""
    inserted = _insert_or_exact_reload(
        connection,
        statement=idempotent_insert_generation_run(generation_run),
        table=schema.generation_runs,
        key_column="generation_run_id",
        key=generation_run.generation_run_id,
        expected=io.generation_run_row(generation_run),
    )
    for node_attempt in node_attempts:
        _insert_or_exact_reload(
            connection,
            statement=idempotent_insert_node_attempt(node_attempt),
            table=schema.node_attempts,
            key_column="node_attempt_id",
            key=node_attempt.node_attempt_id,
            expected=io.node_attempt_row(node_attempt),
        )
    if inserted is not None:
        _invalidate_experiment_acceptance(
            connection, prediction_id=generation_run.prediction_id
        )


_NODE_ATTEMPT_NULLABLE_JSONB_COLUMNS = frozenset(
    {"provider_config", "output", "failure"}
)
_SCORE_ATTEMPT_NULLABLE_JSONB_COLUMNS = frozenset({"metrics"})
_SCORE_HARNESS_FAILURE_NULLABLE_JSONB_COLUMNS = frozenset(
    {"extracted_submission"}
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


def persist_score_attempt(
    connection: Connection,
    *,
    score_attempt: ScoreAttemptRecord,
) -> ScoreAttemptInsertResult:
    inserted_row = _insert_or_exact_reload(
        connection,
        statement=idempotent_insert_score_attempt(score_attempt),
        table=schema.score_attempts,
        key_column="score_attempt_id",
        key=score_attempt.score_attempt_id,
        expected=io.score_attempt_row(score_attempt),
    )
    status = (
        ScoreAttemptInsertStatus.INSERTED
        if inserted_row is not None
        else ScoreAttemptInsertStatus.ALREADY_PRESENT
    )
    if inserted_row is not None:
        _invalidate_experiment_acceptance(
            connection, prediction_id=score_attempt.prediction_id
        )
    return ScoreAttemptInsertResult(
        score_attempt_id=score_attempt.score_attempt_id,
        status=status,
    )


def persist_score_harness_failure(
    connection: Connection,
    *,
    harness_failure: ScoreHarnessFailureRecord,
) -> ScoreAttemptInsertResult:
    inserted_row = _insert_or_exact_reload(
        connection,
        statement=idempotent_insert_score_harness_failure(harness_failure),
        table=schema.score_harness_failures,
        key_column="score_harness_failure_id",
        key=harness_failure.score_harness_failure_id,
        expected=io.score_harness_failure_row(harness_failure),
    )
    status = (
        ScoreAttemptInsertStatus.INSERTED
        if inserted_row is not None
        else ScoreAttemptInsertStatus.ALREADY_PRESENT
    )
    if inserted_row is not None:
        _invalidate_experiment_acceptance(
            connection, prediction_id=harness_failure.prediction_id
        )
    return ScoreAttemptInsertResult(
        score_attempt_id=harness_failure.score_attempt_id,
        status=status,
    )


def idempotent_insert_generation_run(record: GenerationRunRecord) -> Any:
    """Insert a generation run row, ignoring generation_run_id conflicts."""
    return (
        insert(schema.generation_runs)
        .values(io.generation_run_row(record))
        .on_conflict_do_nothing(index_elements=["generation_run_id"])
        .returning(schema.generation_runs.c.generation_run_id)
    )


def idempotent_insert_node_attempt(record: NodeAttemptRecord) -> Any:
    """Insert a node attempt row, ignoring conflicts on ``node_attempt_id``."""
    return (
        insert(schema.node_attempts)
        .values(
            _postgres_insert_values(
                io.node_attempt_row(record),
                nullable_jsonb_columns=_NODE_ATTEMPT_NULLABLE_JSONB_COLUMNS,
            )
        )
        .on_conflict_do_nothing(index_elements=["node_attempt_id"])
        .returning(schema.node_attempts.c.node_attempt_id)
    )


def idempotent_insert_score_attempt(record: ScoreAttemptRecord) -> Any:
    return (
        insert(schema.score_attempts)
        .values(
            _postgres_insert_values(
                io.score_attempt_row(record),
                nullable_jsonb_columns=_SCORE_ATTEMPT_NULLABLE_JSONB_COLUMNS,
            )
        )
        .on_conflict_do_nothing(index_elements=["score_attempt_id"])
        .returning(schema.score_attempts.c.score_attempt_id)
    )


def idempotent_insert_score_harness_failure(
    record: ScoreHarnessFailureRecord,
) -> Any:
    return (
        insert(schema.score_harness_failures)
        .values(
            _postgres_insert_values(
                io.score_harness_failure_row(record),
                nullable_jsonb_columns=(
                    _SCORE_HARNESS_FAILURE_NULLABLE_JSONB_COLUMNS
                ),
            )
        )
        .on_conflict_do_nothing(index_elements=["score_harness_failure_id"])
        .returning(schema.score_harness_failures.c.score_harness_failure_id)
    )


def _insert_or_exact_reload(
    connection: Connection,
    *,
    statement: Any,
    table: Any,
    key_column: str,
    key: str,
    expected: Mapping[str, Any],
) -> Any:
    inserted = connection.execute(statement).first()
    if inserted is not None:
        return inserted
    actual = (
        connection.execute(select(table).where(table.c[key_column] == key))
        .mappings()
        .one()
    )
    if dict(actual) != dict(expected):
        raise ValueError(
            f"append-only identity collision for {key_column}={key}"
        )
    return None


def _invalidate_experiment_acceptance(
    connection: Connection, *, prediction_id: str
) -> None:
    """Serialize a new terminal fact with acceptance pointer invalidation."""
    experiment_name = connection.execute(
        select(schema.prediction_specs.c.experiment_name).where(
            schema.prediction_specs.c.prediction_id == prediction_id
        )
    ).scalar_one()
    experiment = (
        connection.execute(
            select(schema.experiments)
            .where(schema.experiments.c.experiment_name == experiment_name)
            .with_for_update()
        )
        .mappings()
        .one()
    )
    connection.execute(
        update(schema.experiments)
        .where(schema.experiments.c.experiment_name == experiment_name)
        .values(
            acceptance_source_version=experiment["acceptance_source_version"]
            + 1,
            current_acceptance_id=None,
        )
    )


def terminal_submission_text(
    value: Any,
    *,
    status: GenerationRunStatus,
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("terminal output must be a string submission")
    return value


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
