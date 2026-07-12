from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any

from dbos import DBOS, SetWorkflowID
from dr_graph import GraphRunResult, NodeOutput, NodeSpec, execute_graph
from dr_platform import (
    clear_throttle_backoff,
    record_throttle_failure,
    throttle_delay_seconds,
)
from dr_platform.dbos_config import (
    WORKFLOW_START_RACE_ERRORS,
    workflow_start_raced,
)
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import create_engine

from whetstone.clock import now
from whetstone.eval_failures import should_retry_step
from whetstone.platform.node_execution import (
    NodeStepResult,
    attach_node_step_timing_to_exception,
    execute_lm_node,
    failure_metadata_from_exception,
    node_step_error_result_from_failure,
    node_step_timing_from_exception,
    provider_config_ref_for_node,
)
from whetstone.platform.persistence import (
    generation_run_record_from_result,
    load_prediction_spec,
    node_attempt_records_from_steps,
    persist_generation_result,
)
from whetstone.platform.platform_db import PLATFORM_SCHEMA
from whetstone.platform.runtime import resolve_application_database_url
from whetstone.records import (
    FailureMetadataPayload,
    GenerationRunRecord,
    NodeAttemptRecord,
    NodeAttemptStatus,
    PredictionSpecRecord,
    stable_generation_run_id,
    stable_node_attempt_id,
)

PLATFORM_GENERATION_WORKFLOW_NAME = "whetstone_generation"
LOAD_SPEC_STEP_NAME = "whetstone_load_prediction_spec"
GENERATION_STARTED_AT_STEP_NAME = "whetstone_generation_started_at"
GENERATION_COMPLETED_AT_STEP_NAME = "whetstone_generation_completed_at"
NODE_STEP_ERROR_RESULT_STEP_NAME = "whetstone_node_step_error_result"
EXECUTE_NODE_STEP_NAME = "whetstone_execute_lm_node"
THROTTLE_PREFLIGHT_STEP_NAME = "whetstone_throttle_preflight"
PERSIST_RESULT_STEP_NAME = "whetstone_persist_generation_result"
WORKFLOW_ID_PREFIX = "whetstone-generation"
NODE_STEP_MAX_ATTEMPTS = 3
NODE_STEP_RETRY_INTERVAL_SECONDS = 2.0

type RunNodeStep = Callable[
    [PredictionSpecRecord, NodeSpec, Mapping[str, Any]],
    NodeStepResult,
]


class PredictionGraphExecution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generation_run: GenerationRunRecord
    node_attempts: tuple[NodeAttemptRecord, ...]
    graph_result: GraphRunResult
    node_step_results: tuple[NodeStepResult, ...] = Field(
        default_factory=tuple
    )


def execute_prediction_graph(
    *,
    spec: PredictionSpecRecord,
    attempt_index: int,
    generation_run_id: str,
    started_at: datetime,
    completed_at: datetime,
    run_node_step: RunNodeStep,
) -> PredictionGraphExecution:
    graph_result, node_step_results = run_prediction_graph_core(
        spec=spec,
        run_node_step=run_node_step,
    )
    return _records_for_persistence(
        spec=spec,
        generation_run_id=generation_run_id,
        attempt_index=attempt_index,
        graph_result=graph_result,
        node_step_results=node_step_results,
        started_at=started_at,
        completed_at=completed_at,
    )


def run_prediction_graph_core(
    *,
    spec: PredictionSpecRecord,
    run_node_step: RunNodeStep,
) -> tuple[GraphRunResult, tuple[NodeStepResult, ...]]:
    node_step_results: list[NodeStepResult] = []

    def run_node(
        node: NodeSpec,
        node_inputs: Mapping[str, Any],
    ) -> NodeOutput:
        step_result = run_node_step(spec, node, node_inputs)
        node_step_results.append(step_result)
        return step_result.graph_output()

    graph_result = execute_graph(
        graph=spec.graph.graph,
        inputs=spec.task.inputs.values,
        run_node=run_node,
    )
    return graph_result, tuple(node_step_results)


@DBOS.workflow(name=PLATFORM_GENERATION_WORKFLOW_NAME)
def run_prediction_graph_workflow(
    prediction_id: str,
    attempt_index: int = 0,
    execution_recipe_digest: str = "",
) -> str:
    database_url = resolve_application_database_url()
    spec = PredictionSpecRecord.model_validate(
        load_prediction_spec_step(database_url, prediction_id)
    )
    generation_run_id = stable_generation_run_id(
        prediction_id=spec.prediction_id,
        execution_recipe_digest=execution_recipe_digest,
        attempt_index=attempt_index,
    )
    started_at = datetime.fromisoformat(
        generation_started_at_step(generation_run_id)
    )

    def run_node_step(
        step_spec: PredictionSpecRecord,
        node: NodeSpec,
        node_inputs: Mapping[str, Any],
    ) -> NodeStepResult:
        spec_payload = step_spec.model_dump(mode="json")
        node_payload = node.model_dump(mode="json")
        node_inputs_payload = dict(node_inputs)
        try:
            delay_seconds = throttle_preflight_step(
                database_url,
                spec_payload,
                node_payload,
            )
            sleep_for_backoff_seconds(delay_seconds)
            result = execute_lm_node_step(
                database_url,
                spec_payload,
                node_payload,
                node_inputs_payload,
                stable_node_attempt_id(
                    generation_run_id=generation_run_id,
                    node_id=node.id,
                    attempt_index=attempt_index,
                ),
            )
            return NodeStepResult.model_validate(result)
        except Exception as error:
            timing = node_step_timing_from_exception(error)
            result = node_step_error_result_step(
                spec_payload,
                node_payload,
                failure_metadata_from_exception(error).model_dump(mode="json"),
                timing[0].isoformat() if timing is not None else None,
                timing[1].isoformat() if timing is not None else None,
            )
            return NodeStepResult.model_validate(result)

    graph_result, node_step_results = run_prediction_graph_core(
        spec=spec,
        run_node_step=run_node_step,
    )
    completed_at = datetime.fromisoformat(
        generation_completed_at_step(generation_run_id)
    )
    persist_generation_result_step(
        database_url,
        spec.model_dump(mode="json"),
        generation_run_id,
        attempt_index,
        graph_result.model_dump(mode="json"),
        [
            step_result.model_dump(mode="json")
            for step_result in node_step_results
        ],
        started_at.isoformat(),
        completed_at.isoformat(),
    )
    return generation_run_id


def start_prediction_graph_workflow(
    database_url: str,
    prediction_id: str,
    attempt_index: int = 0,
) -> str:
    generation_run_id, _handle = _start_prediction_graph_workflow_handle(
        database_url=database_url,
        prediction_id=prediction_id,
        attempt_index=attempt_index,
    )
    return generation_run_id


def run_prediction_graph_workflow_once(
    database_url: str,
    prediction_id: str,
    attempt_index: int = 0,
) -> str:
    _generation_run_id, handle = _start_prediction_graph_workflow_handle(
        database_url=database_url,
        prediction_id=prediction_id,
        attempt_index=attempt_index,
    )
    result = handle.get_result()
    if not isinstance(result, str):
        raise TypeError("platform graph workflow returned a non-string result")
    return result


def platform_generation_workflow_id(generation_run_id: str) -> str:
    return f"{WORKFLOW_ID_PREFIX}:{generation_run_id}"


def _start_prediction_graph_workflow_handle(
    *,
    database_url: str,
    prediction_id: str,
    attempt_index: int,
) -> tuple[str, Any]:
    generation_run_id = stable_generation_run_id(
        prediction_id=prediction_id,
        execution_recipe_digest="",
        attempt_index=attempt_index,
    )
    workflow_id = platform_generation_workflow_id(generation_run_id)
    with SetWorkflowID(workflow_id):
        try:
            handle = DBOS.start_workflow(
                run_prediction_graph_workflow,
                prediction_id,
                attempt_index,
            )
        except WORKFLOW_START_RACE_ERRORS:
            handle = DBOS.retrieve_workflow(workflow_id)
        except Exception as error:
            if workflow_start_raced(workflow_id=workflow_id, error=error):
                handle = DBOS.retrieve_workflow(workflow_id)
            else:
                raise
    return generation_run_id, handle


@DBOS.step(name=LOAD_SPEC_STEP_NAME)
def load_prediction_spec_step(
    database_url: str,
    prediction_id: str,
) -> dict[str, Any]:
    engine = create_engine(database_url)
    try:
        with engine.begin() as connection:
            spec = load_prediction_spec(
                connection,
                prediction_id=prediction_id,
            )
        return spec.model_dump(mode="json")
    finally:
        engine.dispose()


@DBOS.step(name=GENERATION_STARTED_AT_STEP_NAME)
def generation_started_at_step(generation_run_id: str) -> str:
    return timestamp_now_iso()


@DBOS.step(name=GENERATION_COMPLETED_AT_STEP_NAME)
def generation_completed_at_step(generation_run_id: str) -> str:
    return timestamp_now_iso()


def timestamp_now_iso() -> str:
    return now().isoformat()


def sleep_for_backoff_seconds(seconds: float) -> None:
    if seconds <= 0:
        return
    try:
        DBOS.sleep(seconds)
    except Exception:
        time.sleep(seconds)


@DBOS.step(name=THROTTLE_PREFLIGHT_STEP_NAME)
def throttle_preflight_step(
    database_url: str,
    spec_payload: dict[str, Any],
    node_payload: dict[str, Any],
) -> float:
    return provider_throttle_delay_seconds(
        database_url,
        spec_payload,
        node_payload,
    )


def provider_throttle_delay_seconds(
    database_url: str,
    spec_payload: dict[str, Any],
    node_payload: dict[str, Any],
) -> float:
    provider_ref = provider_config_ref_for_node(
        spec=PredictionSpecRecord.model_validate(spec_payload),
        node=NodeSpec.model_validate(node_payload),
    )
    engine = create_engine(database_url)
    try:
        with engine.begin() as connection:
            return throttle_delay_seconds(
                connection,
                throttle_key=provider_ref.throttle_key,
                now=now(),
                schema=PLATFORM_SCHEMA,
            )
    finally:
        engine.dispose()


@DBOS.step(
    name=EXECUTE_NODE_STEP_NAME,
    retries_allowed=True,
    max_attempts=NODE_STEP_MAX_ATTEMPTS,
    interval_seconds=NODE_STEP_RETRY_INTERVAL_SECONDS,
    should_retry=should_retry_step,
)
def execute_lm_node_step(
    database_url: str,
    spec_payload: dict[str, Any],
    node_payload: dict[str, Any],
    node_inputs: dict[str, Any],
    node_attempt_id: str | None = None,
) -> dict[str, Any]:
    spec = PredictionSpecRecord.model_validate(spec_payload)
    node = NodeSpec.model_validate(node_payload)
    try:
        provider_ref = provider_config_ref_for_node(spec=spec, node=node)
    except Exception:
        provider_ref = None
    step_started_at = now()
    try:
        result = execute_lm_node(
            spec=spec,
            node=node,
            node_inputs=node_inputs,
            idempotency_key=node_attempt_id,
            raise_retryable=True,
        )
    except Exception as error:
        if node_step_timing_from_exception(error) is None:
            attach_node_step_timing_to_exception(
                error,
                started_at=step_started_at,
                completed_at=now(),
            )
        if provider_ref is not None:
            record_throttle_failure_state(
                database_url=database_url,
                throttle_key=provider_ref.throttle_key,
                error=error,
            )
        raise
    if result.status is NodeAttemptStatus.SUCCESS and provider_ref is not None:
        try:
            clear_throttle_backoff_state(
                database_url=database_url,
                throttle_key=provider_ref.throttle_key,
            )
        except Exception:
            pass
    return result.model_dump(mode="json")


def record_throttle_failure_state(
    *,
    database_url: str,
    throttle_key: str,
    error: BaseException,
) -> None:
    from whetstone.eval_failures import summarize_exception

    summary = summarize_exception(error)
    engine = create_engine(database_url)
    try:
        with engine.begin() as connection:
            record_throttle_failure(
                connection,
                throttle_key=throttle_key,
                failure_class=summary.failure_class,
                error_type=summary.failure_exception_type,
                message=summary.message,
                metadata=summary.failure_metadata,
                now=now(),
                schema=PLATFORM_SCHEMA,
            )
    finally:
        engine.dispose()


def clear_throttle_backoff_state(
    *,
    database_url: str,
    throttle_key: str,
) -> None:
    engine = create_engine(database_url)
    try:
        with engine.begin() as connection:
            clear_throttle_backoff(
                connection,
                throttle_key=throttle_key,
                now=now(),
                schema=PLATFORM_SCHEMA,
            )
    finally:
        engine.dispose()


@DBOS.step(name=NODE_STEP_ERROR_RESULT_STEP_NAME)
def node_step_error_result_step(
    spec_payload: dict[str, Any],
    node_payload: dict[str, Any],
    failure_payload: dict[str, Any],
    started_at: str | None = None,
    completed_at: str | None = None,
) -> dict[str, Any]:
    if started_at is not None and completed_at is not None:
        step_started_at = datetime.fromisoformat(started_at)
        step_completed_at = datetime.fromisoformat(completed_at)
    else:
        current_time = now()
        step_started_at = current_time
        step_completed_at = current_time
    result = node_step_error_result_from_failure(
        spec=PredictionSpecRecord.model_validate(spec_payload),
        node=NodeSpec.model_validate(node_payload),
        failure=FailureMetadataPayload.model_validate(failure_payload),
        started_at=step_started_at,
        completed_at=step_completed_at,
    )
    return result.model_dump(mode="json")


@DBOS.step(name=PERSIST_RESULT_STEP_NAME)
def persist_generation_result_step(
    database_url: str,
    spec_payload: dict[str, Any],
    generation_run_id: str,
    attempt_index: int,
    graph_result_payload: dict[str, Any],
    node_step_result_payloads: list[dict[str, Any]],
    started_at: str,
    completed_at: str,
) -> None:
    spec = PredictionSpecRecord.model_validate(spec_payload)
    graph_result = GraphRunResult.model_validate(graph_result_payload)
    node_step_results = tuple(
        NodeStepResult.model_validate(payload)
        for payload in node_step_result_payloads
    )
    execution = _records_for_persistence(
        spec=spec,
        generation_run_id=generation_run_id,
        attempt_index=attempt_index,
        graph_result=graph_result,
        node_step_results=node_step_results,
        started_at=datetime.fromisoformat(started_at),
        completed_at=datetime.fromisoformat(completed_at),
    )
    engine = create_engine(database_url)
    try:
        with engine.begin() as connection:
            persist_generation_result(
                connection,
                generation_run=execution.generation_run,
                node_attempts=execution.node_attempts,
            )
    finally:
        engine.dispose()


def _records_for_persistence(
    *,
    spec: PredictionSpecRecord,
    generation_run_id: str,
    attempt_index: int,
    graph_result: GraphRunResult,
    node_step_results: tuple[NodeStepResult, ...],
    started_at: datetime,
    completed_at: datetime,
) -> PredictionGraphExecution:
    generation_run = generation_run_record_from_result(
        spec=spec,
        generation_run_id=generation_run_id,
        attempt_index=attempt_index,
        result=graph_result,
        started_at=started_at,
        completed_at=completed_at,
    )
    node_attempts = node_attempt_records_from_steps(
        spec=spec,
        generation_run_id=generation_run_id,
        step_results=node_step_results,
    )
    return PredictionGraphExecution(
        generation_run=generation_run,
        node_attempts=node_attempts,
        graph_result=graph_result,
        node_step_results=node_step_results,
    )
