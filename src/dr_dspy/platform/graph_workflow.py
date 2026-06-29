from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any

from dbos import DBOS, SetWorkflowID
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import create_engine

from dr_dspy.eval_failures import should_retry_step
from dr_dspy.graph import GraphRunResult, NodeOutput, NodeSpec, execute_graph
from dr_dspy.platform.node_execution import (
    NodeStepResult,
    execute_lm_node,
    node_step_error_result,
)
from dr_dspy.platform.persistence import (
    generation_run_record_from_result,
    load_prediction_spec,
    node_attempt_records_from_steps,
    persist_generation_result,
)
from dr_dspy.records import (
    GenerationRunRecord,
    NodeAttemptRecord,
    PredictionSpecRecord,
    stable_generation_run_id,
)

PLATFORM_GENERATION_WORKFLOW_NAME = "dr_dspy_platform_graph_generation_v1"
LOAD_SPEC_STEP_NAME = "dr_dspy_platform_load_prediction_spec_v1"
GENERATION_STARTED_AT_STEP_NAME = (
    "dr_dspy_platform_generation_started_at_v1"
)
GENERATION_COMPLETED_AT_STEP_NAME = (
    "dr_dspy_platform_generation_completed_at_v1"
)
NODE_ATTEMPT_STARTED_AT_STEP_NAME = (
    "dr_dspy_platform_node_attempt_started_at_v1"
)
NODE_ATTEMPT_COMPLETED_AT_STEP_NAME = (
    "dr_dspy_platform_node_attempt_completed_at_v1"
)
EXECUTE_NODE_STEP_NAME = "dr_dspy_platform_execute_lm_node_v1"
PERSIST_RESULT_STEP_NAME = "dr_dspy_platform_persist_generation_result_v1"
WORKFLOW_ID_PREFIX = "platform-generate-v1"

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
    database_url: str,
    prediction_id: str,
    attempt_index: int = 0,
) -> str:
    spec = PredictionSpecRecord.model_validate(
        load_prediction_spec_step(database_url, prediction_id)
    )
    generation_run_id = stable_generation_run_id(
        prediction_id=spec.prediction_id,
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
        node_started_at = datetime.fromisoformat(
            node_attempt_started_at_step(generation_run_id, node.id)
        )
        try:
            result = execute_lm_node_step(
                step_spec.model_dump(mode="json"),
                node.model_dump(mode="json"),
                dict(node_inputs),
            )
            return NodeStepResult.model_validate(result)
        except Exception as error:
            node_completed_at = datetime.fromisoformat(
                node_attempt_completed_at_step(generation_run_id, node.id)
            )
            return node_step_error_result(
                spec=step_spec,
                node=node,
                error=error,
                started_at=node_started_at,
                completed_at=node_completed_at,
            )

    graph_result, node_step_results = run_prediction_graph_core(
        spec=spec,
        run_node_step=run_node_step,
    )
    completed_at = datetime.fromisoformat(
        generation_completed_at_step(generation_run_id)
    )
    execution = _records_for_persistence(
        spec=spec,
        attempt_index=attempt_index,
        generation_run_id=generation_run_id,
        graph_result=graph_result,
        node_step_results=node_step_results,
        started_at=started_at,
        completed_at=completed_at,
    )
    persist_generation_result_step(
        database_url,
        spec.model_dump(mode="json"),
        generation_run_id,
        attempt_index,
        execution.graph_result.model_dump(mode="json"),
        [
            step_result.model_dump(mode="json")
            for step_result in execution.node_step_results
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
    generation_run_id = stable_generation_run_id(
        prediction_id=prediction_id,
        attempt_index=attempt_index,
    )
    with SetWorkflowID(f"{WORKFLOW_ID_PREFIX}:{generation_run_id}"):
        DBOS.start_workflow(
            run_prediction_graph_workflow,
            database_url,
            prediction_id,
            attempt_index,
        )
    return generation_run_id


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


@DBOS.step(name=NODE_ATTEMPT_STARTED_AT_STEP_NAME)
def node_attempt_started_at_step(
    generation_run_id: str,
    node_id: str,
) -> str:
    return timestamp_now_iso()


@DBOS.step(name=NODE_ATTEMPT_COMPLETED_AT_STEP_NAME)
def node_attempt_completed_at_step(
    generation_run_id: str,
    node_id: str,
) -> str:
    return timestamp_now_iso()


def timestamp_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@DBOS.step(
    name=EXECUTE_NODE_STEP_NAME,
    retries_allowed=True,
    max_attempts=3,
    interval_seconds=2.0,
    should_retry=should_retry_step,
)
def execute_lm_node_step(
    spec_payload: dict[str, Any],
    node_payload: dict[str, Any],
    node_inputs: dict[str, Any],
) -> dict[str, Any]:
    result = execute_lm_node(
        spec=PredictionSpecRecord.model_validate(spec_payload),
        node=NodeSpec.model_validate(node_payload),
        node_inputs=node_inputs,
        raise_retryable=True,
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
