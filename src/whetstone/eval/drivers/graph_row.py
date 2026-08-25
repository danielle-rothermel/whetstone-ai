from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from dr_graph import GraphRunResult, NodeOutcomeStatus

from whetstone.eval.drivers.graph_execution import (
    METADATA_SUBMISSION_RESULT_KEY,
    bounded_node_error_message,
    cache_marks_from_metadata,
    graph_run_cancelled,
    metadata_prompt,
    node_error_failure_code,
    node_error_row_state,
    node_text,
    require_node_error,
    require_node_success,
    run_rollout_graph,
    telemetry_from_metadata,
)
from whetstone.eval.drivers.row_common import RolloutRowOutput
from whetstone.eval.traces import (
    ExecutedComponentStep,
    ExecutedRowState,
    llm_component_step,
)
from whetstone.experiment.graph.nodes import (
    EVAL_OUTPUT_FIELD,
    PROVIDER_GENERATION_OUTPUT_FIELD,
)
from whetstone.experiment.graph.rollout_template import EVAL_NODE_ID, LLM_NODE_ID

if TYPE_CHECKING:
    from dr_graph import GraphConfig, RunNode

__all__ = [
    "execute_rollout_graph",
    "graph_result_to_component_steps",
    "graph_result_to_row_fields",
]


def execute_rollout_graph(
    *,
    graph: GraphConfig,
    inputs: Mapping[str, Any],
    run_node: RunNode,
) -> GraphRunResult:
    return run_rollout_graph(graph=graph, inputs=inputs, run_node=run_node)


def graph_result_to_component_steps(
    run: GraphRunResult,
    *,
    llm_node_id: str = LLM_NODE_ID,
    llm_output_field: str = PROVIDER_GENERATION_OUTPUT_FIELD,
) -> tuple[ExecutedComponentStep, ...]:
    outcome = run.outcomes.get(llm_node_id)
    if outcome is None or outcome.status is not NodeOutcomeStatus.SUCCESS:
        return ()
    output = require_node_success(outcome)
    prompt = metadata_prompt(output.metadata)
    generation = node_text(output, field=llm_output_field)
    return (
        llm_component_step(
            trace_index=0,
            component_id=llm_node_id,
            prompt=prompt,
            generation=generation,
        ),
    )


def graph_result_to_row_fields(
    run: GraphRunResult,
    *,
    candidate_id: str,
    task_id: str,
    task_index: int,
    seed_index: int,
    llm_node_id: str = LLM_NODE_ID,
    eval_node_id: str = EVAL_NODE_ID,
    llm_output_field: str = PROVIDER_GENERATION_OUTPUT_FIELD,
    eval_output_field: str = EVAL_OUTPUT_FIELD,
) -> RolloutRowOutput:
    if graph_run_cancelled(run):
        return RolloutRowOutput(
            candidate_id=candidate_id,
            task_id=task_id,
            task_index=task_index,
            seed_index=seed_index,
            row_state=ExecutedRowState.MISSING,
            trace_steps=(),
            output_text=None,
            score=None,
            failure_code="cancelled",
        )

    llm_outcome = run.outcomes.get(llm_node_id)
    eval_outcome = run.outcomes.get(eval_node_id)
    component_steps = graph_result_to_component_steps(
        run,
        llm_node_id=llm_node_id,
        llm_output_field=llm_output_field,
    )

    if llm_outcome is not None and llm_outcome.status is NodeOutcomeStatus.ERROR:
        error = require_node_error(llm_outcome)
        telemetry = telemetry_from_metadata(error.metadata)
        return RolloutRowOutput(
            candidate_id=candidate_id,
            task_id=task_id,
            task_index=task_index,
            seed_index=seed_index,
            row_state=node_error_row_state(error),
            trace_steps=component_steps,
            output_text=None,
            score=None,
            failure_code=node_error_failure_code(error),
            finish_reason=telemetry.finish_reason,
            provider_error=(
                None
                if telemetry.provider_error is None
                else dict(telemetry.provider_error)
            ),
            prompt_tokens=telemetry.prompt_tokens,
            completion_tokens=telemetry.completion_tokens,
            provider_cost=telemetry.provider_cost,
            cache_hit=cache_marks_from_metadata(error.metadata).cache_hit,
            error_type=error.error_type,
            error_message=bounded_node_error_message(error.message),
            failed_node_id=llm_node_id,
        )

    output_text: str | None = None
    finish_reason: str | None = None
    provider_error: dict[str, object] | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    provider_cost: float | None = None
    cache_hit = False
    if llm_outcome is not None and llm_outcome.status is NodeOutcomeStatus.SUCCESS:
        llm_output = require_node_success(llm_outcome)
        output_text = node_text(llm_output, field=llm_output_field)
        telemetry = telemetry_from_metadata(llm_output.metadata)
        finish_reason = telemetry.finish_reason
        provider_error = (
            None
            if telemetry.provider_error is None
            else dict(telemetry.provider_error)
        )
        prompt_tokens = telemetry.prompt_tokens
        completion_tokens = telemetry.completion_tokens
        provider_cost = telemetry.provider_cost
        cache_hit = cache_marks_from_metadata(llm_output.metadata).cache_hit

    score: float | None = None
    submission_result: object | None = None
    row_state = ExecutedRowState.SUCCESS
    failure_code = ""
    error_type: str | None = None
    error_message: str | None = None
    failed_node_id: str | None = None

    if eval_outcome is None:
        row_state = ExecutedRowState.MISSING
        failure_code = "missing_eval_outcome"
    elif eval_outcome.status is NodeOutcomeStatus.ERROR:
        error = require_node_error(eval_outcome)
        row_state = node_error_row_state(error)
        failure_code = node_error_failure_code(error)
        error_type = error.error_type
        error_message = bounded_node_error_message(error.message)
        failed_node_id = eval_node_id
    elif eval_outcome.status is NodeOutcomeStatus.SUCCESS:
        eval_output = require_node_success(eval_outcome)
        raw_score = eval_output.values.get(eval_output_field)
        if raw_score is not None and type(raw_score) not in (int, float):
            raise ValueError("eval node score must be numeric or null")
        score = None if raw_score is None else float(raw_score)
        submission_result = eval_output.metadata.get(
            METADATA_SUBMISSION_RESULT_KEY
        )
    else:
        row_state = ExecutedRowState.MISSING
        failure_code = "eval_not_successful"

    return RolloutRowOutput(
        candidate_id=candidate_id,
        task_id=task_id,
        task_index=task_index,
        seed_index=seed_index,
        row_state=row_state,
        trace_steps=component_steps,
        output_text=output_text,
        score=score,
        failure_code=failure_code,
        finish_reason=finish_reason,
        provider_error=provider_error,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        provider_cost=provider_cost,
        cache_hit=cache_hit,
        submission_result=submission_result,
        error_type=error_type,
        error_message=error_message,
        failed_node_id=failed_node_id,
    )
