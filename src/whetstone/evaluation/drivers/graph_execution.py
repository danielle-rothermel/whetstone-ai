from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from dr_graph import (
    GraphRunInterruptedError,
    NodeInputSourceKind,
    NodeOutcomeStatus,
    as_node_input_source_ref,
    execute_graph,
)

from whetstone.evaluation.attribution import (
    AccountingCell,
    AttributedOutcome,
    attribute_outcome,
)
from whetstone.evaluation.traces import ExecutedRowState
from whetstone.execution.call_support import CallTelemetry
from whetstone.execution.prompt_cache import PartialCacheMarks

if TYPE_CHECKING:
    from collections.abc import Mapping

    from dr_graph import (
        GraphConfig,
        GraphRunResult,
        NodeConfig,
        NodeError,
        NodeOutcome,
        NodeOutput,
        RunNode,
    )

__all__ = [
    "METADATA_CACHE_HIT_KEY",
    "METADATA_CACHE_PROVENANCE_KEY",
    "METADATA_CACHE_SOURCE_AT_KEY",
    "METADATA_CACHE_SOURCE_CALL_ID_KEY",
    "METADATA_CACHE_SOURCE_PHASE_KEY",
    "METADATA_CACHE_SOURCE_UNIT_KEY",
    "METADATA_FAILURE_CODE_KEY",
    "METADATA_PROMPT_KEY",
    "METADATA_REDRIVABLE_KEY",
    "METADATA_SUBMISSION_RESULT_KEY",
    "GenerationNodeError",
    "cache_marks_from_metadata",
    "cache_marks_metadata",
    "cancelled_row_state",
    "external_input_field",
    "graph_run_cancelled",
    "metadata_prompt",
    "node_error_failure_code",
    "node_error_redrivable",
    "node_error_row_state",
    "node_text",
    "require_node_error",
    "require_node_success",
    "run_generation_graph",
    "single_node_input",
    "telemetry_from_metadata",
    "telemetry_metadata",
]


METADATA_PROMPT_KEY = "prompt"
METADATA_FAILURE_CODE_KEY = "failure_code"
METADATA_REDRIVABLE_KEY = "redrivable"
METADATA_CACHE_HIT_KEY = "cache_hit"
METADATA_CACHE_PROVENANCE_KEY = "cache_provenance"
METADATA_CACHE_SOURCE_PHASE_KEY = "cache_source_phase"
METADATA_CACHE_SOURCE_UNIT_KEY = "cache_source_unit"
METADATA_CACHE_SOURCE_CALL_ID_KEY = "cache_source_call_id"
METADATA_CACHE_SOURCE_AT_KEY = "cache_source_at"
METADATA_SUBMISSION_RESULT_KEY = "submission_result"

_TELEMETRY_KEYS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "reasoning_tokens",
    "latency_s",
    "finish_reason",
    "provider_error",
)


class GenerationNodeError(Exception):
    failure_class: ClassVar[AttributedOutcome] = (
        AttributedOutcome.INFRASTRUCTURE
    )

    def __init__(
        self, message: str, *, metadata: Mapping[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.metadata: dict[str, Any] = dict(metadata or {})


def telemetry_metadata(telemetry: CallTelemetry) -> dict[str, Any]:
    return {
        "prompt_tokens": telemetry.prompt_tokens,
        "completion_tokens": telemetry.completion_tokens,
        "total_tokens": telemetry.total_tokens,
        "reasoning_tokens": telemetry.reasoning_tokens,
        "latency_s": telemetry.latency_s,
        "finish_reason": telemetry.finish_reason,
        "provider_error": telemetry.provider_error,
    }


def telemetry_from_metadata(metadata: Mapping[str, Any]) -> CallTelemetry:
    values = {key: metadata.get(key) for key in _TELEMETRY_KEYS}
    latency = values["latency_s"]
    return CallTelemetry(
        prompt_tokens=values["prompt_tokens"],
        completion_tokens=values["completion_tokens"],
        total_tokens=values["total_tokens"],
        reasoning_tokens=values["reasoning_tokens"],
        latency_s=None if latency is None else float(latency),
        finish_reason=values["finish_reason"],
        provider_error=values["provider_error"],
    )


def cache_marks_metadata(marks: PartialCacheMarks) -> dict[str, Any]:
    return {
        METADATA_CACHE_HIT_KEY: marks.cache_hit,
        METADATA_CACHE_SOURCE_PHASE_KEY: marks.cache_source_phase,
        METADATA_CACHE_SOURCE_UNIT_KEY: marks.cache_source_unit,
        METADATA_CACHE_SOURCE_CALL_ID_KEY: marks.cache_source_call_id,
        METADATA_CACHE_SOURCE_AT_KEY: marks.cache_source_at,
    }


def cache_marks_from_metadata(
    metadata: Mapping[str, Any],
) -> PartialCacheMarks:
    return PartialCacheMarks(
        cache_hit=metadata.get(METADATA_CACHE_HIT_KEY) is True,
        cache_source_phase=metadata.get(METADATA_CACHE_SOURCE_PHASE_KEY),
        cache_source_unit=metadata.get(METADATA_CACHE_SOURCE_UNIT_KEY),
        cache_source_call_id=metadata.get(METADATA_CACHE_SOURCE_CALL_ID_KEY),
        cache_source_at=metadata.get(METADATA_CACHE_SOURCE_AT_KEY),
    )


def external_input_field(source_ref: str) -> str:
    ref = as_node_input_source_ref(source_ref)
    if ref.kind is not NodeInputSourceKind.GRAPH_EXTERNAL or ref.field is None:
        raise ValueError(
            f"source ref {source_ref!r} is not a graph external input"
        )
    return ref.field


def run_generation_graph(
    *,
    graph: GraphConfig,
    inputs: Mapping[str, Any],
    run_node: RunNode,
) -> GraphRunResult:
    try:
        return execute_graph(graph=graph, inputs=inputs, run_node=run_node)
    except GraphRunInterruptedError as interruption:
        return interruption.partial_result


def graph_run_cancelled(run: GraphRunResult) -> bool:
    return any(
        outcome.status is NodeOutcomeStatus.CANCELLED
        for outcome in run.outcomes.values()
    )


_ROW_STATE_BY_CELL = {
    AccountingCell.FAILED: ExecutedRowState.FAILED,
    AccountingCell.MISSING: ExecutedRowState.MISSING,
}


def _row_state_for_outcome_kind(kind: AttributedOutcome) -> ExecutedRowState:
    cell = attribute_outcome(kind)
    row_state = _ROW_STATE_BY_CELL.get(cell)
    if row_state is None:
        raise ValueError(
            f"outcome kind {kind.value!r} attributes to cell {cell.value!r}, "
            "which has no executed-row state"
        )
    return row_state


def cancelled_row_state() -> ExecutedRowState:
    return _row_state_for_outcome_kind(AttributedOutcome.CANCELLATION)


def node_error_row_state(error: NodeError) -> ExecutedRowState:
    kind = AttributedOutcome.INFRASTRUCTURE
    if error.failure_class is not None:
        try:
            kind = AttributedOutcome(error.failure_class)
        except ValueError:
            kind = AttributedOutcome.INFRASTRUCTURE
    return _row_state_for_outcome_kind(kind)


_NODE_EXECUTION_FAILURE_CODE = "node_execution_error"


def node_error_failure_code(error: NodeError) -> str:
    code = error.metadata.get(METADATA_FAILURE_CODE_KEY)
    if isinstance(code, str) and code:
        return code
    return _NODE_EXECUTION_FAILURE_CODE


def node_error_redrivable(error: NodeError) -> bool:
    return error.metadata.get(METADATA_REDRIVABLE_KEY) is True


def require_node_error(outcome: NodeOutcome) -> NodeError:
    if outcome.status is not NodeOutcomeStatus.ERROR or outcome.error is None:
        raise AssertionError(
            f"node {outcome.node_id!r} was expected to carry an error"
        )
    return outcome.error


def require_node_success(outcome: NodeOutcome) -> NodeOutput:
    if (
        outcome.status is not NodeOutcomeStatus.SUCCESS
        or outcome.output is None
    ):
        raise AssertionError(
            f"node {outcome.node_id!r} was expected to have succeeded"
        )
    return outcome.output


def single_node_input(node: NodeConfig, node_inputs: Mapping[str, Any]) -> str:
    (field,) = node.input_fields()
    value = node_inputs[field.name]
    if type(value) is not str:
        raise ValueError(
            f"node {node.node_id!r} input {field.name!r} must be text"
        )
    return value


def node_text(output: NodeOutput, *, field: str) -> str:
    value = output.values[field]
    if type(value) is not str:
        raise AssertionError(
            f"node output field {field!r} was expected to be text"
        )
    return value


def metadata_prompt(metadata: Mapping[str, Any]) -> str:
    prompt = metadata.get(METADATA_PROMPT_KEY)
    if type(prompt) is not str:
        raise AssertionError("LLM node metadata must record its wire prompt")
    return prompt
