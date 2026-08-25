from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from dr_graph import (
    GraphRunInterruptedError,
    NodeInputSourceKind,
    NodeOutcomeStatus,
    as_node_input_source_ref,
    execute_graph,
)

from whetstone.eval.attribution import (
    AccountingCell,
    AttributedOutcome,
    attribute_generated_row_cell,
    attribute_outcome,
)
from whetstone.eval.traces import ExecutedRowState
from whetstone.execution.call_metadata import (
    METADATA_CACHE_HIT_KEY,
    METADATA_CACHE_PROVENANCE_KEY,
    METADATA_CACHE_SOURCE_AT_KEY,
    METADATA_CACHE_SOURCE_CALL_ID_KEY,
    METADATA_CACHE_SOURCE_PHASE_KEY,
    METADATA_CACHE_SOURCE_UNIT_KEY,
    METADATA_FAILURE_CODE_KEY,
    METADATA_PROMPT_KEY,
    METADATA_REDRIVABLE_KEY,
    METADATA_SUBMISSION_RESULT_KEY,
    cache_marks_from_metadata,
    cache_marks_metadata,
    telemetry_from_metadata,
    telemetry_metadata,
)

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
    "MAX_NODE_ERROR_MESSAGE_CHARS",
    "bounded_node_error_message",
    "node_error_redrivable",
    "node_error_row_state",
    "node_text",
    "require_node_error",
    "require_node_success",
    "run_rollout_graph",
    "single_node_input",
    "telemetry_from_metadata",
    "telemetry_metadata",
]


class GenerationNodeError(Exception):
    failure_class: ClassVar[AttributedOutcome] = (
        AttributedOutcome.INFRASTRUCTURE
    )

    def __init__(
        self, message: str, *, metadata: Mapping[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.metadata: dict[str, Any] = dict(metadata or {})










def external_input_field(source_ref: str) -> str:
    ref = as_node_input_source_ref(source_ref)
    if ref.kind is not NodeInputSourceKind.GRAPH_EXTERNAL or ref.field is None:
        raise ValueError(
            f"source ref {source_ref!r} is not a graph external input"
        )
    return ref.field


def run_rollout_graph(
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


_ROW_STATE_BY_ACCOUNTING_CELL = {
    AccountingCell.FAILED: ExecutedRowState.FAILED,
    AccountingCell.MISSING: ExecutedRowState.MISSING,
    AccountingCell.INVALID: ExecutedRowState.INVALID,
}


def node_error_row_state(error: NodeError) -> ExecutedRowState:
    """Attribute one node failure to the row state the contract assigns it.

    The failure *code* is consulted before the failure *class*. A provider
    that returns a blank generation or refuses the request has not failed
    infrastructurally: the eval contract scores those rows ``invalid``. Only
    codes with no contract attribution fall back to the raised class, and
    finally to ``infrastructure``.
    """
    code = error.metadata.get(METADATA_FAILURE_CODE_KEY)
    if isinstance(code, str) and code:
        cell = attribute_generated_row_cell(code)
        if cell is not None:
            row_state = _ROW_STATE_BY_ACCOUNTING_CELL.get(cell)
            if row_state is not None:
                return row_state
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


#: Longest exception message persisted on a row. A node failure's message is
#: diagnostic evidence, not a payload: bounding it keeps one pathological
#: error from dominating an evidence object that holds hundreds of rows.
MAX_NODE_ERROR_MESSAGE_CHARS = 2000


def bounded_node_error_message(message: str) -> str:
    """Truncate one node error message to its persisted bound."""
    if len(message) <= MAX_NODE_ERROR_MESSAGE_CHARS:
        return message
    return message[:MAX_NODE_ERROR_MESSAGE_CHARS] + "...[truncated]"


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
