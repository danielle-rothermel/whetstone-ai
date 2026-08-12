"""The generation-executor seam onto dr-graph's graph interpreter.

The generation executor interprets the generation graph via
:func:`dr_graph.execute_graph`; whetstone supplies node behaviors as
``RunNode`` callables and carries per-leg trace and usage evidence in
``NodeOutput.metadata``. Node-behavior failures raise
:class:`GenerationNodeError`, whose ``failure_class`` is a whetstone
:class:`~whetstone.evaluation.attribution.AttributedOutcome` value (dr-graph
reads it off the exception type and unwraps the StrEnum), so graph-complete
partial outcomes feed the pinned F4 attribution table directly. Execution is
serial and intra-graph: one graph run per generation row.
"""

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

# In-memory NodeOutput.metadata / NodeError.metadata keys for the per-leg
# evidence channel between node behaviors and the row mapping. These keys
# never cross a persistence boundary; the persisted row outcome models keep
# their own pinned schemas.
METADATA_PROMPT_KEY = "prompt"
METADATA_FAILURE_CODE_KEY = "failure_code"
METADATA_REDRIVABLE_KEY = "redrivable"
METADATA_CACHE_HIT_KEY = "cache_hit"
METADATA_CACHE_PROVENANCE_KEY = "cache_provenance"
METADATA_CACHE_SOURCE_PHASE_KEY = "cache_source_phase"
METADATA_CACHE_SOURCE_UNIT_KEY = "cache_source_unit"
METADATA_CACHE_SOURCE_CALL_ID_KEY = "cache_source_call_id"
METADATA_CACHE_SOURCE_AT_KEY = "cache_source_at"
METADATA_SUBMISSION_RESULT_KEY = "code_submission_result"

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
    """A typed node-behavior failure with its attribution failure class.

    ``failure_class`` aligns with whetstone's attribution inputs
    (:class:`AttributedOutcome`); ``metadata`` carries the per-leg failure
    evidence (telemetry, failure code, redrivability, submission record) the
    row mapping reads back off :class:`dr_graph.NodeError`.
    """

    failure_class: ClassVar[AttributedOutcome] = (
        AttributedOutcome.INFRASTRUCTURE
    )

    def __init__(
        self, message: str, *, metadata: Mapping[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.metadata: dict[str, Any] = dict(metadata or {})


def telemetry_metadata(telemetry: CallTelemetry) -> dict[str, Any]:
    """Pack one call's telemetry into the per-leg metadata channel."""
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
    """Rebuild one call's telemetry from the per-leg metadata channel."""
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
    """Pack one leg's partial-record cache marks into the metadata channel."""
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
    """Rebuild one leg's cache marks from the metadata channel."""
    return PartialCacheMarks(
        cache_hit=metadata.get(METADATA_CACHE_HIT_KEY) is True,
        cache_source_phase=metadata.get(METADATA_CACHE_SOURCE_PHASE_KEY),
        cache_source_unit=metadata.get(METADATA_CACHE_SOURCE_UNIT_KEY),
        cache_source_call_id=metadata.get(METADATA_CACHE_SOURCE_CALL_ID_KEY),
        cache_source_at=metadata.get(METADATA_CACHE_SOURCE_AT_KEY),
    )


def external_input_field(source_ref: str) -> str:
    """The bare external-input field name of a ``task.<field>`` source ref.

    ``execute_graph`` resolves graph external inputs by their bare field
    names, while graph builders declare them in ``task.<field>`` source-ref
    form.
    """
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
    """Interpret one generation graph run, keeping cancellation graph-complete.

    An interrupted run (dr-graph maps ``asyncio.CancelledError`` and
    ``KeyboardInterrupt`` to cancellation) returns its graph-complete partial
    result — the cancelled node plus BLOCKED downstream nodes — instead of
    escaping the row.
    """
    try:
        return execute_graph(graph=graph, inputs=inputs, run_node=run_node)
    except GraphRunInterruptedError as interruption:
        return interruption.partial_result


def graph_run_cancelled(run: GraphRunResult) -> bool:
    """Whether any node outcome of this run was cancelled."""
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
    """The executed-row state the pinned table assigns to cancellation."""
    return _row_state_for_outcome_kind(AttributedOutcome.CANCELLATION)


def node_error_row_state(error: NodeError) -> ExecutedRowState:
    """Derive the executed-row state of a node error via the pinned table."""
    kind = AttributedOutcome.INFRASTRUCTURE
    if error.failure_class is not None:
        try:
            kind = AttributedOutcome(error.failure_class)
        except ValueError:
            kind = AttributedOutcome.INFRASTRUCTURE
    return _row_state_for_outcome_kind(kind)


#: Stable failure code for a node error that carried no explicit code (an
#: unexpected node-behavior exception surfaced by the interpreter).
_NODE_EXECUTION_FAILURE_CODE = "node_execution_error"


def node_error_failure_code(error: NodeError) -> str:
    """The most specific stable failure code carried by a node error."""
    code = error.metadata.get(METADATA_FAILURE_CODE_KEY)
    if isinstance(code, str) and code:
        return code
    return _NODE_EXECUTION_FAILURE_CODE


def node_error_redrivable(error: NodeError) -> bool:
    """Whether the failed leg was a transient transport fault."""
    return error.metadata.get(METADATA_REDRIVABLE_KEY) is True


def require_node_error(outcome: NodeOutcome) -> NodeError:
    """The node's error, asserting the outcome really is an error."""
    if outcome.status is not NodeOutcomeStatus.ERROR or outcome.error is None:
        raise AssertionError(
            f"node {outcome.node_id!r} was expected to carry an error"
        )
    return outcome.error


def require_node_success(outcome: NodeOutcome) -> NodeOutput:
    """The node's output, asserting the outcome really succeeded."""
    if (
        outcome.status is not NodeOutcomeStatus.SUCCESS
        or outcome.output is None
    ):
        raise AssertionError(
            f"node {outcome.node_id!r} was expected to have succeeded"
        )
    return outcome.output


def single_node_input(node: NodeConfig, node_inputs: Mapping[str, Any]) -> str:
    """The one declared input of a single-input node, required to be text."""
    (field,) = node.input_fields()
    value = node_inputs[field.name]
    if type(value) is not str:
        raise ValueError(
            f"node {node.node_id!r} input {field.name!r} must be text"
        )
    return value


def node_text(output: NodeOutput, *, field: str) -> str:
    """A node output's declared text value (e.g. a provider generation)."""
    value = output.values[field]
    if type(value) is not str:
        raise AssertionError(
            f"node output field {field!r} was expected to be text"
        )
    return value


def metadata_prompt(metadata: Mapping[str, Any]) -> str:
    """The exact wire prompt an LLM node behavior recorded for its leg."""
    prompt = metadata.get(METADATA_PROMPT_KEY)
    if type(prompt) is not str:
        raise AssertionError("LLM node metadata must record its wire prompt")
    return prompt
