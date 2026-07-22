"""The Codex CLI agent adapter: exactly one opaque, tool-using Step.

One Codex Optimization Run is **exactly one** opaque, long-running, tool-using
Optimization Step (``codex-agent-run.html``, ``optimizer-briefs.md`` section
5).
The Step Request crosses the process boundary as a prompt plus files and
serializes the Tool Config; the MCP *bridge* exposes the non-serializable
Runtime Tool Handle. The agent loop is opaque -- Whetstone observes no internal
strategy -- and exits on agent-stop or the ``max_evaluation_calls`` maximum.
The
sole Step Result carries the ``returned_proposal_count = 4`` base-bound
proposals plus the accepted/refused/completed Tool Call Store Entries, Tool
Result references, agent conversation evidence, and control cost.

The adapter drives a :class:`CodexRunner`:

* the subprocess runner (in ``codex_runner``) launches ``codex exec``
  non-interactively, configured to use the whetstone MCP server (which exposes
  ``evaluate_candidate``), passes the task prompt, and treats the process
  output
  as the opaque execution. The codex binary and model are configurable.
* the fake runner is the deterministic test double: a scripted MCP client
  driving the same in-process MCP server through the same protocol, with no
  real CLI. All deterministic tests use it.

After the runner returns, the adapter reads the DURABLE evidence the bridge
left
in the Tool Call Store (acceptance/refusal/completion under
``(tool_config_hash, call_id)``) and reconstructs the Tool Result references --
so the sole Step Result references every Tool Result + terminal Entry even
though the calls were made out-of-process. Budgets, status, and the terminal
result are durable outside the opacity.
"""

from __future__ import annotations

from typing import Any, Protocol

from dr_store import ObjectStore

from whetstone.optimization.adapters import AdapterOutput, ToolCallRecord
from whetstone.optimization.mutation import DiffCheckError, diff_check
from whetstone.optimization.schema import (
    Candidate,
    OptimizationStepRequest,
    StepMode,
    StepStatus,
)
from whetstone.optimization.tool_store import (
    ToolCallState,
    ToolCallStore,
    ToolCallStoreEntry,
)
from whetstone.optimization.tools import (
    ToolCall,
    ToolConfig,
    ToolResult,
    tool_result_reference,
)

__all__ = [
    "CodexAdapter",
    "CodexRunResult",
    "CodexRunner",
    "CodexToolCallLog",
    "OpaqueStepError",
]


class OpaqueStepError(Exception):
    """The opaque Codex Step could not produce its sole durable Step Result."""


class CodexToolCallLog:
    """One tool call the opaque agent made, as read back from durable state.

    The agent's calls are opaque while running; after it exits, each accepted
    or refused call is a durable Tool Call Store Entry keyed by
    ``(tool_config_hash, call_id)``. This pairs that call's identity with the
    persisted Tool Result so the adapter can reconstruct a
    :class:`ToolCallRecord` for the Step Result evidence.
    """

    __slots__ = ("call_id", "result")

    def __init__(self, *, call_id: str, result: ToolResult) -> None:
        self.call_id = call_id
        self.result = result


class CodexRunResult:
    """What a :class:`CodexRunner` returns after the opaque Step exits.

    * ``proposals`` -- the ``returned_proposal_count`` ordered, stable-ID,
      base-bound proposals the agent wrote (parsed from its final JSON output).
    * ``tool_calls`` -- the tool calls the agent made, read back from durable
      Tool Call Store state (identity + persisted Tool Result), in agent order.
    * ``conversation_evidence`` -- opaque agent conversation evidence (a
      transcript reference/summary); record-local provenance, never a score.
    * ``control_cost`` -- agent tokens + wall clock; advisory, never a score.
    """

    __slots__ = (
        "control_cost",
        "conversation_evidence",
        "proposals",
        "tool_calls",
    )

    def __init__(
        self,
        *,
        proposals: tuple[Candidate, ...],
        tool_calls: tuple[CodexToolCallLog, ...],
        conversation_evidence: dict[str, Any] | None = None,
        control_cost: dict[str, Any] | None = None,
    ) -> None:
        self.proposals = proposals
        self.tool_calls = tool_calls
        self.conversation_evidence = conversation_evidence or {}
        self.control_cost = control_cost or {}


class CodexRunner(Protocol):
    """Runs the one opaque Codex Step and returns its result.

    The runner owns everything inside the opacity: launching (or emulating) the
    agent, exposing the MCP bridge, and collecting the final proposals +
    durable
    tool evidence + control cost. Whetstone observes only the returned
    :class:`CodexRunResult`.
    """

    def run(
        self, request: OptimizationStepRequest, tool_config: ToolConfig
    ) -> CodexRunResult: ...


class CodexAdapter:
    """The tool-using Codex adapter: EXACTLY ONE opaque Step per run.

    ``mode`` is tool-using so the harness treats it on the tool-using path, but
    the tools are exercised out-of-process by the agent via the MCP bridge; the
    Runtime Tool Handles the harness constructs are therefore not called by
    this
    adapter directly. The adapter drives the :class:`CodexRunner`, validates
    the
    returned proposals against the Mutation Surface + base-binding contract,
    and
    reconstructs the Tool Result evidence from durable Tool Call Store state so
    the sole Step Result references every Tool Result + terminal Entry.
    """

    def __init__(
        self,
        runner: CodexRunner,
        *,
        store: ObjectStore,
        tool_store: ToolCallStore,
    ) -> None:
        self._runner = runner
        self._store = store
        self._tool_store = tool_store

    @property
    def mode(self) -> StepMode:
        return StepMode.TOOL_USING

    def invoke(
        self,
        request: OptimizationStepRequest,
        handles: tuple[Any, ...],
    ) -> AdapterOutput:
        if request.step_index != 0:
            raise OpaqueStepError(
                "the Codex run is exactly one opaque Step; step_index must "
                "be 0"
            )
        if len(request.tool_configs) != 1:
            raise OpaqueStepError(
                "the Codex run serializes exactly one Tool Config "
                "(evaluate_candidate)"
            )
        tool_config = request.tool_configs[0]

        # Launch the opaque agent; it uses the tool via the MCP bridge.
        run_result = self._runner.run(request, tool_config)

        # Reconstruct durable tool evidence (calls were out-of-process).
        records = self._reconstruct_records(run_result, tool_config)

        # Validate the returned proposals (base-bound, template-only).
        proposals = self._validate_proposals(request, run_result.proposals)
        status = (
            StepStatus.COMPLETE
            if proposals is not None
            else StepStatus.FAILED
        )
        accepted = proposals or ()

        state_delta = {
            "codex_step": {
                "step_index": request.step_index,
                "tool_config_hash": tool_config.identity_hash(),
                "max_evaluation_calls": (
                    tool_config.capacity.max_accepted_calls
                ),
                "accepted_call_count": self._tool_store.accepted_count(
                    tool_config.identity_hash()
                ),
                "observed_tool_call_count": len(records),
                "conversation_evidence": run_result.conversation_evidence,
            }
        }
        return AdapterOutput(
            proposed_candidates=accepted,
            accepted_candidates=accepted,
            tool_call_records=tuple(records),
            proposed_status=status,
            state_delta=state_delta,
        )

    # -- evidence reconstruction --------------------------------------------

    def _reconstruct_records(
        self, run_result: CodexRunResult, tool_config: ToolConfig
    ) -> list[ToolCallRecord]:
        records: list[ToolCallRecord] = []
        tool_config_hash = tool_config.identity_hash()
        for logged in run_result.tool_calls:
            entry = self._tool_store.get(tool_config_hash, logged.call_id)
            if entry is None:
                raise OpaqueStepError(
                    "a logged agent tool call has no durable Tool Call Store "
                    f"Entry: {logged.call_id!r}"
                )
            self._assert_evidence_consistent(entry, logged.result)
            call = ToolCall(
                call_id=logged.call_id,
                tool_config_hash=tool_config_hash,
                store_namespace=tool_config.store_namespace,
                args=logged.result.output.get("call_args", {})
                if logged.result.output
                else {},
            )
            records.append(ToolCallRecord(call=call, result=logged.result))
        return records

    @staticmethod
    def _assert_evidence_consistent(
        entry: ToolCallStoreEntry, result: ToolResult
    ) -> None:
        if result.refusal is not None:
            if entry.state is not ToolCallState.REFUSED:
                raise OpaqueStepError(
                    "a refused agent Tool Result must have a refused Store "
                    f"Entry: {result.call_id!r}"
                )
        elif entry.state is not ToolCallState.COMPLETED:
            raise OpaqueStepError(
                "an accepted agent Tool Result must have a completed Store "
                f"Entry: {result.call_id!r}"
            )
        elif entry.tool_result_ref != tool_result_reference(result):
            raise OpaqueStepError(
                "the completed Store Entry references a different Tool Result "
                f"than the agent returned: {result.call_id!r}"
            )

    # -- proposal validation -------------------------------------------------

    def _validate_proposals(
        self,
        request: OptimizationStepRequest,
        proposals: tuple[Candidate, ...],
    ) -> tuple[Candidate, ...] | None:
        target = request.output_contract.returned_proposal_count
        if len(proposals) != target:
            return None
        bases = {candidate.base_ref for candidate in request.candidates}
        base_by_ref = {c.base_ref: c for c in request.candidates}
        seen_bases: set[str] = set()
        validated: list[Candidate] = []
        for proposal in proposals:
            if proposal.base_ref not in bases:
                return None
            if proposal.base_ref in seen_bases:
                # No route duplicated: one proposal per allowed base.
                return None
            seen_bases.add(proposal.base_ref)
            try:
                diff_check(
                    base=base_by_ref[proposal.base_ref], proposed=proposal
                )
            except DiffCheckError:
                return None
            validated.append(proposal)
        # Each allowed base must be bound exactly once (none omitted).
        if seen_bases != bases:
            return None
        return tuple(validated)
