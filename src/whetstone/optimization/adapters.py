"""Algorithm adapters and the algorithm-neutral adapter surface.

The adapter surface is deliberately narrow: an adapter owns only
algorithm-specific proposal or tool-use logic and returns a typed
:class:`AdapterOutput`. The harness owns validation, dispatch, external
evaluation, finalization, restart, idempotency, budgets, evidence, and terminal
assembly.

Three adapter shapes share one output type:

* **pure** (identity) — returns unchanged candidates, no intents, no tools.
* **proposal-only** (COPRO, MIPROv2) — returns candidates + zero-or-more
  immutable Evaluation Intents; performs no evaluation.
* **tool-using** (GEPA, Codex) — may issue Tool Calls through the Runtime Tool
  Handle during execution; returns final candidates + tool-call evidence.

This module ships the :class:`IdentityOptimizerAdapter` (the pure "identity
optimizer"): exactly one pure Step returning its candidates unchanged, with no
tools and no intents.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from whetstone.optimization.schema import (
    Candidate,
    EvaluationIntent,
    OptimizationStepRequest,
    StepKind,
    StepMode,
    StepStatus,
)
from whetstone.optimization.tools import (
    RuntimeToolHandle,
    ToolCall,
    ToolResult,
)

__all__ = [
    "AdapterOutput",
    "IdentityOptimizerAdapter",
    "OptimizerAdapter",
    "ToolCallRecord",
]


class ToolCallRecord(BaseModel):
    """A tool call an adapter issued during execution, paired with its result.

    The adapter issues the Tool Call through the Runtime Tool Handle; the
    harness records the Tool Result and Store Entry as durable evidence.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    call: ToolCall
    result: ToolResult

    @model_validator(mode="after")
    def _validate(self) -> ToolCallRecord:
        if self.call.call_id != self.result.call_id:
            raise ValueError(
                "ToolCallRecord call/result call_id must match"
            )
        return self


class AdapterOutput(BaseModel):
    """The typed output an adapter returns from one Step invocation.

    * ``proposed_candidates`` / ``accepted_candidates`` — the candidate batch.
    * ``evaluation_intents`` — proposal-only measurement requests (never
      carrying a score); empty for pure and tool-using steps.
    * ``tool_call_records`` — the Tool Calls a tool-using adapter issued and
      their results; empty for pure and proposal-only steps.
    * ``proposed_status`` — the adapter's proposed terminal status; the harness
      may override to ``failed`` on a contract failure.
    * ``state_delta`` / ``history_delta`` — opaque JSON deltas the harness
      durably snapshots (never an authoritative mutable pickle).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    proposed_candidates: tuple[Candidate, ...] = ()
    accepted_candidates: tuple[Candidate, ...] = ()
    evaluation_intents: tuple[EvaluationIntent, ...] = ()
    tool_call_records: tuple[ToolCallRecord, ...] = ()
    proposed_status: StepStatus = StepStatus.CONTINUE
    state_delta: dict[str, Any] = Field(default_factory=dict)
    history_delta: dict[str, Any] = Field(default_factory=dict)

    def record_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


@runtime_checkable
class OptimizerAdapter(Protocol):
    """The narrow adapter surface every algorithm implements.

    ``mode`` declares the execution mode the harness dispatches on.
    ``invoke`` receives the immutable Step Request and, for tool-using modes,
    the Runtime Tool Handles constructed at the execution boundary; it returns
    a typed :class:`AdapterOutput`. It performs no evaluation, never persists.
    """

    @property
    def mode(self) -> StepMode: ...

    def invoke(
        self,
        request: OptimizationStepRequest,
        handles: tuple[RuntimeToolHandle, ...],
    ) -> AdapterOutput: ...


class IdentityOptimizerAdapter:
    """The pure identity optimizer: exactly one pure Step, unchanged output.

    Returns its input candidates unchanged, proposes ``complete``, and emits no
    tools and no Evaluation Intents. Official evaluation remains downstream of
    the terminal Optimization Result.
    """

    @property
    def mode(self) -> StepMode:
        return StepMode.PURE

    def invoke(
        self,
        request: OptimizationStepRequest,
        handles: tuple[RuntimeToolHandle, ...],
    ) -> AdapterOutput:
        if request.kind is not StepKind.IDENTITY:
            raise ValueError(
                "the identity optimizer runs only the identity kind"
            )
        if handles:
            raise ValueError(
                "the identity optimizer needs no Runtime Tool Handle"
            )
        # Unchanged candidates: proposed == accepted == input, no intents.
        return AdapterOutput(
            proposed_candidates=request.candidates,
            accepted_candidates=request.candidates,
            proposed_status=StepStatus.COMPLETE,
        )
