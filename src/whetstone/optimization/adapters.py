"""Generic optimizer adapter contracts and the pure identity adapter."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator

from whetstone.optimization.identity import TypedRef, reject_non_json
from whetstone.optimization.schema import (
    BudgetDelta,
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
    "AdapterCheckpoint",
    "AdapterOutput",
    "AdapterRegistry",
    "IdentityOptimizerAdapter",
    "MappingAdapterRegistry",
    "OptimizerAdapter",
    "ToolCallRecord",
]


class ToolCallRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    call: ToolCall
    result: ToolResult

    @model_validator(mode="after")
    def _validate(self) -> ToolCallRecord:
        if self.call.call_id != self.result.call_id:
            raise ValueError("Tool Call and Result call_id must match")
        if self.call.tool_config_hash != self.result.tool_config_hash:
            raise ValueError("Tool Call and Result config identity must match")
        return self


class AdapterOutput(BaseModel):
    """Serializable output checkpointed before downstream effects."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    proposed_candidates: tuple[Candidate, ...] = ()
    accepted_candidates: tuple[Candidate, ...] = ()
    evaluation_intents: tuple[EvaluationIntent, ...] = ()
    tool_call_records: tuple[ToolCallRecord, ...] = ()
    budget_delta: BudgetDelta = Field(default_factory=BudgetDelta)
    proposed_status: StepStatus = StepStatus.CONTINUE
    state_delta: dict[str, Any] = Field(default_factory=dict)
    history_delta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate(self) -> AdapterOutput:
        reject_non_json(self.state_delta, field="state_delta")
        reject_non_json(self.history_delta, field="history_delta")
        return self

    def record_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class AdapterCheckpoint(BaseModel):
    """Bound checkpoint proving which request and adapter produced output."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_ref: TypedRef
    adapter_key: StrictStr
    output: AdapterOutput

    @model_validator(mode="after")
    def _validate(self) -> AdapterCheckpoint:
        if not self.adapter_key:
            raise ValueError("adapter_key must be non-empty")
        return self

    def record_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


@runtime_checkable
class OptimizerAdapter(Protocol):
    @property
    def key(self) -> str: ...

    @property
    def mode(self) -> StepMode: ...

    def invoke(
        self,
        request: OptimizationStepRequest,
        handles: tuple[RuntimeToolHandle, ...],
    ) -> AdapterOutput: ...


class AdapterRegistry(Protocol):
    """Injected lookup seam used by the core harness."""

    def resolve(self, adapter_key: str) -> OptimizerAdapter: ...


@dataclass(frozen=True, slots=True)
class MappingAdapterRegistry:
    """Small immutable registry useful to applications and tests."""

    adapters: Mapping[str, OptimizerAdapter]

    def __post_init__(self) -> None:
        copied = MappingProxyType(dict(self.adapters))
        if any(not key for key in copied):
            raise ValueError("adapter keys must be non-empty")
        for key, adapter in copied.items():
            if adapter.key != key:
                raise ValueError(
                    f"registry key {key!r} does not match adapter key "
                    f"{adapter.key!r}"
                )
        object.__setattr__(self, "adapters", copied)

    def resolve(self, adapter_key: str) -> OptimizerAdapter:
        try:
            return self.adapters[adapter_key]
        except KeyError as exc:
            raise KeyError(
                f"no optimizer adapter registered for {adapter_key!r}"
            ) from exc


class IdentityOptimizerAdapter:
    """One pure step, unchanged candidates, and no measurement."""

    @property
    def key(self) -> str:
        return "identity"

    @property
    def mode(self) -> StepMode:
        return StepMode.PURE

    def invoke(
        self,
        request: OptimizationStepRequest,
        handles: tuple[RuntimeToolHandle, ...],
    ) -> AdapterOutput:
        if request.kind is not StepKind.IDENTITY:
            raise ValueError("identity runs only the identity step kind")
        if handles:
            raise ValueError("identity receives no Runtime Tool Handles")
        return AdapterOutput(
            proposed_candidates=request.candidates,
            accepted_candidates=request.candidates,
            budget_delta=BudgetDelta(),
            proposed_status=StepStatus.COMPLETE,
        )
