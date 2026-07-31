"""Generic optimizer adapter contracts and the pure identity adapter."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from whetstone.optimization.identity import (
    ImmutableJsonObject,
    OpaqueKey,
    TerminalFailure,
    TypedRef,
)
from whetstone.optimization.schema import (
    BudgetDelta,
    Candidate,
    EvaluationIntent,
    OptimizationStepRequest,
    StepKind,
    StepMode,
    StepStatus,
)
from whetstone.optimization.tools import RuntimeToolHandle

__all__ = [
    "AdapterCheckpoint",
    "AdapterOutput",
    "AdapterRegistry",
    "IdentityOptimizerAdapter",
    "MappingAdapterRegistry",
    "OptimizerAdapter",
]


class AdapterOutput(BaseModel):
    """Serializable output checkpointed before downstream effects."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    proposed_candidates: tuple[Candidate, ...] = ()
    accepted_candidates: tuple[Candidate, ...] = ()
    evaluation_intents: tuple[EvaluationIntent, ...] = ()
    budget_delta: BudgetDelta = Field(default_factory=BudgetDelta)
    proposed_status: StepStatus = StepStatus.CONTINUE
    terminal_failure: TerminalFailure | None = None
    state_delta: ImmutableJsonObject = Field(
        default_factory=lambda: ImmutableJsonObject({})
    )
    history_delta: ImmutableJsonObject = Field(
        default_factory=lambda: ImmutableJsonObject({})
    )

    @model_validator(mode="after")
    def _validate(self) -> AdapterOutput:
        if (self.proposed_status is StepStatus.FAILED) != (
            self.terminal_failure is not None
        ):
            raise ValueError(
                "a failed Adapter Output requires exactly one shared "
                "terminal failure"
            )
        if self.proposed_status is StepStatus.FAILED:
            if self.accepted_candidates:
                raise ValueError(
                    "a failed Adapter Output claims no accepted candidates"
                )
            if self.evaluation_intents:
                raise ValueError(
                    "a failed Adapter Output requests no Evaluations"
                )
        return self

    def record_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class AdapterCheckpoint(BaseModel):
    """Bound checkpoint proving which request and adapter produced output."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_ref: TypedRef
    adapter_key: OpaqueKey
    output: AdapterOutput

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
