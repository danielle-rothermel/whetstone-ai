from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from whetstone.core.effects.authority import ReplayPolicy
from whetstone.core.identity import (
    ImmutableJsonObject,
    OpaqueKey,
    TerminalFailure,
    TypedRef,
)
from whetstone.experiment.candidate import Candidate
from whetstone.optimization.contracts import (
    BudgetDelta,
    EvaluationIntent,
    OptimizationStepRequest,
    StepKind,
    StepMode,
    StepStatus,
)
from whetstone.optimization.tools.contracts import RuntimeToolHandle

__all__ = [
    "AdapterCheckpoint",
    "AdapterOutput",
    "AdapterRegistry",
    "AdapterReplayPolicyMismatchError",
    "MappingAdapterRegistry",
    "OptimizerAdapter",
]


class AdapterReplayPolicyMismatchError(ValueError):
    def __init__(
        self,
        *,
        adapter_key: str,
        configured_policy: ReplayPolicy,
        required_policy: ReplayPolicy,
    ) -> None:
        self.adapter_key = adapter_key
        self.configured_policy = configured_policy
        self.required_policy = required_policy
        super().__init__(
            f"adapter {adapter_key!r} requires replay policy "
            f"{required_policy.value!r}; configured policy is "
            f"{configured_policy.value!r}"
        )


class AdapterOutput(BaseModel):
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

    @property
    def required_replay_policy(self) -> ReplayPolicy: ...

    def invoke(
        self,
        request: OptimizationStepRequest,
        handles: tuple[RuntimeToolHandle, ...],
    ) -> AdapterOutput: ...


class AdapterRegistry(Protocol):
    def resolve(self, adapter_key: str) -> OptimizerAdapter: ...


@dataclass(frozen=True, slots=True)
class MappingAdapterRegistry:
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

