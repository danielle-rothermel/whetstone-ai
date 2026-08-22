from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

from whetstone.core.leasing import ReplayPolicy
from whetstone.core.identity import (
    ImmutableJsonObject,
    OpaqueKey,
    TerminalFailure,
    TypedRef,
)
from whetstone.experiment.candidate import Candidate
from whetstone.optim.cost import ProposerCallUsage
from whetstone.optim.contracts import (
    BudgetDelta,
    OptimEvalRequest,
    SearchEvidence,
    OptimStepRequest,
    StepKind,
    StepMode,
    StepStatus,
)
from whetstone.optim.tools.contracts import RuntimeToolHandle

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
    optim_eval_requests: tuple[OptimEvalRequest, ...] = ()
    #: Evidence for evaluations the adapter drove inside its own search.
    #: The harness verifies each entry's run and step against the Step
    #: Request, then carries it onto the Step Result unchanged.
    search_evidence: tuple[SearchEvidence, ...] = ()
    budget_delta: BudgetDelta = Field(default_factory=BudgetDelta)
    proposed_status: StepStatus = StepStatus.CONTINUE
    terminal_failure: TerminalFailure | None = None
    #: A COMPLETE Step that accepted no improvement over the seed candidate.
    #: Only a contract with search-dependent terminal cardinality may claim
    #: it, and only for the run's own initial candidate; the harness checks
    #: both against the Step Request.
    seed_retained: StrictBool = False
    #: The candidate a ``seed_retained`` Step kept. Present exactly when
    #: ``seed_retained`` is set, so the harness can verify the claim names
    #: the run's seed rather than any candidate the search happened to like.
    retained_candidate: Candidate | None = None
    #: Usage for each proposer-model call this Step made, in call order.
    #: Every optimizer reports through this one field so run-level spend has
    #: a single shape rather than a per-optimizer state layout.
    proposer_usage: tuple[ProposerCallUsage, ...] = ()
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
            if self.optim_eval_requests:
                raise ValueError(
                    "a failed Adapter Output requests no Evaluations"
                )
        if self.seed_retained:
            if self.proposed_status is not StepStatus.COMPLETE:
                raise ValueError(
                    "only a COMPLETE Adapter Output may retain the seed"
                )
            if self.accepted_candidates:
                raise ValueError(
                    "a seed-retaining Adapter Output accepts no candidates"
                )
            if self.retained_candidate is None:
                raise ValueError(
                    "a seed-retaining Adapter Output must name the retained "
                    "candidate"
                )
        elif self.retained_candidate is not None:
            raise ValueError(
                "only a seed-retaining Adapter Output names a retained "
                "candidate"
            )
        search_ids = [
            evidence.eval_request_id for evidence in self.search_evidence
        ]
        if len(set(search_ids)) != len(search_ids):
            raise ValueError(
                "Adapter Output search evidence Eval Request IDs must be "
                "unique"
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
        request: OptimStepRequest,
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

