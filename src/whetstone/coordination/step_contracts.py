"""Per-optimizer step contracts, resolved by adapter key.

Each optimizer owns one ``StepContractProvider``: it declares the first
Step Request for a run, derives the next Step Request from the prior
results, and parses its own persisted control payload. ``StepRequestBuilder``
and ``HarnessRunController`` dispatch on adapter key through this registry
instead of forking per optimizer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from dr_store import ObjectStore

from whetstone.experiment.candidate import Candidate
from whetstone.optim.contracts import (
    OptimRunRef,
    OptimStepRequest,
    OptimStepResult,
)

if TYPE_CHECKING:
    from whetstone.core.identity import TypedRef

__all__ = [
    "OptimizerControl",
    "StepContractProvider",
    "resolve_step_contract_provider",
    "step_contract_provider_keys",
]


class OptimizerControl(Protocol):
    """The persisted, identity-bearing configuration of one optimizer run."""

    def identity_hash(self) -> str: ...

    def model_dump(self, *, mode: str = ...) -> dict[str, Any]: ...


class StepContractProvider(Protocol):
    """Everything the harness loop needs to drive one optimizer."""

    @property
    def adapter_key(self) -> str: ...

    def parse_control(self, payload: dict[str, Any]) -> OptimizerControl:
        """Deserialize this optimizer's persisted launch control payload."""
        ...

    def requires_control(self) -> bool:
        """Whether a run of this optimizer cannot start without a control."""
        ...

    def build_first(
        self,
        *,
        store: ObjectStore,
        run: OptimRunRef,
        initial_candidate: Candidate,
        control: OptimizerControl | None,
        extra_pools: dict[str, Any] | None,
    ) -> OptimStepRequest: ...

    def build_next(
        self,
        *,
        store: ObjectStore,
        prior: OptimStepResult,
        prior_ref: TypedRef,
        prior_results: tuple[OptimStepResult, ...],
        control: OptimizerControl,
        mutation_field: str,
        extra_pools: dict[str, Any] | None,
    ) -> OptimStepRequest: ...


def _providers() -> dict[str, StepContractProvider]:
    from whetstone.optim.codex.step_contract import CodexStepContractProvider
    from whetstone.optim.copro.step_contract import CoproStepContractProvider
    from whetstone.optim.gepa.step_contract import GepaStepContractProvider
    from whetstone.optim.miprov2.step_contract import (
        Miprov2StepContractProvider,
    )

    return {
        provider.adapter_key: provider
        for provider in (
            CodexStepContractProvider(),
            CoproStepContractProvider(),
            GepaStepContractProvider(),
            Miprov2StepContractProvider(),
        )
    }


def resolve_step_contract_provider(adapter_key: str) -> StepContractProvider:
    """Return the step-contract provider registered for ``adapter_key``."""
    try:
        return _providers()[adapter_key]
    except KeyError:
        raise ValueError(
            f"no optimizer step contract registered for {adapter_key!r}"
        ) from None


def step_contract_provider_keys() -> tuple[str, ...]:
    """Ordered adapter keys with a registered step contract."""
    return tuple(sorted(_providers()))
