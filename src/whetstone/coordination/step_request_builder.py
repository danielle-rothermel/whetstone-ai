from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dr_store import ObjectStore

from whetstone.coordination.step_contracts import (
    resolve_step_contract_provider,
)
from whetstone.experiment.candidate import Candidate
from whetstone.optim.contracts import (
    OptimRunRef,
    OptimStepRequest,
    OptimStepResult,
    step_result_reference,
)

if TYPE_CHECKING:
    from whetstone.core.identity import TypedRef
    from whetstone.optim.codex.control import CodexControl
    from whetstone.optim.copro.control import CoproControl
    from whetstone.optim.gepa.control import GepaControl
    from whetstone.optim.miprov2.control import Miprov2Control


class StepRequestBuilder:
    """Build durable ``OptimStepRequest`` values for harness step loops.

    Per-optimizer contracts live in that optimizer's step-contract provider;
    this builder resolves one by adapter key and delegates.
    """

    def __init__(self, *, store: ObjectStore) -> None:
        self._store = store

    def build_first(
        self,
        *,
        run: OptimRunRef,
        adapter_key: str,
        initial_candidate: Candidate,
        control: (
            CodexControl | CoproControl | GepaControl | Miprov2Control | None
        ) = None,
        extra_pools: dict[str, Any] | None = None,
    ) -> OptimStepRequest:
        provider = resolve_step_contract_provider(adapter_key)
        return provider.build_first(
            store=self._store,
            run=run,
            initial_candidate=initial_candidate,
            control=control,
            extra_pools=extra_pools,
        )

    def build_next(
        self,
        *,
        prior: OptimStepResult,
        prior_ref: TypedRef,
        prior_results: tuple[OptimStepResult, ...],
        control: (
            CodexControl | CoproControl | GepaControl | Miprov2Control
        ),
        mutation_field: str,
        extra_pools: dict[str, Any] | None = None,
    ) -> OptimStepRequest:
        if prior_ref != step_result_reference(prior).record_ref:
            raise ValueError("prior step result ref is not exact")
        adapter_key = prior.request.record.run.record.adapter_key
        provider = resolve_step_contract_provider(adapter_key)
        return provider.build_next(
            store=self._store,
            prior=prior,
            prior_ref=prior_ref,
            prior_results=prior_results,
            control=control,
            mutation_field=mutation_field,
            extra_pools=extra_pools,
        )

    def validate_copro_history(self, request: OptimStepRequest) -> None:
        from whetstone.optim.copro.adapter import attempt_history_entries

        attempt_history_entries(request)


__all__ = ["StepRequestBuilder"]
