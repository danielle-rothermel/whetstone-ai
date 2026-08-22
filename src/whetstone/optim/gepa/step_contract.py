"""GEPA's step contracts.

Upstream ``optimize`` decides for itself when a round exhausts the metric
budget, so any GEPA step may terminalize. Every GEPA step therefore binds
a contract that returns nothing while continuing and the run terminal
cardinality when it completes, and a terminal step that keeps the seed
reports ``seed_retained`` instead of an accepted candidate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dr_store import ObjectStore

from whetstone.core.identity import ImmutableJsonObject
from whetstone.experiment.candidate import Candidate
from whetstone.optim.contracts import (
    BudgetState,
    OptimRunRef,
    OptimStepRequest,
    OptimStepResult,
    OutputContract,
    StepKind,
    StepStatus,
)
from whetstone.optim.gepa.harness_adapter import (
    GEPA_ADAPTER_KEY,
    GEPA_SKIPPED_MUTATIONS_KEY,
)
from whetstone.optim.gepa.step_engine import GEPA_STATE_KEY

if TYPE_CHECKING:
    from whetstone.core.identity import TypedRef
    from whetstone.optim.gepa.control import GepaControl

__all__ = ["GepaStepContractProvider", "gepa_step_output_contract"]


def gepa_step_output_contract(run: OptimRunRef) -> OutputContract:
    """The contract every GEPA step binds.

    A continuing step returns nothing; a completing step returns the run
    terminal cardinality, or nothing when it retains the seed.
    """
    terminal = run.record.terminal_output_contract
    return OutputContract(
        returned_proposal_count=0,
        # The run's COMPLETE cardinality, not its continuing count: a run
        # whose terminal contract sets the two differently would otherwise
        # bind a step contract that fails ``honors_terminal``, rejecting
        # every honest completing step.
        terminal_proposal_count=terminal.accepted_count_for(
            StepStatus.COMPLETE
        ),
        require_distinct_bases=terminal.require_distinct_bases,
    )


def _gepa_prior_state(
    store: ObjectStore, prior: OptimStepResult
) -> dict[str, Any]:
    prior_state = (
        {}
        if prior.state_ref is None
        else store.get(prior.state_ref.reference)
    )
    if not isinstance(prior_state, dict):
        return {}
    checkpoint = prior_state.get(GEPA_STATE_KEY, {})
    skipped = prior_state.get(GEPA_SKIPPED_MUTATIONS_KEY, [])
    pools: dict[str, Any] = {
        GEPA_STATE_KEY: checkpoint if isinstance(checkpoint, dict) else {},
    }
    if isinstance(skipped, list):
        pools[GEPA_SKIPPED_MUTATIONS_KEY] = skipped
    return pools


class GepaStepContractProvider:
    @property
    def adapter_key(self) -> str:
        return GEPA_ADAPTER_KEY

    def requires_control(self) -> bool:
        return True

    def parse_control(self, payload: dict[str, Any]) -> GepaControl:
        from whetstone.optim.gepa.control import GepaControl

        return GepaControl.model_validate(payload)

    def build_first(
        self,
        *,
        store: ObjectStore,
        run: OptimRunRef,
        initial_candidate: Candidate,
        control: GepaControl | None,
        extra_pools: dict[str, Any] | None,
    ) -> OptimStepRequest:
        del store
        if control is None:
            raise ValueError("GEPA initial step requires the exact control")
        from whetstone.optim.gepa.control import GepaControl

        if not isinstance(control, GepaControl):
            raise TypeError("GEPA initial step requires GepaControl")
        pools: dict[str, Any] = {}
        if extra_pools:
            pools.update(extra_pools)
        return OptimStepRequest(
            run=run,
            step_id=f"{run.record.run_id}:gepa:0",
            kind=StepKind.PROPOSAL,
            kind_label="gepa_iteration",
            step_index=0,
            candidates=(initial_candidate,),
            pools=ImmutableJsonObject(pools),
            hyperparameters=ImmutableJsonObject(
                control.step_hyperparameters(iteration=0)
            ),
            budget=BudgetState(
                remaining=ImmutableJsonObject(
                    {"metric_calls": control.resolved_max_metric_calls}
                ),
            ),
            step_output_contract=gepa_step_output_contract(run),
        )

    def build_next(
        self,
        *,
        store: ObjectStore,
        prior: OptimStepResult,
        prior_ref: TypedRef,
        prior_results: tuple[OptimStepResult, ...],
        control: GepaControl,
        mutation_field: str,
        extra_pools: dict[str, Any] | None,
    ) -> OptimStepRequest:
        del prior_results, mutation_field
        from whetstone.optim.gepa.control import GepaControl

        if not isinstance(control, GepaControl):
            raise TypeError("GEPA continuation requires GepaControl")
        step_index = prior.step_index + 1
        run = prior.request.record.run
        pools = _gepa_prior_state(store, prior)
        if extra_pools:
            pools.update(extra_pools)
        return OptimStepRequest(
            run=run,
            step_id=f"{prior.run_id}:gepa:{step_index}",
            kind=StepKind.PROPOSAL,
            kind_label="gepa_iteration",
            step_index=step_index,
            prior_step_result_ref=prior_ref,
            prior_state_ref=prior.state_ref,
            prior_history_ref=prior.history_ref,
            candidates=(prior.request.record.candidates[0],),
            pools=ImmutableJsonObject(pools),
            hyperparameters=ImmutableJsonObject(
                control.step_hyperparameters(iteration=step_index)
            ),
            budget=prior.budget,
            step_output_contract=gepa_step_output_contract(run),
        )
