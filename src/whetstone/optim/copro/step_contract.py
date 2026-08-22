"""COPRO's step contracts: seed round, history rounds, then finalize."""

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
)
from whetstone.optim.copro.adapter import COPRO_ADAPTER_KEY

if TYPE_CHECKING:
    from whetstone.core.identity import TypedRef
    from whetstone.optim.copro.control import CoproControl

__all__ = [
    "CoproStepContractProvider",
    "copro_attempt_history",
    "copro_completed_rounds",
    "copro_total_proposal_calls",
]


def copro_total_proposal_calls(control: CoproControl) -> int:
    if control.depth < 1:
        raise ValueError("COPRO depth must be at least 1")
    total = control.breadth - 1
    if control.depth > 1:
        total += control.breadth * (control.depth - 1)
    return total


def copro_attempt_history(
    *,
    prior_results: tuple[OptimStepResult, ...],
    control: CoproControl,
    mutation_field: str,
) -> tuple[Any, ...]:
    from whetstone.optim.copro.adapter import CoproAttempt

    attempts: list[CoproAttempt] = []
    for prior in prior_results:
        round_index = prior.step_index
        round_start = round_index * control.breadth
        for offset, resolution in enumerate(prior.resolved_intents):
            attempts.append(
                CoproAttempt.from_resolution(
                    occurrence_ordinal=round_start + offset,
                    round_index=round_index,
                    resolution=resolution,
                    expected_run_id=prior.run_id,
                    expected_eval_config_ref=control.eval_config_ref,
                    expected_eval_role=control.eval_role,
                    expected_provider_execution_policy_ref=(
                        control.provider_execution_policy_ref
                    ),
                    expected_reward_policy_hash=(
                        control.expected_reward_policy_hash
                    ),
                    mutation_field=mutation_field,
                )
            )
    return tuple(attempts)


def copro_completed_rounds(
    *,
    control: CoproControl,
    attempt_history: tuple[Any, ...],
) -> int:
    if len(attempt_history) % control.breadth:
        raise ValueError("COPRO attempt history ends with a partial round")
    return len(attempt_history) // control.breadth


class CoproStepContractProvider:
    """COPRO returns ``breadth`` proposals per round and top-k at the end."""

    @property
    def adapter_key(self) -> str:
        return COPRO_ADAPTER_KEY

    def requires_control(self) -> bool:
        return True

    def parse_control(self, payload: dict[str, Any]) -> CoproControl:
        from whetstone.optim.copro.control import CoproControl

        return CoproControl.model_validate(payload)

    def build_first(
        self,
        *,
        store: ObjectStore,
        run: OptimRunRef,
        initial_candidate: Candidate,
        control: CoproControl | None,
        extra_pools: dict[str, Any] | None,
    ) -> OptimStepRequest:
        del store
        if control is None:
            raise ValueError("COPRO initial step requires the exact control")
        pools: dict[str, Any] = {"attempt_history": []}
        if extra_pools:
            pools.update(extra_pools)
        return OptimStepRequest(
            run=run,
            step_id=f"{run.record.run_id}:copro:0",
            kind=StepKind.PROPOSAL,
            kind_label="seed_proposal",
            step_index=0,
            candidates=(initial_candidate,),
            pools=ImmutableJsonObject(pools),
            hyperparameters=ImmutableJsonObject(
                control.step_hyperparameters(iteration=0)
            ),
            budget=BudgetState(
                remaining=ImmutableJsonObject(
                    {
                        "proposal_calls": copro_total_proposal_calls(control),
                    }
                ),
            ),
            step_output_contract=OutputContract(
                returned_proposal_count=control.breadth - 1,
            ),
        )

    def build_next(
        self,
        *,
        store: ObjectStore,
        prior: OptimStepResult,
        prior_ref: TypedRef,
        prior_results: tuple[OptimStepResult, ...],
        control: CoproControl,
        mutation_field: str,
        extra_pools: dict[str, Any] | None,
    ) -> OptimStepRequest:
        del store
        step_index = prior.step_index + 1
        attempt_history = copro_attempt_history(
            prior_results=prior_results,
            control=control,
            mutation_field=mutation_field,
        )
        completed_rounds = copro_completed_rounds(
            control=control,
            attempt_history=attempt_history,
        )
        history_payload = [
            item.model_dump(mode="json") for item in attempt_history
        ]
        finalizing = completed_rounds >= control.depth
        pools: dict[str, Any] = {"attempt_history": history_payload}
        if extra_pools:
            pools.update(extra_pools)
        return OptimStepRequest(
            run=prior.request.record.run,
            step_id=f"{prior.run_id}:copro:{step_index}",
            kind=StepKind.PROPOSAL,
            kind_label=(
                "copro_finalize" if finalizing else "history_proposal"
            ),
            step_index=step_index,
            prior_step_result_ref=prior_ref,
            prior_state_ref=prior.state_ref,
            prior_history_ref=prior.history_ref,
            candidates=(prior.request.record.candidates[0],),
            pools=ImmutableJsonObject(pools),
            hyperparameters=ImmutableJsonObject(
                control.step_hyperparameters(
                    iteration=(
                        min(control.depth - 1, step_index)
                        if finalizing
                        else step_index
                    )
                )
            ),
            budget=prior.budget,
            step_output_contract=(
                prior.request.record.run.record.terminal_output_contract
                if finalizing
                else OutputContract(
                    returned_proposal_count=control.breadth,
                )
            ),
        )
