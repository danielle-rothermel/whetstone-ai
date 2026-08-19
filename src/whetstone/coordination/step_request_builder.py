from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dr_store import ObjectStore

from whetstone.core.identity import ImmutableJsonObject, TypedRef
from whetstone.experiment.candidate import Candidate
from whetstone.optim.contracts import (
    BudgetState,
    OptimRunRef,
    OptimStepRequest,
    OptimStepResult,
    OutputContract,
    StepKind,
    step_result_reference,
)

if TYPE_CHECKING:
    from whetstone.optim.copro.control import CoproControl
    from whetstone.optim.gepa.control import GepaControl


def _copro_total_proposal_calls(control: CoproControl) -> int:
    if control.depth < 1:
        raise ValueError("COPRO depth must be at least 1")
    total = control.breadth - 1
    if control.depth > 1:
        total += control.breadth * (control.depth - 1)
    return total


def _copro_attempt_history(
    *,
    request: OptimStepRequest,
    prior_results: tuple[OptimStepResult, ...],
    control: CoproControl,
    mutation_field: str,
) -> tuple[Any, ...]:
    from whetstone.optim.copro.adapter import CoproAttempt

    del request
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


def _copro_completed_rounds(
    *,
    control: CoproControl,
    attempt_history: tuple[Any, ...],
) -> int:
    if len(attempt_history) % control.breadth:
        raise ValueError("COPRO attempt history ends with a partial round")
    return len(attempt_history) // control.breadth


class StepRequestBuilder:
    """Build durable ``OptimStepRequest`` values for harness step loops."""

    def __init__(self, *, store: ObjectStore) -> None:
        self._store = store

    def build_first(
        self,
        *,
        run: OptimRunRef,
        adapter_key: str,
        initial_candidate: Candidate,
        control: CoproControl | GepaControl | None = None,
        extra_pools: dict[str, Any] | None = None,
    ) -> OptimStepRequest:
        from whetstone.optim.copro.adapter import COPRO_ADAPTER_KEY
        from whetstone.optim.gepa.harness_adapter import GEPA_ADAPTER_KEY

        if adapter_key == GEPA_ADAPTER_KEY:
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
                step_output_contract=(
                    run.record.terminal_output_contract
                    if control.resolved_max_metric_calls == 0
                    else OutputContract(returned_proposal_count=0)
                ),
            )

        if adapter_key != COPRO_ADAPTER_KEY:
            raise ValueError(
                f"unsupported adapter key for initial step: {adapter_key!r}"
            )
        if control is None:
            raise ValueError("COPRO initial step requires the exact control")
        pools = {"attempt_history": []}
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
                        "proposal_calls": _copro_total_proposal_calls(control),
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
        prior: OptimStepResult,
        prior_ref: TypedRef,
        prior_results: tuple[OptimStepResult, ...],
        control: CoproControl | GepaControl,
        mutation_field: str,
    ) -> OptimStepRequest:
        from whetstone.optim.copro.adapter import COPRO_ADAPTER_KEY
        from whetstone.optim.gepa.harness_adapter import GEPA_ADAPTER_KEY
        from whetstone.optim.gepa.step_engine import GEPA_STATE_KEY

        if prior_ref != step_result_reference(prior).record_ref:
            raise ValueError("prior step result ref is not exact")
        adapter_key = prior.request.record.run.record.adapter_key
        if adapter_key == GEPA_ADAPTER_KEY:
            from whetstone.optim.gepa.control import GepaControl

            if not isinstance(control, GepaControl):
                raise TypeError("GEPA continuation requires GepaControl")
            step_index = prior.step_index + 1
            prior_state = (
                {}
                if prior.state_ref is None
                else self._store.get(prior.state_ref.reference)
            )
            checkpoint = prior_state.get(GEPA_STATE_KEY, {})
            if isinstance(checkpoint, dict):
                consumed = int(checkpoint.get("metric_calls_consumed", 0))
                checkpoint_terminal = bool(checkpoint.get("terminal", False))
            else:
                consumed = 0
                checkpoint_terminal = False
            terminal_next = (
                control.resolved_max_metric_calls == 0
                or checkpoint_terminal
                or consumed + 1 >= control.resolved_max_metric_calls
            )
            return OptimStepRequest(
                run=prior.request.record.run,
                step_id=f"{prior.run_id}:gepa:{step_index}",
                kind=StepKind.PROPOSAL,
                kind_label="gepa_iteration",
                step_index=step_index,
                prior_step_result_ref=prior_ref,
                prior_state_ref=prior.state_ref,
                prior_history_ref=prior.history_ref,
                candidates=(prior.request.record.candidates[0],),
                pools=ImmutableJsonObject({GEPA_STATE_KEY: checkpoint}),
                hyperparameters=ImmutableJsonObject(
                    control.step_hyperparameters(iteration=step_index)
                ),
                budget=prior.budget,
                step_output_contract=(
                    prior.request.record.run.record.terminal_output_contract
                    if terminal_next
                    else OutputContract(returned_proposal_count=0)
                ),
            )
        if adapter_key != COPRO_ADAPTER_KEY:
            raise ValueError(
                f"unsupported adapter key for continuation: {adapter_key!r}"
            )
        step_index = prior.step_index + 1
        attempt_history = _copro_attempt_history(
            request=prior.request.record,
            prior_results=prior_results,
            control=control,
            mutation_field=mutation_field,
        )
        completed_rounds = _copro_completed_rounds(
            control=control,
            attempt_history=attempt_history,
        )
        history_payload = [
            item.model_dump(mode="json") for item in attempt_history
        ]
        if completed_rounds >= control.depth:
            return OptimStepRequest(
                run=prior.request.record.run,
                step_id=f"{prior.run_id}:copro:{step_index}",
                kind=StepKind.PROPOSAL,
                kind_label="copro_finalize",
                step_index=step_index,
                prior_step_result_ref=prior_ref,
                prior_state_ref=prior.state_ref,
                prior_history_ref=prior.history_ref,
                candidates=(prior.request.record.candidates[0],),
                pools=ImmutableJsonObject(
                    {"attempt_history": history_payload}
                ),
                hyperparameters=ImmutableJsonObject(
                    control.step_hyperparameters(
                        iteration=min(control.depth - 1, step_index)
                    )
                ),
                budget=prior.budget,
                step_output_contract=(
                    prior.request.record.run.record.terminal_output_contract
                ),
            )
        return OptimStepRequest(
            run=prior.request.record.run,
            step_id=f"{prior.run_id}:copro:{step_index}",
            kind=StepKind.PROPOSAL,
            kind_label="history_proposal",
            step_index=step_index,
            prior_step_result_ref=prior_ref,
            prior_state_ref=prior.state_ref,
            prior_history_ref=prior.history_ref,
            candidates=(prior.request.record.candidates[0],),
            pools=ImmutableJsonObject({"attempt_history": history_payload}),
            hyperparameters=ImmutableJsonObject(
                control.step_hyperparameters(iteration=step_index)
            ),
            budget=prior.budget,
            step_output_contract=OutputContract(
                returned_proposal_count=control.breadth,
            ),
        )

    def validate_copro_history(self, request: OptimStepRequest) -> None:
        from whetstone.optim.copro.adapter import attempt_history_entries

        attempt_history_entries(request)


__all__ = [
    "StepRequestBuilder",
    "_copro_completed_rounds",
    "_copro_total_proposal_calls",
]
