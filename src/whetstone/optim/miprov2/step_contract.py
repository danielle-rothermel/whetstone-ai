"""MIPROv2's step contracts.

MIPROv2 drives a long multi-phase search -- bootstrap, instruction
proposal, baseline, then an Optuna study of sampled and promoted trials --
one harness Step at a time. Which phase the next Step runs is not a
function of the step index: it is whatever the durable
:class:`~whetstone.optim.miprov2.runtime.Miprov2State` says comes next.
So unlike COPRO's fixed round schedule, every MIPROv2 Step Request is
derived by asking the driver to preview the next plan against that state.

The state lives in the adapter's ``state_delta`` under
:data:`~whetstone.optim.miprov2.adapter.MIPROV2_STATE_KEY`, mirroring GEPA,
and carries its own RNG checkpoint and durable route bindings, so resuming
a run needs nothing but the prior Step Result.
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
    step_result_reference,
)
from whetstone.optim.miprov2.adapter import (
    MIPROV2_ADAPTER_KEY,
    MIPROV2_COMPLETE,
    MIPROV2_FAILED,
    MIPROV2_STATE_KEY,
    fold_prior_resolutions,
)
from whetstone.optim.miprov2.runtime import Miprov2Driver, Miprov2State

if TYPE_CHECKING:
    from whetstone.core.identity import TypedRef
    from whetstone.optim.miprov2.control import Miprov2Control

__all__ = [
    "MIPROV2_INITIAL_STATE_POOL_KEY",
    "Miprov2StepContractProvider",
    "miprov2_step_output_contract",
    "miprov2_state_from_prior",
]

#: Pool key carrying the run's opening MIPROv2 state. Only the first Step
#: reads it; every later Step reads the prior Step Result's state snapshot.
MIPROV2_INITIAL_STATE_POOL_KEY = MIPROV2_STATE_KEY


def miprov2_step_output_contract(
    run: OptimRunRef,
    *,
    kind_label: str,
) -> OutputContract:
    """The contract a MIPROv2 Step binds, given the phase it will run.

    Only the completing Step returns the run's terminal cardinality. Every
    other phase -- eval binding, bootstrap, proposal, baseline, sample,
    promotion -- is interior search that returns no proposal, and a failed
    Step returns none either.
    """

    if kind_label != MIPROV2_COMPLETE:
        return OutputContract(returned_proposal_count=0)
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


def miprov2_state_from_prior(
    store: ObjectStore,
    prior: OptimStepResult,
) -> Miprov2State:
    """Load the MIPROv2 state the prior Step persisted.

    The state snapshot is the run's only durable authority, so a resumed
    run reads it here rather than reconstructing anything from the request.
    """

    state_ref = prior.state_ref
    if state_ref is None:
        raise ValueError("prior MIPROv2 Step Result has no state snapshot")
    snapshot = store.get(state_ref.reference)
    if not isinstance(snapshot, dict):
        raise ValueError("MIPROv2 state snapshot must be a JSON object")
    try:
        payload = snapshot[MIPROV2_STATE_KEY]
    except KeyError:
        raise ValueError(
            "prior MIPROv2 state snapshot has no "
            f"{MIPROV2_STATE_KEY!r} entry"
        ) from None
    return Miprov2State.model_validate(payload)


def _step_candidates(state: Miprov2State) -> tuple[Candidate, ...]:
    """The candidates a Step carries: the base, plus a distinct teacher.

    Every intent the Step may emit must cite a candidate the request already
    carries, and a bootstrap intent evaluates the teacher.
    """

    base = state.control.base_candidate
    teacher = state.control.teacher_candidate
    if teacher == base:
        return (base.record,)
    return (base.record, teacher.record)


def _phase_label(driver: Miprov2Driver, state: Miprov2State) -> str:
    """Preview which phase the next Step runs, without causing effects.

    ``plan`` is a pure projection of the durable state, so previewing it to
    label the Step and choose its output contract is free of side effects.
    """

    if state.phase == MIPROV2_FAILED:
        return MIPROV2_FAILED
    return driver.plan(state).kind


class Miprov2StepContractProvider:
    """Derive each MIPROv2 Step Request from the durable search state."""

    def __init__(self, *, driver: Miprov2Driver | None = None) -> None:
        self._driver = driver or Miprov2Driver()

    @property
    def adapter_key(self) -> str:
        return MIPROV2_ADAPTER_KEY

    def requires_control(self) -> bool:
        return True

    def parse_control(self, payload: dict[str, Any]) -> Miprov2Control:
        from whetstone.optim.miprov2.control import Miprov2Control

        return Miprov2Control.model_validate(payload)

    def build_first(
        self,
        *,
        store: ObjectStore,
        run: OptimRunRef,
        initial_candidate: Candidate,
        control: Miprov2Control | None,
        extra_pools: dict[str, Any] | None,
    ) -> OptimStepRequest:
        del store
        if control is None:
            raise ValueError("MIPROv2 initial step requires the exact control")
        from whetstone.optim.miprov2.control import Miprov2Control

        if not isinstance(control, Miprov2Control):
            raise TypeError("MIPROv2 initial step requires Miprov2Control")
        pools = dict(extra_pools or {})
        try:
            state_payload = pools.pop(MIPROV2_INITIAL_STATE_POOL_KEY)
        except KeyError:
            raise ValueError(
                "MIPROv2 initial step requires the opening state at pool key "
                f"{MIPROV2_INITIAL_STATE_POOL_KEY!r}; build it with "
                "prepare_miprov2_run"
            ) from None
        state = Miprov2State.model_validate(state_payload)
        if state.control.reference() != control.reference():
            raise ValueError(
                "MIPROv2 opening state does not bind the exact launch control"
            )
        if state.run != run:
            raise ValueError(
                "MIPROv2 opening state does not bind the exact run"
            )
        if state.control.base_candidate.record != initial_candidate:
            raise ValueError(
                "MIPROv2 launch candidate is not the control base candidate"
            )
        return self._request(
            run=run,
            state=state,
            step_index=0,
            budget=_opening_budget(state),
            pools={MIPROV2_STATE_KEY: state.model_dump(mode="json"), **pools},
            prior_step_result_ref=None,
            prior_state_ref=None,
            prior_history_ref=None,
        )

    def build_next(
        self,
        *,
        store: ObjectStore,
        prior: OptimStepResult,
        prior_ref: TypedRef,
        prior_results: tuple[OptimStepResult, ...],
        control: Miprov2Control,
        mutation_field: str,
        extra_pools: dict[str, Any] | None,
    ) -> OptimStepRequest:
        del prior_results, mutation_field, extra_pools
        from whetstone.optim.miprov2.control import Miprov2Control

        if not isinstance(control, Miprov2Control):
            raise TypeError("MIPROv2 continuation requires Miprov2Control")
        if prior_ref != step_result_reference(prior).record_ref:
            raise ValueError("prior MIPROv2 Step Result ref is not exact")
        state = miprov2_state_from_prior(store, prior)
        if state.control.reference() != control.reference():
            raise ValueError(
                "persisted MIPROv2 state does not bind the run's control"
            )
        # The prior Step's Intent Resolutions have not been folded into the
        # persisted snapshot yet -- the adapter folds them when it runs the
        # Step. The contract must label this Step by the phase that will
        # actually run, so it folds the same resolutions through the same
        # evidence resolver first.
        return self._request(
            run=prior.request.record.run,
            state=fold_prior_resolutions(store, state, prior),
            step_index=prior.step_index + 1,
            budget=prior.budget,
            pools={},
            prior_step_result_ref=prior_ref,
            prior_state_ref=prior.state_ref,
            prior_history_ref=prior.history_ref,
        )

    def _request(
        self,
        *,
        run: OptimRunRef,
        state: Miprov2State,
        step_index: int,
        budget: BudgetState,
        pools: dict[str, Any],
        prior_step_result_ref: TypedRef | None,
        prior_state_ref: TypedRef | None,
        prior_history_ref: TypedRef | None,
    ) -> OptimStepRequest:
        kind_label = _phase_label(self._driver, state)
        return OptimStepRequest(
            run=run,
            step_id=f"{state.run_id}:miprov2:{step_index}",
            kind=StepKind.PROPOSAL,
            kind_label=kind_label,
            step_index=step_index,
            prior_step_result_ref=prior_step_result_ref,
            prior_state_ref=prior_state_ref,
            prior_history_ref=prior_history_ref,
            candidates=_step_candidates(state),
            pools=ImmutableJsonObject(pools),
            hyperparameters=ImmutableJsonObject(
                {
                    "demo_mode": state.control.demo_mode.value,
                    "seed": state.control.seed,
                }
            ),
            budget=budget,
            step_output_contract=miprov2_step_output_contract(
                run,
                kind_label=kind_label,
            ),
        )


def _opening_budget(state: Miprov2State) -> BudgetState:
    """The run's opening budget, taken from the control's effect ceilings."""

    budget = state.budget
    dimensions = {
        "bootstrap_generations": budget.bootstrap_generations,
        "proposal_calls": budget.proposal_calls,
        "evaluations": budget.evaluations,
        "task_rows": budget.task_rows,
    }
    return BudgetState(
        remaining=ImmutableJsonObject(dimensions),
        consumed=ImmutableJsonObject(dict.fromkeys(dimensions, 0)),
    )
