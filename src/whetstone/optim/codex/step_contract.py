"""Codex's step contract.

Codex runs exactly one opaque Step, and that Step is always terminal: the
CLI either names a ``call_id`` it evaluated, or it keeps the seed. The
contract therefore returns nothing while continuing -- which never happens
-- and the run terminal cardinality when it completes, so a terminal step
that keeps the seed may report ``seed_retained``.

``build_next`` raises: "one step only" is encoded here, and the adapter's
``step_index != 0`` guard is the same rule seen from the other side.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dr_store import ObjectStore

from whetstone.core.identity import ImmutableJsonObject
from whetstone.experiment.candidate import Candidate
from whetstone.optim.codex.adapter import CODEX_ADAPTER_KEY
from whetstone.optim.contracts import (
    BudgetState,
    OptimRunRef,
    OptimStepRequest,
    OptimStepResult,
    OutputContract,
    StepKind,
    StepStatus,
)

if TYPE_CHECKING:
    from whetstone.core.identity import TypedRef
    from whetstone.optim.codex.control import CodexControl

__all__ = [
    "CODEX_STEP_KIND_LABEL",
    "CODEX_TOOL_CALLS_BUDGET_LABEL",
    "CodexStepContractProvider",
    "codex_step_output_contract",
]

#: The budget label the Issued Tool Call ledger reads and debits. It must be
#: spelled exactly this way: ``_IssuedToolCallLedger`` keys its hard limit
#: and its ``budget_delta`` injection on this literal.
CODEX_TOOL_CALLS_BUDGET_LABEL = "tool_calls"
CODEX_STEP_KIND_LABEL = "codex_direct"


def codex_step_output_contract(run: OptimRunRef) -> OutputContract:
    """The contract the single Codex step binds.

    It returns the run terminal cardinality on completion, or nothing when
    it retains the seed -- the search-dependent shape that
    ``terminal_proposal_count`` exists to express.
    """
    terminal = run.record.terminal_output_contract
    return OutputContract(
        returned_proposal_count=0,
        terminal_proposal_count=terminal.accepted_count_for(
            StepStatus.COMPLETE
        ),
        require_distinct_bases=terminal.require_distinct_bases,
    )


class CodexStepContractProvider:
    @property
    def adapter_key(self) -> str:
        return CODEX_ADAPTER_KEY

    def requires_control(self) -> bool:
        return True

    def parse_control(self, payload: dict[str, Any]) -> CodexControl:
        from whetstone.optim.codex.control import CodexControl

        return CodexControl.model_validate(payload)

    def build_first(
        self,
        *,
        store: ObjectStore,
        run: OptimRunRef,
        initial_candidate: Candidate,
        control: CodexControl | None,
        extra_pools: dict[str, Any] | None,
    ) -> OptimStepRequest:
        del store
        if control is None:
            raise ValueError("Codex initial step requires the exact control")
        from whetstone.optim.codex.control import CodexControl

        if not isinstance(control, CodexControl):
            raise TypeError("Codex initial step requires CodexControl")
        pools: dict[str, Any] = {}
        if extra_pools:
            pools.update(extra_pools)
        return OptimStepRequest(
            run=run,
            step_id=f"{run.record.run_id}:codex:0",
            kind=StepKind.TOOL,
            kind_label=CODEX_STEP_KIND_LABEL,
            step_index=0,
            candidates=(initial_candidate,),
            pools=ImmutableJsonObject(pools),
            hyperparameters=ImmutableJsonObject(
                control.step_hyperparameters(iteration=0)
            ),
            budget=BudgetState(
                remaining=ImmutableJsonObject(
                    {
                        CODEX_TOOL_CALLS_BUDGET_LABEL: (
                            control.max_tool_calls
                        )
                    }
                ),
            ),
            step_output_contract=codex_step_output_contract(run),
        )

    def build_next(
        self,
        *,
        store: ObjectStore,
        prior: OptimStepResult,
        prior_ref: TypedRef,
        prior_results: tuple[OptimStepResult, ...],
        control: CodexControl,
        mutation_field: str,
        extra_pools: dict[str, Any] | None,
    ) -> OptimStepRequest:
        del store, prior_ref, prior_results, control, mutation_field
        del extra_pools
        raise ValueError(
            "Codex runs exactly one opaque step; step "
            f"{prior.step_index + 1} has no contract"
        )
