"""Strict Adapter Output failure-boundary contracts."""

import pytest
from pydantic import ValidationError

from whetstone.evaluation_role import EvaluationRole
from whetstone.optimization.adapters import AdapterOutput
from whetstone.optimization.identity import TerminalFailure
from whetstone.optimization.schema import (
    BudgetDelta,
    EvaluationBinding,
    EvaluationIntent,
    StepStatus,
    candidate_reference,
    eval_config_reference,
)

from .support import (
    candidate,
    eval_config,
    internal_reward_policy,
)


def _failure() -> TerminalFailure:
    return TerminalFailure(
        code="adapter_exhausted",
        message="all adapter attempts failed",
        details={"attempts": 3},
    )


def test_failed_adapter_output_requires_shared_terminal_failure() -> None:
    with pytest.raises(
        ValidationError,
        match="requires exactly one shared terminal failure",
    ):
        AdapterOutput(proposed_status=StepStatus.FAILED)


@pytest.mark.parametrize(
    "status",
    [StepStatus.CONTINUE, StepStatus.COMPLETE],
)
def test_nonfailed_adapter_output_rejects_terminal_failure(
    status: StepStatus,
) -> None:
    with pytest.raises(
        ValidationError,
        match="requires exactly one shared terminal failure",
    ):
        AdapterOutput(
            proposed_status=status,
            terminal_failure=_failure(),
        )


def test_failed_adapter_output_rejects_accepted_candidates() -> None:
    proposed = candidate("diagnostic")
    with pytest.raises(
        ValidationError,
        match="claims no accepted candidates",
    ):
        AdapterOutput(
            proposed_candidates=(proposed,),
            accepted_candidates=(proposed,),
            proposed_status=StepStatus.FAILED,
            terminal_failure=_failure(),
        )


def test_failed_adapter_output_rejects_evaluation_intents() -> None:
    proposed = candidate("diagnostic")
    config = eval_config_reference(eval_config())
    intent = EvaluationIntent(
        intent_id="diagnostic-evaluation",
        candidate=candidate_reference(proposed),
        target_eval_config=config,
        evaluation_binding=EvaluationBinding(
            eval_config=config,
            role=EvaluationRole.INTERNAL,
            campaign="adapter-contract",
        ),
        purpose="diagnose adapter failure",
        run_id="run-proposal",
        step_index=0,
        expected_reward_policy_hash=internal_reward_policy().identity_hash(),
    )
    with pytest.raises(
        ValidationError,
        match="requests no Evaluations",
    ):
        AdapterOutput(
            proposed_candidates=(proposed,),
            evaluation_intents=(intent,),
            proposed_status=StepStatus.FAILED,
            terminal_failure=_failure(),
        )


def test_failed_adapter_output_round_trips_exact_shared_failure() -> None:
    failure = _failure()
    diagnostic = candidate("diagnostic")
    output = AdapterOutput(
        proposed_candidates=(diagnostic,),
        budget_delta=BudgetDelta(consumed={"tool_calls": 1}),
        proposed_status=StepStatus.FAILED,
        terminal_failure=failure,
        state_delta={"last_candidate": "diagnostic"},
        history_delta={"attempts": 3},
    )

    round_tripped = AdapterOutput.model_validate_json(output.model_dump_json())

    assert round_tripped == output
    assert type(round_tripped.terminal_failure) is TerminalFailure
    assert round_tripped.terminal_failure == failure
