"""Proposal evaluation is outside the checkpointed adapter invocation."""

import pytest

from whetstone.optimization import (
    IntentOutcome,
    OptimizationHarness,
    StepStatus,
)

from .support import (
    CountingProposalAdapter,
    RecordingEvaluationService,
    make_store,
    proposal_request,
    registry,
)


class CrashOnceEvaluationService:
    def __init__(self) -> None:
        self.calls = 0

    def resolve_evaluation_intent(self, intent):
        del intent
        self.calls += 1
        raise RuntimeError("crash during external evaluation")


def test_fresh_sqlite_restart_reuses_adapter_checkpoint(tmp_path) -> None:
    adapter = CountingProposalAdapter()
    request = proposal_request()
    crashed_store = make_store(tmp_path)
    crashed = OptimizationHarness(
        store=crashed_store,
        adapter_registry=registry(adapter),
        evaluation_service=CrashOnceEvaluationService(),
    )
    with pytest.raises(RuntimeError, match="crash during"):
        crashed.run_step(request)
    assert adapter.invocations == 1
    assert crashed.resolve_step_result(request.run_id, 0) is None

    fresh_store = make_store(tmp_path)
    fresh = OptimizationHarness(
        store=fresh_store,
        adapter_registry=registry(adapter),
        evaluation_service=RecordingEvaluationService(fresh_store),
    )
    result, result_ref = fresh.run_step(request)
    assert adapter.invocations == 1
    assert result.resolved_intents[0].outcome is IntentOutcome.COMPLETED
    assert fresh.resolve_step_result(request.run_id, 0) == result_ref


def test_candidate_local_failure_does_not_erase_successful_steps(
    tmp_path,
) -> None:
    store = make_store(tmp_path)
    first_adapter = CountingProposalAdapter(status=StepStatus.CONTINUE)
    harness = OptimizationHarness(
        store=store,
        adapter_registry=registry(first_adapter),
        evaluation_service=RecordingEvaluationService(store),
    )
    first_request = proposal_request()
    first, first_ref = harness.run_step(first_request)

    failed_service = RecordingEvaluationService(
        store, outcome=IntentOutcome.FAILED
    )
    second_adapter = CountingProposalAdapter(status=StepStatus.COMPLETE)
    second = OptimizationHarness(
        store=store,
        adapter_registry=registry(second_adapter),
        evaluation_service=failed_service,
    )
    second_request = proposal_request(
        step_index=1,
        prior_step_result_ref=first_ref,
        budget=first.budget,
    )
    second_result, second_ref = second.run_step(second_request)
    assert second_result.status is StepStatus.COMPLETE
    assert second_result.resolved_intents[0].outcome is IntentOutcome.FAILED

    terminal, _ = second.terminalize(
        run_id=first_request.run_id,
        step_result_refs=(first_ref, second_ref),
    )
    assert terminal.step_result_refs == (first_ref, second_ref)
    assert len(terminal.proposals) == 1


def test_pre_execution_rejection_is_recorded_without_evidence(
    tmp_path,
) -> None:
    store = make_store(tmp_path)
    adapter = CountingProposalAdapter()
    harness = OptimizationHarness(
        store=store,
        adapter_registry=registry(adapter),
        evaluation_service=RecordingEvaluationService(
            store, outcome=IntentOutcome.REJECTED
        ),
    )
    result, _ = harness.run_step(proposal_request())
    resolution = result.resolved_intents[0]
    assert resolution.outcome is IntentOutcome.REJECTED
    assert resolution.evaluation_evidence_refs == ()
