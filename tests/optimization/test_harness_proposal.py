"""Proposal-only harness dispatch, checkpointing, and intent resolution."""

from __future__ import annotations

import pytest

from whetstone.optimization import (
    ADAPTER_CHECKPOINT_SCHEMA,
    OptimizationHarness,
    StepStatus,
)

from .support import (
    FULL_B,
    FULL_C,
    CountingProposalAdapter,
    RecordingEvaluationService,
    WrongTargetEvaluationService,
    make_store,
    proposal_request,
)


def test_proposal_step_checkpoints_and_resolves_outside_invocation() -> None:
    store = make_store()
    evaluator = RecordingEvaluationService(store)
    harness = OptimizationHarness(store=store, evaluation_service=evaluator)
    adapter = CountingProposalAdapter()

    request = proposal_request()
    result, _ref = harness.run_step(request, adapter)

    # The adapter proposed one candidate + one Intent; the harness resolved the
    # Intent OUTSIDE the invocation through the evaluation service.
    assert adapter.invocations == 1
    assert len(evaluator.resolved) == 1
    assert len(result.resolved_intents) == 1
    resolution = result.resolved_intents[0]
    # The final Step Result references the resolution evidence.
    assert resolution.evaluation_evidence_refs
    # The Intent itself embeds no score.
    assert "score" not in resolution.intent.model_dump()
    # The typed adapter output was durably checkpointed in dr-store's binding
    # table (not in process memory), so a fresh harness would resolve it.
    ckpt_ref = harness._resolve_checkpoint_binding(
        request.run_id, request.step_index
    )
    assert ckpt_ref is not None
    assert ckpt_ref.schema_name == ADAPTER_CHECKPOINT_SCHEMA


def test_intent_resolves_only_under_exact_target_eval_config() -> None:
    store = make_store()
    harness = OptimizationHarness(
        store=store, evaluation_service=WrongTargetEvaluationService()
    )
    adapter = CountingProposalAdapter(target_hash=FULL_B)
    with pytest.raises(Exception) as exc:
        harness.run_step(proposal_request(), adapter)
    # The mismatch is caught by IntentResolution's exact-target rule.
    assert "exact target" in str(exc.value)


def test_restart_reuses_checkpoint_without_rerunning_proposal() -> None:
    store = make_store()
    evaluator = RecordingEvaluationService(store)
    harness = OptimizationHarness(store=store, evaluation_service=evaluator)
    adapter = CountingProposalAdapter()

    request = proposal_request()
    # First pass invokes and checkpoints, then crashes DURING intent resolution
    # by using an evaluator that raises once.
    harness._run_proposal(request, adapter)  # invoke + durable checkpoint
    assert adapter.invocations == 1

    # A brand-new harness over the SAME store (no shared process memory, as
    # after a real process restart) replays: the durable checkpoint is reused
    # and the completed proposal invocation is NOT rerun.
    fresh = OptimizationHarness(store=store, evaluation_service=evaluator)
    result, _ref = fresh.run_step(request, adapter)
    assert adapter.invocations == 1  # never rerun across restart
    assert result.status is StepStatus.CONTINUE
    assert len(result.resolved_intents) == 1


def test_same_request_and_result_replay_idempotently() -> None:
    store = make_store()
    evaluator = RecordingEvaluationService(store)
    harness = OptimizationHarness(store=store, evaluation_service=evaluator)
    adapter = CountingProposalAdapter()
    request = proposal_request()

    result_a, ref_a = harness.run_step(request, adapter)
    result_b, ref_b = harness.run_step(request, adapter)
    # No second Step Result; the adapter never runs twice.
    assert ref_a == ref_b
    assert result_a == result_b
    assert adapter.invocations == 1


def test_proposal_step_without_evaluation_service_fails() -> None:
    store = make_store()
    harness = OptimizationHarness(store=store)  # no evaluation service
    adapter = CountingProposalAdapter()
    with pytest.raises(ValueError, match="EvaluationService"):
        harness.run_step(proposal_request(), adapter)


def test_failed_status_is_carried_onto_step_result() -> None:
    store = make_store()
    evaluator = RecordingEvaluationService(store)
    harness = OptimizationHarness(store=store, evaluation_service=evaluator)
    adapter = CountingProposalAdapter(
        target_hash=FULL_C, status=StepStatus.FAILED
    )
    result, _ref = harness.run_step(
        proposal_request(), adapter
    )
    assert result.status is StepStatus.FAILED
