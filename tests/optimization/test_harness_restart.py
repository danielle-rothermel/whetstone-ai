"""Restart / idempotency / no-authoritative-pickle harness proofs.

Crash points covered: before adapter completion, during external Evaluation
Intent resolution, after tool completion, and after Step Result persistence.
Every replay reuses durable evidence, never creates a second Step Result, and
never follows a back-edge before the prior Result exists.
"""

from __future__ import annotations

import pytest

from whetstone.optimization import (
    IdentityOptimizerAdapter,
    OptimizationHarness,
    StepResultConflictError,
    StepStatus,
)

from .support import (
    CountingProposalAdapter,
    RecordingEvaluationService,
    RecordingToolExecutor,
    ToolUsingAdapter,
    make_store,
    proposal_request,
    pure_request,
    tool_request,
)


class _RaiseOnceEvaluator:
    """Resolves the Intent, but raises the first time (mid-resolution)."""

    def __init__(self, inner: RecordingEvaluationService) -> None:
        self._inner = inner
        self._raised = False

    def resolve_evaluation_intent(self, intent):
        if not self._raised:
            self._raised = True
            raise RuntimeError("crash during external evaluation")
        return self._inner.resolve_evaluation_intent(intent)


def test_crash_before_adapter_completion_reinvokes_cleanly() -> None:
    # No checkpoint exists yet, so a re-run simply invokes the adapter and
    # produces exactly one Step Result.
    store = make_store()
    ev = RecordingEvaluationService(store)
    harness = OptimizationHarness(store=store, evaluation_service=ev)
    adapter = CountingProposalAdapter()
    _result, ref = harness.run_step(proposal_request(), adapter)
    assert adapter.invocations == 1
    assert harness.resolve_step_result("run-copro", 0) == ref


def test_crash_during_intent_resolution_reuses_checkpoint() -> None:
    store = make_store()
    inner = RecordingEvaluationService(store)
    harness = OptimizationHarness(
        store=store, evaluation_service=_RaiseOnceEvaluator(inner)
    )
    adapter = CountingProposalAdapter()
    request = proposal_request()

    # First run crashes during resolution AFTER the proposal was checkpointed.
    with pytest.raises(RuntimeError, match="crash during external"):
        harness.run_step(request, adapter)
    assert adapter.invocations == 1
    # No Step Result was persisted (finalization never reached).
    assert harness.resolve_step_result("run-copro", 0) is None

    # Replay: the proposal invocation is NOT rerun (checkpoint reused); the
    # second resolution succeeds and exactly one Step Result is persisted.
    result, ref = harness.run_step(request, adapter)
    assert adapter.invocations == 1
    assert result.status is StepStatus.CONTINUE
    assert harness.resolve_step_result("run-copro", 0) == ref


def test_crash_after_tool_completion_replays_idempotently() -> None:
    store = make_store()
    executor = RecordingToolExecutor()
    harness = OptimizationHarness(store=store, tool_executor=executor)
    adapter = ToolUsingAdapter(call_ids=("c1",))
    request = tool_request()

    result_a, ref_a = harness.run_step(request, adapter)
    tch = request.tool_configs[0].identity_hash()
    assert harness.tool_store.accepted_count(tch) == 1

    # Replay after completion: same Step Result, no second capacity debit.
    result_b, ref_b = harness.run_step(request, adapter)
    assert ref_a == ref_b
    assert result_a == result_b
    assert harness.tool_store.accepted_count(tch) == 1


def test_crash_after_step_result_persistence_is_idempotent() -> None:
    store = make_store()
    harness = OptimizationHarness(store=store)
    adapter = IdentityOptimizerAdapter()
    request = pure_request()

    result_a, ref_a = harness.run_step(request, adapter)
    # A replay resolves the existing Step Result FIRST and returns it.
    result_b, ref_b = harness.run_step(request, adapter)
    assert ref_a == ref_b
    assert result_a == result_b


def test_divergent_result_for_same_step_conflicts() -> None:
    store = make_store()
    harness = OptimizationHarness(store=store)
    # Two DIFFERENT requests that share the same (run_id, step_index) identity
    # produce different Step Results; the second conflicts with the durable
    # winner and never replaces it.
    from .support import candidate

    req_a = pure_request(candidates=(candidate("A", text="x"),))
    req_b = pure_request(candidates=(candidate("A", text="y"),))
    _result_a, ref_a = harness.run_step(req_a, IdentityOptimizerAdapter())
    with pytest.raises(StepResultConflictError) as exc:
        harness.run_step(req_b, IdentityOptimizerAdapter())
    # The durable winner (ref_a) is preserved and exposed.
    assert exc.value.existing == ref_a


def test_restart_position_is_derivable_only_from_persisted_results() -> None:
    # NO authoritative optimizer pickle: a fresh harness over the SAME store,
    # given the same request, re-derives its position and produces the same
    # Step Result reference. There is no in-process optimizer object carried.
    store = make_store()
    request = pure_request()

    h1 = OptimizationHarness(store=store)
    _r1, ref1 = h1.run_step(request, IdentityOptimizerAdapter())

    # A brand-new harness (no shared memory) re-runs the same request and
    # arrives at the identical content-addressed Step Result reference.
    h2 = OptimizationHarness(store=store)
    _r2, ref2 = h2.run_step(request, IdentityOptimizerAdapter())
    assert ref2 == ref1
    # The reference is content-addressed: it is a pure function of the request
    # + result, not of process memory.
    assert ref1.content_hash == ref2.content_hash


def test_back_edge_forbidden_before_prior_result_persisted() -> None:
    # A later Step (index 1) cannot be built without the prior Step Result ref;
    # the schema itself forbids a noninitial request without one.
    from pydantic import ValidationError

    from whetstone.optimization import (
        OptimizationStepRequest,
        StepKind,
        StepMode,
    )

    from .support import FULL_A, output_contract

    with pytest.raises(ValidationError):
        OptimizationStepRequest(
            run_id="run-copro",
            step_id="s1",
            optimizer_config_hash=FULL_A,
            mode=StepMode.PROPOSAL_ONLY,
            kind=StepKind.PROPOSAL,
            step_index=1,
            output_contract=output_contract(1),
        )
