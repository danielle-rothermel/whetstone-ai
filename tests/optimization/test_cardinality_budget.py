"""Cardinality per mode, budget carry-forward, and status/terminal proofs."""

from __future__ import annotations

from whetstone.optimization import (
    BudgetState,
    IdentityOptimizerAdapter,
    OptimizationHarness,
    StepStatus,
    step_result_reference,
)

from .support import (
    CountingProposalAdapter,
    RecordingEvaluationService,
    RecordingToolExecutor,
    ToolUsingAdapter,
    candidate,
    make_store,
    proposal_request,
    pure_request,
    tool_request,
)


def test_identity_run_has_exactly_one_pure_step() -> None:
    store = make_store()
    harness = OptimizationHarness(store=store)
    _r, ref = harness.run_step(pure_request(), IdentityOptimizerAdapter())
    terminal = harness.terminalize(run_id="run-pure", step_result_refs=(ref,))
    assert len(terminal.step_result_refs) == 1


def test_codex_run_has_exactly_one_tool_using_step() -> None:
    store = make_store()
    executor = RecordingToolExecutor()
    harness = OptimizationHarness(store=store, tool_executor=executor)
    _r, ref = harness.run_step(tool_request(), ToolUsingAdapter())
    terminal = harness.terminalize(run_id="run-tool", step_result_refs=(ref,))
    assert len(terminal.step_result_refs) == 1
    assert terminal.status is StepStatus.COMPLETE


def test_copro_style_run_has_many_proposal_steps() -> None:
    # Two proposal steps whose budgets advance only through immutable Results.
    store = make_store()
    ev = RecordingEvaluationService(store)
    harness = OptimizationHarness(store=store, evaluation_service=ev)

    s0 = proposal_request(step_index=0)
    # Attach an initial budget: 100 rollouts remaining.
    s0 = s0.model_copy(
        update={"budget": BudgetState(remaining={"rollouts": 100})}
    )
    r0, ref0 = harness.run_step(s0, CountingProposalAdapter())
    assert r0.status is StepStatus.CONTINUE

    # The next request references the prior Step Result and carries the budget
    # forward from that immutable Result.
    carried = OptimizationHarness.carry_budget_forward(r0)
    s1 = proposal_request(step_index=1, prior_step_result_ref=ref0)
    s1 = s1.model_copy(update={"budget": carried})
    _r1, ref1 = harness.run_step(s1, CountingProposalAdapter())

    terminal = harness.terminalize(
        run_id="run-copro", step_result_refs=(ref0, ref1)
    )
    assert len(terminal.step_result_refs) == 2
    # The prior Step Result reference resolves the immutable prior result.
    assert s1.prior_step_result_ref == step_result_reference(r0)


def test_status_continue_complete_failed_all_representable() -> None:
    store = make_store()
    ev = RecordingEvaluationService(store)
    harness = OptimizationHarness(store=store, evaluation_service=ev)

    cont = CountingProposalAdapter(status=StepStatus.CONTINUE)
    r_cont, _ = harness.run_step(proposal_request(run_id="rc"), cont)
    assert r_cont.status is StepStatus.CONTINUE

    comp = CountingProposalAdapter(status=StepStatus.COMPLETE)
    r_comp, _ = harness.run_step(proposal_request(run_id="rp"), comp)
    assert r_comp.status is StepStatus.COMPLETE

    fail = CountingProposalAdapter(status=StepStatus.FAILED)
    r_fail, ref_fail = harness.run_step(proposal_request(run_id="rf"), fail)
    assert r_fail.status is StepStatus.FAILED

    # A failed terminal run blocks official materialization: no proposals.
    terminal = harness.terminalize(
        run_id="rf", step_result_refs=(ref_fail,)
    )
    assert terminal.status is StepStatus.FAILED
    assert terminal.proposals == ()


def test_budget_cannot_be_reconstructed_from_process_memory() -> None:
    # The next Step's budget is exactly the prior Result's budget object, never
    # recomputed. Prove carry_forward is a pure function of the Result.
    store = make_store()
    harness = OptimizationHarness(store=store)
    request = pure_request(candidates=(candidate("A"),))
    request = request.model_copy(
        update={
            "budget": BudgetState(
                consumed={"rollouts": 5}, remaining={"rollouts": 95}
            )
        }
    )
    result, _ref = harness.run_step(request, IdentityOptimizerAdapter())
    carried = OptimizationHarness.carry_budget_forward(result)
    assert carried.consumed == {"rollouts": 5}
    assert carried.remaining == {"rollouts": 95}
    assert carried == result.budget
