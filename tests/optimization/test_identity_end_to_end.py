"""End-to-end identity optimizer run: request -> step -> result -> terminal."""

from __future__ import annotations

import pytest

from whetstone.optimization import (
    Candidate,
    IdentityOptimizerAdapter,
    OptimizationHarness,
    OptimizationRun,
    OptimizationStepRequest,
    OutputContract,
    StepKind,
    StepMode,
    StepStatus,
)

from .support import FULL_A, make_store


def _starting_candidates(n: int = 8) -> tuple[Candidate, ...]:
    return tuple(
        Candidate(
            candidate_id=f"cand-{i}",
            base_ref="encoder-base",
            payload={"user_prompt_template": f"template-{i}"},
        )
        for i in range(n)
    )


def test_identity_run_request_step_result_terminal() -> None:
    store = make_store()
    harness = OptimizationHarness(store=store)
    candidates = _starting_candidates(8)
    contract = OutputContract(returned_proposal_count=8)

    run = OptimizationRun(
        run_id="eval-run-1",
        optimizer_config_hash=FULL_A,
        mode=StepMode.PURE,
        output_contract=contract,
    )
    assert len(run.identity_hash()) == 64

    request = OptimizationStepRequest(
        run_id="eval-run-1",
        step_id="eval-run-1-s0",
        optimizer_config_hash=FULL_A,
        mode=StepMode.PURE,
        kind=StepKind.IDENTITY,
        step_index=0,
        candidates=candidates,
        output_contract=contract,
    )

    result, result_ref = harness.run_step(request, IdentityOptimizerAdapter())

    # Exactly one pure Step: unchanged candidates, no tools, no intents.
    assert result.status is StepStatus.COMPLETE
    assert result.accepted_candidates == candidates
    assert result.proposed_candidates == candidates
    assert result.resolved_intents == ()
    assert result.tool_evidence == ()
    # The Step Result references its request by Content Hash.
    assert result.request_ref.content_hash

    # Terminalize: the terminal Optimization Result carries the N ordered
    # unchanged proposals, the sole Step Result ref, and makes no official
    # claim (it has no official evaluation fields).
    terminal = harness.terminalize(
        run_id="eval-run-1",
        step_result_refs=(result_ref,),
        cost={"wall_clock_s": 0.0},
    )
    assert terminal.status is StepStatus.COMPLETE
    assert len(terminal.proposals) == 8
    assert [p.candidate_id for p in terminal.proposals] == [
        c.candidate_id for c in candidates
    ]
    assert terminal.step_result_refs == (result_ref,)
    dumped = terminal.model_dump()
    assert "official" not in dumped
    assert "objective" not in dumped


def test_identity_adapter_rejects_tools_and_intents() -> None:
    adapter = IdentityOptimizerAdapter()
    assert adapter.mode is StepMode.PURE
    # The adapter emits neither intents nor tool calls.
    request = OptimizationStepRequest(
        run_id="r",
        step_id="s0",
        optimizer_config_hash=FULL_A,
        mode=StepMode.PURE,
        kind=StepKind.IDENTITY,
        step_index=0,
        candidates=_starting_candidates(2),
        output_contract=OutputContract(returned_proposal_count=2),
    )
    output = adapter.invoke(request, ())
    assert output.evaluation_intents == ()
    assert output.tool_call_records == ()
    assert output.accepted_candidates == request.candidates


def test_identity_adapter_refuses_runtime_handles() -> None:
    from whetstone.optimization import RuntimeToolHandle

    from .support import make_tool_definition_config

    adapter = IdentityOptimizerAdapter()
    request = OptimizationStepRequest(
        run_id="r",
        step_id="s0",
        optimizer_config_hash=FULL_A,
        mode=StepMode.PURE,
        kind=StepKind.IDENTITY,
        step_index=0,
        candidates=_starting_candidates(1),
        output_contract=OutputContract(returned_proposal_count=1),
    )
    handle = RuntimeToolHandle(
        make_tool_definition_config(),
        lambda call: None,  # type: ignore
    )
    with pytest.raises(ValueError, match="needs no Runtime Tool Handle"):
        adapter.invoke(request, (handle,))
