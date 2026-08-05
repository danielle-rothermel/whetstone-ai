"""Central output-contract and durable budget enforcement."""

import pytest

from whetstone.core.identity import TypedRef
from whetstone.optimization.contracts import (
    BudgetDelta,
    BudgetState,
    OptimizationRunRef,
    OutputContract,
)
from whetstone.optimization.harness import OptimizationHarness

from .support import (
    FULL_D,
    CountingProposalAdapter,
    RecordingEvaluationService,
    candidate,
    make_harness,
    make_store,
    proposal_request,
    registry,
)


def _harness(store, adapter, run: OptimizationRunRef):
    return make_harness(
        store=store,
        adapter_registry=registry(adapter),
        run=run,
        evaluation_service=RecordingEvaluationService(store),
    )


def test_budget_delta_is_validated_debited_and_carried(tmp_path) -> None:
    store = make_store(tmp_path)
    adapter = CountingProposalAdapter(
        budget_delta=BudgetDelta(consumed={"rollouts": 3})
    )
    request = proposal_request(
        budget=BudgetState(
            consumed={"rollouts": 2},
            remaining={"rollouts": 8},
        )
    )
    result, _ = _harness(store, adapter, request.run).run_step(request)
    assert result.budget_delta.consumed == {"rollouts": 3}
    assert result.budget.consumed == {"rollouts": 5}
    assert result.budget.remaining == {"rollouts": 5}
    assert OptimizationHarness.carry_budget_forward(result) == result.budget


@pytest.mark.parametrize(
    ("delta", "message"),
    [
        (BudgetDelta(consumed={"unknown": 1}), "undeclared"),
        (BudgetDelta(consumed={"rollouts": 11}), "only 10"),
    ],
)
def test_invalid_budget_delta_never_binds_result(
    tmp_path, delta, message
) -> None:
    store = make_store(tmp_path)
    adapter = CountingProposalAdapter(budget_delta=delta)
    request = proposal_request()
    harness = _harness(store, adapter, request.run)
    with pytest.raises(ValueError, match=message):
        harness.run_step(request)
    assert harness.resolve_step_result(request.run_id, 0) is None


def test_exact_cardinality_is_enforced_centrally(tmp_path) -> None:
    store = make_store(tmp_path)
    adapter = CountingProposalAdapter(
        candidates=(candidate("P1"), candidate("P2"))
    )
    request = proposal_request()
    harness = _harness(store, adapter, request.run)
    with pytest.raises(ValueError, match="cardinality"):
        harness.run_step(request)


def test_distinct_bases_is_conditional(tmp_path) -> None:
    repeated = (
        candidate("P1", base="same", text="first"),
        candidate("P2", base="same", text="second"),
    )
    store = make_store(tmp_path, "allowed.sqlite")
    adapter = CountingProposalAdapter(candidates=repeated)
    allowed = proposal_request(
        contract=OutputContract(
            returned_proposal_count=2,
            require_distinct_bases=False,
        )
    )
    result, _ = _harness(store, adapter, allowed.run).run_step(allowed)
    assert len(result.accepted_candidates) == 2

    strict_store = make_store(tmp_path, "strict.sqlite")
    strict_adapter = CountingProposalAdapter(candidates=repeated)
    strict = proposal_request(
        run_id="strict",
        contract=OutputContract(
            returned_proposal_count=2,
            require_distinct_bases=True,
        ),
    )
    with pytest.raises(ValueError, match="distinct-base"):
        _harness(strict_store, strict_adapter, strict.run).run_step(strict)


def test_prior_ref_must_match_actual_preceding_binding(tmp_path) -> None:
    store = make_store(tmp_path)
    first_adapter = CountingProposalAdapter()
    first_request = proposal_request()
    first, _ = _harness(store, first_adapter, first_request.run).run_step(
        first_request
    )
    forged = proposal_request(
        step_index=1,
        prior_step_result_ref=TypedRef(
            schema_name="whetstone.optimization_step_result",
            content_hash=FULL_D,
        ),
        budget=first.budget,
    )
    with pytest.raises(ValueError, match="actual preceding"):
        _harness(store, CountingProposalAdapter(), first_request.run).run_step(
            forged
        )
