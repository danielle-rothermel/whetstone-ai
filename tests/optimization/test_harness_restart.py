"""Step and terminal bindings replay from fresh SQLite-backed instances."""

import pytest

from whetstone.optimization import (
    OptimizationHarness,
    OptimizationResultConflictError,
)

from .support import make_store, pure_request, registry


def test_terminal_result_is_persisted_bound_and_replayed(tmp_path) -> None:
    store = make_store(tmp_path)
    request = pure_request()
    harness = OptimizationHarness(
        store=store,
        adapter_registry=registry(),
    )
    _, step_ref = harness.run_step(request)
    result_a, result_ref_a = harness.terminalize(
        run_id=request.run_id,
        step_result_refs=(step_ref,),
        cost={"calls": 0},
    )

    class ExplodingRegistry:
        def resolve(self, adapter_key):
            del adapter_key
            raise AssertionError("terminal replay must not resolve an adapter")

    fresh = OptimizationHarness(
        store=make_store(tmp_path),
        adapter_registry=ExplodingRegistry(),
    )
    result_b, result_ref_b = fresh.terminalize(
        run_id=request.run_id,
        step_result_refs=(step_ref,),
        cost={"calls": 0},
    )
    assert (result_b, result_ref_b) == (result_a, result_ref_a)
    assert fresh.resolve_optimization_result(request.run_id) == result_ref_a


def test_divergent_terminal_result_preserves_winner(tmp_path) -> None:
    store = make_store(tmp_path)
    request = pure_request()
    harness = OptimizationHarness(
        store=store,
        adapter_registry=registry(),
    )
    _, step_ref = harness.run_step(request)
    _, winner = harness.terminalize(
        run_id=request.run_id,
        step_result_refs=(step_ref,),
        cost={"calls": 0},
    )
    with pytest.raises(OptimizationResultConflictError) as exc:
        harness.terminalize(
            run_id=request.run_id,
            step_result_refs=(step_ref,),
            cost={"calls": 1},
        )
    assert exc.value.existing == winner
