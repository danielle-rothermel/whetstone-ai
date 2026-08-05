"""Step and terminal bindings replay from fresh SQLite-backed instances."""

import pytest

from whetstone.optimization.contracts import step_result_reference
from whetstone.optimization.run_store import OptimizationResultConflictError

from .support import make_harness, make_store, pure_request, registry


def test_terminal_result_is_persisted_bound_and_replayed(tmp_path) -> None:
    store = make_store(tmp_path)
    request = pure_request()
    harness = make_harness(
        store=store,
        adapter_registry=registry(),
        run=request.run,
    )
    step, _step_ref = harness.run_step(request)
    result_a, result_ref_a = harness.terminalize(
        run=request.run,
        step_results=(step_result_reference(step),),
        cost={"calls": 0},
    )

    class ExplodingRegistry:
        def resolve(self, adapter_key):
            del adapter_key
            raise AssertionError("terminal replay must not resolve an adapter")

    fresh = make_harness(
        store=make_store(tmp_path),
        adapter_registry=ExplodingRegistry(),
        run=request.run,
    )
    result_b, result_ref_b = fresh.terminalize(
        run=request.run,
        step_results=result_a.step_results,
        cost={"calls": 0},
    )
    assert (result_b, result_ref_b) == (result_a, result_ref_a)
    assert fresh.resolve_optimization_result(request.run_id) == result_ref_a


def test_divergent_terminal_result_preserves_winner(tmp_path) -> None:
    store = make_store(tmp_path)
    request = pure_request()
    harness = make_harness(
        store=store,
        adapter_registry=registry(),
        run=request.run,
    )
    step, _step_ref = harness.run_step(request)
    _, winner = harness.terminalize(
        run=request.run,
        step_results=(step_result_reference(step),),
        cost={"calls": 0},
    )
    with pytest.raises(OptimizationResultConflictError) as exc:
        harness.terminalize(
            run=request.run,
            step_results=(step_result_reference(step),),
            cost={"calls": 1},
        )
    assert exc.value.existing == winner
