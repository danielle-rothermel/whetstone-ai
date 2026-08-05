"""Step and terminal bindings replay from fresh SQLite-backed instances."""

import pytest
from dr_store import BindingConflictError

from whetstone.optimization.contracts import (
    step_request_reference,
    step_result_reference,
)
from whetstone.optimization.run_store import (
    OptimizationResultConflictError,
    OptimizationRunConflictError,
    StepResultConflictError,
)

from .support import (
    candidate,
    make_harness,
    make_store,
    output_contract,
    pure_request,
    pure_run,
    registry,
)


def test_divergent_run_binding_preserves_first_writer_across_restart(
    tmp_path,
) -> None:
    run_a = pure_run(run_id="restart-run-conflict")
    run_b = pure_run(
        run_id=run_a.record.run_id,
        contract=output_contract(2),
    )
    make_harness(
        store=make_store(tmp_path),
        adapter_registry=registry(),
        run=run_a,
    )

    with pytest.raises(OptimizationRunConflictError) as exc:
        make_harness(
            store=make_store(tmp_path),
            adapter_registry=registry(),
            run=run_b,
        )

    assert exc.value.existing == run_a.record_ref
    assert exc.value.requested == run_b.record_ref
    persisted = make_store(tmp_path)
    assert (
        persisted.resolve(f"whetstone.optimization_run:{run_a.record.run_id}")
        == run_a.record_ref.reference
    )
    replay = make_harness(
        store=persisted,
        adapter_registry=registry(),
        run=run_a,
    )
    assert replay.bind_run(run_a) == run_a


def test_divergent_step_request_preserves_first_result_across_restart(
    tmp_path,
) -> None:
    run = pure_run(run_id="restart-step-conflict")
    request_a = pure_request(
        run=run,
        candidates=(candidate("A", text="first"),),
    )
    request_b = pure_request(
        run=run,
        candidates=(candidate("B", text="second"),),
    )
    first = make_harness(
        store=make_store(tmp_path),
        adapter_registry=registry(),
        run=run,
    )
    result_a, result_ref_a = first.run_step(request_a)

    fresh = make_harness(
        store=make_store(tmp_path),
        adapter_registry=registry(),
        run=run,
    )
    with pytest.raises(StepResultConflictError) as exc:
        fresh.run_step(request_b)

    assert exc.value.existing == result_ref_a
    assert exc.value.requested == step_request_reference(request_b).record_ref
    assert fresh.resolve_step_result(run.record.run_id, 0) == result_ref_a
    assert fresh.run_step(request_a) == (result_a, result_ref_a)


def test_run_bind_detects_immediate_resolution_mismatch(
    tmp_path, monkeypatch
) -> None:
    store = make_store(tmp_path)
    run_a = pure_run(run_id="post-bind-run-conflict")
    run_b = pure_run(
        run_id=run_a.record.run_id,
        contract=output_contract(2),
    )
    run_key = f"whetstone.optimization_run:{run_a.record.run_id}"
    durable_resolve = store.resolve

    def resolve_mismatched_run(key: str):
        if key == run_key:
            return run_b.record_ref.reference
        return durable_resolve(key)

    with monkeypatch.context() as patch:
        patch.setattr(store, "resolve", resolve_mismatched_run)
        with pytest.raises(OptimizationRunConflictError) as exc:
            make_harness(
                store=store,
                adapter_registry=registry(),
                run=run_a,
            )

    assert exc.value.existing == run_b.record_ref
    assert exc.value.requested == run_a.record_ref
    assert durable_resolve(run_key) == run_a.record_ref.reference
    make_harness(
        store=store,
        adapter_registry=registry(),
        run=run_a,
    )


def test_final_step_binding_conflict_translation_preserves_refs(
    tmp_path, monkeypatch
) -> None:
    store = make_store(tmp_path)
    run = pure_run(run_id="final-bind-step-conflict")
    request_a = pure_request(
        run=run,
        candidates=(candidate("A", text="first"),),
    )
    request_b = pure_request(
        run=run,
        candidates=(candidate("B", text="second"),),
    )
    first = make_harness(
        store=store,
        adapter_registry=registry(),
        run=run,
    )
    result_a, result_ref_a = first.run_step(request_a)
    result_key = f"whetstone.optimization_step_result:v2:{run.record.run_id}#0"
    durable_resolve = store.resolve
    durable_bind = store.bind
    attempted_refs = []

    def hide_existing_result(key: str):
        if key == result_key:
            return None
        return durable_resolve(key)

    def capture_result_bind(key: str, reference):
        if key == result_key:
            attempted_refs.append(reference)
        return durable_bind(key, reference)

    with monkeypatch.context() as patch:
        patch.setattr(store, "resolve", hide_existing_result)
        patch.setattr(store, "bind", capture_result_bind)
        fresh = make_harness(
            store=store,
            adapter_registry=registry(),
            run=run,
        )
        with pytest.raises(StepResultConflictError) as exc:
            fresh.run_step(request_b)

    assert len(attempted_refs) == 1
    attempted = attempted_refs[0]
    assert exc.value.existing == result_ref_a
    assert exc.value.requested.reference == attempted
    assert exc.value.requested != result_ref_a
    conflict = exc.value.__cause__
    assert isinstance(conflict, BindingConflictError)
    assert conflict.existing == result_ref_a.reference
    assert conflict.requested == attempted
    replay = make_harness(
        store=make_store(tmp_path),
        adapter_registry=registry(),
        run=run,
    )
    assert replay.resolve_step_result(run.record.run_id, 0) == result_ref_a
    assert replay.run_step(request_a) == (result_a, result_ref_a)


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
