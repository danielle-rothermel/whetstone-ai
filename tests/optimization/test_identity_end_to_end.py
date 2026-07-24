"""The zero-optimization baseline is pure and durable."""

import pytest

from whetstone.optimization import (
    CANDIDATE_RECORD_SCHEMA,
    IdentityOptimizerAdapter,
    OptimizationHarness,
    RuntimeToolHandle,
    StepStatus,
    ToolResult,
)

from .support import (
    candidate,
    make_store,
    make_tool_definition_config,
    pure_request,
    registry,
)


def test_identity_persists_candidates_and_terminal_result(tmp_path) -> None:
    store = make_store(tmp_path)
    harness = OptimizationHarness(
        store=store,
        adapter_registry=registry(),
    )
    request = pure_request(candidates=(candidate("A"), candidate("B")))
    step, step_ref = harness.run_step(request)
    assert step.status is StepStatus.COMPLETE
    assert step.resolved_intents == ()
    assert step.tool_evidence == ()
    assert all(
        ref.record_ref.schema_name == CANDIDATE_RECORD_SCHEMA
        for ref in step.accepted_candidates
    )
    terminal, terminal_ref = harness.terminalize(
        run_id=request.run_id,
        step_result_refs=(step_ref,),
    )
    assert [p.candidate.record.candidate_id for p in terminal.proposals] == [
        "A",
        "B",
    ]
    assert harness.resolve_optimization_result(request.run_id) == terminal_ref


def test_identity_replay_never_invokes_registry_adapter(tmp_path) -> None:
    store = make_store(tmp_path)
    request = pure_request()
    first = OptimizationHarness(
        store=store,
        adapter_registry=registry(),
    )
    result_a, ref_a = first.run_step(request)

    class MissingRegistry:
        def resolve(self, adapter_key):
            del adapter_key
            raise AssertionError(
                "replay must not resolve or invoke an adapter"
            )

    fresh = OptimizationHarness(
        store=make_store(tmp_path),
        adapter_registry=MissingRegistry(),
    )
    result_b, ref_b = fresh.run_step(request)
    assert (result_b, ref_b) == (result_a, ref_a)


def test_identity_adapter_refuses_runtime_handles() -> None:
    config = make_tool_definition_config()
    handle = RuntimeToolHandle(
        config,
        lambda call: ToolResult(
            call_id=call.call_id,
            tool_config_ref=config.tool_definition_ref,
            tool_config_hash=config.identity_hash(),
            store_namespace=config.store_namespace,
            refusal={"refusal_class": "validation", "reason": "unused"},
        ),
    )
    with pytest.raises(ValueError, match="no Runtime"):
        IdentityOptimizerAdapter().invoke(pure_request(), (handle,))
