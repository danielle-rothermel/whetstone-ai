"""The zero-optimization baseline is pure and durable."""

import pytest

from whetstone.optimization import (
    CANDIDATE_RECORD_SCHEMA,
    IdentityOptimizerAdapter,
    RuntimeToolHandle,
    StepStatus,
    ToolCapacityScope,
    ToolResult,
    step_result_reference,
    tool_call_reference,
)
from whetstone.optimization.tools import tool_capacity_binding

from .support import (
    candidate,
    make_harness,
    make_store,
    make_tool_definition_config,
    pure_request,
    registry,
    tool_run,
)


def test_identity_persists_candidates_and_terminal_result(tmp_path) -> None:
    store = make_store(tmp_path)
    request = pure_request(candidates=(candidate("A"), candidate("B")))
    harness = make_harness(
        store=store,
        adapter_registry=registry(),
        run=request.run,
    )
    step, _step_ref = harness.run_step(request)
    persisted_request = store.get(step.request.record_ref.reference)
    assert isinstance(persisted_request, dict)
    assert persisted_request == request.model_dump(mode="json")
    assert persisted_request["run"] == request.run.model_dump(mode="json")
    assert step.status is StepStatus.COMPLETE
    assert step.resolved_intents == ()
    assert step.tool_evidence == ()
    assert all(
        ref.record_ref.schema_name == CANDIDATE_RECORD_SCHEMA
        for ref in step.accepted_candidates
    )
    terminal, terminal_ref = harness.terminalize(
        run=request.run,
        step_results=(step_result_reference(step),),
    )
    assert [p.candidate.record.candidate_id for p in terminal.proposals] == [
        "A",
        "B",
    ]
    assert harness.resolve_optimization_result(request.run_id) == terminal_ref


def test_identity_replay_never_invokes_registry_adapter(tmp_path) -> None:
    store = make_store(tmp_path)
    request = pure_request()
    first = make_harness(
        store=store,
        adapter_registry=registry(),
        run=request.run,
    )
    result_a, ref_a = first.run_step(request)

    class MissingRegistry:
        def resolve(self, adapter_key):
            del adapter_key
            raise AssertionError(
                "replay must not resolve or invoke an adapter"
            )

    fresh = make_harness(
        store=make_store(tmp_path),
        adapter_registry=MissingRegistry(),
        run=request.run,
    )
    result_b, ref_b = fresh.run_step(request)
    assert (result_b, ref_b) == (result_a, ref_a)


def test_identity_adapter_refuses_runtime_handles() -> None:
    config = make_tool_definition_config()
    handle = RuntimeToolHandle(
        config,
        tool_capacity_binding(
            ToolCapacityScope.RUN, tool_run(config=config).record_ref
        ),
        lambda call: ToolResult(
            call=tool_call_reference(call),
            refusal={"refusal_class": "validation", "reason": "unused"},
        ),
    )
    with pytest.raises(ValueError, match="no Runtime"):
        IdentityOptimizerAdapter().invoke(pure_request(), (handle,))
