"""Tool-using output checkpoints and terminal evidence."""

import pytest

from whetstone.optimization import (
    BudgetState,
    OptimizationHarness,
    ToolCallState,
)

from .support import (
    RecordingToolExecutor,
    ToolUsingAdapter,
    make_store,
    registry,
    tool_request,
)


def test_tool_results_and_store_entries_are_step_evidence(tmp_path) -> None:
    store = make_store(tmp_path)
    adapter = ToolUsingAdapter(call_ids=("c1",))
    executor = RecordingToolExecutor()
    harness = OptimizationHarness(
        store=store,
        adapter_registry=registry(adapter),
        tool_executor=executor,
    )
    request = tool_request()
    result, _ = harness.run_step(request)
    evidence = result.tool_evidence[0]
    assert evidence.store_entry.state is ToolCallState.COMPLETED
    assert evidence.store_entry.tool_result_ref == evidence.tool_result_ref
    assert adapter.invocations == 1
    assert executor.handles_built == 1


def test_tool_output_checkpoint_prevents_redrive_before_result(
    tmp_path,
) -> None:
    adapter = ToolUsingAdapter(call_ids=("c1",))
    executor = RecordingToolExecutor()
    request = tool_request().model_copy(
        update={"budget": BudgetState(remaining={"tool_calls": 0})}
    )
    first_store = make_store(tmp_path)
    first = OptimizationHarness(
        store=first_store,
        adapter_registry=registry(adapter),
        tool_executor=executor,
    )
    with pytest.raises(ValueError, match="only 0"):
        first.run_step(request)
    assert adapter.invocations == 1
    config_hash = request.tool_configs[0].identity_hash()
    assert first.tool_store.accepted_count(config_hash) == 1

    fresh = OptimizationHarness(
        store=make_store(tmp_path),
        adapter_registry=registry(adapter),
        tool_executor=executor,
    )
    with pytest.raises(ValueError, match="only 0"):
        fresh.run_step(request)
    assert adapter.invocations == 1
    assert executor.handles_built == 1
    assert fresh.tool_store.accepted_count(config_hash) == 1


def test_tool_step_requires_executor_before_invocation(tmp_path) -> None:
    store = make_store(tmp_path)
    adapter = ToolUsingAdapter()
    harness = OptimizationHarness(
        store=store,
        adapter_registry=registry(adapter),
    )
    with pytest.raises(ValueError, match="ToolExecutor"):
        harness.run_step(tool_request())
    assert adapter.invocations == 0
