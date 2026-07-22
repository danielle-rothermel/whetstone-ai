"""Tool-using harness: handles at execution only, evidence recorded."""

from __future__ import annotations

import pytest

from whetstone.optimization import (
    OptimizationHarness,
    StepStatus,
    ToolCallState,
)

from .support import (
    RecordingToolExecutor,
    ToolUsingAdapter,
    make_store,
    make_tool_definition_config,
    tool_request,
)


def test_tool_handles_constructed_only_at_execution_boundary() -> None:
    store = make_store()
    executor = RecordingToolExecutor()
    tool_store_obj = None
    harness = OptimizationHarness(
        store=store, tool_executor=executor
    )
    tool_store_obj = harness.tool_store
    adapter = ToolUsingAdapter(call_ids=("c1", "c2"))

    request = tool_request()
    # No handle exists before the step runs.
    assert executor.handles_built == 0
    result, _ref = harness.run_step(request, adapter)
    # The handle was constructed exactly once, at execution.
    assert executor.handles_built == 1
    assert adapter.invocations == 1
    # Two calls accepted; capacity debited exactly twice.
    tch = request.tool_configs[0].identity_hash()
    assert tool_store_obj.accepted_count(tch) == 2
    assert result.status is StepStatus.COMPLETE


def test_step_result_references_every_tool_result_and_store_entry() -> None:
    store = make_store()
    executor = RecordingToolExecutor()
    harness = OptimizationHarness(store=store, tool_executor=executor)
    adapter = ToolUsingAdapter(call_ids=("c1", "c2"))

    result, _ref = harness.run_step(tool_request(), adapter)
    assert len(result.tool_evidence) == 2
    for ev in result.tool_evidence:
        # Each evidence pairs a Tool Result ref with its completed Store Entry.
        assert ev.store_entry.state is ToolCallState.COMPLETED
        assert ev.store_entry.tool_result_ref == ev.tool_result_ref
        assert ev.tool_result_ref.schema_name == "whetstone.tool_result"


def test_tool_using_requires_complete_tool_config() -> None:
    # The Tool Config carries the typed Tool Definition ref + Identity Hash.
    cfg = make_tool_definition_config()
    assert len(cfg.tool_definition_identity_hash) == 64
    assert len(cfg.identity_hash()) == 64


def test_tool_using_step_needs_a_tool_executor() -> None:
    store = make_store()
    harness = OptimizationHarness(store=store)  # no executor
    adapter = ToolUsingAdapter()
    with pytest.raises(ValueError, match="ToolExecutor"):
        harness.run_step(tool_request(), adapter)


def test_capacity_exhaustion_refuses_further_calls() -> None:
    store = make_store()
    executor = RecordingToolExecutor()
    harness = OptimizationHarness(store=store, tool_executor=executor)
    # Capacity 1, but the adapter issues two distinct calls: the second is
    # refused (typed non-execution outcome), not a measurement.
    cfg = make_tool_definition_config(capacity=1)
    adapter = ToolUsingAdapter(call_ids=("c1", "c2"))
    result, _ref = harness.run_step(
        tool_request(config=cfg), adapter
    )
    states = [ev.store_entry.state for ev in result.tool_evidence]
    assert ToolCallState.COMPLETED in states
    # The refused call surfaces as a refused Tool Result, referenced as
    # evidence but debiting no capacity.
    refused = [
        ev for ev in result.tool_evidence
        if ev.store_entry.state is ToolCallState.REFUSED
    ]
    assert len(refused) == 1
    assert refused[0].store_entry.refusal is not None
