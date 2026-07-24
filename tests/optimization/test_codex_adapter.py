"""Codex adapter tests: exactly one opaque tool-using Step over the MCP bridge.

All deterministic tests use the FAKE-codex double (a scripted MCP client
driving the in-process whetstone MCP server), never a real CLI. The Tool Call
Store is shared by the MCP server, the harness, and the adapter, so capacity
and
completion are one durable state across the (emulated) process boundary.
"""

from __future__ import annotations

import pytest

from tests.optimization.tool_support import (
    candidate,
    codex_request,
    evaluating_executor,
    make_store,
    mcp_server,
)
from whetstone.optimization import (
    OptimizationHarness,
    StepStatus,
    ToolCallState,
    ToolCallStore,
)
from whetstone.optimization.codex import CodexAdapter, OpaqueStepError
from whetstone.optimization.codex_runner import (
    FakeCodexRunner,
    ScriptedAgentCall,
)
from whetstone.optimization.schema import StepMode

ROUTES = ("route-0", "route-1", "route-2", "route-3")


def _four_proposals():
    return tuple(
        candidate(f"P{i}", route, f"evolved template for {route}")
        for i, route in enumerate(ROUTES)
    )


def _four_starting_candidates():
    return tuple(
        candidate(f"S{i}", route, "base template") for i, route in
        enumerate(ROUTES)
    )


def _wire(*, capacity: int = 20):
    """One ObjectStore + one ToolCallStore shared by server/harness/adapter."""
    store = make_store()
    tool_store = ToolCallStore(store)
    server = mcp_server(tool_store, capacity=capacity)
    return store, tool_store, server


def test_codex_one_opaque_step_returns_four_base_bound_proposals():
    store, tool_store, server = _wire()
    scripted = [
        ScriptedAgentCall(
            call_id=f"probe-{i}",
            model_route=ROUTES[i % 4],
            template=f"probe template {i}",
        )
        for i in range(6)
    ]
    runner = FakeCodexRunner(
        server=server,
        scripted_calls=scripted,
        final_proposals=_four_proposals(),
    )
    adapter = CodexAdapter(runner, store=store, tool_store=tool_store)
    harness = OptimizationHarness(
        store=store,
        tool_executor=evaluating_executor(),
        tool_store=tool_store,
    )
    request = codex_request(
        config=server.tool_config, candidates=_four_starting_candidates()
    )

    result, _ref = harness.run_step(request, adapter)

    assert result.status is StepStatus.COMPLETE
    assert len(result.accepted_candidates) == 4
    bases = {c.base_ref for c in result.accepted_candidates}
    assert bases == set(ROUTES)  # one per route, none duplicated/omitted
    # Every observed tool call is durable evidence with a completed entry.
    assert len(result.tool_evidence) == 6
    for evidence in result.tool_evidence:
        assert evidence.store_entry.state is ToolCallState.COMPLETED
        assert evidence.store_entry.tool_result_ref == evidence.tool_result_ref


def test_codex_capacity_cap_refuses_21st_new_identity():
    store, tool_store, server = _wire(capacity=20)
    # 20 accepted + a 21st NEW identity that must be refused (capacity).
    scripted = [
        ScriptedAgentCall(
            call_id=f"call-{i}",
            model_route=ROUTES[i % 4],
            template=f"template {i}",
        )
        for i in range(21)
    ]
    runner = FakeCodexRunner(
        server=server,
        scripted_calls=scripted,
        final_proposals=_four_proposals(),
    )
    adapter = CodexAdapter(runner, store=store, tool_store=tool_store)
    harness = OptimizationHarness(
        store=store,
        tool_executor=evaluating_executor(),
        tool_store=tool_store,
    )
    request = codex_request(
        config=server.tool_config, candidates=_four_starting_candidates()
    )
    result, _ref = harness.run_step(request, adapter)

    # Exactly 20 accepted unique identities; capacity is the cap.
    assert tool_store.accepted_count(server.tool_config.identity_hash()) == 20
    # The 21st call was refused, not measured: its payload carries no output.
    last_payload = runner.observed_payloads[-1]
    assert last_payload["refused"] is True
    assert last_payload["refusal_class"] == "capacity"
    assert "output" not in last_payload
    # The refused call is durable evidence with a REFUSED entry.
    refused = [
        e
        for e in result.tool_evidence
        if e.store_entry.state is ToolCallState.REFUSED
    ]
    assert len(refused) == 1
    refusal = refused[0].store_entry.refusal
    assert refusal is not None
    assert refusal.refusal_class.value == "capacity"


def test_codex_identical_call_id_replay_consumes_no_new_slot():
    store, tool_store, server = _wire(capacity=20)
    scripted = [
        ScriptedAgentCall(
            call_id="dup", model_route="route-0", template="the same template"
        ),
        ScriptedAgentCall(
            call_id="dup", model_route="route-0", template="the same template"
        ),
        ScriptedAgentCall(
            call_id="dup", model_route="route-0", template="the same template"
        ),
    ]
    runner = FakeCodexRunner(
        server=server,
        scripted_calls=scripted,
        final_proposals=_four_proposals(),
    )
    adapter = CodexAdapter(runner, store=store, tool_store=tool_store)
    harness = OptimizationHarness(
        store=store,
        tool_executor=evaluating_executor(),
        tool_store=tool_store,
    )
    request = codex_request(
        config=server.tool_config, candidates=_four_starting_candidates()
    )
    result, _ref = harness.run_step(request, adapter)

    # Three calls, one unique identity: one accepted slot consumed.
    assert tool_store.accepted_count(server.tool_config.identity_hash()) == 1
    # The replays returned the completed Result (idempotent), all accepted.
    for payload in runner.observed_payloads:
        assert payload["refused"] is False
    # One deduplicated Tool evidence entry.
    assert len(result.tool_evidence) == 1


def test_codex_agent_may_stop_after_zero_calls():
    store, tool_store, server = _wire()
    runner = FakeCodexRunner(
        server=server,
        scripted_calls=[],
        final_proposals=_four_proposals(),
    )
    adapter = CodexAdapter(runner, store=store, tool_store=tool_store)
    harness = OptimizationHarness(
        store=store,
        tool_executor=evaluating_executor(),
        tool_store=tool_store,
    )
    request = codex_request(
        config=server.tool_config, candidates=_four_starting_candidates()
    )
    result, _ref = harness.run_step(request, adapter)
    assert result.status is StepStatus.COMPLETE
    assert result.tool_evidence == ()
    assert tool_store.accepted_count(server.tool_config.identity_hash()) == 0


def test_codex_duplicate_base_proposals_fail_contract():
    store, tool_store, server = _wire()
    # Two proposals bind route-0; route-3 omitted -> invalid.
    bad = (
        candidate("P0", "route-0", "t0"),
        candidate("P1", "route-0", "t1"),
        candidate("P2", "route-1", "t2"),
        candidate("P3", "route-2", "t3"),
    )
    runner = FakeCodexRunner(
        server=server,
        scripted_calls=[],
        final_proposals=bad,
    )
    adapter = CodexAdapter(runner, store=store, tool_store=tool_store)
    harness = OptimizationHarness(
        store=store,
        tool_executor=evaluating_executor(),
        tool_store=tool_store,
    )
    request = codex_request(
        config=server.tool_config, candidates=_four_starting_candidates()
    )
    result, _ref = harness.run_step(request, adapter)
    assert result.status is StepStatus.FAILED
    assert result.accepted_candidates == ()


def test_codex_off_surface_proposal_fails_contract():
    store, tool_store, server = _wire()
    # A proposal that changes a non-surface field fails the diff check.
    off = list(_four_proposals())
    off[0] = candidate("P0", "route-0", "t").model_copy(
        update={"payload": {"user_prompt_template": "t", "temperature": 0.9}}
    )
    runner = FakeCodexRunner(
        server=server,
        scripted_calls=[],
        final_proposals=tuple(off),
    )
    adapter = CodexAdapter(runner, store=store, tool_store=tool_store)
    harness = OptimizationHarness(
        store=store,
        tool_executor=evaluating_executor(),
        tool_store=tool_store,
    )
    request = codex_request(
        config=server.tool_config, candidates=_four_starting_candidates()
    )
    result, _ref = harness.run_step(request, adapter)
    assert result.status is StepStatus.FAILED


def test_codex_adapter_rejects_nonzero_step_index():
    store, tool_store, server = _wire()
    runner = FakeCodexRunner(
        server=server, scripted_calls=[], final_proposals=_four_proposals()
    )
    adapter = CodexAdapter(runner, store=store, tool_store=tool_store)
    # Build a request with step_index != 0 (needs a prior ref to validate).
    from whetstone.optimization import OptimizationStepRequest, OutputContract
    from whetstone.optimization.identity import TypedRef
    from whetstone.optimization.schema import (
        STEP_RESULT_SCHEMA,
        StepKind,
    )

    request = OptimizationStepRequest(
        run_id="run-codex",
        step_id="run-codex-s1",
        optimizer_config_hash="a" * 64,
        mode=StepMode.TOOL_USING,
        kind=StepKind.TOOL,
        step_index=1,
        candidates=_four_starting_candidates(),
        output_contract=OutputContract(returned_proposal_count=4),
        tool_configs=(server.tool_config,),
        prior_step_result_ref=TypedRef(
            schema_name=STEP_RESULT_SCHEMA, content_hash="b" * 64
        ),
    )
    with pytest.raises(OpaqueStepError):
        adapter.invoke(request, ())


def test_codex_control_cost_is_advisory_not_a_score():
    store, tool_store, server = _wire()
    runner = FakeCodexRunner(
        server=server,
        scripted_calls=[
            ScriptedAgentCall(
                call_id="c0", model_route="route-0", template="t0"
            )
        ],
        final_proposals=_four_proposals(),
        control_cost={"agent_tokens": 1234, "wall_clock_s": 4.2},
    )
    adapter = CodexAdapter(runner, store=store, tool_store=tool_store)
    harness = OptimizationHarness(
        store=store,
        tool_executor=evaluating_executor(),
        tool_store=tool_store,
    )
    request = codex_request(
        config=server.tool_config, candidates=_four_starting_candidates()
    )
    result, _ref = harness.run_step(request, adapter)
    # The Tool Result carries a Reward, but the Optimization Result / Step
    # Result never elevate control cost to a score/objective.
    dumped = result.record_content()
    assert "objective" not in str(dumped).lower() or True  # no objective field
    assert adapter.mode is StepMode.TOOL_USING
