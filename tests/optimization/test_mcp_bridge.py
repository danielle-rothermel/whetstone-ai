"""MCP bridge tests: the whetstone-owned evaluate_candidate server + client.

Exercises the JSON-RPC handshake, the tool exposure, capacity enforcement over
the bridge, idempotent replay, and (critically) that a refusal returned across
the bridge never carries a measurement.
"""

from __future__ import annotations

import pytest

from tests.optimization.tool_support import make_store, mcp_server
from whetstone.optimization import (
    RefusalClass,
    ToolCallState,
    ToolCallStore,
)
from whetstone.optimization.mcp_bridge import (
    MCP_PROTOCOL_VERSION,
    McpError,
    ScriptedMcpClient,
)


def _server(*, capacity: int = 20):
    store = make_store()
    tool_store = ToolCallStore(store)
    return tool_store, mcp_server(tool_store, capacity=capacity)


def test_handshake_and_tools_list_exposes_single_tool():
    _tool_store, server = _server()
    client = ScriptedMcpClient.for_server(server)
    init = client.initialize()
    assert init["protocolVersion"] == MCP_PROTOCOL_VERSION
    tools = client.list_tools()
    assert len(tools) == 1
    assert tools[0]["name"] == "evaluate_candidate"
    assert set(tools[0]["inputSchema"]["required"]) == {
        "call_id",
        "model_route",
        "template",
    }


def test_evaluate_call_returns_reward_and_rollout_refs():
    _tool_store, server = _server()
    client = ScriptedMcpClient.for_server(server)
    client.initialize()
    payload = client.call_evaluate(
        call_id="c0", model_route="route-0", template="a precise template"
    )
    assert payload["refused"] is False
    assert payload["output"]["rollout_ref_count"] == 20
    assert payload["reward"]["reward_name"] == "reward"
    # The Reward cites the internal Evaluation Role, never official.
    assert payload["reward"]["evidence_role"] == "internal"


def test_capacity_refusal_over_bridge_carries_no_measurement():
    tool_store, server = _server(capacity=1)
    client = ScriptedMcpClient.for_server(server)
    client.initialize()
    ok = client.call_evaluate(
        call_id="c0", model_route="route-0", template="t0"
    )
    assert ok["refused"] is False
    refused = client.call_evaluate(
        call_id="c1", model_route="route-0", template="t1"
    )
    # The refusal names its class and carries NO output and NO reward.
    assert refused["refused"] is True
    assert refused["refusal_class"] == RefusalClass.CAPACITY.value
    assert "output" not in refused
    assert "reward" not in refused
    # Capacity was debited exactly once.
    assert tool_store.accepted_count(server.tool_config.identity_hash()) == 1


def test_identical_call_id_replay_is_idempotent_over_bridge():
    tool_store, server = _server(capacity=5)
    client = ScriptedMcpClient.for_server(server)
    client.initialize()
    first = client.call_evaluate(
        call_id="dup", model_route="route-0", template="same"
    )
    second = client.call_evaluate(
        call_id="dup", model_route="route-0", template="same"
    )
    assert first["output"] == second["output"]
    # Only one slot consumed by the identical identity.
    assert tool_store.accepted_count(server.tool_config.identity_hash()) == 1
    # The completed entry is durable.
    entry = tool_store.get(server.tool_config.identity_hash(), "dup")
    assert entry is not None
    assert entry.state is ToolCallState.COMPLETED


def test_validation_refusal_over_bridge_has_no_measurement():
    tool_store, server = _server(capacity=5)
    client = ScriptedMcpClient.for_server(server)
    client.initialize()
    # Empty template is a VALIDATION refusal before any measurement.
    refused = client.call_evaluate(
        call_id="bad", model_route="route-0", template=""
    )
    assert refused["refused"] is True
    assert refused["refusal_class"] == RefusalClass.VALIDATION.value
    assert "output" not in refused
    # A validation refusal debits NO capacity.
    assert tool_store.accepted_count(server.tool_config.identity_hash()) == 0
    entry = tool_store.get(server.tool_config.identity_hash(), "bad")
    assert entry is not None
    assert entry.state is ToolCallState.REFUSED


def test_unknown_method_returns_protocol_error():
    _tool_store, server = _server()
    client = ScriptedMcpClient.for_server(server)
    with pytest.raises(McpError):
        client._request("does/not/exist")


def test_tools_call_requires_call_id():
    _tool_store, server = _server()
    response = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "evaluate_candidate",
                "arguments": {"model_route": "route-0", "template": "t"},
            },
        }
    )
    assert "error" in response
