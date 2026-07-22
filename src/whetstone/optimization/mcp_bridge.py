"""The whetstone-owned MCP bridge for the Codex agent run.

The Codex CLI is a process we do not own. The *bridge*
(``codex-agent-run.html`` #local-bridge) is the transport that exposes
Whetstone's toolkit across that process boundary: a whetstone-owned MCP server,
launched by ``codex exec``, exposing exactly one tool --
``evaluate_candidate``.
Codex issues MCP ``tools/call`` requests; the server routes each to the
authoritative Tool Call Store and the internal-role
:class:`~whetstone.optimization.tool_eval.ToolEvaluator`, and returns the Tool
Result across the boundary. The ``max_evaluation_calls`` cap is enforced by
Tool
Capacity + the Tool Call Store, restored across a bridge restart, and
identical-``call_id`` replay returns the completed Result without spending
another slot.

This module implements a minimal, dependency-free MCP server over stdio
(newline-delimited JSON-RPC 2.0) sufficient for the ``initialize`` /
``tools/list`` / ``tools/call`` handshake Codex performs, plus a scripted MCP
*client* (:class:`ScriptedMcpClient`) that speaks the same protocol so the
FAKE-codex test double can drive the server deterministically with no real CLI.

The server's non-serializable
:class:`~whetstone.optimization.tools.RuntimeToolHandle` is constructed from
the
serialized Tool Config only at the execution boundary, exactly as an in-process
tool-using Step does -- the boundary is a process boundary here, but nothing
about the Tool Config, Tool Call Store key ``(tool_config_hash, call_id)``,
capacity accounting, or refusal semantics changes.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, TextIO

from whetstone.optimization.tool_eval import EvaluatingToolExecutor
from whetstone.optimization.tool_store import ToolCallStore
from whetstone.optimization.tools import (
    RuntimeToolHandle,
    ToolCall,
    ToolConfig,
    ToolResult,
)

__all__ = [
    "MCP_PROTOCOL_VERSION",
    "EvaluateCandidateServer",
    "McpError",
    "ScriptedMcpClient",
    "serve_stdio",
    "tool_result_to_mcp_content",
]

MCP_PROTOCOL_VERSION = "2024-11-05"


class McpError(Exception):
    """A JSON-RPC/MCP protocol-level error surfaced to the client."""

    def __init__(self, code: int, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")


def tool_result_to_mcp_content(result: ToolResult) -> dict[str, Any]:
    """Render a :class:`ToolResult` as an MCP ``tools/call`` result payload.

    A refused result is returned as an ``isError`` MCP result whose text names
    the typed refusal class and reason, so the agent sees a refusal as a
    non-execution outcome -- never as a measured Tool Result. An accepted,
    completed result returns its typed output + Reward as the structured
    content the agent reads back.
    """
    if result.refusal is not None:
        return {
            "isError": True,
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "refused": True,
                            "refusal_class": (
                                result.refusal.refusal_class.value
                            ),
                            "reason": result.refusal.reason,
                            "call_id": result.call_id,
                        }
                    ),
                }
            ],
        }
    payload = {
        "refused": False,
        "call_id": result.call_id,
        "output": result.output,
        "reward": result.reward,
    }
    return {
        "isError": False,
        "content": [{"type": "text", "text": json.dumps(payload)}],
        "structuredContent": payload,
    }


class EvaluateCandidateServer:
    """The whetstone-owned MCP server exposing the single evaluate tool.

    It owns the serialized :class:`ToolConfig`, the authoritative
    :class:`ToolCallStore` (over a shared durable backend so capacity survives
    a
    bridge restart), and an :class:`EvaluatingToolExecutor` that constructs the
    :class:`RuntimeToolHandle` at the execution boundary. Each MCP
    ``tools/call`` for ``evaluate_candidate`` becomes one Tool Call through
    that
    handle; the terminal Tool Call Store Entry and Tool Result are the durable
    evidence the whetstone process reads back after Codex exits.
    """

    def __init__(
        self,
        *,
        tool_config: ToolConfig,
        store: ToolCallStore,
        executor: EvaluatingToolExecutor,
    ) -> None:
        self._config = tool_config
        self._store = store
        self._executor = executor
        # The Runtime Tool Handle is constructed at the execution boundary and
        # never serialized; here that boundary is the server's own process.
        self._handle: RuntimeToolHandle = executor.runtime_handle(
            tool_config, store
        )
        # Tool Results the server produced this session (evidence for tests).
        self.produced_results: list[ToolResult] = []

    @property
    def tool_config(self) -> ToolConfig:
        return self._config

    # -- protocol dispatch ---------------------------------------------------

    def handle_request(
        self, message: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Dispatch one JSON-RPC request; return a response (or None for a
        notification)."""
        method = message.get("method")
        request_id = message.get("id")
        try:
            if method == "initialize":
                result = self._initialize()
            elif method == "notifications/initialized":
                return None
            elif method == "tools/list":
                result = self._tools_list()
            elif method == "tools/call":
                result = self._tools_call(message.get("params") or {})
            elif method == "ping":
                result = {}
            else:
                raise McpError(-32601, f"method not found: {method!r}")
        except McpError as exc:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": exc.code, "message": exc.message},
            }
        if request_id is None:
            return None
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    def _initialize(self) -> dict[str, Any]:
        return {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {
                "name": "whetstone-evaluate-candidate",
                "version": "1",
            },
        }

    def _tools_list(self) -> dict[str, Any]:
        return {
            "tools": [
                {
                    "name": self._config.tool_name,
                    "description": (
                        "Evaluate one encoder template on one Model Route "
                        "under the internal-role Eval Config and return its "
                        "internal Reward and Rollout references."
                    ),
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "call_id": {"type": "string"},
                            "model_route": {"type": "string"},
                            "template": {"type": "string"},
                        },
                        "required": ["call_id", "model_route", "template"],
                    },
                }
            ]
        }

    def _tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        if name != self._config.tool_name:
            raise McpError(-32602, f"unknown tool: {name!r}")
        arguments = params.get("arguments") or {}
        call_id = arguments.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            raise McpError(-32602, "evaluate_candidate requires a call_id")
        call = ToolCall(
            call_id=call_id,
            tool_config_hash=self._config.identity_hash(),
            store_namespace=self._config.store_namespace,
            args={
                "model_route": arguments.get("model_route", ""),
                "template": arguments.get("template", ""),
            },
        )
        result = self._handle(call)
        self.produced_results.append(result)
        # Accepted results are transitioned to completed in the store here (the
        # bridge owns completion), so the terminal Entry is durable evidence
        # and an identical-call_id replay returns the completed Result.
        if result.refusal is None:
            self._store.complete(result.tool_config_hash, result)
        return tool_result_to_mcp_content(result)


class ScriptedMcpClient:
    """A deterministic MCP client that speaks the stdio JSON-RPC protocol.

    It is the transport half of the FAKE-codex test double: given a server (or
    a subprocess speaking the same protocol), it performs the ``initialize`` /
    ``tools/list`` / ``tools/call`` handshake exactly as a real MCP client
    (Codex) would, with no real CLI and no network. Tests script the sequence
    of ``evaluate_candidate`` calls and read the structured results.
    """

    def __init__(
        self, send: Callable[[dict[str, Any]], dict[str, Any] | None]
    ) -> None:
        self._send = send
        self._next_id = 0

    @classmethod
    def for_server(
        cls, server: EvaluateCandidateServer
    ) -> ScriptedMcpClient:
        """Bind the client directly to an in-process server (no subprocess)."""
        return cls(server.handle_request)

    def _request(
        self, method: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        self._next_id += 1
        message: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": self._next_id,
            "method": method,
        }
        if params is not None:
            message["params"] = params
        response = self._send(message)
        if response is None:
            raise McpError(-32603, f"no response for {method!r}")
        if "error" in response:
            error = response["error"]
            raise McpError(error["code"], error["message"])
        return response["result"]

    def initialize(self) -> dict[str, Any]:
        result = self._request(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "fake-codex", "version": "1"},
            },
        )
        # A real client sends the initialized notification (no response).
        self._send(
            {"jsonrpc": "2.0", "method": "notifications/initialized"}
        )
        return result

    def list_tools(self) -> list[dict[str, Any]]:
        return list(self._request("tools/list")["tools"])

    def call_evaluate(
        self, *, call_id: str, model_route: str, template: str
    ) -> dict[str, Any]:
        """Call ``evaluate_candidate`` once and return the parsed payload.

        The returned dict has ``refused`` plus, when accepted, ``output`` and
        ``reward``; a refusal carries ``refusal_class`` and ``reason`` and
        never an ``output`` -- the agent cannot mistake a refusal for a
        measurement.
        """
        result = self._request(
            "tools/call",
            {
                "name": "evaluate_candidate",
                "arguments": {
                    "call_id": call_id,
                    "model_route": model_route,
                    "template": template,
                },
            },
        )
        content = result["content"][0]["text"]
        return json.loads(content)


def serve_stdio(
    server: EvaluateCandidateServer,
    *,
    stdin: TextIO,
    stdout: TextIO,
) -> None:
    """Run the MCP server over newline-delimited JSON-RPC on stdio.

    This is the entry the launched subprocess runs: read one JSON message per
    line, dispatch it, and write one JSON response line (notifications produce
    no line). It returns on EOF.
    """
    for raw in stdin:
        line = raw.strip()
        if not line:
            continue
        message = json.loads(line)
        response = server.handle_request(message)
        if response is not None:
            stdout.write(json.dumps(response) + "\n")
            stdout.flush()
