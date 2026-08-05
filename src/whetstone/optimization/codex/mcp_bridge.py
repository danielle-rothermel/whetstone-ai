"""Minimal actual JSON-RPC bridge for the external Codex MCP step."""

from __future__ import annotations

import json
from collections.abc import Callable
from io import StringIO
from typing import Any, TextIO

from whetstone.core.identity import ImmutableJsonObject, NonEmptyId
from whetstone.optimization.tools.contracts import (
    RuntimeToolHandle,
    ToolCall,
    ToolResult,
)

MCP_PROTOCOL_VERSION = "2024-11-05"


class McpError(RuntimeError):
    def __init__(self, code: int, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")


def tool_result_to_mcp_content(result: ToolResult) -> dict[str, Any]:
    if result.refusal is not None:
        payload = {
            "refused": True,
            "call_id": str(result.call_id),
            "refusal_class": result.refusal.refusal_class.value,
            "reason": str(result.refusal.reason),
        }
        return {
            "isError": True,
            "content": [{"type": "text", "text": json.dumps(payload)}],
        }
    payload = {
        "refused": False,
        "call_id": str(result.call_id),
        "output": None if result.output is None else result.output.to_json(),
        "reward": (
            None
            if result.reward is None
            else result.reward.model_dump(mode="json")
        ),
    }
    return {
        "isError": False,
        "content": [{"type": "text", "text": json.dumps(payload)}],
        "structuredContent": payload,
    }


class EvaluateCandidateServer:
    """Expose exactly one externally modeled evaluation tool."""

    def __init__(self, *, handle: RuntimeToolHandle) -> None:
        self.tool_config = handle.config
        self._handle = handle

    @property
    def handle(self) -> RuntimeToolHandle:
        """The exact admitted Tool Handle this server dispatches through."""
        return self._handle

    def handle_request(self, message: dict[str, Any]) -> dict[str, Any] | None:
        request_id = message.get("id")
        try:
            method = message.get("method")
            if method == "initialize":
                result = {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "whetstone", "version": "1"},
                }
            elif method == "notifications/initialized":
                return None
            elif method == "tools/list":
                result = {"tools": [self._tool_definition()]}
            elif method == "tools/call":
                result = self._call(message.get("params") or {})
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

    def _tool_definition(self) -> dict[str, Any]:
        return {
            "name": self.tool_config.tool_name,
            "description": (
                "Evaluate a candidate using Whetstone's canonical engine."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "call_id": {"type": "string"},
                    "base_ref": {
                        "type": "object",
                        "properties": {
                            "schema_name": {"type": "string"},
                            "content_hash": {"type": "string"},
                        },
                        "required": ["schema_name", "content_hash"],
                    },
                    "model_route": {"type": "string"},
                    "template": {"type": "string"},
                },
                "required": [
                    "call_id",
                    "base_ref",
                    "model_route",
                    "template",
                ],
            },
        }

    def _call(self, params: dict[str, Any]) -> dict[str, Any]:
        if params.get("name") != self.tool_config.tool_name:
            raise McpError(-32602, "unknown tool")
        arguments = params.get("arguments")
        if not isinstance(arguments, dict):
            raise McpError(-32602, "tool arguments must be an object")
        call_id = arguments.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            raise McpError(-32602, "call_id must be non-empty")
        base_ref = arguments.get("base_ref")
        if not isinstance(base_ref, dict):
            raise McpError(-32602, "base_ref must be a typed reference object")
        try:
            args = ImmutableJsonObject(
                {
                    "base_ref": base_ref,
                    "model_route": arguments.get("model_route", ""),
                    "template": arguments.get("template", ""),
                }
            )
        except ValueError as exc:
            raise McpError(-32602, str(exc)) from exc
        call = ToolCall(
            call_id=NonEmptyId(call_id),
            tool_config=self._handle.tool_config_ref,
            capacity_binding=self._handle.binding,
            args=args,
        )
        return tool_result_to_mcp_content(self._handle(call))


class JsonRpcClient:
    """MCP client over an injected line-oriented process boundary."""

    def __init__(
        self,
        exchange: Callable[[str], str | None],
        *,
        tool_name: str,
    ) -> None:
        self._exchange = exchange
        self._tool_name = tool_name
        self._next_id = 0

    @property
    def tool_name(self) -> str:
        """The exact server-advertised tool name this client calls."""
        return self._tool_name

    def _send(
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
        raw = self._exchange(json.dumps(message))
        if raw is None:
            raise McpError(-32603, "MCP process returned no response")
        response = json.loads(raw)
        if "error" in response:
            error = response["error"]
            raise McpError(int(error["code"]), str(error["message"]))
        return response["result"]

    def initialize(self) -> None:
        self._send(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "codex", "version": "1"},
            },
        )
        self._exchange(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                }
            )
        )

    def list_tools(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._send("tools/list")["tools"])

    def evaluate(
        self,
        *,
        call_id: str,
        base_ref: dict[str, Any],
        model_route: str,
        template: str,
    ) -> dict[str, Any]:
        result = self._send(
            "tools/call",
            {
                "name": self._tool_name,
                "arguments": {
                    "call_id": call_id,
                    "base_ref": base_ref,
                    "model_route": model_route,
                    "template": template,
                },
            },
        )
        return json.loads(result["content"][0]["text"])


class InProcessMcpProcess:
    """Fake process boundary through the actual stdio JSON-RPC server."""

    def __init__(self, server: EvaluateCandidateServer) -> None:
        self._server = server

    def exchange(self, raw: str) -> str | None:
        stdin = StringIO(raw + "\n")
        stdout = StringIO()
        serve_stdio(self._server, stdin=stdin, stdout=stdout)
        response = stdout.getvalue().strip()
        return response or None


def _protocol_error(code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": None,
        "error": {"code": code, "message": message},
    }


def serve_stdio(
    server: EvaluateCandidateServer, *, stdin: TextIO, stdout: TextIO
) -> None:
    """Serve one request per line; malformed lines never end the session."""
    for raw in stdin:
        if not raw.strip():
            continue
        response: dict[str, Any] | None
        try:
            message = json.loads(raw)
        except json.JSONDecodeError as exc:
            response = _protocol_error(-32700, f"parse error: {exc.msg}")
        else:
            if isinstance(message, dict):
                response = server.handle_request(message)
            else:
                response = _protocol_error(
                    -32600, "request must be a JSON object"
                )
        if response is not None:
            stdout.write(json.dumps(response) + "\n")
            stdout.flush()


__all__ = [
    "MCP_PROTOCOL_VERSION",
    "EvaluateCandidateServer",
    "InProcessMcpProcess",
    "JsonRpcClient",
    "McpError",
    "serve_stdio",
    "tool_result_to_mcp_content",
]
