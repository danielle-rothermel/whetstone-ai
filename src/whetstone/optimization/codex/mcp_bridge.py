from __future__ import annotations

import json
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
    """Expose one admitted evaluation tool over JSON-RPC."""

    def __init__(self, *, handle: RuntimeToolHandle) -> None:
        self.tool_config = handle.config
        self._handle = handle

    @property
    def handle(self) -> RuntimeToolHandle:
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


def _protocol_error(code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": None,
        "error": {"code": code, "message": message},
    }


def serve_stdio(
    server: EvaluateCandidateServer, *, stdin: TextIO, stdout: TextIO
) -> None:
    """Return protocol errors for invalid JSON and non-object requests."""
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
    "McpError",
    "serve_stdio",
    "tool_result_to_mcp_content",
]
