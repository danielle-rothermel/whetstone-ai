from __future__ import annotations

import json
from collections.abc import Callable
from io import StringIO
from typing import Any, Protocol

from whetstone.optimization.codex.mcp_bridge import (
    MCP_PROTOCOL_VERSION,
    EvaluateCandidateServer,
    McpError,
    serve_stdio,
)


class JsonRpcProcess(Protocol):
    def exchange(self, raw: str) -> str | None: ...


class JsonRpcClient:
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
    def __init__(self, server: EvaluateCandidateServer) -> None:
        self._server = server

    def exchange(self, raw: str) -> str | None:
        stdin = StringIO(raw + "\n")
        stdout = StringIO()
        serve_stdio(self._server, stdin=stdin, stdout=stdout)
        response = stdout.getvalue().strip()
        return response or None
