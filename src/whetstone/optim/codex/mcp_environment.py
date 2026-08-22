"""Persisted environment keys for the Codex MCP evaluation server.

The Codex CLI receives these as ``mcp_servers.whetstone.env.*`` entries and
the server reads them back by name, so the literal spellings are a
persisted format with a golden test rather than incidental strings.
"""

from enum import UNIQUE, StrEnum, verify


@verify(UNIQUE)
class McpEnvironmentKey(StrEnum):
    SQLITE_PATH = "WS_MCP_SQLITE_PATH"
    TOOL_CONFIG = "WS_MCP_TOOL_CONFIG"
    CAPACITY_BINDING = "WS_MCP_CAPACITY_BINDING"
    RUNTIME_CONFIG = "WS_MCP_RUNTIME_CONFIG"
    RUNTIME_CONFIG_CLASS = "WS_MCP_RUNTIME_CONFIG_CLASS"
    REWARD_POLICY = "WS_MCP_REWARD_POLICY"
    #: Run-scoped token the adapter mints per Step. The server echoes only
    #: its hash through the output artifact, so a stale or foreign server
    #: process cannot pass off an artifact for this run.
    RUN_LEASE_TOKEN = "WS_MCP_RUN_LEASE_TOKEN"


__all__ = ["McpEnvironmentKey"]
