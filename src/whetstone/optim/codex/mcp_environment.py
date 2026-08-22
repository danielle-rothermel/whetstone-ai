"""Persisted environment keys for the Codex MCP evaluation server.

whetstone hosts the evaluation server itself and configures it through
these variables. They never reach the Codex process, which is given only
the endpoint URL and a bearer token. The server reads them back by name,
so the literal spellings are a persisted format with a golden test rather
than incidental strings.
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
    #: Run-scoped token the adapter mints per Step. The agent echoes only
    #: its hash through the output artifact, which shows the artifact came
    #: from a process that received this Step's prompt.
    RUN_LEASE_TOKEN = "WS_MCP_RUN_LEASE_TOKEN"
    #: Digest binding RUN_LEASE_TOKEN to this run's exact Tool Config and
    #: capacity binding. The server recomputes it from its own
    #: configuration and refuses to start when it disagrees, so a token
    #: minted for another run cannot bring up a server for this one.
    RUN_LEASE_BINDING = "WS_MCP_RUN_LEASE_BINDING"


__all__ = ["McpEnvironmentKey"]
