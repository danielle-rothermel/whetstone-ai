"""Stdio MCP child reconstructed from serialized canonical runtime config."""

from __future__ import annotations

import os
import sys

from dr_store import ObjectStore, SqliteBackend

from whetstone.evaluation.tool import EngineToolEvaluator
from whetstone.execution.mode import EvaluationRuntimeConfig
from whetstone.optimization.mcp_bridge import (
    EvaluateCandidateServer,
    serve_stdio,
)
from whetstone.optimization.reward import RewardPolicy
from whetstone.optimization.tool_eval import EvaluatingToolExecutor
from whetstone.optimization.tool_store import ToolCallStore
from whetstone.optimization.tools import ToolConfig


def build_server_from_env(
    environ: dict[str, str] | None = None,
) -> EvaluateCandidateServer:
    env = environ if environ is not None else dict(os.environ)
    store = ObjectStore(SqliteBackend(env["WS_MCP_SQLITE_PATH"]))
    tool_config = ToolConfig.model_validate_json(env["WS_MCP_TOOL_CONFIG"])
    runtime = EvaluationRuntimeConfig.model_validate_json(
        env["WS_MCP_RUNTIME_CONFIG"]
    )
    reward_policy = RewardPolicy.model_validate_json(
        env["WS_MCP_REWARD_POLICY"]
    )
    engine = runtime.build_engine(store)
    if reward_policy.identity_hash() != tool_config.reward_policy_ref:
        raise ValueError("MCP reward policy does not match Tool Config")
    return EvaluateCandidateServer(
        tool_config=tool_config,
        store=ToolCallStore(store),
        executor=EvaluatingToolExecutor(
            EngineToolEvaluator(engine), reward_policy
        ),
    )


def main() -> None:
    serve_stdio(build_server_from_env(), stdin=sys.stdin, stdout=sys.stdout)


if __name__ == "__main__":
    main()


__all__ = ["build_server_from_env", "main"]
