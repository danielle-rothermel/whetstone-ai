from __future__ import annotations

import os
import sys

from dr_store import ObjectStore, SqliteBackend

from whetstone.core.effects.authority import (
    EffectAuthority,
    ReplayPolicy,
)
from whetstone.experiment.reward import RewardPolicy
from whetstone.optimization.codex.mcp_bridge import (
    EvaluateCandidateServer,
    serve_stdio,
)
from whetstone.optimization.codex.mcp_environment import McpEnvironmentKey
from whetstone.optimization.codex.runtime import EvaluationRuntimeConfig
from whetstone.optimization.tools.contracts import (
    ToolCapacityBinding,
    ToolConfig,
)
from whetstone.optimization.tools.evaluator import EngineToolEvaluator
from whetstone.optimization.tools.execution import EvaluatingToolExecutor
from whetstone.optimization.tools.facade import (
    ToolAdmissionAuthority,
    ToolCallStore,
)


def build_server_from_env(
    environ: dict[str, str] | None = None,
) -> EvaluateCandidateServer:
    env = environ if environ is not None else dict(os.environ)
    sqlite_path = env[McpEnvironmentKey.SQLITE_PATH]
    store = ObjectStore(SqliteBackend(sqlite_path))
    tool_config = ToolConfig.model_validate_json(
        env[McpEnvironmentKey.TOOL_CONFIG]
    )
    binding = ToolCapacityBinding.model_validate_json(
        env[McpEnvironmentKey.CAPACITY_BINDING]
    )
    runtime = EvaluationRuntimeConfig.model_validate_json(
        env[McpEnvironmentKey.RUNTIME_CONFIG]
    )
    reward_policy = RewardPolicy.model_validate_json(
        env[McpEnvironmentKey.REWARD_POLICY]
    )
    engine = runtime.build_engine(store)
    if reward_policy.identity_hash() != tool_config.reward_policy_hash:
        raise ValueError("MCP reward policy does not match Tool Config")
    effect_authority = EffectAuthority.sqlite(sqlite_path)
    tool_store = ToolCallStore(
        store,
        ToolAdmissionAuthority.sqlite(sqlite_path),
        effect_authority,
    )
    executor = EvaluatingToolExecutor(
        EngineToolEvaluator(engine),
        reward_policy,
        effect_authority,
        owner_id=f"whetstone-mcp-{os.getpid()}",
        replay_policy=(
            ReplayPolicy.IDEMPOTENT
            if tool_config.idempotent_replay
            else ReplayPolicy.NO_REDRIVE
        ),
    )
    return EvaluateCandidateServer(
        handle=executor.runtime_handle(tool_config, tool_store, binding)
    )


def main() -> None:
    serve_stdio(build_server_from_env(), stdin=sys.stdin, stdout=sys.stdout)


if __name__ == "__main__":
    main()


__all__ = ["build_server_from_env", "main"]
