from __future__ import annotations

import os

from dr_serialize import decode_strict_json_bytes
from dr_store.sync import close_persistent, persistent_sqlite

from whetstone.core.effects.authority import (
    EffectAuthority,
    ReplayPolicy,
)
from whetstone.experiment.reward import RewardPolicy
from whetstone.optim.codex.mcp_bridge import (
    EvaluateCandidateServer,
)
from whetstone.optim.codex.mcp_environment import McpEnvironmentKey
from whetstone.optim.codex.config import load_runtime_config
from whetstone.optim.tools.contracts import (
    ToolCapacityBinding,
    ToolConfig,
)
from whetstone.optim.tools.evaluator import EngineToolEvaluator
from whetstone.optim.tools.execution import EvaluatingToolExecutor
from whetstone.optim.tools.facade import (
    ToolAdmissionAuthority,
    ToolCallStore,
)


def build_server_from_env(
    environ: dict[str, str] | None = None,
) -> EvaluateCandidateServer:
    env = environ if environ is not None else dict(os.environ)
    sqlite_path = env[McpEnvironmentKey.SQLITE_PATH]
    store = persistent_sqlite(sqlite_path)
    tool_config_raw = _strict_env_json(env[McpEnvironmentKey.TOOL_CONFIG])
    tool_config = ToolConfig.model_validate_json(tool_config_raw)
    binding_raw = _strict_env_json(env[McpEnvironmentKey.CAPACITY_BINDING])
    binding = ToolCapacityBinding.model_validate_json(binding_raw)
    runtime_raw = _strict_env_json(env[McpEnvironmentKey.RUNTIME_CONFIG])
    runtime_class = env[McpEnvironmentKey.RUNTIME_CONFIG_CLASS]
    runtime = load_runtime_config(class_path=runtime_class, raw=runtime_raw)
    reward_policy_raw = _strict_env_json(env[McpEnvironmentKey.REWARD_POLICY])
    reward_policy = RewardPolicy.model_validate_json(reward_policy_raw)
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


def _strict_env_json(raw: str) -> bytes:
    encoded = raw.encode()
    decode_strict_json_bytes(
        encoded,
        max_bytes=len(encoded),
        max_depth=len(encoded),
    )
    return encoded


def main() -> None:
    sqlite_path = os.environ[McpEnvironmentKey.SQLITE_PATH]
    try:
        build_server_from_env().run(transport="stdio")
    finally:
        close_persistent(sqlite_path)


if __name__ == "__main__":
    main()


__all__ = ["build_server_from_env", "main"]
