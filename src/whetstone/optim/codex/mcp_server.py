from __future__ import annotations

import os
import sys
from hmac import compare_digest

from dr_serialize import decode_strict_json_bytes
from dr_store.sync import close_persistent, persistent_sqlite

from whetstone.core.leasing import (
    EffectLeaseAuthority,
    ReplayPolicy,
)
from whetstone.experiment.reward import RewardPolicy
from whetstone.optim.codex.adapter import codex_run_lease_binding
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
    # Bind this server process to this run. A non-empty token proves only
    # that whoever started the server held some token; the binding digest
    # is over the token *and* this server's own Tool Config and capacity
    # binding, so a token minted for another run -- or replayed at a
    # server configured for a different one -- does not verify and the
    # server refuses to start rather than admitting that run's calls.
    token = env.get(McpEnvironmentKey.RUN_LEASE_TOKEN)
    if not token:
        raise ValueError("MCP server requires the run lease token")
    presented = env.get(McpEnvironmentKey.RUN_LEASE_BINDING)
    if not presented:
        raise ValueError("MCP server requires the run lease binding")
    expected = codex_run_lease_binding(
        token=token,
        store_namespace_key=str(tool_config.store_namespace_key),
        tool_config_hash=str(tool_config.identity_hash()),
        capacity_scope=binding.scope.value,
        capacity_subject=_capacity_subject_key(binding),
    )
    if not compare_digest(presented, expected):
        raise ValueError(
            "MCP server run lease token is not bound to this run's exact "
            "Tool Config and capacity binding"
        )
    effect_authority = EffectLeaseAuthority.sqlite(sqlite_path)
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


def _capacity_subject_key(binding: ToolCapacityBinding) -> str:
    subject = binding.subject_ref
    if subject is None:
        return ""
    return f"{subject.schema_name}@{subject.content_hash}"


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
        # Suppress close failures so they cannot replace an exception
        # from the server run itself.
        try:
            close_persistent(sqlite_path)
        except Exception as exc:  # noqa: BLE001 - shutdown best effort
            print(f"store close failed during shutdown: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()


__all__ = ["build_server_from_env", "main"]
