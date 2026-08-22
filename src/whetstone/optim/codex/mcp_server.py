from __future__ import annotations

import os
from hmac import compare_digest

from dr_serialize import decode_strict_json_bytes
from dr_store.sync import persistent_sqlite

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
    # This server rebuilds the engine from the runtime config alone, so a
    # config that does not carry the launch's rendering settings would
    # evaluate under the toy defaults while the harness used the launch's.
    # The Tool Config pins the field independently, so it is the check: a
    # different one makes every call fail preflight, and a silently
    # defaulted render contract would score a different prompt than the
    # harness declares. Refuse rather than evaluate the wrong thing.
    persisted_field = getattr(runtime, "mutation_field", None)
    if persisted_field is None:
        raise ValueError(
            "MCP runtime config does not carry the launch mutation field; "
            "the evaluation server cannot rebuild the launch's engine and "
            "would silently evaluate under the toy defaults"
        )
    if persisted_field != tool_config.candidate_template_field:
        raise ValueError(
            "MCP runtime config mutation field "
            f"{persisted_field!r} does not match the Tool Config's "
            f"{tool_config.candidate_template_field!r}"
        )
    # The render contract is the other half of the rendering identity,
    # and it defaults just as silently: an absent one falls through to
    # the toy contract, so the server would render the agent's candidate
    # under different rules than the harness scored the baseline with and
    # report the result as comparable. Nothing downstream catches that --
    # both renders succeed -- so it is refused here, symmetric with the
    # mutation field. Unlike the field, the Tool Config pins no rendering
    # identity to cross-check against, so presence is the whole check.
    if getattr(runtime, "render_contract", None) is None:
        raise ValueError(
            "MCP runtime config does not carry the launch render "
            "contract; the evaluation server cannot rebuild the launch's "
            "engine and would silently evaluate under the toy defaults"
        )
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
    server = EvaluateCandidateServer(
        handle=executor.runtime_handle(tool_config, tool_store, binding)
    )
    # The persistent session opened above is process-lifetime and keyed by
    # path, so nothing releases it on its own. whetstone now hosts the
    # server in-process and outlives every Step it runs, so the host needs
    # to know which session it owns in order to close it on teardown.
    server.sqlite_path = sqlite_path
    return server


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


__all__ = ["build_server_from_env"]
