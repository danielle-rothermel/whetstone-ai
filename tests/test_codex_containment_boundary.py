"""What the Codex agent process may reach, asserted without a sandbox.

Every guarantee here is enforced by whetstone code rather than by macOS
``sandbox-exec``, so these tests run on every platform. The sandbox-only
facts live in :mod:`tests.test_codex_sandbox_profile`.
"""

from __future__ import annotations

import pytest

from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.optim.codex.mcp_environment import McpEnvironmentKey
from whetstone.optim.codex.runner import SubprocessCodexRunner
from whetstone.testing.toy.experiment import (
    TOY_MUTATION_FIELD,
    toy_template_render_contract,
)


_TASK_MODEL_KEY_VALUE = "sk-TASK-MODEL-KEY"
_CODEX_AGENT_KEY_VALUE = "sk-codex-agent-key"


def _runner(
    tmp_path,
    sqlite_store,
    *,
    environment: dict[str, str],
) -> SubprocessCodexRunner:
    """A runner built only far enough to expose its environment grants.

    Nothing here executes: the tests read the two environments the runner
    computes at construction, which is where the boundary is decided.
    """
    # Production always carries both rendering settings onto the config
    # the out-of-process server rebuilds its engine from, and the server
    # refuses a config missing either.
    runtime_config = ReferenceEvalRuntimeConfig(
        mutation_field=TOY_MUTATION_FIELD,
        render_contract=toy_template_render_contract(),
    )
    return SubprocessCodexRunner(
        executor=_NeverRunExecutor(),
        sqlite_path=str((tmp_path / "store.sqlite").resolve()),
        runtime_config=runtime_config,
        runtime_config_class=(
            "whetstone.eval.reference_runtime:ReferenceEvalRuntimeConfig"
        ),
        reward_policy=runtime_config.build_engine(sqlite_store).reward_policy,
        environment=environment,
    )


class _NeverRunExecutor:
    def run_blocking(self, job):
        raise AssertionError("this test must not spawn a Codex process")


@pytest.fixture
def task_model_key_env() -> str:
    policy = ReferenceEvalRuntimeConfig().execution_policy
    return policy.transport_policy.api_key_env


def test_the_task_model_key_never_enters_the_codex_process_environment(
    tmp_path,
    sqlite_store,
    task_model_key_env,
) -> None:
    """The agent must not be able to score candidates outside the Tool.

    The task-model API key is the eval transport's credential. The Codex
    process is a general-purpose agent with network access, so holding
    that key would let it call the task model directly and return a
    candidate it scored itself, unadmitted and unledgered.
    """
    runner = _runner(
        tmp_path,
        sqlite_store,
        environment={
            "PATH": "/usr/bin",
            "OPENAI_API_KEY": _CODEX_AGENT_KEY_VALUE,
            task_model_key_env: _TASK_MODEL_KEY_VALUE,
        },
    )

    granted = runner.codex_process_environment()

    assert task_model_key_env not in granted
    assert _TASK_MODEL_KEY_VALUE not in granted.values()
    # The Codex CLI's own credential is still granted: it must reach its
    # own model.
    assert granted["OPENAI_API_KEY"] == _CODEX_AGENT_KEY_VALUE


def test_the_task_model_key_is_granted_to_the_evaluation_server(
    tmp_path,
    sqlite_store,
    task_model_key_env,
) -> None:
    """The eval driver still needs the key; only the server gets it.

    whetstone hosts that server in its own environment, so the key never
    crosses into the sandbox at all.
    """
    runner = _runner(
        tmp_path,
        sqlite_store,
        environment={
            "PATH": "/usr/bin",
            "OPENAI_API_KEY": _CODEX_AGENT_KEY_VALUE,
            task_model_key_env: _TASK_MODEL_KEY_VALUE,
        },
    )

    server_env = runner.mcp_server_secret_environment()

    assert server_env[task_model_key_env] == _TASK_MODEL_KEY_VALUE


def test_an_absent_task_model_key_grants_nothing(
    tmp_path,
    sqlite_store,
    task_model_key_env,
) -> None:
    runner = _runner(
        tmp_path, sqlite_store, environment={"PATH": "/usr/bin"}
    )

    assert runner.mcp_server_secret_environment() == {}
    assert task_model_key_env not in runner.codex_process_environment()


def test_the_run_lease_token_is_not_a_codex_process_variable(
    tmp_path,
    sqlite_store,
) -> None:
    """The token reaches the server through the MCP env block only."""
    runner = _runner(
        tmp_path,
        sqlite_store,
        environment={
            "PATH": "/usr/bin",
            McpEnvironmentKey.RUN_LEASE_TOKEN: "leaked",
        },
    )

    assert (
        McpEnvironmentKey.RUN_LEASE_TOKEN
        not in runner.codex_process_environment()
    )


def _server_environment(tmp_path, sqlite_store, *, run_id: str) -> dict:
    """The exact environment the runner grants one run's MCP server."""
    from tests.codex_support import (
        toy_capacity_binding,
        toy_codex_control,
        toy_codex_run,
    )
    from whetstone.optim.codex.adapter import codex_run_lease_binding
    from whetstone.optim.codex.runner import _capacity_subject_key

    engine = ReferenceEvalRuntimeConfig().build_engine(sqlite_store)
    control = toy_codex_control(engine=engine, max_tool_calls=2)
    run, config, _candidate = toy_codex_run(
        control=control, engine=engine, run_id=run_id
    )
    binding = toy_capacity_binding(run)
    token = f"token-for-{run_id}"
    # Production always carries both rendering settings onto the config
    # the out-of-process server rebuilds its engine from, and the server
    # refuses a config missing either.
    runtime_config = ReferenceEvalRuntimeConfig(
        mutation_field=TOY_MUTATION_FIELD,
        render_contract=toy_template_render_contract(),
    )
    return {
        McpEnvironmentKey.SQLITE_PATH: str(tmp_path / "server.sqlite"),
        McpEnvironmentKey.TOOL_CONFIG: config.model_dump_json(),
        McpEnvironmentKey.CAPACITY_BINDING: binding.model_dump_json(),
        McpEnvironmentKey.RUNTIME_CONFIG: runtime_config.model_dump_json(),
        McpEnvironmentKey.RUNTIME_CONFIG_CLASS: (
            "whetstone.eval.reference_runtime:ReferenceEvalRuntimeConfig"
        ),
        McpEnvironmentKey.REWARD_POLICY: (
            engine.reward_policy.model_dump_json()
        ),
        McpEnvironmentKey.RUN_LEASE_TOKEN: token,
        McpEnvironmentKey.RUN_LEASE_BINDING: codex_run_lease_binding(
            token=token,
            store_namespace_key=str(config.store_namespace_key),
            tool_config_hash=str(config.identity_hash()),
            capacity_scope=binding.scope.value,
            capacity_subject=_capacity_subject_key(binding),
        ),
    }


def test_the_mcp_server_refuses_a_token_bound_to_another_run(
    tmp_path,
    sqlite_store,
) -> None:
    """A non-empty token is not proof that the server serves this run.

    The binding digest covers the token together with the server's own
    Tool Config and capacity binding, so a token minted for a different
    run does not verify against this server's configuration.
    """
    from whetstone.optim.codex.mcp_server import build_server_from_env

    mine = _server_environment(tmp_path, sqlite_store, run_id="run-mine")
    theirs = _server_environment(
        tmp_path, sqlite_store, run_id="run-theirs"
    )
    foreign = {
        **mine,
        McpEnvironmentKey.RUN_LEASE_TOKEN: theirs[
            McpEnvironmentKey.RUN_LEASE_TOKEN
        ],
        McpEnvironmentKey.RUN_LEASE_BINDING: theirs[
            McpEnvironmentKey.RUN_LEASE_BINDING
        ],
    }

    with pytest.raises(ValueError, match="not bound to this run"):
        build_server_from_env(foreign)


def test_the_mcp_server_refuses_a_replayed_token_without_its_binding(
    tmp_path,
    sqlite_store,
) -> None:
    """A stale server started from a leaked token alone is refused."""
    from whetstone.optim.codex.mcp_server import build_server_from_env

    mine = _server_environment(tmp_path, sqlite_store, run_id="run-mine")
    without_binding = dict(mine)
    without_binding.pop(McpEnvironmentKey.RUN_LEASE_BINDING)

    with pytest.raises(ValueError, match="requires the run lease binding"):
        build_server_from_env(without_binding)

    tampered = {**mine, McpEnvironmentKey.RUN_LEASE_BINDING: "0" * 64}
    with pytest.raises(ValueError, match="not bound to this run"):
        build_server_from_env(tampered)


def test_the_mcp_server_accepts_its_own_run_binding(
    tmp_path,
    sqlite_store,
) -> None:
    """The success side: the runner's own binding verifies."""
    from whetstone.optim.codex.mcp_server import build_server_from_env

    mine = _server_environment(tmp_path, sqlite_store, run_id="run-mine")

    assert build_server_from_env(mine) is not None
