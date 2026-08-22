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
    runtime_config = ReferenceEvalRuntimeConfig()
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


def test_the_task_model_key_is_granted_to_the_mcp_server_child(
    tmp_path,
    sqlite_store,
    task_model_key_env,
) -> None:
    """The eval driver still needs the key; only the child gets it.

    It travels as one more ``mcp_servers.whetstone.env.*`` entry, which
    the CLI applies to the server process alone.
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
