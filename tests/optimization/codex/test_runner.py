from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
from dr_store import ObjectStore, SqliteBackend

from tests.optimization.codex.support import (
    binding,
    engine,
    executor,
    request,
    runtime_config,
    tool_config,
)
from tests.optimization.support import memory_tool_call_store
from whetstone.core.effects.authority import EffectAuthority
from whetstone.envs.factory import EnvExperiment
from whetstone.optimization.codex.adapter import (
    CodexOutputArtifact,
    OpaqueStepError,
)
from whetstone.optimization.codex.runner import (
    _CODEX_DENIED_FEATURES,
    SubprocessCodexRunner,
    _MacOsProcessIsolation,
)
from whetstone.optimization.contracts import OptimizationStepRequest
from whetstone.optimization.tools.contracts import (
    RuntimeToolHandle,
    ToolConfig,
)
from whetstone.optimization.tools.facade import ToolCallStore


@dataclass(frozen=True, slots=True)
class SubprocessBoundary:
    store: ObjectStore
    config: ToolConfig
    runner: SubprocessCodexRunner
    request: OptimizationStepRequest
    handle: RuntimeToolHandle
    tool_store: ToolCallStore
    partial_log_path: Path | None


def _subprocess_boundary(
    tmp_path: Path,
    experiment: EnvExperiment,
    mode: str,
) -> SubprocessBoundary:
    store = ObjectStore(SqliteBackend(tmp_path / f"subprocess-{mode}.sqlite"))
    evaluation_engine = engine(store, experiment)
    config = tool_config(
        evaluation_engine,
        experiment,
        f"codex-subprocess-{mode}",
    )
    mcp_state = tmp_path / f"mcp-{mode}"
    mcp_state.mkdir()
    partial_path = mcp_state / "partials" if mode == "proposal" else None
    runtime = runtime_config(
        evaluation_engine,
        partial_log_path=None if partial_path is None else str(partial_path),
    )
    executable = tmp_path / f"fake-codex-{mode}"
    executable.write_text(
        (
            f"#!{sys.executable}\n"
            "from tests.optimization.codex.fake_cli import app\n"
            "app()\n"
        ),
        encoding="utf-8",
    )
    executable.chmod(0o755)
    outside_secret = mcp_state / "neighbor-secret.txt"
    outside_secret.write_text("host secret", encoding="utf-8")
    runner = SubprocessCodexRunner(
        sqlite_path=str(mcp_state / "store.sqlite"),
        runtime_config=runtime,
        reward_policy=experiment.reward_policy,
        codex_binary=str(executable),
        environment={
            "CODEX_HOME": str(tmp_path / "codex-auth"),
            "OPENROUTER_API_KEY": "provider-secret",
            "AWS_SECRET_ACCESS_KEY": "forbidden",
            "UNRELATED_VALUE": "forbidden",
        },
    )
    step_request = request(
        experiment.initial_candidate,
        config,
        proposal_count=1 if mode == "proposal" else 0,
        run_id=f"codex-run-{mode}",
        hyperparameters={
            "adversarial_outside_path": str(outside_secret),
            "search_temperature": 0.7,
        },
    )
    authority = EffectAuthority.memory()
    tool_store = memory_tool_call_store(store, authority)
    handle = executor(evaluation_engine, experiment, authority).runtime_handle(
        config,
        tool_store,
        binding(step_request),
    )
    return SubprocessBoundary(
        store=store,
        config=config,
        runner=runner,
        request=step_request,
        handle=handle,
        tool_store=tool_store,
        partial_log_path=partial_path,
    )


@pytest.fixture
def cross_platform_fake_codex_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if sys.platform == "darwin":
        return

    def wrap_without_seatbelt(
        _isolation: _MacOsProcessIsolation,
        command: list[str],
        *,
        profile_path: Path,
        readable_paths: tuple[Path, ...],
        writable_paths: tuple[Path, ...],
    ) -> list[str]:
        profile_path.write_text(
            _MacOsProcessIsolation._profile(
                readable_paths=readable_paths,
                writable_paths=writable_paths,
            ),
            encoding="utf-8",
        )
        return command

    monkeypatch.setattr(
        _MacOsProcessIsolation,
        "wrap",
        wrap_without_seatbelt,
    )


def test_subprocess_uses_typed_artifact_and_restricted_authority(
    tmp_path,
    codex_experiment: EnvExperiment,
    cross_platform_fake_codex_boundary,
) -> None:
    boundary = _subprocess_boundary(tmp_path, codex_experiment, "success")

    run = boundary.runner.run(boundary.request, boundary.handle)

    artifact = CodexOutputArtifact.model_validate(run.artifact)
    events = artifact.conversation_evidence["jsonl_events"]
    assert isinstance(events, list)
    invocation = events[0]
    assert isinstance(invocation, dict)
    argv = invocation["argv"]
    assert isinstance(argv, list)
    assert invocation["schema_exists"] is True
    assert invocation["schema_run_id"] == boundary.request.run_id
    assert (
        Path(invocation["cwd"]).resolve()
        == Path(argv[argv.index("--cd") + 1]).resolve()
    )
    assert argv[argv.index("--sandbox") + 1] == "read-only"
    for flag in (
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--strict-config",
        "--output-schema",
        "--output-last-message",
    ):
        assert flag in argv
    assert 'shell_environment_policy.inherit="none"' in argv
    disabled = {
        argv[index + 1]
        for index, item in enumerate(argv)
        if item == "--disable"
    }
    assert disabled == set(_CODEX_DENIED_FEATURES)
    platform_environment = {"LC_CTYPE", "__CF_USER_TEXT_ENCODING"}
    assert set(invocation["env_keys"]) - platform_environment == {
        "CODEX_HOME",
        "OPENROUTER_API_KEY",
        "PYTHONPATH",
    }
    assert "AWS_SECRET_ACCESS_KEY" not in invocation["env_keys"]
    assert "UNRELATED_VALUE" not in invocation["env_keys"]
    assert "provider-secret" not in " ".join(str(item) for item in argv)
    assert events[1]["item"]["type"] == "agent_message"
    assert artifact.conversation_evidence["agent"] == {"agent": "final"}
    isolation = artifact.conversation_evidence["isolation"]
    assert isolation["strategy"] == "macos_sandbox_exec"
    profile = isolation["profile"]
    outside_path = boundary.request.hyperparameters["adversarial_outside_path"]
    assert isinstance(outside_path, str)
    assert outside_path not in profile
    source_root = Path(__file__).resolve().parents[3] / "src"
    assert source_root.is_dir()
    assert str(source_root) not in profile
    assert artifact.control_cost == {"agent_tokens": 7}


@pytest.mark.parametrize("mode", ["missing", "malformed"])
def test_subprocess_rejects_missing_or_malformed_artifact(
    tmp_path,
    codex_experiment: EnvExperiment,
    mode: str,
    cross_platform_fake_codex_boundary,
) -> None:
    boundary = _subprocess_boundary(tmp_path, codex_experiment, mode)

    with pytest.raises(OpaqueStepError):
        boundary.runner.run(boundary.request, boundary.handle)


def test_subprocess_proposal_uses_the_mcp_evaluation_path(
    tmp_path,
    codex_experiment: EnvExperiment,
    cross_platform_fake_codex_boundary,
) -> None:
    boundary = _subprocess_boundary(tmp_path, codex_experiment, "proposal")

    run = boundary.runner.run(boundary.request, boundary.handle)

    assert len(run.artifact.proposals) == 1
    assert boundary.partial_log_path is not None
    assert boundary.partial_log_path.is_dir()
    mcp_result = run.artifact.conversation_evidence["agent"]["mcp_result"]
    assert mcp_result["refused"] is False


@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="neighboring-secret enforcement requires macOS Seatbelt",
)
def test_macos_process_isolation_denies_neighboring_state_file(
    tmp_path, codex_experiment: EnvExperiment
) -> None:
    boundary = _subprocess_boundary(tmp_path, codex_experiment, "success")

    result = boundary.runner.run(boundary.request, boundary.handle)

    events = result.artifact.conversation_evidence["jsonl_events"]
    assert events[0]["outside_read"] == "denied"


def test_process_isolation_fails_closed_on_unsupported_platform(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")

    with pytest.raises(OpaqueStepError):
        _MacOsProcessIsolation().wrap(
            ["/bin/true"],
            profile_path=tmp_path / "profile.sb",
            readable_paths=(),
            writable_paths=(tmp_path,),
        )


@pytest.mark.parametrize(
    ("sqlite_path", "partial_log_path", "prompt_cache_path"),
    [
        pytest.param("relative/store.sqlite", None, None, id="sqlite"),
        pytest.param(
            "/abs/store.sqlite",
            "relative/partials.jsonl",
            None,
            id="partial-log",
        ),
        pytest.param(
            "/abs/store.sqlite",
            None,
            "relative/cache",
            id="prompt-cache",
        ),
    ],
)
def test_subprocess_runner_rejects_relative_runtime_paths(
    tmp_path,
    codex_experiment: EnvExperiment,
    sqlite_path: str,
    partial_log_path: str | None,
    prompt_cache_path: str | None,
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "relative-paths.sqlite"))
    evaluation_engine = engine(store, codex_experiment)

    with pytest.raises(OpaqueStepError):
        SubprocessCodexRunner(
            sqlite_path=sqlite_path,
            runtime_config=runtime_config(
                evaluation_engine,
                partial_log_path=partial_log_path,
                prompt_cache_path=prompt_cache_path,
            ),
            reward_policy=codex_experiment.reward_policy,
            environment={},
        )
