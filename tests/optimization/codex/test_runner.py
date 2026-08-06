from __future__ import annotations

import importlib.machinery
import json
import sys
from collections.abc import Callable
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
    _MCP_TOOLS_APPROVAL_MODE,
    SubprocessCodexRunner,
    _decode_stderr,
    _MacOsProcessIsolation,
    _parse_jsonl_events,
    _parse_output_artifact,
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
    *,
    tool_name: str = "evaluate_candidate",
    prompt_builder: Callable[[OptimizationStepRequest], str] | None = None,
) -> SubprocessBoundary:
    store = ObjectStore(SqliteBackend(tmp_path / f"subprocess-{mode}.sqlite"))
    evaluation_engine = engine(store, experiment)
    config = tool_config(
        evaluation_engine,
        experiment,
        f"codex-subprocess-{mode}",
        tool_name=tool_name,
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
        prompt_builder=prompt_builder,
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
    assert (
        "mcp_servers.whetstone.default_tools_approval_mode="
        + json.dumps(_MCP_TOOLS_APPROVAL_MODE)
        in argv
    )
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


@pytest.mark.parametrize(
    "stdout",
    [b'{"value":1,"value":2}\n', b'{"value":NaN}\n', b"\xff\n"],
)
def test_jsonl_events_reject_non_strict_json(stdout: bytes) -> None:
    with pytest.raises(OpaqueStepError, match="JSONL event 1 is malformed"):
        _parse_jsonl_events(stdout)


@pytest.mark.parametrize(
    "artifact_bytes",
    [b'{"value":1,"value":2}', b'{"value":NaN}', b"\xff"],
)
def test_output_artifact_rejects_non_strict_json(
    tmp_path: Path,
    artifact_bytes: bytes,
) -> None:
    path = tmp_path / "artifact.json"
    path.write_bytes(artifact_bytes)

    with pytest.raises(OpaqueStepError, match="schema validation"):
        _parse_output_artifact(
            path,
            stdout=b"",
            stderr="",
            run_id="run",
        )


def test_stderr_requires_utf8() -> None:
    with pytest.raises(OpaqueStepError, match="stderr is not valid UTF-8"):
        _decode_stderr(b"\xff")


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


def test_partial_log_profile_preserves_declared_parent_authority(
    tmp_path,
    codex_experiment: EnvExperiment,
) -> None:
    boundary = _subprocess_boundary(tmp_path, codex_experiment, "proposal")
    runtime_root = tmp_path / "isolated-runtime"
    runtime_root.mkdir()

    writable_paths = boundary.runner._writable_runtime_paths(runtime_root)

    partial_path = boundary.partial_log_path
    assert partial_path is not None
    resolved_partial = partial_path.resolve()
    lock_path = resolved_partial.with_name(f".{resolved_partial.name}.lock")
    assert {resolved_partial, lock_path, resolved_partial.parent} <= set(
        writable_paths
    )
    profile = _MacOsProcessIsolation._profile(
        readable_paths=(),
        writable_paths=writable_paths,
    )
    assert (
        "(allow file-write* (subpath "
        f"{json.dumps(str(resolved_partial.parent))}))" in profile
    )
    for ancestor in resolved_partial.parents:
        assert (
            "(allow file-read* file-test-existence (literal "
            f"{json.dumps(str(ancestor))}))" in profile
        )


def test_default_prompt_names_configured_mcp_tool(
    tmp_path,
    codex_experiment: EnvExperiment,
    cross_platform_fake_codex_boundary,
) -> None:
    tool_name = "score_candidate_draft"
    boundary = _subprocess_boundary(
        tmp_path,
        codex_experiment,
        "success",
        tool_name=tool_name,
    )

    run = boundary.runner.run(boundary.request, boundary.handle)

    events = run.artifact.conversation_evidence["jsonl_events"]
    prompt = events[0]["argv"][-1]
    instruction = prompt.split("\nOPTIMIZATION_STEP_REQUEST_JSON=", 1)[0]
    assert tool_name in instruction
    assert "evaluate_candidate" not in instruction


def test_custom_prompt_builder_remains_request_only(
    tmp_path,
    codex_experiment: EnvExperiment,
    cross_platform_fake_codex_boundary,
) -> None:
    observed_requests: list[OptimizationStepRequest] = []

    def build_prompt(step_request: OptimizationStepRequest) -> str:
        observed_requests.append(step_request)
        context = json.dumps(step_request.model_dump(mode="json"))
        return f"custom prompt\nOPTIMIZATION_STEP_REQUEST_JSON={context}"

    boundary = _subprocess_boundary(
        tmp_path,
        codex_experiment,
        "success",
        prompt_builder=build_prompt,
    )

    run = boundary.runner.run(boundary.request, boundary.handle)

    events = run.artifact.conversation_evidence["jsonl_events"]
    assert events[0]["argv"][-1].startswith("custom prompt\n")
    assert observed_requests == [boundary.request]


def test_runtime_staging_merges_namespace_package_locations(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    codex_experiment: EnvExperiment,
) -> None:
    namespace = "row_job_namespace"
    first = tmp_path / "first" / namespace
    second = tmp_path / "second" / namespace
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "unrelated.py").write_text("VALUE = 'first'\n", encoding="utf-8")
    (first / "collision.py").write_text("VALUE = 'first'\n", encoding="utf-8")
    (second / "configured.py").write_text(
        "def run():\n    return 'configured'\n",
        encoding="utf-8",
    )
    (second / "collision.py").write_text(
        "VALUE = 'second'\n", encoding="utf-8"
    )
    spec = importlib.machinery.ModuleSpec(
        namespace,
        loader=None,
        is_package=True,
    )
    assert spec.submodule_search_locations is not None
    spec.submodule_search_locations.extend((str(first), str(second)))
    monkeypatch.setattr(
        "whetstone.optimization.codex.runner.importlib.util.find_spec",
        lambda _name: spec,
    )
    store = ObjectStore(SqliteBackend(tmp_path / "namespace.sqlite"))
    evaluation_engine = engine(store, codex_experiment)
    runner = SubprocessCodexRunner(
        sqlite_path=str(tmp_path / "mcp.sqlite"),
        runtime_config=runtime_config(evaluation_engine).model_copy(
            update={"row_job_entrypoint": f"{namespace}.configured:run"}
        ),
        reward_policy=codex_experiment.reward_policy,
        environment={},
    )
    destination = tmp_path / "runtime"
    destination.mkdir()

    runner._stage_runtime(destination)

    staged_namespace = destination / namespace
    assert (staged_namespace / "unrelated.py").is_file()
    assert (staged_namespace / "configured.py").is_file()
    assert (staged_namespace / "collision.py").read_text(encoding="utf-8") == (
        "VALUE = 'first'\n"
    )


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
