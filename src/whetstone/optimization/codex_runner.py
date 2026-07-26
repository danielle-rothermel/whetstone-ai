"""External and fake-process runners for the opaque Codex step."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import ValidationError

from whetstone.optimization.codex import (
    CodexOutputArtifact,
    CodexRunResult,
    OpaqueStepError,
)
from whetstone.optimization.mcp_bridge import JsonRpcClient
from whetstone.optimization.reward import RewardPolicy
from whetstone.optimization.schema import Candidate, OptimizationStepRequest
from whetstone.optimization.tools import ToolConfig

if TYPE_CHECKING:
    from whetstone.execution.mode import EvaluationRuntimeConfig

_MACOS_SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")
_CODEX_DENIED_FEATURES = (
    "apps",
    "browser_use",
    "browser_use_external",
    "code_mode",
    "code_mode_host",
    "code_mode_only",
    "collaboration_modes",
    "computer_use",
    "default_mode_request_user_input",
    "goals",
    "hooks",
    "image_generation",
    "in_app_browser",
    "memories",
    "multi_agent",
    "multi_agent_v2",
    "plugins",
    "remote_plugin",
    "request_permissions_tool",
    "shell_snapshot",
    "shell_tool",
    "skill_mcp_dependency_install",
    "skill_search",
    "standalone_web_search",
    "tool_call_mcp_elicitation",
    "tool_suggest",
    "unified_exec",
    "web_search_cached",
    "web_search_request",
    "workspace_dependencies",
)


class JsonRpcProcess(Protocol):
    def exchange(self, raw: str) -> str | None: ...


@dataclass(frozen=True, slots=True)
class ScriptedAgentCall:
    call_id: str
    base_ref: str
    model_route: str
    template: str


@dataclass(frozen=True, slots=True)
class _MacOsProcessIsolation:
    """Fail-closed outer filesystem boundary for Codex and descendants."""

    def wrap(
        self,
        command: list[str],
        *,
        profile_path: Path,
        readable_paths: tuple[Path, ...],
        writable_paths: tuple[Path, ...],
    ) -> list[str]:
        if sys.platform != "darwin" or not _MACOS_SANDBOX_EXEC.is_file():
            raise OpaqueStepError(
                "Codex process isolation requires macOS sandbox-exec; "
                "no insecure fallback is available"
            )
        profile_path.write_text(
            self._profile(
                readable_paths=readable_paths,
                writable_paths=writable_paths,
            ),
            encoding="utf-8",
        )
        return [
            str(_MACOS_SANDBOX_EXEC),
            "-f",
            str(profile_path),
            *command,
        ]

    @staticmethod
    def _profile(
        *,
        readable_paths: tuple[Path, ...],
        writable_paths: tuple[Path, ...],
    ) -> str:
        def rule(operation: str, path: Path) -> str:
            resolved = path.resolve()
            selector = "subpath" if resolved.is_dir() else "literal"
            return (
                f"(allow {operation} ({selector} {json.dumps(str(resolved))}))"
            )

        platform_reads = (
            Path("/System"),
            Path("/Library/Apple"),
            Path("/Library/Preferences"),
            Path("/private/etc"),
            Path("/private/var/db/timezone"),
            Path("/usr/lib"),
            Path("/usr/share"),
            Path("/opt/homebrew/lib"),
            Path("/usr/local/lib"),
            Path("/bin"),
            Path("/sbin"),
            Path("/usr/bin"),
            Path("/usr/sbin"),
            Path("/usr/libexec"),
        )
        read_rules = [
            rule("file-read* file-test-existence", path)
            for path in (*platform_reads, *readable_paths, *writable_paths)
        ]
        executable_rules = [
            rule("file-map-executable", path)
            for path in (*platform_reads, *readable_paths)
        ]
        write_rules = [rule("file-write*", path) for path in writable_paths]
        return "\n".join(
            [
                "(version 1)",
                "(deny default)",
                "(allow process*)",
                "(allow signal (target same-sandbox))",
                "(allow process-info* (target same-sandbox))",
                "(allow network*)",
                "(allow mach*)",
                "(allow ipc-posix*)",
                "(allow sysctl-read)",
                "(allow system*)",
                "(allow iokit-open)",
                "(allow user-preference-read)",
                '(allow file-read-metadata file-test-existence (subpath "/"))',
                '(allow file-read* file-test-existence (literal "/"))',
                '(allow file-read* file-write* file-ioctl (subpath "/dev"))',
                *read_rules,
                *executable_rules,
                *write_rules,
                "",
            ]
        )


class FakeCodexRunner:
    """Fake only the opaque process; speak real serialized MCP JSON-RPC."""

    def __init__(
        self,
        *,
        process: JsonRpcProcess,
        scripted_calls: Sequence[ScriptedAgentCall],
        final_proposals: Sequence[Candidate],
    ) -> None:
        self._process = process
        self._calls = tuple(scripted_calls)
        self._proposals = tuple(final_proposals)
        self.observed_payloads: list[dict[str, Any]] = []

    def run(
        self, request: OptimizationStepRequest, tool_config: ToolConfig
    ) -> CodexRunResult:
        client = JsonRpcClient(self._process.exchange)
        client.initialize()
        tools = client.list_tools()
        if not any(tool["name"] == tool_config.tool_name for tool in tools):
            raise OpaqueStepError("external MCP process omitted the tool")
        for call in self._calls:
            self.observed_payloads.append(
                client.evaluate(
                    call_id=call.call_id,
                    base_ref=call.base_ref,
                    model_route=call.model_route,
                    template=call.template,
                )
            )
        return CodexRunResult(
            artifact=CodexOutputArtifact(
                run_id=request.run_id,
                proposals=self._proposals,
                conversation_evidence={
                    "process": "fake",
                    "jsonrpc_call_count": len(self._calls),
                },
                control_cost={"agent_tokens": 0},
            )
        )


def build_codex_command(
    *,
    prompt: str,
    codex_binary: str,
    model: str,
    mcp_env: dict[str, str],
    output_schema_path: str,
    output_artifact_path: str,
    working_directory: str,
) -> list[str]:
    argv = [
        codex_binary,
        "exec",
        "--json",
        "--color",
        "never",
        "--skip-git-repo-check",
        "--strict-config",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--sandbox",
        "read-only",
        "--cd",
        working_directory,
        "--output-schema",
        output_schema_path,
        "--output-last-message",
        output_artifact_path,
        "-c",
        'shell_environment_policy.inherit="none"',
    ]
    for feature in _CODEX_DENIED_FEATURES:
        argv.extend(["--disable", feature])
    if model:
        argv.extend(["--model", model])
    argv.extend(
        [
            "-c",
            f"mcp_servers.whetstone.command={json.dumps(sys.executable)}",
            "-c",
            "mcp_servers.whetstone.args="
            + json.dumps(["-m", "whetstone.optimization.mcp_server"]),
        ]
    )
    for key, value in sorted(mcp_env.items()):
        argv.extend(
            [
                "-c",
                f"mcp_servers.whetstone.env.{key}={json.dumps(value)}",
            ]
        )
    argv.append(prompt)
    return argv


class SubprocessCodexRunner:
    """Launch Codex through the macOS-only, fail-closed MCP sandbox."""

    def __init__(
        self,
        *,
        sqlite_path: str,
        runtime_config: EvaluationRuntimeConfig,
        reward_policy: RewardPolicy,
        codex_binary: str = "codex",
        model: str = "",
        timeout_seconds: float = 600.0,
        environment: Mapping[str, str] | None = None,
        prompt_builder: (
            Callable[[OptimizationStepRequest], str] | None
        ) = None,
    ) -> None:
        self._sqlite_path = sqlite_path
        self._runtime = runtime_config
        self._reward_policy = reward_policy
        self._binary = codex_binary
        self._model = model
        self._timeout = timeout_seconds
        self._prompt_builder = prompt_builder or _default_prompt
        source_environment = (
            dict(os.environ) if environment is None else dict(environment)
        )
        allowed = {
            "PATH",
            "CODEX_HOME",
            "OPENAI_API_KEY",
            "SSL_CERT_FILE",
            "SSL_CERT_DIR",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "NO_PROXY",
            self._runtime.execution_policy.transport_policy.api_key_env,
        }
        self._environment = {
            key: value
            for key, value in source_environment.items()
            if key in allowed
        }

    def run(
        self, request: OptimizationStepRequest, tool_config: ToolConfig
    ) -> CodexRunResult:
        resolved_binary = shutil.which(
            self._binary, path=self._environment.get("PATH")
        )
        if resolved_binary is None:
            raise OpaqueStepError(
                f"Codex binary {self._binary!r} was not found"
            )
        with tempfile.TemporaryDirectory(
            prefix="whetstone-codex-"
        ) as working_directory:
            root = Path(working_directory)
            runtime_root = root / "runtime"
            runtime_root.mkdir()
            self._stage_runtime(runtime_root)
            codex_home = root / "codex-home"
            codex_home.mkdir()
            self._stage_auth(codex_home)
            isolated_environment = {
                **self._environment,
                "CODEX_HOME": str(codex_home),
                "PYTHONPATH": str(runtime_root),
            }
            schema_path = root / "output-schema.json"
            artifact_path = root / "last-message.json"
            schema = CodexOutputArtifact.model_json_schema()
            run_id_schema = schema["properties"]["run_id"]
            assert isinstance(run_id_schema, dict)
            run_id_schema["const"] = request.run_id
            schema_path.write_text(
                json.dumps(schema, sort_keys=True), encoding="utf-8"
            )
            command = build_codex_command(
                prompt=self._prompt_builder(request),
                codex_binary=resolved_binary,
                model=self._model,
                mcp_env={
                    "WS_MCP_SQLITE_PATH": self._sqlite_path,
                    "WS_MCP_TOOL_CONFIG": tool_config.model_dump_json(),
                    "WS_MCP_RUNTIME_CONFIG": self._runtime.model_dump_json(),
                    "WS_MCP_REWARD_POLICY": (
                        self._reward_policy.model_dump_json()
                    ),
                    "PYTHONPATH": str(runtime_root),
                },
                output_schema_path=str(schema_path),
                output_artifact_path=str(artifact_path),
                working_directory=working_directory,
            )
            profile_path = root / "codex.sb"
            isolation = _MacOsProcessIsolation()
            command = isolation.wrap(
                command,
                profile_path=profile_path,
                readable_paths=self._readable_runtime_paths(
                    resolved_binary=Path(resolved_binary),
                    runtime_root=runtime_root,
                ),
                writable_paths=self._writable_runtime_paths(root),
            )
            completed = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                cwd=working_directory,
                env=isolated_environment,
                check=False,
            )
            if completed.returncode:
                raise OpaqueStepError(
                    f"Codex exited {completed.returncode}: "
                    f"{completed.stderr[-2000:]}"
                )
            artifact = _parse_output_artifact(
                artifact_path,
                stdout=completed.stdout,
                stderr=completed.stderr,
                run_id=request.run_id,
                isolation={
                    "strategy": "macos_sandbox_exec",
                    "profile": profile_path.read_text(encoding="utf-8"),
                    "denied_features": list(_CODEX_DENIED_FEATURES),
                },
            )
        return CodexRunResult(artifact=artifact)

    def _stage_auth(self, destination: Path) -> None:
        source_raw = self._environment.get("CODEX_HOME")
        if source_raw is None:
            return
        source = Path(source_raw)
        for name in ("auth.json", ".credentials.json"):
            candidate = source / name
            if candidate.is_file():
                shutil.copy2(candidate, destination / name)

    def _stage_runtime(self, destination: Path) -> None:
        package_root = Path(__file__).resolve().parents[1]
        shutil.copytree(
            package_root,
            destination / package_root.name,
            dirs_exist_ok=True,
        )
        module_name = self._runtime.transport_factory.partition(":")[0]
        top_level = module_name.partition(".")[0]
        spec = importlib.util.find_spec(top_level)
        if spec is None:
            raise OpaqueStepError(
                f"transport factory package {top_level!r} was not found"
            )
        if spec.submodule_search_locations:
            source = Path(next(iter(spec.submodule_search_locations)))
            target = destination / top_level
            if source.resolve() != package_root.resolve():
                shutil.copytree(source, target, dirs_exist_ok=True)
        elif spec.origin is not None:
            shutil.copy2(spec.origin, destination / Path(spec.origin).name)

    def _readable_runtime_paths(
        self,
        *,
        resolved_binary: Path,
        runtime_root: Path,
    ) -> tuple[Path, ...]:
        paths = {
            resolved_binary.resolve(),
            Path(sys.executable).resolve(),
            Path(sys.prefix).resolve(),
            Path(sys.base_prefix).resolve(),
            runtime_root.resolve(),
        }
        for key in ("SSL_CERT_FILE", "SSL_CERT_DIR"):
            raw = self._environment.get(key)
            if raw:
                paths.add(Path(raw).resolve())
        return tuple(sorted(paths, key=str))

    def _writable_runtime_paths(self, root: Path) -> tuple[Path, ...]:
        state_paths = {root.resolve()}
        sqlite_path = Path(self._sqlite_path).resolve()
        if not sqlite_path.parent.is_dir():
            raise OpaqueStepError(
                "Codex MCP SQLite parent directory does not exist"
            )
        state_paths.update(
            {
                sqlite_path,
                Path(f"{sqlite_path}-journal"),
                Path(f"{sqlite_path}-shm"),
                Path(f"{sqlite_path}-wal"),
            }
        )
        if self._runtime.partial_log_path:
            partial_path = Path(self._runtime.partial_log_path).resolve()
            if not partial_path.parent.is_dir():
                raise OpaqueStepError(
                    "Codex partial-log parent directory does not exist"
                )
            state_paths.add(partial_path)
        if self._runtime.prompt_cache_path:
            cache_path = Path(self._runtime.prompt_cache_path).resolve()
            if not cache_path.is_dir():
                raise OpaqueStepError(
                    "Codex prompt-cache directory does not exist"
                )
            state_paths.add(cache_path)
        return tuple(sorted(state_paths, key=str))


def _default_prompt(request: OptimizationStepRequest) -> str:
    context = json.dumps(
        request.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        "Use only the external evaluate_candidate MCP tool for measurements. "
        "Do not call any built-in tool. Build proposals from the exact "
        "candidate base_ref, model route, payload template, Tool Config, "
        "capacity, budget, pools, hyperparameters, and output contract in "
        "the serialized request below. Evaluate candidate drafts through "
        "MCP before selecting them. Write the schema-conforming final "
        "artifact with exactly the requested proposal count.\n"
        f"OPTIMIZATION_STEP_REQUEST_JSON={context}"
    )


def _parse_jsonl_events(stdout: str) -> tuple[dict[str, Any], ...]:
    events: list[dict[str, Any]] = []
    for ordinal, raw in enumerate(stdout.splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise OpaqueStepError(
                f"Codex JSONL event {ordinal} is malformed"
            ) from exc
        if not isinstance(value, dict):
            raise OpaqueStepError(
                f"Codex JSONL event {ordinal} is not an object"
            )
        events.append(value)
    return tuple(events)


def _parse_output_artifact(
    path: Path,
    *,
    stdout: str,
    stderr: str,
    run_id: str,
    isolation: dict[str, Any] | None = None,
) -> CodexOutputArtifact:
    if not path.is_file():
        raise OpaqueStepError("Codex produced no final output artifact")
    try:
        artifact = CodexOutputArtifact.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as exc:
        raise OpaqueStepError(
            "Codex final output artifact failed schema validation"
        ) from exc
    if artifact.run_id != run_id:
        raise OpaqueStepError("Codex final output artifact has the wrong run")
    process_evidence = {
        "agent": artifact.conversation_evidence,
        "jsonl_events": list(_parse_jsonl_events(stdout)),
        "stderr": stderr,
        "isolation": isolation or {},
    }
    return artifact.model_copy(
        update={"conversation_evidence": process_evidence}
    )


__all__ = [
    "FakeCodexRunner",
    "JsonRpcProcess",
    "ScriptedAgentCall",
    "SubprocessCodexRunner",
    "build_codex_command",
]
