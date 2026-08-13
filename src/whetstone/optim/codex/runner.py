from __future__ import annotations

import importlib.util
import json
import math
import os
import shutil
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import UNIQUE, StrEnum, verify
from pathlib import Path
from subprocess import TimeoutExpired
from typing import TYPE_CHECKING, Any, Final
from uuid import uuid4

from dr_exec import (
    BudgetAxis,
    BudgetExceededOutcome,
    Budgets,
    ContainmentProfile,
    EnvGrant,
    ExecutionJob,
    Executor,
    ExecutorFailure,
    ExitedOutcome,
    FiniteDurationLimit,
    JobId,
    RetainedPayloadStream,
    SignaledOutcome,
    SpawnAbsentOutcome,
    SpawnFailedOutcome,
    UnbudgetedLimit,
    UntrustedCommandTarget,
)
from dr_serialize import StrictJsonDecodeError, decode_strict_json_bytes
from pydantic import ValidationError

from whetstone.experiment.reward import RewardPolicy
from whetstone.optim.codex.adapter import (
    CodexOutputArtifact,
    CodexRunResult,
    OpaqueStepError,
)
from whetstone.optim.codex.mcp_environment import McpEnvironmentKey
from whetstone.optim.contracts import OptimStepRequest
from whetstone.optim.tools.contracts import RuntimeToolHandle

if TYPE_CHECKING:
    from whetstone.optim.codex.config import EvalRuntimeConfig

_MACOS_SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")


_MCP_TOOLS_APPROVAL_MODE = "auto"
_DIRECT_EXEC_SOURCE: Final = (
    "import os,sys;os.chdir(sys.argv[1]);os.execv(sys.argv[2],sys.argv[2:])"
)


@verify(UNIQUE)
class _CodexDeniedFeature(StrEnum):
    APPS = "apps"
    BROWSER_USE = "browser_use"
    BROWSER_USE_EXTERNAL = "browser_use_external"
    CODE_MODE = "code_mode"
    CODE_MODE_HOST = "code_mode_host"
    CODE_MODE_ONLY = "code_mode_only"
    COLLABORATION_MODES = "collaboration_modes"
    COMPUTER_USE = "computer_use"
    DEFAULT_MODE_REQUEST_USER_INPUT = "default_mode_request_user_input"
    GOALS = "goals"
    HOOKS = "hooks"
    IMAGE_GENERATION = "image_generation"
    IN_APP_BROWSER = "in_app_browser"
    MEMORIES = "memories"
    MULTI_AGENT = "multi_agent"
    MULTI_AGENT_V2 = "multi_agent_v2"
    PLUGINS = "plugins"
    REMOTE_PLUGIN = "remote_plugin"
    REQUEST_PERMISSIONS_TOOL = "request_permissions_tool"
    SHELL_SNAPSHOT = "shell_snapshot"
    SHELL_TOOL = "shell_tool"
    SKILL_MCP_DEPENDENCY_INSTALL = "skill_mcp_dependency_install"
    SKILL_SEARCH = "skill_search"
    STANDALONE_WEB_SEARCH = "standalone_web_search"
    TOOL_CALL_MCP_ELICITATION = "tool_call_mcp_elicitation"
    TOOL_SUGGEST = "tool_suggest"
    UNIFIED_EXEC = "unified_exec"
    WEB_SEARCH_CACHED = "web_search_cached"
    WEB_SEARCH_REQUEST = "web_search_request"
    WORKSPACE_DEPENDENCIES = "workspace_dependencies"


_CODEX_DENIED_FEATURES = (
    _CodexDeniedFeature.APPS,
    _CodexDeniedFeature.BROWSER_USE,
    _CodexDeniedFeature.BROWSER_USE_EXTERNAL,
    _CodexDeniedFeature.CODE_MODE,
    _CodexDeniedFeature.CODE_MODE_HOST,
    _CodexDeniedFeature.CODE_MODE_ONLY,
    _CodexDeniedFeature.COLLABORATION_MODES,
    _CodexDeniedFeature.COMPUTER_USE,
    _CodexDeniedFeature.DEFAULT_MODE_REQUEST_USER_INPUT,
    _CodexDeniedFeature.GOALS,
    _CodexDeniedFeature.HOOKS,
    _CodexDeniedFeature.IMAGE_GENERATION,
    _CodexDeniedFeature.IN_APP_BROWSER,
    _CodexDeniedFeature.MEMORIES,
    _CodexDeniedFeature.MULTI_AGENT,
    _CodexDeniedFeature.MULTI_AGENT_V2,
    _CodexDeniedFeature.PLUGINS,
    _CodexDeniedFeature.REMOTE_PLUGIN,
    _CodexDeniedFeature.REQUEST_PERMISSIONS_TOOL,
    _CodexDeniedFeature.SHELL_SNAPSHOT,
    _CodexDeniedFeature.SHELL_TOOL,
    _CodexDeniedFeature.SKILL_MCP_DEPENDENCY_INSTALL,
    _CodexDeniedFeature.SKILL_SEARCH,
    _CodexDeniedFeature.STANDALONE_WEB_SEARCH,
    _CodexDeniedFeature.TOOL_CALL_MCP_ELICITATION,
    _CodexDeniedFeature.TOOL_SUGGEST,
    _CodexDeniedFeature.UNIFIED_EXEC,
    _CodexDeniedFeature.WEB_SEARCH_CACHED,
    _CodexDeniedFeature.WEB_SEARCH_REQUEST,
    _CodexDeniedFeature.WORKSPACE_DEPENDENCIES,
)


@dataclass(frozen=True, slots=True)
class _MacOsProcessIsolation:
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

        def literal_rule(operation: str, path: Path) -> str:
            return (
                f"(allow {operation} (literal "
                f"{json.dumps(str(path.resolve()))}))"
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

        traversal_paths = {
            parent
            for path in (*readable_paths, *writable_paths)
            for parent in path.resolve().parents
        }
        traversal_rules = [
            literal_rule("file-read* file-test-existence", path)
            for path in sorted(traversal_paths, key=str)
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
                *traversal_rules,
                *read_rules,
                *executable_rules,
                *write_rules,
                "",
            ]
        )


def build_codex_command(
    *,
    prompt: str,
    codex_binary: str,
    model: str,
    mcp_env: dict[str, str] | None,
    mcp_server_module: str = "whetstone.optim.codex.mcp_server",
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
    if mcp_env is not None:
        argv.extend(
            [
                "-c",
                f"mcp_servers.whetstone.command={json.dumps(sys.executable)}",
                "-c",
                "mcp_servers.whetstone.args="
                + json.dumps(
                    ["-m", mcp_server_module]
                ),
                "-c",
                "mcp_servers.whetstone.default_tools_approval_mode="
                + json.dumps(_MCP_TOOLS_APPROVAL_MODE),
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


def _require_absolute(field: str, raw: str | None, *, optional: bool) -> None:
    if not raw:
        if optional:
            return
        raise OpaqueStepError(f"Codex {field} is required")
    if not Path(raw).is_absolute():
        raise OpaqueStepError(
            f"Codex {field} must be absolute; {raw!r} resolves differently "
            "in the host and in the isolated child process"
        )


def _codex_budgets(timeout_seconds: float) -> Budgets:
    unbudgeted = UnbudgetedLimit()
    return Budgets(
        wall_time=FiniteDurationLimit(
            max_ns=max(1, math.ceil(timeout_seconds * 1_000_000_000))
        ),
        input_bytes=unbudgeted,
        payload_output=unbudgeted,
        memory_bytes=unbudgeted,
        cpu_time=unbudgeted,
        process_count=unbudgeted,
        file_size_bytes=unbudgeted,
        open_file_count=unbudgeted,
        disk_bytes=unbudgeted,
    )


def _retained_bytes(
    stream: RetainedPayloadStream,
    *,
    stream_name: str,
) -> bytes:
    if stream.dropped_bytes:
        raise OpaqueStepError(
            f"Codex {stream_name} was truncated despite unbudgeted output"
        )
    return stream.head + stream.tail


@dataclass(frozen=True, slots=True)
class CodexStructuredExecution:
    artifact_bytes: bytes
    stdout: bytes
    stderr: str
    isolation: dict[str, Any]


class CodexStructuredExecutionFailure(OpaqueStepError):
    def __init__(
        self,
        message: str,
        *,
        stdout: bytes,
        stderr: bytes,
        artifact_bytes: bytes = b"",
        isolation: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr
        self.artifact_bytes = artifact_bytes
        self.isolation = isolation or {}


class SubprocessCodexRunner:
    def __init__(
        self,
        *,
        executor: Executor,
        sqlite_path: str | None = None,
        runtime_config: EvalRuntimeConfig | None = None,
        runtime_config_class: str | None = None,
        reward_policy: RewardPolicy | None = None,
        codex_binary: str = "codex",
        model: str = "",
        mcp_server_module: str = "whetstone.optim.codex.mcp_server",
        timeout_seconds: float = 600.0,
        environment: Mapping[str, str] | None = None,
        prompt_builder: (
            Callable[[OptimStepRequest], str] | None
        ) = None,
    ) -> None:
        mcp_values = (sqlite_path, runtime_config, reward_policy)
        if any(value is not None for value in mcp_values) and not all(
            value is not None for value in mcp_values
        ):
            raise ValueError(
                "Codex MCP runner requires sqlite_path, runtime_config, and "
                "reward_policy together"
            )
        if sqlite_path is not None:
            _require_absolute("sqlite_path", sqlite_path, optional=False)
        if runtime_config is not None:
            _require_absolute(
                "partial_log_path",
                runtime_config.partial_log_path,
                optional=True,
            )
            _require_absolute(
                "prompt_cache_path",
                runtime_config.prompt_cache_path,
                optional=True,
            )
        if runtime_config is not None and runtime_config_class is None:
            raise ValueError(
                "runtime_config_class is required when runtime_config is set"
            )
        if runtime_config_class is not None:
            module_name, separator, class_name = runtime_config_class.partition(
                ":"
            )
            if not separator or not module_name or not class_name:
                raise ValueError(
                    "runtime_config_class must be module:Class"
                )
        self._sqlite_path = sqlite_path
        self._executor = executor
        self._runtime = runtime_config
        self._runtime_config_class = runtime_config_class
        self._reward_policy = reward_policy
        self._mcp_server_module = mcp_server_module
        self._binary = codex_binary
        self._model = model
        self._timeout = timeout_seconds
        self._prompt_builder = prompt_builder
        if environment is None:
            source_environment = dict(os.environ)
            configured_home = source_environment.get("CODEX_HOME")
            auth_source = (
                Path(configured_home)
                if configured_home is not None
                else Path.home() / ".codex"
            )
        else:
            source_environment = dict(environment)
            configured_home = source_environment.get("CODEX_HOME")
            auth_source = (
                Path(configured_home) if configured_home is not None else None
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
        }
        if self._runtime is not None:
            allowed.add(
                self._runtime.execution_policy.transport_policy.api_key_env
            )
        self._environment = {
            key: value
            for key, value in source_environment.items()
            if key in allowed
        }
        self._auth_source = auth_source

    def run(
        self, request: OptimStepRequest, handle: RuntimeToolHandle
    ) -> CodexRunResult:
        if (
            self._sqlite_path is None
            or self._runtime is None
            or self._runtime_config_class is None
            or self._reward_policy is None
        ):
            raise OpaqueStepError(
                "Codex optimizer run requires its MCP runtime configuration"
            )
        schema = CodexOutputArtifact.model_json_schema()
        run_id_schema = schema["properties"]["run_id"]
        assert isinstance(run_id_schema, dict)
        run_id_schema["const"] = request.run_id
        prompt = (
            _default_prompt(request, tool_name=handle.config.tool_name)
            if self._prompt_builder is None
            else self._prompt_builder(request)
        )
        execution = self._execute_structured(
            prompt=prompt,
            output_schema=schema,
            mcp_env={
                McpEnvironmentKey.SQLITE_PATH: self._sqlite_path,
                McpEnvironmentKey.TOOL_CONFIG: handle.config.model_dump_json(),
                McpEnvironmentKey.CAPACITY_BINDING: (
                    handle.binding.model_dump_json()
                ),
                McpEnvironmentKey.RUNTIME_CONFIG: (
                    self._runtime.model_dump_json()
                ),
                McpEnvironmentKey.RUNTIME_CONFIG_CLASS: (
                    self._runtime_config_class
                ),
                McpEnvironmentKey.REWARD_POLICY: (
                    self._reward_policy.model_dump_json()
                ),
            },
            stage_runtime=True,
        )
        artifact = _parse_output_artifact_bytes(
            execution.artifact_bytes,
            stdout=execution.stdout,
            stderr=execution.stderr,
            run_id=request.run_id,
            isolation=execution.isolation,
        )
        return CodexRunResult(artifact=artifact)

    def run_structured_prompt(
        self,
        *,
        prompt: str,
        output_schema: dict[str, Any],
    ) -> CodexStructuredExecution:

        return self._execute_structured(
            prompt=prompt,
            output_schema=output_schema,
            mcp_env=None,
            stage_runtime=False,
        )

    def _execute_structured(
        self,
        *,
        prompt: str,
        output_schema: dict[str, Any],
        mcp_env: dict[str, str] | None,
        stage_runtime: bool,
    ) -> CodexStructuredExecution:
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
            runtime_root: Path | None = None
            if stage_runtime:
                runtime_root = root / "runtime"
                runtime_root.mkdir()
                self._stage_runtime(runtime_root)
            codex_home = root / "codex-home"
            codex_home.mkdir()
            self._stage_auth(codex_home)
            isolated_environment = {
                **self._environment,
                "CODEX_HOME": str(codex_home),
            }
            exact_mcp_env = mcp_env
            if runtime_root is not None:
                isolated_environment["PYTHONPATH"] = str(runtime_root)
                assert exact_mcp_env is not None
                exact_mcp_env = {
                    **exact_mcp_env,
                    "PYTHONPATH": str(runtime_root),
                }
            schema_path = root / "output-schema.json"
            artifact_path = root / "last-message.json"
            schema_path.write_text(
                json.dumps(output_schema, sort_keys=True), encoding="utf-8"
            )
            command = build_codex_command(
                prompt=prompt,
                codex_binary=resolved_binary,
                model=self._model,
                mcp_env=exact_mcp_env,
                mcp_server_module=self._mcp_server_module,
                output_schema_path=str(schema_path),
                output_artifact_path=str(artifact_path),
                working_directory=working_directory,
            )
            profile_path = root / "codex.sb"
            direct_exec_command = [
                sys.executable,
                "-I",
                "-c",
                _DIRECT_EXEC_SOURCE,
                working_directory,
                *command,
            ]
            sandbox_wrapped_command = _MacOsProcessIsolation().wrap(
                direct_exec_command,
                profile_path=profile_path,
                readable_paths=self._readable_runtime_paths(
                    resolved_binary=Path(resolved_binary),
                    runtime_root=runtime_root,
                ),
                writable_paths=(
                    self._writable_runtime_paths(root)
                    if stage_runtime
                    else (root.resolve(),)
                ),
            )
            job = ExecutionJob(
                job_id=JobId(uuid4()),
                target=UntrustedCommandTarget(
                    argv=tuple(sandbox_wrapped_command),
                    stdin=b"",
                    containment_profile=(
                        ContainmentProfile.PROCESS_BOUNDARY_ONLY
                    ),
                ),
                env=EnvGrant.fixed(isolated_environment),
                budgets=_codex_budgets(self._timeout),
            )
            try:
                completed = self._executor.run(job)
            except ExecutorFailure as exc:
                raise OpaqueStepError("Codex execution failed") from exc
            stdout = _retained_bytes(
                completed.result.payload_outputs.stdout,
                stream_name="stdout",
            )
            stderr_bytes = _retained_bytes(
                completed.result.payload_outputs.stderr,
                stream_name="stderr",
            )
            isolation = {
                "strategy": "macos_sandbox_exec",
                "profile": profile_path.read_text(encoding="utf-8"),
                "denied_features": list(_CODEX_DENIED_FEATURES),
            }
            outcome = completed.result.outcome
            if isinstance(outcome, BudgetExceededOutcome):
                if outcome.axis is BudgetAxis.WALL_TIME:
                    raise TimeoutExpired(
                        sandbox_wrapped_command,
                        self._timeout,
                        output=stdout,
                        stderr=stderr_bytes,
                    )
                raise CodexStructuredExecutionFailure(
                    "Codex execution failed with an unexpected budget outcome",
                    stdout=stdout,
                    stderr=stderr_bytes,
                    isolation=isolation,
                )
            if isinstance(outcome, SpawnAbsentOutcome | SpawnFailedOutcome):
                raise CodexStructuredExecutionFailure(
                    "Codex process could not be spawned",
                    stdout=stdout,
                    stderr=stderr_bytes,
                    isolation=isolation,
                )
            if isinstance(outcome, ExitedOutcome):
                return_code = outcome.exit_code
            elif isinstance(outcome, SignaledOutcome):
                return_code = -outcome.signal_number
            else:
                raise CodexStructuredExecutionFailure(
                    f"Codex execution failed with outcome {outcome.kind}",
                    stdout=stdout,
                    stderr=stderr_bytes,
                    isolation=isolation,
                )
            try:
                stderr = _decode_stderr(stderr_bytes)
            except OpaqueStepError as exc:
                raise CodexStructuredExecutionFailure(
                    str(exc),
                    stdout=stdout,
                    stderr=stderr_bytes,
                    isolation=isolation,
                ) from exc
            if return_code:
                try:
                    artifact_bytes = (
                        artifact_path.read_bytes()
                        if artifact_path.is_file()
                        else b""
                    )
                except OSError as exc:
                    raise CodexStructuredExecutionFailure(
                        "Codex final output artifact could not be read",
                        stdout=stdout,
                        stderr=stderr_bytes,
                        isolation=isolation,
                    ) from exc
                raise CodexStructuredExecutionFailure(
                    f"Codex exited {return_code}: {stderr[-2000:]}",
                    stdout=stdout,
                    stderr=stderr_bytes,
                    artifact_bytes=artifact_bytes,
                    isolation=isolation,
                )
            if not artifact_path.is_file():
                raise CodexStructuredExecutionFailure(
                    "Codex produced no final output artifact",
                    stdout=stdout,
                    stderr=stderr_bytes,
                    isolation=isolation,
                )
            try:
                artifact_bytes = artifact_path.read_bytes()
            except OSError as exc:
                raise CodexStructuredExecutionFailure(
                    "Codex final output artifact could not be read",
                    stdout=stdout,
                    stderr=stderr_bytes,
                    isolation=isolation,
                ) from exc
            return CodexStructuredExecution(
                artifact_bytes=artifact_bytes,
                stdout=stdout,
                stderr=stderr,
                isolation=isolation,
            )

    def _stage_auth(self, destination: Path) -> None:
        source = self._auth_source
        if source is None:
            return
        for name in ("auth.json", ".credentials.json"):
            candidate = source / name
            if candidate.is_file():
                shutil.copy2(candidate, destination / name)

    def _stage_runtime(self, destination: Path) -> None:
        if self._runtime is None:
            raise OpaqueStepError("Codex MCP runtime is not configured")
        package_root = Path(__file__).resolve().parents[2]
        shutil.copytree(
            package_root,
            destination / package_root.name,
            dirs_exist_ok=True,
        )
        module_name = self._runtime.row_job_entrypoint.partition(":")[0]
        top_level = module_name.partition(".")[0]
        spec = importlib.util.find_spec(top_level)
        if spec is None:
            raise OpaqueStepError(
                f"transport factory package {top_level!r} was not found"
            )
        if spec.submodule_search_locations:
            target = destination / top_level

            for location in reversed(spec.submodule_search_locations):
                source = Path(location)
                if source.resolve() != package_root.resolve():
                    shutil.copytree(source, target, dirs_exist_ok=True)
        elif spec.origin is not None:
            shutil.copy2(spec.origin, destination / Path(spec.origin).name)

    def _readable_runtime_paths(
        self,
        *,
        resolved_binary: Path,
        runtime_root: Path | None,
    ) -> tuple[Path, ...]:
        paths = {
            resolved_binary.resolve(),
            Path(sys.executable).resolve(),
            Path(sys.prefix).resolve(),
            Path(sys.base_prefix).resolve(),
        }
        if runtime_root is not None:
            paths.add(runtime_root.resolve())
        for key in ("SSL_CERT_FILE", "SSL_CERT_DIR"):
            raw = self._environment.get(key)
            if raw:
                paths.add(Path(raw).resolve())
        return tuple(sorted(paths, key=str))

    def _writable_runtime_paths(self, root: Path) -> tuple[Path, ...]:
        if self._sqlite_path is None or self._runtime is None:
            raise OpaqueStepError("Codex MCP runtime is not configured")
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

            state_paths.add(
                partial_path.with_name(f".{partial_path.name}.lock")
            )
            state_paths.add(partial_path.parent)
        if self._runtime.prompt_cache_path:
            cache_path = Path(self._runtime.prompt_cache_path).resolve()
            if not cache_path.is_dir():
                raise OpaqueStepError(
                    "Codex prompt-cache directory does not exist"
                )
            state_paths.add(cache_path)
        return tuple(sorted(state_paths, key=str))


def _default_prompt(
    request: OptimStepRequest, *, tool_name: str
) -> str:
    context = json.dumps(
        request.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        f"Use only the external {tool_name} MCP tool for measurements. "
        "Do not call any built-in tool. Build proposals from the exact "
        "candidate base_ref, model route, payload template, Tool Config, "
        "capacity, budget, pools, hyperparameters, and output contract in "
        "the serialized request below. Guidance, not a checked requirement: "
        "evaluating candidate drafts through MCP before selecting them "
        "produces better proposals, but the artifact is accepted on its "
        "proposal contract alone. Write the schema-conforming final "
        "artifact with exactly the requested proposal count.\n"
        f"OPTIM_STEP_REQUEST_JSON={context}"
    )


def _parse_jsonl_events(stdout: bytes) -> tuple[dict[str, Any], ...]:
    events: list[dict[str, Any]] = []
    for ordinal, raw in enumerate(stdout.splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            value = decode_strict_json_bytes(
                raw,
                max_bytes=len(raw),
                max_depth=len(raw),
            )
        except StrictJsonDecodeError as exc:
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
    stdout: bytes,
    stderr: str,
    run_id: str,
    isolation: dict[str, Any] | None = None,
) -> CodexOutputArtifact:
    if not path.is_file():
        raise OpaqueStepError("Codex produced no final output artifact")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise OpaqueStepError(
            "Codex final output artifact failed schema validation"
        ) from exc
    return _parse_output_artifact_bytes(
        raw,
        stdout=stdout,
        stderr=stderr,
        run_id=run_id,
        isolation=isolation,
    )


def _parse_output_artifact_bytes(
    raw: bytes,
    *,
    stdout: bytes,
    stderr: str,
    run_id: str,
    isolation: dict[str, Any] | None = None,
) -> CodexOutputArtifact:
    try:
        decode_strict_json_bytes(
            raw,
            max_bytes=len(raw),
            max_depth=len(raw),
        )
        artifact = CodexOutputArtifact.model_validate_json(raw)
    except (StrictJsonDecodeError, ValidationError) as exc:
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


def _decode_stderr(stderr: bytes) -> str:
    try:
        return stderr.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise OpaqueStepError("Codex stderr is not valid UTF-8") from exc


__all__ = [
    "CodexStructuredExecution",
    "CodexStructuredExecutionFailure",
    "SubprocessCodexRunner",
    "build_codex_command",
]
