from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
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
    FiniteOutput,
    JobId,
    OutputOverflowPolicy,
    PayloadRetentionBudget,
    RetainedPayloadStream,
    SignaledOutcome,
    SpawnAbsentOutcome,
    SpawnFailedOutcome,
    StreamRetentionBudget,
    UnbudgetedLimit,
    UntrustedCommandTarget,
    WorkingDirectoryGrant,
)
from dr_serialize import StrictJsonDecodeError, decode_strict_json_bytes
from pydantic import ValidationError

from whetstone.experiment.reward import RewardPolicy
from whetstone.optim.codex.adapter import (
    CodexMcpHostFailure,
    CodexOutputArtifact,
    CodexRunResult,
    CodexStructuredExecutionFailure,
    CodexWallBudgetExceeded,
    OpaqueStepError,
    codex_lease_token_hash,
    codex_run_lease_binding,
)
from whetstone.optim.codex.containment import (
    CODEX_AUTH_FILENAMES,
    CODEX_CONTAINMENT_PROFILE,
    CODEX_DEFAULT_MAX_OUTPUT_BYTES,
    CODEX_DENIED_FEATURES,
    CODEX_FILESYSTEM_POLICY,
    CODEX_NETWORK_POLICY,
)
from whetstone.optim.codex.mcp_environment import McpEnvironmentKey
from whetstone.optim.codex.mcp_host import CodexMcpEndpoint, CodexMcpHost
from whetstone.optim.codex.mcp_server import build_server_from_env
from whetstone.optim.contracts import OptimStepRequest
from whetstone.optim.tools.contracts import (
    RuntimeToolHandle,
    ToolCapacityBinding,
)

if TYPE_CHECKING:
    from whetstone.optim.codex.config import EvalRuntimeConfig

_MACOS_SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")


_MCP_TOOLS_APPROVAL_MODE = "auto"

#: The variable the Codex CLI reads the evaluation endpoint's bearer
#: token from. It crosses into the CLI's configuration, so it is pinned.
CODEX_MCP_TOKEN_ENV = "WS_MCP_BEARER_TOKEN"

#: The Codex config key that carries the control's reasoning effort. It
#: crosses into a foreign CLI and nothing derives it, so it is pinned by
#: a golden test.
_REASONING_EFFORT_CONFIG_KEY = "model_reasoning_effort"

#: dr-exec v1 rejects a finite limit on these axes, so the artifact records
#: them as unbudgeted rather than claiming containment it cannot enforce.
_CODEX_UNBUDGETED_AXES = (
    BudgetAxis.INPUT_BYTES.value,
    BudgetAxis.MEMORY_BYTES.value,
    BudgetAxis.CPU_TIME.value,
    BudgetAxis.PROCESS_COUNT.value,
    BudgetAxis.FILE_SIZE_BYTES.value,
    BudgetAxis.OPEN_FILE_COUNT.value,
    BudgetAxis.DISK_BYTES.value,
)
_DIRECT_EXEC_SOURCE: Final = (
    "import os,sys;os.execv(sys.argv[1],sys.argv[1:])"
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
    reasoning_effort: str,
    mcp_endpoint: CodexMcpEndpoint | None,
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
    for feature in CODEX_DENIED_FEATURES:
        argv.extend(["--disable", feature])
    if model:
        argv.extend(["--model", model])
    if reasoning_effort:
        # The CLI has no reasoning-effort flag; it is a config key, and
        # --strict-config rejects an unknown one, so a misspelling here
        # fails the launch rather than being silently ignored.
        argv.extend(
            [
                "-c",
                f"{_REASONING_EFFORT_CONFIG_KEY}={json.dumps(reasoning_effort)}",
            ]
        )
    if mcp_endpoint is not None:
        # A URL, not a command: the agent connects to a server whetstone
        # already runs. It never spawns the server, so it never inherits
        # a profile that can write the store. The bearer token comes from
        # the agent's environment rather than the argv, which is world
        # readable through the process table.
        argv.extend(
            [
                "-c",
                "mcp_servers.whetstone.url="
                + json.dumps(mcp_endpoint.url),
                "-c",
                "mcp_servers.whetstone.bearer_token_env_var="
                + json.dumps(CODEX_MCP_TOKEN_ENV),
                "-c",
                "mcp_servers.whetstone.default_tools_approval_mode="
                + json.dumps(_MCP_TOOLS_APPROVAL_MODE),
            ]
        )
    argv.append(prompt)
    return argv


def _capacity_subject_key(binding: ToolCapacityBinding) -> str:
    """The capacity subject as one exact string for the lease binding."""
    subject = binding.subject_ref
    if subject is None:
        return ""
    return f"{subject.schema_name}@{subject.content_hash}"


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


def _codex_budgets(
    *,
    wall_seconds: float,
    max_output_bytes: int,
) -> Budgets:
    """Bound the Codex job on every axis dr-exec v1 can enforce.

    ``wall_time`` is the hard stop and ``payload_output`` bounds retention.
    ``process_count`` and the resource axes stay unbudgeted because dr-exec
    v1 rejects a finite limit on them; the process boundary and the macOS
    sandbox profile are the containment, and the wall budget terminates a
    run that spawns without bound.
    """
    unbudgeted = UnbudgetedLimit()
    # dr-exec requires the four retention windows to sum to max_bytes
    # exactly; the remainder lands on the stdout head, which is where the
    # JSONL event stream the runner parses begins.
    quarter = max_output_bytes // 4
    stdout_head = max_output_bytes - 3 * quarter
    return Budgets(
        wall_time=FiniteDurationLimit.from_seconds(wall_seconds),
        input_bytes=unbudgeted,
        payload_output=FiniteOutput(
            max_bytes=max_output_bytes,
            overflow_policy=OutputOverflowPolicy.MARKED_TRUNCATION,
            retention=PayloadRetentionBudget(
                stdout=StreamRetentionBudget(
                    head_bytes=stdout_head,
                    tail_bytes=quarter,
                ),
                stderr=StreamRetentionBudget(
                    head_bytes=quarter,
                    tail_bytes=quarter,
                ),
            ),
        ),
        memory_bytes=unbudgeted,
        cpu_time=unbudgeted,
        process_count=unbudgeted,
        file_size_bytes=unbudgeted,
        open_file_count=unbudgeted,
        disk_bytes=unbudgeted,
    )


#: Identifies whetstone's elision marker inside an otherwise-Codex
#: stream. The parser has to tell this synthetic line from the agent's
#: own output, and a human-readable prefix cannot carry that: Codex
#: streams reasoning, and a line opening with a bracketed aside is
#: ordinary prose. Matching one loosely drops a genuine record silently
#: and misdirects the stitch-boundary search. This fixed token is not
#: something an agent emits by accident.
CODEX_ELISION_SENTINEL = "whetstone-elision-53f1c0e2-8a47-4d19-9b6e-2c7d0f5a1b83"
#: The marker whetstone writes between a retained head and tail. It
#: leads the line so a reader can find it without parsing, and it is not
#: valid JSON, so a JSONL consumer fails loudly on it rather than
#: silently treating a stitched stream as contiguous.
CODEX_ELIDED_MARKER_PREFIX = f"[... {CODEX_ELISION_SENTINEL} ".encode()
#: Matches exactly the marker ``_retained_bytes`` writes, anchored on the
#: sentinel and on the whole line -- never a prefix.
CODEX_ELIDED_MARKER_PATTERN = re.compile(
    rb"^\[\.\.\. "
    + re.escape(CODEX_ELISION_SENTINEL.encode())
    + rb" \d+ bytes elided \.\.\.\]$"
)


def _is_elision_marker(raw: bytes) -> bool:
    """Is this line whetstone's elision marker, and not agent output?"""
    return CODEX_ELIDED_MARKER_PATTERN.match(raw) is not None


@contextlib.contextmanager
def _entered_mcp_host(
    build: Callable[[], CodexMcpHost],
) -> Iterator[CodexMcpEndpoint]:
    """Bring the evaluation host up, and take it back down, under one code.

    This exists to keep ``codex_mcp_host_failed`` meaning what the ledger
    reads it as: whetstone's own server, not the agent. Only the two
    phases that are whetstone's are wrapped.

    Startup -- building the server and entering the host -- fails before
    the agent runs, so nothing was paid for. Teardown failures are
    wrapped too, but only when the block itself succeeded: the host is
    still whetstone's, yet the run did happen, so the message says so
    rather than implying the agent never started. When the block raised,
    that exception is the real failure and a teardown error must not
    displace it.

    Neither ``CodexMcpHostError`` nor a lifespan error is an
    ``OpaqueStepError``, so unwrapped they would pass the adapter's
    checkpoint and leave this ``NO_REDRIVE`` effect nonterminal until the
    lease lapsed.
    """
    try:
        host = build()
        endpoint = host.__enter__()
    except Exception as exc:
        raise CodexMcpHostFailure(
            "Codex MCP evaluation host failed to start", cause=exc
        ) from exc
    try:
        yield endpoint
    except BaseException:
        # The block's own failure is the real one. Let the host tear down
        # and let that exception continue; a teardown error here must not
        # mask it.
        host.__exit__(*sys.exc_info())
        raise
    try:
        host.__exit__(None, None, None)
    except Exception as exc:
        raise CodexMcpHostFailure(
            "Codex MCP evaluation host failed to shut down after a "
            "completed run",
            cause=exc,
        ) from exc


@dataclass(frozen=True, slots=True)
class _RetainedStream:
    """One captured stream plus whether the output budget truncated it."""

    data: bytes
    dropped_bytes: int

    @property
    def truncated(self) -> bool:
        return self.dropped_bytes > 0


def _retained_bytes(stream: RetainedPayloadStream) -> _RetainedStream:
    """Read a retained stream, reporting truncation instead of raising.

    Under a finite ``payload_output`` budget truncation is an expected
    outcome, not a contract violation: the retained head and tail are still
    exact, and the artifact records how many bytes the budget dropped.

    The head and tail were not adjacent in the real stream, so joining
    them bare would fabricate a line the process never emitted -- the
    truncated last head line running straight into the truncated first
    tail line. Parsed as JSONL that is either a malformed event at a
    boundary Codex never produced, or a well-formed event that never
    happened. An explicit marker line separates them, on its own line in
    both directions, so nothing downstream can read the join as
    contiguous.
    """
    if stream.dropped_bytes <= 0:
        return _RetainedStream(
            data=stream.head + stream.tail,
            dropped_bytes=stream.dropped_bytes,
        )
    marker = (
        f"{CODEX_ELIDED_MARKER_PREFIX.decode()}"
        f"{stream.dropped_bytes} bytes elided ...]"
    ).encode()
    head = stream.head
    if head and not head.endswith(b"\n"):
        head += b"\n"
    tail = stream.tail
    return _RetainedStream(
        data=head + marker + b"\n" + tail,
        dropped_bytes=stream.dropped_bytes,
    )


def _stdout_was_truncated(isolation: dict[str, Any] | None) -> bool:
    """Did the output budget stitch this run's stdout?

    The runner already records truncation in the isolation evidence, so
    the JSONL parser reads its tolerance from the same place rather than
    from a second flag that could disagree with the recorded artifact.
    """
    if not isolation:
        return False
    truncation = isolation.get("output_truncation")
    if not isinstance(truncation, dict):
        return False
    return bool(truncation.get("stdout_truncated"))


@dataclass(frozen=True, slots=True)
class CodexStructuredExecution:
    artifact_bytes: bytes
    stdout: bytes
    stderr: str
    isolation: dict[str, Any]


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
        reasoning_effort: str = "",
        timeout_seconds: float = 600.0,
        max_output_bytes: int = CODEX_DEFAULT_MAX_OUTPUT_BYTES,
        environment: Mapping[str, str] | None = None,
        extra_environment_keys: frozenset[str] = frozenset(),
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
        self._binary = codex_binary
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._timeout = timeout_seconds
        if max_output_bytes < 4:
            raise ValueError("max_output_bytes must leave room for retention")
        self._max_output_bytes = max_output_bytes
        self._prompt_builder = prompt_builder
        source_environment = (
            dict(os.environ) if environment is None else dict(environment)
        )
        # Where this run's credentials are copied from. An explicit
        # CODEX_HOME names it; otherwise it is the location the Codex CLI
        # itself uses, and a user who logged in normally has it there.
        # Resolving it the same way whether or not the caller passed an
        # explicit environment is what lets the preflight probe -- which
        # always passes one -- see the credentials the real run will use.
        # The path is a *source to copy from*: CODEX_HOME is rewritten
        # per run to the scratch directory, and nothing here widens the
        # environment the untrusted agent receives.
        configured_home = source_environment.get("CODEX_HOME")
        auth_source = (
            Path(configured_home)
            if configured_home is not None
            else Path.home() / ".codex"
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
        # The allowlist is deliberately narrow: the Codex process is
        # untrusted and inherits nothing by default. A caller that must
        # grant one more variable names it explicitly.
        allowed.update(extra_environment_keys)
        # The task-model API key is the eval transport's credential and is
        # deliberately absent from ``allowed``. The Codex process is a
        # general-purpose agent with network access, so holding that key
        # would let it score candidates directly -- unadmitted, unleased,
        # and outside the ledger. Only the MCP evaluation server needs it,
        # and whetstone hosts that server in its own environment.
        secret_key_env = (
            self._runtime.execution_policy.transport_policy.api_key_env
            if self._runtime is not None
            else None
        )
        self._mcp_secret_environment = (
            {secret_key_env: source_environment[secret_key_env]}
            if secret_key_env is not None
            and secret_key_env in source_environment
            else {}
        )
        self._environment = {
            key: value
            for key, value in source_environment.items()
            if key in allowed and key != secret_key_env
        }
        self._auth_source = auth_source

    def codex_process_environment(self) -> dict[str, str]:
        """Exactly what the untrusted Codex process is granted.

        ``CODEX_HOME`` is rewritten per run to point at that run's
        scratch directory; everything else is fixed here.
        """
        return dict(self._environment)

    def mcp_server_secret_environment(self) -> dict[str, str]:
        """The credentials only the evaluation server receives.

        These never enter :meth:`codex_process_environment`, so the agent
        cannot read them out of its own environment.
        """
        return dict(self._mcp_secret_environment)

    def run(
        self,
        request: OptimStepRequest,
        handle: RuntimeToolHandle,
        *,
        lease_token: str,
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
        token_hash = codex_lease_token_hash(lease_token)
        schema = CodexOutputArtifact.model_json_schema()
        # Pin the two fields the adapter checks before it reads anything
        # else, so a non-conforming artifact fails at the CLI boundary
        # rather than as a Step terminal failure.
        for field, constant in (
            ("run_id", request.run_id),
            ("lease_token_hash", token_hash),
        ):
            field_schema = schema["properties"][field]
            assert isinstance(field_schema, dict)
            field_schema["const"] = constant
        prompt = (
            _default_prompt(
                request,
                tool_name=handle.config.tool_name,
                lease_token_hash=token_hash,
                max_tool_calls=handle.config.capacity.max_accepted_calls,
            )
            if self._prompt_builder is None
            else self._prompt_builder(request)
        )
        # whetstone hosts the evaluation server itself, before the sandbox
        # exists, and hands the agent only a loopback URL and this run's
        # bearer token. The server is the sole writer of the whetstone
        # store, and SQLite needs real write access to that file: running
        # it under the agent's profile would hand the agent the ledger and
        # the admission-capacity rows that cap paid evaluations.
        #
        # Building the server and bringing the host up are the two ways
        # this Step fails *before* the agent ever runs: a malformed or
        # mismatched runtime config, a squatted port, a bind or lifespan
        # failure, a startup that misses its deadline. None of those are
        # OpaqueStepError, so they would unwind past the adapter's
        # checkpoint and leave this NO_REDRIVE effect nonterminal until
        # the lease lapsed, so they are normalized. The host closes its
        # own thread and socket before raising, so that strands nothing.
        #
        # Only those phases are the host's. codex_mcp_host_failed tells
        # the ledger one specific thing -- whetstone's own server never
        # came up, so the agent never ran and this Step paid for nothing.
        # Covering the agent execution under it too would let any
        # unforeseen failure inside the Codex run claim that,
        # misreporting a Step that may have spent money as one that
        # never started, and burying the real defect behind a host
        # diagnostic. Every failure still terminalizes; they differ only
        # in which code they carry.
        host = _entered_mcp_host(
            lambda: CodexMcpHost(
                build_server_from_env(
                    self._mcp_server_environment(handle, lease_token)
                ),
                auth_token=lease_token,
            )
        )
        with host as endpoint:
            try:
                execution = self._execute_structured(
                    prompt=prompt,
                    output_schema=schema,
                    mcp_endpoint=endpoint,
                )
            except OpaqueStepError:
                # The agent's own failures already carry their taxonomy
                # and their isolation evidence; re-wrapping loses both,
                # and the adapter terminalizes them as they are.
                raise
            except Exception as exc:
                # An unforeseen defect inside the execution path. It
                # still has to terminalize -- the harness releases the
                # effect lease only once the adapter returns, so
                # unwinding here wedges this NO_REDRIVE run until the
                # lease lapses -- but under the execution taxonomy,
                # because the host is up and the agent may have run.
                raise OpaqueStepError(
                    f"Codex execution failed unexpectedly: {exc}"
                ) from exc
        artifact = _parse_output_artifact_bytes(
            execution.artifact_bytes,
            stdout=execution.stdout,
            stderr=execution.stderr,
            run_id=request.run_id,
            isolation=execution.isolation,
            stdout_truncated=_stdout_was_truncated(execution.isolation),
        )
        return CodexRunResult(artifact=artifact)

    def _mcp_server_environment(
        self,
        handle: RuntimeToolHandle,
        lease_token: str,
    ) -> dict[str, str]:
        """Configure the evaluation server this Step hosts.

        This never reaches the Codex process. It stays in whetstone's own
        environment, which is why the task-model credential and the store
        path are safe to name here.
        """
        assert self._sqlite_path is not None
        assert self._runtime is not None
        assert self._runtime_config_class is not None
        assert self._reward_policy is not None
        return {
            McpEnvironmentKey.SQLITE_PATH: self._sqlite_path,
            McpEnvironmentKey.TOOL_CONFIG: handle.config.model_dump_json(),
            McpEnvironmentKey.CAPACITY_BINDING: (
                handle.binding.model_dump_json()
            ),
            McpEnvironmentKey.RUNTIME_CONFIG: self._runtime.model_dump_json(),
            McpEnvironmentKey.RUNTIME_CONFIG_CLASS: (
                self._runtime_config_class
            ),
            McpEnvironmentKey.REWARD_POLICY: (
                self._reward_policy.model_dump_json()
            ),
            McpEnvironmentKey.RUN_LEASE_TOKEN: lease_token,
            McpEnvironmentKey.RUN_LEASE_BINDING: codex_run_lease_binding(
                token=lease_token,
                store_namespace_key=str(handle.config.store_namespace_key),
                tool_config_hash=str(handle.config.identity_hash()),
                capacity_scope=handle.binding.scope.value,
                capacity_subject=_capacity_subject_key(handle.binding),
            ),
            **self._mcp_secret_environment,
        }

    def run_structured_prompt(
        self,
        *,
        prompt: str,
        output_schema: dict[str, Any],
    ) -> CodexStructuredExecution:

        return self._execute_structured(
            prompt=prompt,
            output_schema=output_schema,
            mcp_endpoint=None,
        )

    def _execute_structured(
        self,
        *,
        prompt: str,
        output_schema: dict[str, Any],
        mcp_endpoint: CodexMcpEndpoint | None,
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
            codex_home = root / "codex-home"
            codex_home.mkdir()
            self.stage_auth(codex_home)
            isolated_environment = {
                **self._environment,
                "CODEX_HOME": str(codex_home),
            }
            if mcp_endpoint is not None:
                # The only MCP material the agent gets: a bearer token for
                # an endpoint whetstone already runs.
                isolated_environment[CODEX_MCP_TOKEN_ENV] = (
                    mcp_endpoint.auth_token
                )
            schema_path = root / "output-schema.json"
            artifact_path = root / "last-message.json"
            schema_path.write_text(
                json.dumps(output_schema, sort_keys=True), encoding="utf-8"
            )
            command = build_codex_command(
                prompt=prompt,
                codex_binary=resolved_binary,
                model=self._model,
                reasoning_effort=self._reasoning_effort,
                mcp_endpoint=mcp_endpoint,
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
                *command,
            ]
            sandbox_wrapped_command = _MacOsProcessIsolation().wrap(
                direct_exec_command,
                profile_path=profile_path,
                readable_paths=self._readable_runtime_paths(
                    resolved_binary=Path(resolved_binary),
                ),
                writable_paths=self._writable_runtime_paths(root),
            )
            budgets = _codex_budgets(
                wall_seconds=self._timeout,
                max_output_bytes=self._max_output_bytes,
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
                workspace=WorkingDirectoryGrant.caller(root),
                budgets=budgets,
            )
            try:
                # dr-exec's Executor.run is a coroutine; the optimizer
                # harness drives Steps synchronously, so this path takes
                # the blocking entry point.
                completed = self._executor.run_blocking(job)
            except ExecutorFailure as exc:
                raise OpaqueStepError("Codex execution failed") from exc
            stdout_stream = _retained_bytes(
                completed.result.payload_outputs.stdout
            )
            stderr_stream = _retained_bytes(
                completed.result.payload_outputs.stderr
            )
            stdout = stdout_stream.data
            stderr_bytes = stderr_stream.data
            isolation = {
                "strategy": "macos_sandbox_exec",
                "profile": profile_path.read_text(encoding="utf-8"),
                "denied_features": list(CODEX_DENIED_FEATURES),
                "network_policy": CODEX_NETWORK_POLICY,
                "filesystem_policy": CODEX_FILESYSTEM_POLICY,
                "containment_profile": CODEX_CONTAINMENT_PROFILE,
                "budgets": {
                    "wall_seconds": self._timeout,
                    "max_output_bytes": self._max_output_bytes,
                    # dr-exec v1 accepts no finite limit on these axes; the
                    # wall budget and the process boundary are the stop.
                    "unbudgeted_axes": list(_CODEX_UNBUDGETED_AXES),
                },
                # A truncated stream is a stitched head+tail carrying an
                # elision marker, not a contiguous capture. Anything
                # reading these bytes back has to know that.
                "output_truncation": {
                    "stdout_truncated": stdout_stream.truncated,
                    "stdout_dropped_bytes": stdout_stream.dropped_bytes,
                    "stderr_truncated": stderr_stream.truncated,
                    "stderr_dropped_bytes": stderr_stream.dropped_bytes,
                },
            }
            outcome = completed.result.outcome
            if isinstance(outcome, BudgetExceededOutcome):
                if outcome.axis is BudgetAxis.WALL_TIME:
                    # A wall stop is the most likely end of a long-running
                    # paid agent, so it terminalizes the Step through the
                    # adapter's own failure taxonomy. Raising the raw
                    # subprocess error here would unwind past the harness's
                    # effect-lease maintenance and leave the lease
                    # non-terminal until it lapsed.
                    raise CodexWallBudgetExceeded(
                        "Codex exceeded its wall budget of "
                        f"{self._timeout} seconds",
                        wall_seconds=self._timeout,
                        stdout=stdout,
                        stderr=stderr_bytes,
                        isolation=isolation,
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

    @property
    def auth_source(self) -> Path:
        """The directory this run's Codex credentials are copied from."""
        return self._auth_source

    def stage_auth(self, destination: Path) -> None:
        """Copy the run's Codex credentials into its scratch CODEX_HOME.

        The credentials reach the untrusted agent as files inside the
        scratch home it is given, never as environment values.
        """
        for name in CODEX_AUTH_FILENAMES:
            candidate = self._auth_source / name
            if candidate.is_file():
                shutil.copy2(candidate, destination / name)

    def _readable_runtime_paths(
        self,
        *,
        resolved_binary: Path,
    ) -> tuple[Path, ...]:
        paths = {
            resolved_binary.resolve(),
            Path(sys.executable).resolve(),
            Path(sys.prefix).resolve(),
            Path(sys.base_prefix).resolve(),
        }
        for key in ("SSL_CERT_FILE", "SSL_CERT_DIR"):
            raw = self._environment.get(key)
            if raw:
                paths.add(Path(raw).resolve())
        # Read-only, and only what the caller already chose to grant: a
        # PYTHONPATH entry is in the environment because the allowlist
        # admitted it, and the profile must let the interpreter reach it.
        # Nothing here becomes writable.
        for entry in self._environment.get("PYTHONPATH", "").split(os.pathsep):
            if entry:
                paths.add(Path(entry).resolve())
        return tuple(sorted(paths, key=str))

    def _writable_runtime_paths(self, root: Path) -> tuple[Path, ...]:
        """The only paths the Codex process may write: its own scratch.

        The whetstone store is deliberately absent. It is the durable
        ledger and it holds the admission-capacity rows that cap paid
        evaluations, and SQLite needs real write access to that file --
        so granting it here would let the agent forge Tool Results or
        clear its own budget. The evaluation server is the sole writer,
        and whetstone runs it outside this profile.
        """
        return (root.resolve(),)


def _default_prompt(
    request: OptimStepRequest,
    *,
    tool_name: str,
    lease_token_hash: str,
    max_tool_calls: int,
) -> str:
    """The instruction Codex receives.

    Evaluating through the Tool is mandatory, not guidance: the artifact
    carries no candidate body, so the only way to return a candidate is to
    name the ``call_id`` of a call that was actually admitted and scored.
    """
    context = json.dumps(
        request.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        f"Use only the external {tool_name} MCP tool for measurements. "
        "Do not call any built-in tool. Build candidate templates from the "
        "exact candidate base_ref, model route, payload template, Tool "
        "Config, capacity, budget, pools, hyperparameters, and output "
        "contract in the serialized request below.\n"
        "Evaluating through the MCP tool is mandatory. Every candidate you "
        "consider must be submitted to the tool with a call_id you choose; "
        f"you may make at most {max_tool_calls} calls, and every further "
        "call is refused with refused=true and refusal_class=capacity. A "
        "refusal does not end the run: stop evaluating and write your "
        "artifact.\n"
        "Write a schema-conforming final artifact naming every call_id you "
        "evaluated in evaluated_call_ids, and selected_call_id set to the "
        "call_id whose candidate you chose. The artifact carries no "
        "candidate body: a template that was never evaluated through the "
        "tool cannot be returned. Set selected_call_id to null to keep the "
        "run's seed candidate. Copy lease_token_hash verbatim as "
        f"{lease_token_hash!r}.\n"
        f"OPTIM_STEP_REQUEST_JSON={context}"
    )


@dataclass(frozen=True, slots=True)
class _ParsedJsonlEvents:
    """The complete events in a stream, plus what truncation cost."""

    events: tuple[dict[str, Any], ...]
    #: Lines the output budget cut mid-record, dropped rather than parsed.
    dropped_partial_lines: int


def _parse_jsonl_events(
    stdout: bytes,
    *,
    truncated: bool,
) -> _ParsedJsonlEvents:
    """Read the JSONL event stream, tolerating only budget damage.

    An untruncated stream is exactly what Codex wrote, so every line must
    parse and a malformed one is a contract violation. A truncated stream
    is a stitched head+tail: it carries the elision marker, which is
    deliberately not JSON, and the budget may have cut the head's last
    line and the tail's first line mid-record. Those are artifacts of
    retention, not of what Codex emitted, so they are skipped and counted
    rather than failing a run whose final artifact is valid.

    The tolerance is bounded to the two lines a stitch can damage: a
    truncated stream that loses more than that is still malformed.
    """
    events: list[dict[str, Any]] = []
    lines = [raw for raw in stdout.splitlines() if raw.strip()]
    dropped_partial = 0
    for ordinal, raw in enumerate(lines, start=1):
        if truncated and _is_elision_marker(raw):
            continue
        try:
            value = decode_strict_json_bytes(
                raw,
                max_bytes=len(raw),
                max_depth=len(raw),
            )
        except StrictJsonDecodeError as exc:
            if truncated and _is_cut_stitch_boundary(ordinal, lines):
                dropped_partial += 1
                continue
            raise OpaqueStepError(
                f"Codex JSONL event {ordinal} is malformed"
            ) from exc
        if not isinstance(value, dict):
            raise OpaqueStepError(
                f"Codex JSONL event {ordinal} is not an object"
            )
        events.append(value)
    return _ParsedJsonlEvents(
        events=tuple(events),
        dropped_partial_lines=dropped_partial,
    )


def _is_cut_stitch_boundary(ordinal: int, lines: list[bytes]) -> bool:
    """Was this line demonstrably cut mid-record by the retention window?

    Adjacency to the marker is necessary but not sufficient. Retention
    can end or begin exactly on a record boundary, and then the line
    beside the marker is a *whole* record that Codex emitted. Forgiving
    it on position alone silently deletes genuine malformed process
    output and reports it as retention damage, so the persisted
    ``jsonl_events`` no longer match the retained stream.

    On the head side a cut is demonstrable from the line's own shape:
    the head's last line is the *front* of a record whose end the budget
    removed, so it opens a record it never closes. A line that is
    balanced -- ``not json``, or a whole object followed by trailing
    garbage -- is not a partial record, so it stays a contract violation
    whatever it sits next to.

    The tail side has no such self-evidence, so it reads the head's.
    """
    index = ordinal - 1
    for marker_index, raw in enumerate(lines):
        if not _is_elision_marker(raw):
            continue
        if index == marker_index - 1:
            return _is_cut_record_head(lines[index])
        if index == marker_index + 1:
            return _is_cut_record_tail(
                lines[index],
                head=lines[marker_index - 1] if marker_index > 0 else None,
            )
        return False
    return False


def _is_cut_record_head(raw: bytes) -> bool:
    """Is this the opening fragment of a record cut before its end?

    A Codex JSONL record is one object per line, so an intact record
    ends on its closing brace. A head fragment opens the object and
    stops wherever the budget fell, leaving more braces open than
    closed.
    """
    stripped = raw.strip()
    if not stripped.startswith(b"{"):
        return False
    return _brace_balance(stripped) > 0


def _is_cut_record_tail(raw: bytes, *, head: bytes | None) -> bool:
    """Is this the closing fragment of a record cut after its start?

    Two independent things must hold, because neither is sufficient.

    The line must be incapable of being a whole record. A record Codex
    emitted whole opens on ``{`` and ends on ``}``; a fragment that
    survived from somewhere inside a record through its closing brace
    reaches the end without the start.

    And the retained stream must actually show a record spanning the
    elision, which only the head side can witness. The tail fragment's
    first retained byte lands mid-token, so its own shape proves
    nothing further: brace counting is unreliable once a cut lands
    inside a string value, and a complete malformed line like
    ``not json}`` closes without opening exactly as a real fragment
    does. The head's last line opening a record it never closes is the
    stream's own evidence that the budget fell inside a record. Without
    it, a malformed tail line is indistinguishable from genuine
    malformed process output, and this parser rejects the run rather
    than delete that output and report retention damage in its place.
    """
    stripped = raw.strip()
    if stripped.startswith(b"{"):
        return False
    if not stripped.endswith(b"}"):
        return False
    return head is not None and _is_cut_record_head(head)


def _brace_balance(raw: bytes) -> int:
    """Opening braces minus closing ones, ignoring braces inside strings.

    Only structure outside string literals says whether a record is
    complete; a brace in a message body is content. Escapes are honored
    so an escaped quote does not end the string it sits in.
    """
    depth = 0
    in_string = False
    escaped = False
    for byte in raw:
        if in_string:
            if escaped:
                escaped = False
            elif byte == 0x5C:  # backslash
                escaped = True
            elif byte == 0x22:  # double quote
                in_string = False
            continue
        if byte == 0x22:
            in_string = True
        elif byte == 0x7B:  # {
            depth += 1
        elif byte == 0x7D:  # }
            depth -= 1
    # An unterminated string is itself a cut: the record stopped inside
    # a value, so whatever braces it had opened are still open.
    if in_string and depth == 0:
        return 1
    return depth


def _parse_output_artifact_bytes(
    raw: bytes,
    *,
    stdout: bytes,
    stderr: str,
    run_id: str,
    isolation: dict[str, Any] | None = None,
    stdout_truncated: bool = False,
) -> CodexOutputArtifact:
    try:
        decode_strict_json_bytes(
            raw,
            max_bytes=len(raw),
            max_depth=len(raw),
        )
        artifact = CodexOutputArtifact.model_validate_json(raw)
    except (StrictJsonDecodeError, ValidationError) as exc:
        # The process really ran under the sandbox and really spent its
        # output budget, so the failure carries the same isolation record
        # every other post-execution failure does. Raising the bare base
        # error here dropped it, and the terminalized Step then recorded
        # an empty ``codex_isolation``: no profile, no budgets, no
        # truncation flags to explain a stitched transcript.
        raise CodexStructuredExecutionFailure(
            "Codex final output artifact failed schema validation",
            stdout=stdout,
            stderr=stderr.encode("utf-8", "surrogateescape"),
            artifact_bytes=raw,
            isolation=isolation,
        ) from exc
    if artifact.run_id != run_id:
        raise CodexStructuredExecutionFailure(
            "Codex final output artifact has the wrong run",
            stdout=stdout,
            stderr=stderr.encode("utf-8", "surrogateescape"),
            artifact_bytes=raw,
            isolation=isolation,
        )
    try:
        parsed = _parse_jsonl_events(stdout, truncated=stdout_truncated)
    except OpaqueStepError as exc:
        # Same reason as the schema failure above: a malformed event
        # stream is exactly the failure whose isolation record -- the
        # truncation flags and dropped-byte counts in particular -- is
        # what tells a reader whether the budget caused it. The helper
        # stays a pure parser and the evidence is attached here.
        raise CodexStructuredExecutionFailure(
            str(exc),
            stdout=stdout,
            stderr=stderr.encode("utf-8", "surrogateescape"),
            artifact_bytes=raw,
            isolation=isolation,
        ) from exc
    process_evidence = {
        "agent": artifact.conversation_evidence,
        "jsonl_events": list(parsed.events),
        # What the output budget cost this transcript. Without it a reader
        # cannot tell a complete event stream from a stitched one.
        "jsonl_dropped_partial_lines": parsed.dropped_partial_lines,
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
    "SubprocessCodexRunner",
    "build_codex_command",
]
