from __future__ import annotations

import importlib.machinery
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from subprocess import TimeoutExpired
from uuid import uuid4

import pytest
from dr_exec import (
    AttemptId,
    BudgetAxis,
    BudgetExceededOutcome,
    CancelledOutcome,
    CancelToken,
    CompletedExecution,
    ContainmentProfile,
    DirectoryRunStore,
    EnvGrantKind,
    ExecutionAttribution,
    ExecutionId,
    ExecutionJob,
    ExecutionMeasurements,
    ExecutionOutcome,
    ExecutionResult,
    Executor,
    ExitedOutcome,
    FailureOwner,
    FakeExecutor,
    FakeRecordReceipt,
    FinalizedRecord,
    FiniteDurationLimit,
    IsolatedHostPythonRuntime,
    LimitKind,
    PayloadOutputs,
    ProcessExecutor,
    ProtocolFailedOutcome,
    ProtocolFailureCode,
    RetainedPayloadStream,
    SignaledOutcome,
    SpawnAbsentOutcome,
    SpawnFailedOutcome,
    UntrustedCommandTarget,
    UntrustedCommandTargetRecord,
)
from dr_store import ObjectStore, SqliteBackend

import whetstone.optimization.codex.runner as runner_module
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
    _DIRECT_EXEC_SOURCE,
    _MCP_TOOLS_APPROVAL_MODE,
    CodexStructuredExecutionFailure,
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
    execution_executor: Executor
    runner: SubprocessCodexRunner
    request: OptimizationStepRequest
    handle: RuntimeToolHandle
    tool_store: ToolCallStore
    partial_log_path: Path | None


def _option(args: list[str], name: str) -> str:
    return args[args.index(name) + 1]


def _codex_argv(job: ExecutionJob) -> list[str]:
    target = job.target
    assert isinstance(target, UntrustedCommandTarget)
    argv = target.argv
    assert argv[1] == "-f"
    assert argv[4:7] == ("-I", "-c", _DIRECT_EXEC_SOURCE)
    return list(argv[8:])


def _retained_stream(
    data: bytes, *, dropped_bytes: int = 0
) -> RetainedPayloadStream:
    split = len(data) // 2
    return RetainedPayloadStream(
        head=data[:split],
        tail=data[split:],
        produced_bytes=len(data) + dropped_bytes,
        dropped_bytes=dropped_bytes,
    )


def _completed(
    job: ExecutionJob,
    *,
    outcome: ExecutionOutcome,
    stdout: bytes,
    stderr: bytes,
    stdout_dropped_bytes: int = 0,
) -> CompletedExecution:
    execution_id = ExecutionId(
        job_id=job.job_id,
        attempt_id=AttemptId(uuid4()),
    )
    moment = datetime.now(UTC)
    return CompletedExecution(
        result=ExecutionResult(
            execution_id=execution_id,
            outcome=outcome,
            attribution=ExecutionAttribution(owner=FailureOwner.NONE),
            protocol_outputs=(),
            payload_outputs=PayloadOutputs(
                stdout=_retained_stream(
                    stdout, dropped_bytes=stdout_dropped_bytes
                ),
                stderr=_retained_stream(stderr),
            ),
            measurements=ExecutionMeasurements(
                started_at=moment,
                finished_at=moment,
                duration_ns=0,
                teardown_duration_ns=0,
                input_bytes=0,
                protocol_bytes_received=0,
            ),
        ),
        record_receipt=FakeRecordReceipt(execution_id=execution_id),
    )


def _fake_responder(
    mode: str,
    *,
    outcome: ExecutionOutcome | None = None,
    stderr: bytes = b"complete stderr evidence",
    stdout_dropped_bytes: int = 0,
) -> Callable[[ExecutionJob, CancelToken | None], CompletedExecution]:
    def respond(
        job: ExecutionJob, _cancellation: CancelToken | None
    ) -> CompletedExecution:
        codex_argv = _codex_argv(job)
        args = codex_argv[1:]
        schema_path = Path(_option(args, "--output-schema"))
        artifact_path = Path(_option(args, "--output-last-message"))
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        target = job.target
        assert isinstance(target, UntrustedCommandTarget)
        environment = {item.name: item.value for item in job.env.variables}
        events = (
            {
                "type": "turn.started",
                "argv": args,
                "cwd": target.argv[7],
                "env_keys": sorted(environment),
                "runtime_has_dr_exec": (
                    Path(environment["PYTHONPATH"]) / "dr_exec"
                ).exists(),
                "schema_run_id": schema["properties"]["run_id"]["const"],
                "schema_exists": schema_path.is_file(),
            },
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": "final"},
            },
        )
        stdout = b"".join(
            json.dumps(event, sort_keys=True).encode("utf-8") + b"\n"
            for event in events
        )
        if mode == "malformed":
            artifact_path.write_text("{malformed", encoding="utf-8")
        elif mode != "missing":
            artifact_path.write_text(
                json.dumps(
                    {
                        "run_id": schema["properties"]["run_id"]["const"],
                        "proposals": [],
                        "conversation_evidence": {"agent": "final"},
                        "control_cost": {"agent_tokens": 7},
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        return _completed(
            job,
            outcome=outcome or ExitedOutcome(exit_code=0),
            stdout=stdout,
            stderr=stderr,
            stdout_dropped_bytes=stdout_dropped_bytes,
        )

    return respond


def _subprocess_boundary(
    tmp_path: Path,
    experiment: EnvExperiment,
    mode: str,
    *,
    execution_executor: Executor | None = None,
    tool_name: str = "evaluate_candidate",
    prompt_builder: Callable[[OptimizationStepRequest], str] | None = None,
    codex_binary: str | None = None,
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
    outside_state = tmp_path / f"outside-{mode}"
    outside_state.mkdir()
    outside_secret = outside_state / "secret.txt"
    outside_secret.write_text("host secret", encoding="utf-8")
    resolved_executor = execution_executor or FakeExecutor(
        responder=_fake_responder(mode)
    )
    runner = SubprocessCodexRunner(
        executor=resolved_executor,
        sqlite_path=str(mcp_state / "store.sqlite"),
        runtime_config=runtime,
        reward_policy=experiment.reward_policy,
        codex_binary=str(executable) if codex_binary is None else codex_binary,
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
        execution_executor=resolved_executor,
        runner=runner,
        request=step_request,
        handle=handle,
        tool_store=tool_store,
        partial_log_path=partial_path,
    )


@pytest.fixture
def declared_seatbelt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    sandbox_exec = tmp_path / "sandbox-exec"
    sandbox_exec.touch()
    monkeypatch.setattr(runner_module, "_MACOS_SANDBOX_EXEC", sandbox_exec)
    monkeypatch.setattr(sys, "platform", "darwin")
    return sandbox_exec


def test_subprocess_declares_typed_execution_and_preserves_artifact_evidence(
    tmp_path: Path,
    codex_experiment: EnvExperiment,
    declared_seatbelt: Path,
) -> None:
    boundary = _subprocess_boundary(tmp_path, codex_experiment, "success")

    run = boundary.runner.run(boundary.request, boundary.handle)

    fake = boundary.execution_executor
    assert isinstance(fake, FakeExecutor)
    (job,) = fake.calls
    target = job.target
    assert isinstance(target, UntrustedCommandTarget)
    assert target.stdin == b""
    assert (
        target.containment_profile is ContainmentProfile.PROCESS_BOUNDARY_ONLY
    )
    assert target.argv[:3] == (
        str(declared_seatbelt),
        "-f",
        target.argv[2],
    )
    assert target.argv[3:7] == (
        sys.executable,
        "-I",
        "-c",
        _DIRECT_EXEC_SOURCE,
    )
    working_directory = target.argv[7]
    codex_argv = list(target.argv[8:])
    assert codex_argv[0].endswith("fake-codex-success")
    assert codex_argv[codex_argv.index("--cd") + 1] == working_directory
    assert codex_argv[codex_argv.index("--sandbox") + 1] == "read-only"
    for flag in (
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--strict-config",
        "--output-schema",
        "--output-last-message",
    ):
        assert flag in codex_argv
    assert 'shell_environment_policy.inherit="none"' in codex_argv
    disabled = {
        codex_argv[index + 1]
        for index, item in enumerate(codex_argv)
        if item == "--disable"
    }
    assert disabled == set(_CODEX_DENIED_FEATURES)
    assert (
        "mcp_servers.whetstone.default_tools_approval_mode="
        + json.dumps(_MCP_TOOLS_APPROVAL_MODE)
        in codex_argv
    )
    assert "provider-secret" not in " ".join(codex_argv)

    assert job.env.kind is EnvGrantKind.FIXED
    environment = {item.name: item.value for item in job.env.variables}
    assert set(environment) == {
        "CODEX_HOME",
        "OPENROUTER_API_KEY",
        "PYTHONPATH",
    }
    assert environment["OPENROUTER_API_KEY"] == "provider-secret"
    assert "AWS_SECRET_ACCESS_KEY" not in environment
    assert "UNRELATED_VALUE" not in environment

    assert job.budgets.wall_time == FiniteDurationLimit(max_ns=600_000_000_000)
    assert job.budgets.input_bytes.kind is LimitKind.UNBUDGETED
    assert job.budgets.payload_output.kind is LimitKind.UNBUDGETED
    for axis in (
        "memory_bytes",
        "cpu_time",
        "process_count",
        "file_size_bytes",
        "open_file_count",
        "disk_bytes",
    ):
        assert getattr(job.budgets, axis).kind is LimitKind.UNBUDGETED

    artifact = CodexOutputArtifact.model_validate(run.artifact)
    events = artifact.conversation_evidence["jsonl_events"]
    assert [event["type"] for event in events] == [
        "turn.started",
        "item.completed",
    ]
    invocation = events[0]
    assert invocation["schema_exists"] is True
    assert invocation["schema_run_id"] == boundary.request.run_id
    assert invocation["cwd"] == working_directory
    assert invocation["runtime_has_dr_exec"] is False
    assert artifact.conversation_evidence["agent"] == {"agent": "final"}
    assert artifact.conversation_evidence["stderr"] == (
        "complete stderr evidence"
    )
    isolation = artifact.conversation_evidence["isolation"]
    assert isolation["strategy"] == "macos_sandbox_exec"
    assert Path(target.argv[2]).name == "codex.sb"
    profile = isolation["profile"]
    outside_path = boundary.request.hyperparameters["adversarial_outside_path"]
    assert isinstance(outside_path, str)
    assert outside_path not in profile
    source_root = Path(__file__).resolve().parents[3] / "src"
    assert source_root.is_dir()
    assert str(source_root) not in profile
    assert artifact.control_cost == {"agent_tokens": 7}
    assert str(job.job_id) not in artifact.model_dump_json()


@pytest.mark.parametrize("mode", ["missing", "malformed"])
def test_subprocess_rejects_missing_or_malformed_artifact(
    tmp_path: Path,
    codex_experiment: EnvExperiment,
    mode: str,
    declared_seatbelt: Path,
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


@pytest.mark.parametrize(
    ("outcome", "message"),
    [
        (ExitedOutcome(exit_code=7), "Codex exited 7: exact stderr"),
        (SignaledOutcome(signal_number=9), "Codex exited -9: exact stderr"),
    ],
)
def test_exit_outcomes_preserve_return_code_and_stderr(
    tmp_path: Path,
    codex_experiment: EnvExperiment,
    declared_seatbelt: Path,
    outcome: ExecutionOutcome,
    message: str,
) -> None:
    fake = FakeExecutor(
        responder=_fake_responder(
            "success", outcome=outcome, stderr=b"exact stderr"
        )
    )
    boundary = _subprocess_boundary(
        tmp_path,
        codex_experiment,
        "success",
        execution_executor=fake,
    )

    with pytest.raises(
        CodexStructuredExecutionFailure, match=f"^{message}$"
    ) as error:
        boundary.runner.run(boundary.request, boundary.handle)

    assert error.value.stderr == b"exact stderr"
    assert error.value.artifact_bytes


@pytest.mark.parametrize(
    "outcome",
    [
        SpawnAbsentOutcome(executable="/missing/sandbox-exec"),
        SpawnFailedOutcome(errno=13, error_message="exec"),
    ],
)
def test_spawn_outcomes_are_opaque_codex_failures(
    tmp_path: Path,
    codex_experiment: EnvExperiment,
    declared_seatbelt: Path,
    outcome: ExecutionOutcome,
) -> None:
    fake = FakeExecutor(responder=_fake_responder("success", outcome=outcome))
    boundary = _subprocess_boundary(
        tmp_path,
        codex_experiment,
        "success",
        execution_executor=fake,
    )

    with pytest.raises(OpaqueStepError, match="could not be spawned"):
        boundary.runner.run(boundary.request, boundary.handle)


@pytest.mark.parametrize(
    "outcome",
    [
        BudgetExceededOutcome(axis=BudgetAxis.INPUT_BYTES),
        CancelledOutcome(),
        ProtocolFailedOutcome(
            failure_code=ProtocolFailureCode.INCOMPLETE_STREAM,
            failure_detail="incomplete",
            accepted_output_count=0,
        ),
    ],
)
def test_unexpected_typed_outcomes_are_opaque_codex_failures(
    tmp_path: Path,
    codex_experiment: EnvExperiment,
    declared_seatbelt: Path,
    outcome: ExecutionOutcome,
) -> None:
    fake = FakeExecutor(responder=_fake_responder("success", outcome=outcome))
    boundary = _subprocess_boundary(
        tmp_path,
        codex_experiment,
        "success",
        execution_executor=fake,
    )

    with pytest.raises(OpaqueStepError, match="Codex execution failed"):
        boundary.runner.run(boundary.request, boundary.handle)


def test_wall_budget_preserves_timeout_expired_compatibility(
    tmp_path: Path,
    codex_experiment: EnvExperiment,
    declared_seatbelt: Path,
) -> None:
    fake = FakeExecutor(
        responder=_fake_responder(
            "success",
            outcome=BudgetExceededOutcome(axis=BudgetAxis.WALL_TIME),
            stderr=b"timeout stderr",
        )
    )
    boundary = _subprocess_boundary(
        tmp_path,
        codex_experiment,
        "success",
        execution_executor=fake,
    )

    with pytest.raises(TimeoutExpired) as raised:
        boundary.runner.run(boundary.request, boundary.handle)

    assert raised.value.timeout == 600.0
    assert raised.value.stderr == b"timeout stderr"
    assert raised.value.output.startswith(b'{"argv":')
    (job,) = fake.calls
    target = job.target
    assert isinstance(target, UntrustedCommandTarget)
    assert raised.value.cmd == list(target.argv)


def test_unbudgeted_payload_output_rejects_truncation(
    tmp_path: Path,
    codex_experiment: EnvExperiment,
    declared_seatbelt: Path,
) -> None:
    fake = FakeExecutor(
        responder=_fake_responder("success", stdout_dropped_bytes=1)
    )
    boundary = _subprocess_boundary(
        tmp_path,
        codex_experiment,
        "success",
        execution_executor=fake,
    )

    with pytest.raises(OpaqueStepError, match="despite unbudgeted output"):
        boundary.runner.run(boundary.request, boundary.handle)


def test_executor_failure_is_an_opaque_codex_failure(
    tmp_path: Path,
    codex_experiment: EnvExperiment,
    declared_seatbelt: Path,
) -> None:
    fake = FakeExecutor()
    boundary = _subprocess_boundary(
        tmp_path,
        codex_experiment,
        "success",
        execution_executor=fake,
    )

    with pytest.raises(OpaqueStepError, match="Codex execution failed"):
        boundary.runner.run(boundary.request, boundary.handle)


def test_codex_binary_preflight_precedes_executor(
    tmp_path: Path,
    codex_experiment: EnvExperiment,
) -> None:
    fake = FakeExecutor()
    boundary = _subprocess_boundary(
        tmp_path,
        codex_experiment,
        "success",
        execution_executor=fake,
        codex_binary=str(tmp_path / "absent-codex"),
    )

    with pytest.raises(OpaqueStepError, match="was not found"):
        boundary.runner.run(boundary.request, boundary.handle)

    assert fake.calls == ()


def test_partial_log_profile_preserves_declared_parent_authority(
    tmp_path: Path,
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
    tmp_path: Path,
    codex_experiment: EnvExperiment,
    declared_seatbelt: Path,
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
    tmp_path: Path,
    codex_experiment: EnvExperiment,
    declared_seatbelt: Path,
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
    tmp_path: Path,
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
        executor=FakeExecutor(),
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
    reason="real Codex qualification requires macOS process primitives",
)
@pytest.mark.process_integration
def test_macos_process_executor_qualifies_codex_boundary(
    tmp_path: Path,
    codex_experiment: EnvExperiment,
) -> None:
    record_root = (tmp_path / "dr-exec-records").resolve()
    record_root.mkdir()
    run_store = DirectoryRunStore(root=record_root)
    process_executor = ProcessExecutor(
        runtime=IsolatedHostPythonRuntime(Path(sys.executable)),
        run_store=run_store,
    )
    boundary = _subprocess_boundary(
        tmp_path,
        codex_experiment,
        "proposal",
        execution_executor=process_executor,
    )

    run = boundary.runner.run(boundary.request, boundary.handle)

    assert len(run.artifact.proposals) == 1
    assert boundary.partial_log_path is not None
    assert boundary.partial_log_path.is_dir()
    mcp_result = run.artifact.conversation_evidence["agent"]["mcp_result"]
    assert mcp_result["refused"] is False
    events = run.artifact.conversation_evidence["jsonl_events"]
    invocation = events[0]
    argv = invocation["argv"]
    assert invocation["outside_read"] == "denied"
    assert (
        Path(invocation["cwd"]).resolve()
        == Path(argv[argv.index("--cd") + 1]).resolve()
    )
    profile = run.artifact.conversation_evidence["isolation"]["profile"]
    assert str(record_root) not in profile
    (record_dir,) = tuple(record_root.iterdir())
    record = run_store.load(record_dir)
    assert isinstance(record, FinalizedRecord)
    assert isinstance(record.declaration.target, UntrustedCommandTargetRecord)
    assert (
        record.declaration.target.containment_profile
        is ContainmentProfile.PROCESS_BOUNDARY_ONLY
    )


@pytest.mark.precheck
def test_process_isolation_fails_closed_on_unsupported_platform(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    codex_experiment: EnvExperiment,
) -> None:
    fake = FakeExecutor(responder=_fake_responder("success"))
    boundary = _subprocess_boundary(
        tmp_path,
        codex_experiment,
        "success",
        execution_executor=fake,
    )
    monkeypatch.setattr(sys, "platform", "linux")

    with pytest.raises(OpaqueStepError, match="no insecure fallback"):
        boundary.runner.run(boundary.request, boundary.handle)

    assert fake.calls == ()


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
    tmp_path: Path,
    codex_experiment: EnvExperiment,
    sqlite_path: str,
    partial_log_path: str | None,
    prompt_cache_path: str | None,
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "relative-paths.sqlite"))
    evaluation_engine = engine(store, codex_experiment)

    with pytest.raises(OpaqueStepError):
        SubprocessCodexRunner(
            executor=FakeExecutor(),
            sqlite_path=sqlite_path,
            runtime_config=runtime_config(
                evaluation_engine,
                partial_log_path=partial_log_path,
                prompt_cache_path=prompt_cache_path,
            ),
            reward_policy=codex_experiment.reward_policy,
            environment={},
        )
