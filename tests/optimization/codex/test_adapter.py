from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path

import pytest
from dr_store import ObjectStore, SqliteBackend

from tests.envs.support import execution_policy, process_row_job_factory
from tests.optimization.support import (
    make_harness,
    memory_tool_call_store,
    optimizer_config_ref,
    python_format_contract,
    registry,
)
from whetstone.core.effects.authority import EffectAuthority, ReplayPolicy
from whetstone.core.identity import TypedRef
from whetstone.envs.factory import build_env_experiment
from whetstone.evaluation.engine import EvaluationEngine
from whetstone.execution.partials import PartialLog
from whetstone.experiment.candidate import Candidate, candidate_reference
from whetstone.optimization.codex.adapter import (
    CODEX_OUTPUT_ARTIFACT_SCHEMA,
    CodexAdapter,
    CodexOutputArtifact,
    OpaqueStepError,
)
from whetstone.optimization.codex.mcp_bridge import (
    EvaluateCandidateServer,
    InProcessMcpProcess,
    JsonRpcClient,
    McpError,
    serve_stdio,
)
from whetstone.optimization.codex.mcp_server import build_server_from_env
from whetstone.optimization.codex.runner import (
    _CODEX_DENIED_FEATURES,
    _MCP_TOOLS_APPROVAL_MODE,
    FakeCodexRunner,
    ScriptedAgentCall,
    SubprocessCodexRunner,
    _MacOsProcessIsolation,
    build_codex_command,
)
from whetstone.optimization.codex.runtime import EvaluationRuntimeConfig
from whetstone.optimization.contracts import (
    BudgetState,
    OptimizationRun,
    OptimizationStepRequest,
    OutputContract,
    StepKind,
    StepMode,
    StepStatus,
    optimization_run_reference,
)
from whetstone.optimization.tools.contracts import (
    ToolCapacity,
    ToolCapacityScope,
    ToolConfig,
    ToolDefinition,
    tool_capacity_binding,
    tool_config_reference,
    tool_definition_reference,
)
from whetstone.optimization.tools.evaluator import EngineToolEvaluator
from whetstone.optimization.tools.execution import EvaluatingToolExecutor

ROW_JOB_ENTRYPOINT = "tests.envs.process_workers:drive_internal_success"
MODEL_ROUTE = "openai/test"


def _experiment():
    return build_env_experiment(
        "c18",
        model="openai/test",
        pool_n_per_stratum=2,
        split_sizes=(1, 1, 1),
        repeats=1,
    )


def _engine(store: ObjectStore, experiment) -> EvaluationEngine:
    return EvaluationEngine(
        store=store,
        experiment=experiment,
        sampling=experiment.eval_configs.internal,
        execution_policy=execution_policy(),
        row_job_factory=process_row_job_factory(ROW_JOB_ENTRYPOINT),
    )


def _tool_config(
    engine: EvaluationEngine,
    experiment,
    namespace: str,
    *,
    tool_name: str = "evaluate_candidate",
) -> ToolConfig:
    definition = ToolDefinition(
        tool_name=tool_name,
        input_fields=("base_ref", "model_route", "template"),
        output_fields=(
            "evaluation_evidence_ref",
            "output_artifact_ref",
            "per_task_values",
            "per_task_counts",
            "row_accounting",
        ),
    )
    return ToolConfig(
        definition=tool_definition_reference(definition),
        endpoint_key="mcp://whetstone/evaluate_candidate",
        eval_config=engine.eval_config_ref.record,
        reward_policy_hash=experiment.reward_policy.identity_hash(),
        capacity=ToolCapacity(
            max_accepted_calls=4,
            scope=ToolCapacityScope.RUN,
        ),
        store_namespace_key=namespace,
        idempotent_replay=False,
    )


def _proposals(base: Candidate) -> tuple[Candidate, Candidate]:
    base_record_ref = candidate_reference(base).record_ref
    return (
        Candidate(
            candidate_id="codex-a",
            base_ref=base_record_ref,
            payload={
                "user_prompt_template": (
                    "{question}\n{query}\nRespond True or False."
                )
            },
        ),
        Candidate(
            candidate_id="codex-b",
            base_ref=base_record_ref,
            payload={
                "user_prompt_template": (
                    "{question}\n{query}\nOnly True or False."
                )
            },
        ),
    )


def _run(config: ToolConfig, contract: OutputContract, run_id: str):
    return optimization_run_reference(
        OptimizationRun(
            run_id=run_id,
            optimizer_config=optimizer_config_ref("codex"),
            adapter_key="codex",
            mode=StepMode.TOOL_USING,
            terminal_output_contract=contract,
            template_render_contract=python_format_contract(
                available_fields=("question", "query")
            ),
            tool_configs=(tool_config_reference(config),),
        )
    )


def _request(
    base: Candidate,
    config: ToolConfig,
    *,
    distinct: bool = False,
    proposal_count: int = 2,
    run_id: str = "codex-run",
    hyperparameters: dict | None = None,
) -> OptimizationStepRequest:
    contract = OutputContract(
        returned_proposal_count=proposal_count,
        require_distinct_bases=distinct,
    )
    return OptimizationStepRequest(
        run=_run(config, contract, run_id),
        step_id="codex-opaque",
        kind=StepKind.TOOL,
        step_index=0,
        candidates=(base,),
        step_output_contract=contract,
        budget=BudgetState(remaining={"tool_calls": 4}),
        hyperparameters=hyperparameters or {},
    )


def _binding(request: OptimizationStepRequest):
    return tool_capacity_binding(ToolCapacityScope.RUN, request.run.record_ref)


def _executor(
    engine: EvaluationEngine, experiment, authority: EffectAuthority
) -> EvaluatingToolExecutor:
    return EvaluatingToolExecutor(
        EngineToolEvaluator(engine),
        experiment.reward_policy,
        authority,
        owner_id="codex-test-owner",
        replay_policy=ReplayPolicy.NO_REDRIVE,
    )


def _runner(base: Candidate, *, call_id: str = "agent-call-1"):
    template = base.payload["user_prompt_template"]
    assert isinstance(template, str)
    return FakeCodexRunner(
        scripted_calls=(
            ScriptedAgentCall(
                call_id=call_id,
                base_ref=base.base_ref,
                model_route=MODEL_ROUTE,
                template=template,
            ),
        ),
        final_proposals=_proposals(base),
    )


def _stack(tmp_path, *, namespace: str = "codex-durable"):
    database = tmp_path / "codex.sqlite"
    store = ObjectStore(SqliteBackend(database))
    experiment = _experiment()
    engine = _engine(store, experiment)
    config = _tool_config(engine, experiment, namespace)
    authority = EffectAuthority.memory()
    tool_store = memory_tool_call_store(store, authority)
    executor = _executor(engine, experiment, authority)
    base = experiment.initial_candidate
    runner = _runner(base)
    return (
        database,
        store,
        experiment,
        config,
        tool_store,
        executor,
        runner,
        base,
    )


def test_fake_process_actual_jsonrpc_artifact_and_restart(tmp_path) -> None:
    (
        _database,
        store,
        _experiment_value,
        config,
        tool_store,
        executor,
        runner,
        base,
    ) = _stack(tmp_path)
    adapter = CodexAdapter(runner, store=store, tool_store=tool_store)
    request = _request(base, config)
    harness = make_harness(
        store=store,
        adapter_registry=registry(adapter),
        run=request.run,
        tool_store=tool_store,
        effect_authority=tool_store.effect_authority,
        tool_executor=executor,
        adapter_replay_policy=ReplayPolicy.NO_REDRIVE,
    )

    result, result_ref = harness.run_step(request)

    assert result.status is StepStatus.COMPLETE
    assert len(result.accepted_candidates) == 2
    assert {item.record.base_ref for item in result.accepted_candidates} == {
        candidate_reference(base).record_ref
    }
    assert len(result.tool_evidence) == 1
    assert runner.observed_payloads[0]["refused"] is False
    assert result.state_ref is not None
    state = store.get(result.state_ref.reference)
    assert isinstance(state, dict)
    artifact_ref = TypedRef.model_validate(state["codex_output_artifact_ref"])
    assert artifact_ref.schema_name == CODEX_OUTPUT_ARTIFACT_SCHEMA
    assert store.get(artifact_ref.reference)["run_id"] == request.run_id
    assert state["harness_store_accepted_call_count"] == 1
    assert state["tool_namespace"] == str(config.store_namespace_key)

    class ExplodingRegistry:
        def resolve(self, adapter_key: str):
            raise AssertionError(f"resolved {adapter_key}")

    replay, replay_ref = make_harness(
        store=store,
        adapter_registry=ExplodingRegistry(),
        run=request.run,
        tool_store=tool_store,
        effect_authority=tool_store.effect_authority,
        tool_executor=executor,
        adapter_replay_policy=ReplayPolicy.NO_REDRIVE,
    ).run_step(request)
    assert (replay, replay_ref) == (result, result_ref)


def test_distinct_bases_is_conditional_not_unconditional(tmp_path) -> None:
    (
        _database,
        store,
        _experiment_value,
        config,
        tool_store,
        executor,
        _runner_value,
        base,
    ) = _stack(tmp_path)

    allowed_request = _request(base, config, distinct=False)
    allowed_handle = executor.runtime_handle(
        config, tool_store, _binding(allowed_request)
    )
    allowed = CodexAdapter(
        _runner(base, call_id="agent-call-allowed"),
        store=store,
        tool_store=tool_store,
    ).invoke(allowed_request, (allowed_handle,))

    rejected_request = _request(
        base, config, distinct=True, run_id="codex-run-distinct"
    )
    rejected_handle = executor.runtime_handle(
        config, tool_store, _binding(rejected_request)
    )
    rejected = CodexAdapter(
        _runner(base, call_id="agent-call-rejected"),
        store=store,
        tool_store=tool_store,
    ).invoke(rejected_request, (rejected_handle,))

    assert len(allowed.accepted_candidates) == 2
    assert allowed.proposed_status is StepStatus.COMPLETE
    assert rejected.accepted_candidates == ()
    assert rejected.proposed_status is StepStatus.FAILED


def test_codex_requires_exactly_one_runtime_tool_handle(tmp_path) -> None:
    (
        _database,
        store,
        _experiment_value,
        config,
        tool_store,
        _executor_value,
        runner,
        base,
    ) = _stack(tmp_path)
    adapter = CodexAdapter(runner, store=store, tool_store=tool_store)

    with pytest.raises(OpaqueStepError, match="one Runtime Tool Handle"):
        adapter.invoke(_request(base, config), ())


def test_serialized_runtime_reconstructs_real_engine(tmp_path) -> None:
    experiment = _experiment()
    parent_store = ObjectStore(SqliteBackend(tmp_path / "parent.sqlite"))
    parent_engine = _engine(parent_store, experiment)
    config = _tool_config(parent_engine, experiment, "codex-child")
    runtime = EvaluationRuntimeConfig(
        env_name="c18",
        model="openai/test",
        pool_n_per_stratum=2,
        split_sizes=(1, 1, 1),
        repeats=1,
        expected_eval_config_hash=(
            parent_engine.eval_config_ref.identity_hash
        ),
        execution_policy=execution_policy(),
        row_job_entrypoint=ROW_JOB_ENTRYPOINT,
        partial_log_path=str(tmp_path / "child-partials.jsonl"),
        prompt_cache_path=str(tmp_path / "child-cache"),
    )
    (tmp_path / "child-cache").mkdir()
    request = _request(experiment.initial_candidate, config, proposal_count=0)
    child_db = tmp_path / "child.sqlite"
    server = build_server_from_env(
        {
            "WS_MCP_SQLITE_PATH": str(child_db),
            "WS_MCP_TOOL_CONFIG": config.model_dump_json(),
            "WS_MCP_CAPACITY_BINDING": _binding(request).model_dump_json(),
            "WS_MCP_RUNTIME_CONFIG": runtime.model_dump_json(),
            "WS_MCP_REWARD_POLICY": (
                experiment.reward_policy.model_dump_json()
            ),
        }
    )
    runner = FakeCodexRunner(
        process=InProcessMcpProcess(server),
        scripted_calls=(
            ScriptedAgentCall(
                call_id="child-call",
                base_ref=experiment.initial_candidate.base_ref,
                model_route=MODEL_ROUTE,
                template=experiment.initial_candidate.payload[
                    "user_prompt_template"
                ],
            ),
        ),
        final_proposals=(),
    )

    output = runner.run(
        request,
        server.handle,
    )

    assert output.artifact.run_id == request.run_id
    assert runner.observed_payloads[0]["refused"] is False
    assert server.handle.config == config


def _subprocess_boundary(tmp_path, mode: str):
    experiment = _experiment()
    store = ObjectStore(SqliteBackend(tmp_path / f"subprocess-{mode}.sqlite"))
    engine = _engine(store, experiment)
    config = _tool_config(engine, experiment, f"codex-subprocess-{mode}")
    runtime = EvaluationRuntimeConfig(
        env_name="c18",
        model="openai/test",
        pool_n_per_stratum=2,
        split_sizes=(1, 1, 1),
        repeats=1,
        expected_eval_config_hash=engine.eval_config_ref.identity_hash,
        execution_policy=execution_policy(),
        row_job_entrypoint=ROW_JOB_ENTRYPOINT,
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
    mcp_state = tmp_path / f"mcp-{mode}"
    mcp_state.mkdir()
    mcp_path = mcp_state / "store.sqlite"
    outside_secret = tmp_path / f"outside-secret-{mode}.txt"
    outside_secret.write_text("host secret", encoding="utf-8")
    runner = SubprocessCodexRunner(
        sqlite_path=str(mcp_path),
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
    request = _request(
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
    handle = _executor(engine, experiment, authority).runtime_handle(
        config, tool_store, _binding(request)
    )
    return store, config, runner, request, handle, tool_store, mcp_path


@pytest.fixture
def cross_platform_fake_codex_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve policy generation when Seatbelt cannot enforce the test run."""
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
    tmp_path, cross_platform_fake_codex_boundary
) -> None:
    store, config, runner, request, handle, tool_store, _mcp_path = (
        _subprocess_boundary(tmp_path, "success")
    )
    adapter = CodexAdapter(runner, store=store, tool_store=tool_store)

    output = adapter.invoke(request, (handle,))

    artifact_ref = TypedRef.model_validate(
        output.state_delta["codex_output_artifact_ref"]
    )
    artifact = CodexOutputArtifact.model_validate(
        store.get(artifact_ref.reference)
    )
    events = artifact.conversation_evidence["jsonl_events"]
    assert isinstance(events, list)
    invocation = events[0]
    assert isinstance(invocation, dict)
    argv = invocation["argv"]
    assert isinstance(argv, list)
    assert invocation["schema_exists"] is True
    assert invocation["schema_run_id"] == request.run_id
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
    assert "(deny default)" in profile
    outside_path = request.hyperparameters["adversarial_outside_path"]
    assert isinstance(outside_path, str)
    assert outside_path not in profile
    source_root = Path(__file__).resolve().parents[2] / "src"
    assert str(source_root) not in profile
    assert artifact.control_cost == {"agent_tokens": 7}
    assert output.proposed_status is StepStatus.COMPLETE
    assert (
        str(config.store_namespace_key)
        == (output.state_delta["tool_namespace"])
    )


@pytest.mark.parametrize("mode", ["missing", "malformed"])
def test_subprocess_rejects_missing_or_malformed_artifact(
    tmp_path, mode: str, cross_platform_fake_codex_boundary
) -> None:
    _store, _config, runner, request, handle, _tool_store, _mcp_path = (
        _subprocess_boundary(tmp_path, mode)
    )

    with pytest.raises(OpaqueStepError, match="final output artifact"):
        runner.run(request, handle)


def test_subprocess_nonzero_proposal_uses_mcp_evidence(
    tmp_path, cross_platform_fake_codex_boundary
) -> None:
    store, _config, runner, request, handle, tool_store, _mcp_path = (
        _subprocess_boundary(tmp_path, "proposal")
    )
    adapter = CodexAdapter(runner, store=store, tool_store=tool_store)

    output = adapter.invoke(request, (handle,))

    assert output.proposed_status is StepStatus.COMPLETE
    assert len(output.accepted_candidates) == 1
    assert output.accepted_candidates[0].base_ref == (
        candidate_reference(request.candidates[0]).record_ref
    )


@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="neighboring-secret enforcement requires macOS Seatbelt",
)
def test_macos_process_isolation_denies_neighboring_secret(tmp_path) -> None:
    _store, _config, runner, request, handle, _tool_store, _mcp_path = (
        _subprocess_boundary(tmp_path, "success")
    )

    result = runner.run(request, handle)

    events = result.artifact.conversation_evidence["jsonl_events"]
    assert isinstance(events, list)
    invocation = events[0]
    assert isinstance(invocation, dict)
    assert invocation["outside_read"] == "denied"


def test_process_isolation_fails_closed_on_unsupported_platform(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")

    with pytest.raises(OpaqueStepError, match="no insecure fallback"):
        _MacOsProcessIsolation().wrap(
            ["/bin/true"],
            profile_path=tmp_path / "profile.sb",
            readable_paths=(),
            writable_paths=(tmp_path,),
        )


def test_client_calls_the_handles_configured_tool_name(tmp_path) -> None:
    database = tmp_path / "renamed-tool.sqlite"
    store = ObjectStore(SqliteBackend(database))
    experiment = _experiment()
    engine = _engine(store, experiment)
    config = _tool_config(
        engine,
        experiment,
        "codex-renamed-tool",
        tool_name="score_candidate_draft",
    )
    assert config.tool_name != "evaluate_candidate"
    authority = EffectAuthority.memory()
    tool_store = memory_tool_call_store(store, authority)
    executor = _executor(engine, experiment, authority)
    base = experiment.initial_candidate
    request = _request(base, config, run_id="codex-run-renamed-tool")
    handle = executor.runtime_handle(config, tool_store, _binding(request))
    runner = _runner(base, call_id="renamed-tool-call")

    output = CodexAdapter(runner, store=store, tool_store=tool_store).invoke(
        request, (handle,)
    )

    assert output.proposed_status is StepStatus.COMPLETE
    assert runner.observed_payloads[0]["refused"] is False
    assert tool_store.accepted_count(config, handle.binding) == 1


def test_client_rejects_a_tool_name_the_server_does_not_serve(
    tmp_path,
) -> None:
    database = tmp_path / "mismatched-tool.sqlite"
    store = ObjectStore(SqliteBackend(database))
    experiment = _experiment()
    engine = _engine(store, experiment)
    config = _tool_config(engine, experiment, "codex-mismatched-tool")
    authority = EffectAuthority.memory()
    tool_store = memory_tool_call_store(store, authority)
    executor = _executor(engine, experiment, authority)
    base = experiment.initial_candidate
    request = _request(base, config, run_id="codex-run-mismatched-tool")
    handle = executor.runtime_handle(config, tool_store, _binding(request))
    client = JsonRpcClient(
        InProcessMcpProcess(EvaluateCandidateServer(handle=handle)).exchange,
        tool_name="not_the_served_tool",
    )
    client.initialize()

    with pytest.raises(McpError, match="unknown tool"):
        client.evaluate(
            call_id="mismatched-tool-call",
            base_ref=base.base_ref.model_dump(mode="json"),
            model_route=MODEL_ROUTE,
            template="{question}\n{query}\nRespond True or False.",
        )


def _runtime_config(
    engine: EvaluationEngine,
    *,
    partial_log_path: str | None = None,
    prompt_cache_path: str | None = None,
) -> EvaluationRuntimeConfig:
    return EvaluationRuntimeConfig(
        env_name="c18",
        model="openai/test",
        pool_n_per_stratum=2,
        split_sizes=(1, 1, 1),
        repeats=1,
        expected_eval_config_hash=engine.eval_config_ref.identity_hash,
        execution_policy=execution_policy(),
        row_job_entrypoint=ROW_JOB_ENTRYPOINT,
        partial_log_path=partial_log_path,
        prompt_cache_path=prompt_cache_path,
    )


@pytest.mark.parametrize(
    ("field", "sqlite_path", "partial_log_path", "prompt_cache_path"),
    [
        ("sqlite_path", "relative/store.sqlite", None, None),
        (
            "partial_log_path",
            "/abs/store.sqlite",
            "relative/partials.jsonl",
            None,
        ),
        ("prompt_cache_path", "/abs/store.sqlite", None, "relative/cache"),
    ],
)
def test_subprocess_runner_rejects_relative_runtime_paths(
    tmp_path,
    field: str,
    sqlite_path: str,
    partial_log_path: str | None,
    prompt_cache_path: str | None,
) -> None:
    experiment = _experiment()
    store = ObjectStore(SqliteBackend(tmp_path / "relative-paths.sqlite"))
    engine = _engine(store, experiment)

    with pytest.raises(OpaqueStepError, match=f"{field} must be absolute"):
        SubprocessCodexRunner(
            sqlite_path=sqlite_path,
            runtime_config=_runtime_config(
                engine,
                partial_log_path=partial_log_path,
                prompt_cache_path=prompt_cache_path,
            ),
            reward_policy=experiment.reward_policy,
            environment={},
        )


def test_subprocess_runner_accepts_absolute_runtime_paths(tmp_path) -> None:
    experiment = _experiment()
    store = ObjectStore(SqliteBackend(tmp_path / "absolute-paths.sqlite"))
    engine = _engine(store, experiment)

    runner = SubprocessCodexRunner(
        sqlite_path=str(tmp_path / "mcp-store.sqlite"),
        runtime_config=_runtime_config(
            engine,
            partial_log_path=str(tmp_path / "partials.jsonl"),
            prompt_cache_path=str(tmp_path / "cache"),
        ),
        reward_policy=experiment.reward_policy,
        environment={},
    )

    assert isinstance(runner, SubprocessCodexRunner)


def _malformed_line_server(tmp_path):
    experiment = _experiment()
    store = ObjectStore(SqliteBackend(tmp_path / "serve-stdio.sqlite"))
    engine = _engine(store, experiment)
    config = _tool_config(engine, experiment, "codex-serve-stdio")
    authority = EffectAuthority.memory()
    tool_store = memory_tool_call_store(store, authority)
    request = _request(
        experiment.initial_candidate, config, run_id="codex-run-serve-stdio"
    )
    handle = _executor(engine, experiment, authority).runtime_handle(
        config, tool_store, _binding(request)
    )
    return EvaluateCandidateServer(handle=handle)


@pytest.mark.parametrize(
    ("malformed", "code"),
    [("not-json", -32700), ("[]", -32600)],
)
def test_serve_stdio_answers_malformed_lines_and_keeps_serving(
    tmp_path, malformed: str, code: int
) -> None:
    server = _malformed_line_server(tmp_path)
    stdin = StringIO(
        f"{malformed}\n"
        + json.dumps({"jsonrpc": "2.0", "id": 7, "method": "ping"})
        + "\n"
    )
    stdout = StringIO()

    serve_stdio(server, stdin=stdin, stdout=stdout)

    responses = [
        json.loads(line) for line in stdout.getvalue().splitlines() if line
    ]
    assert len(responses) == 2
    assert responses[0]["id"] is None
    assert responses[0]["error"]["code"] == code
    assert responses[1] == {"jsonrpc": "2.0", "id": 7, "result": {}}


def test_codex_command_pre_approves_the_whetstone_mcp_tools() -> None:
    argv = build_codex_command(
        prompt="proposal prompt",
        codex_binary="/usr/bin/codex",
        model="gpt-5",
        mcp_env={"WS_MCP_SQLITE_PATH": "/abs/store.sqlite"},
        output_schema_path="/abs/schema.json",
        output_artifact_path="/abs/last-message.json",
        working_directory="/abs/work",
    )

    # stdin is DEVNULL, so the evaluation tool must not need an approval turn.
    assert (
        "mcp_servers.whetstone.default_tools_approval_mode="
        + json.dumps(_MCP_TOOLS_APPROVAL_MODE)
    ) in argv
    # Pinned against the Codex config parser, which accepts only these.
    assert _MCP_TOOLS_APPROVAL_MODE in {"auto", "prompt", "writes", "approve"}


def test_sandbox_allows_the_partial_log_lock_and_parent(tmp_path) -> None:
    experiment = _experiment()
    store = ObjectStore(SqliteBackend(tmp_path / "partial-lock.sqlite"))
    engine = _engine(store, experiment)
    partial_root = tmp_path / "partial-root"
    partial_root.mkdir()
    partial_path = partial_root / "partials"
    runner = SubprocessCodexRunner(
        sqlite_path=str(tmp_path / "mcp-store.sqlite"),
        runtime_config=_runtime_config(
            engine, partial_log_path=str(partial_path)
        ),
        reward_policy=experiment.reward_policy,
        environment={},
    )

    writable = runner._writable_runtime_paths(tmp_path / "root")
    profile = _MacOsProcessIsolation._profile(
        readable_paths=(), writable_paths=writable
    )

    # PartialLog locks on this exact sibling and creates the record
    # directory through its parent descriptor.
    lock_path = partial_root / f".{partial_path.name}.lock"
    assert PartialLog(partial_path)._lock_path == lock_path
    assert lock_path in writable
    assert partial_root.resolve() in writable
    assert json.dumps(str(lock_path)) in profile
    parent_rule = json.dumps(str(partial_root.resolve()))
    assert f"(allow file-write* (subpath {parent_rule}))" in profile
