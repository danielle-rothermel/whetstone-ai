from __future__ import annotations

import sys
from pathlib import Path

import pytest
from dr_store import ObjectStore, SqliteBackend

from tests.envs.support import (
    FakeTransport,
    constant_reply,
    execution_policy,
)
from whetstone.envs.factory import build_env_experiment
from whetstone.evaluation import EngineToolEvaluator, EvaluationEngine
from whetstone.execution.mode import EvaluationRuntimeConfig
from whetstone.optimization import (
    CODEX_OUTPUT_ARTIFACT_SCHEMA,
    Candidate,
    CodexAdapter,
    CodexOutputArtifact,
    EvaluateCandidateServer,
    EvaluatingToolExecutor,
    FakeCodexRunner,
    InProcessMcpProcess,
    OpaqueStepError,
    OptimizationHarness,
    OptimizationStepRequest,
    OutputContract,
    ScriptedAgentCall,
    StepKind,
    StepMode,
    StepStatus,
    SubprocessCodexRunner,
    ToolCallStore,
    ToolCapacity,
    ToolConfig,
    ToolDefinition,
    TypedRef,
)
from whetstone.optimization.codex_runner import (
    _CODEX_DENIED_FEATURES,
    _MacOsProcessIsolation,
)
from whetstone.optimization.mcp_server import build_server_from_env

from .support import FULL_A


def runtime_transport() -> FakeTransport:
    """Provider-boundary fake reconstructed by the MCP child config."""
    return FakeTransport(reply=constant_reply("wrong"))


def _experiment():
    return build_env_experiment(
        "c18",
        model="openai/test",
        pool_n_per_stratum=2,
        split_sizes=(1, 1, 1),
        repeats=1,
    )


def _tool_config(engine: EvaluationEngine, namespace: str) -> ToolConfig:
    definition = ToolDefinition(
        tool_name="evaluate_candidate",
        input_fields=("call_id", "base_ref", "model_route", "template"),
        output_fields=("evaluation_evidence_ref", "output_artifact_ref"),
    )
    return ToolConfig(
        tool_name=definition.tool_name,
        tool_definition_ref="tooldef://evaluate_candidate",
        tool_definition_identity_hash=definition.identity_hash(),
        endpoint="mcp://whetstone/evaluate_candidate",
        eval_config_ref=engine.eval_config_ref.record_ref.content_hash,
        eval_config_identity_hash=engine.eval_config_ref.identity_hash,
        reward_policy_ref=engine.experiment.reward_policy.identity_hash(),
        capacity=ToolCapacity(max_accepted_calls=4),
        store_namespace=namespace,
    )


def _proposals(base: Candidate) -> tuple[Candidate, Candidate]:
    return (
        Candidate(
            candidate_id="codex-a",
            base_ref=base.base_ref,
            payload={
                "user_prompt_template": (
                    "{question}\n{query}\nRespond True or False."
                )
            },
        ),
        Candidate(
            candidate_id="codex-b",
            base_ref=base.base_ref,
            payload={
                "user_prompt_template": (
                    "{question}\n{query}\nOnly True or False."
                )
            },
        ),
    )


def _request(
    base: Candidate,
    config: ToolConfig,
    *,
    distinct: bool = False,
) -> OptimizationStepRequest:
    return OptimizationStepRequest(
        run_id="codex-run",
        step_id="codex-opaque",
        optimizer_config_hash=FULL_A,
        adapter_key="codex",
        mode=StepMode.TOOL_USING,
        kind=StepKind.TOOL,
        step_index=0,
        candidates=(base,),
        output_contract=OutputContract(
            returned_proposal_count=2,
            require_distinct_bases=distinct,
        ),
        tool_configs=(config,),
    )


def _stack(tmp_path):
    database = tmp_path / "codex.sqlite"
    store = ObjectStore(SqliteBackend(database))
    experiment = _experiment()
    transport = runtime_transport()
    engine = EvaluationEngine(
        store=store,
        experiment=experiment,
        sampling=experiment.eval_configs.internal,
        execution_policy=execution_policy(),
        transport=transport,
    )
    config = _tool_config(engine, "codex-durable")
    tool_store = ToolCallStore(store)
    executor = EvaluatingToolExecutor(
        EngineToolEvaluator(engine), experiment.reward_policy
    )
    server = EvaluateCandidateServer(
        tool_config=config,
        store=tool_store,
        executor=executor,
    )
    base = experiment.initial_candidate
    runner = FakeCodexRunner(
        process=InProcessMcpProcess(server),
        scripted_calls=(
            ScriptedAgentCall(
                call_id="agent-call-1",
                base_ref=base.base_ref,
                model_route=base.base_ref,
                template=base.payload["user_prompt_template"],
            ),
        ),
        final_proposals=_proposals(base),
    )
    return (
        database,
        store,
        experiment,
        transport,
        config,
        tool_store,
        executor,
        runner,
        base,
    )


def test_fake_process_actual_jsonrpc_artifact_and_restart(tmp_path) -> None:
    (
        database,
        store,
        _experiment_value,
        transport,
        config,
        tool_store,
        executor,
        runner,
        base,
    ) = _stack(tmp_path)
    adapter = CodexAdapter(runner, store=store, tool_store=tool_store)
    request = _request(base, config)
    harness = OptimizationHarness(
        store=store,
        adapter_registry=_Registry(adapter),
        tool_executor=executor,
        tool_store=tool_store,
    )

    result, result_ref = harness.run_step(request)

    assert result.status is StepStatus.COMPLETE
    assert len(result.accepted_candidates) == 2
    assert {item.record.base_ref for item in result.accepted_candidates} == {
        base.base_ref
    }
    assert len(result.tool_evidence) == 1
    assert runner.observed_payloads[0]["refused"] is False
    assert result.state_ref is not None
    state = store.get(result.state_ref.reference)
    assert isinstance(state, dict)
    artifact_ref = TypedRef.model_validate(state["codex_output_artifact_ref"])
    assert artifact_ref.schema_name == CODEX_OUTPUT_ARTIFACT_SCHEMA
    assert store.get(artifact_ref.reference)["run_id"] == request.run_id

    fresh_store = ObjectStore(SqliteBackend(database))
    fresh_calls = ToolCallStore(fresh_store).namespace_calls(
        config.store_namespace, config.identity_hash()
    )
    assert [call.call_id for call in fresh_calls] == ["agent-call-1"]

    class ExplodingRegistry:
        def resolve(self, adapter_key: str):
            raise AssertionError(f"resolved {adapter_key}")

    replay, replay_ref = OptimizationHarness(
        store=fresh_store,
        adapter_registry=ExplodingRegistry(),
    ).run_step(request)
    assert (replay, replay_ref) == (result, result_ref)
    assert len(transport.served) == 1


def test_distinct_bases_is_conditional_not_unconditional(tmp_path) -> None:
    (
        _database,
        store,
        _experiment_value,
        _transport,
        config,
        tool_store,
        _executor,
        runner,
        base,
    ) = _stack(tmp_path)
    adapter = CodexAdapter(runner, store=store, tool_store=tool_store)

    allowed = adapter.invoke(_request(base, config, distinct=False), ())
    rejected = adapter.invoke(_request(base, config, distinct=True), ())

    assert len(allowed.accepted_candidates) == 2
    assert allowed.proposed_status is StepStatus.COMPLETE
    assert rejected.accepted_candidates == ()
    assert rejected.proposed_status is StepStatus.FAILED


def test_serialized_runtime_reconstructs_real_engine(tmp_path) -> None:
    experiment = _experiment()
    parent_store = ObjectStore(SqliteBackend(tmp_path / "parent.sqlite"))
    parent_engine = EvaluationEngine(
        store=parent_store,
        experiment=experiment,
        sampling=experiment.eval_configs.internal,
        execution_policy=execution_policy(),
        transport=runtime_transport(),
    )
    config = _tool_config(parent_engine, "codex-child")
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
        transport_factory=(
            "tests.optimization.test_codex_adapter:runtime_transport"
        ),
        partial_log_path=str(tmp_path / "child-partials.jsonl"),
        prompt_cache_path=str(tmp_path / "child-cache"),
    )
    child_db = tmp_path / "child.sqlite"
    server = build_server_from_env(
        {
            "WS_MCP_SQLITE_PATH": str(child_db),
            "WS_MCP_TOOL_CONFIG": config.model_dump_json(),
            "WS_MCP_RUNTIME_CONFIG": runtime.model_dump_json(),
            "WS_MCP_REWARD_POLICY": (
                experiment.reward_policy.model_dump_json()
            ),
        }
    )
    process = InProcessMcpProcess(server)
    runner = FakeCodexRunner(
        process=process,
        scripted_calls=(
            ScriptedAgentCall(
                call_id="child-call",
                base_ref=experiment.initial_candidate.base_ref,
                model_route=experiment.initial_candidate.base_ref,
                template=experiment.initial_candidate.payload[
                    "user_prompt_template"
                ],
            ),
        ),
        final_proposals=(),
    )
    request = _request(experiment.initial_candidate, config).model_copy(
        update={"output_contract": OutputContract(returned_proposal_count=0)}
    )

    output = runner.run(request, config)

    assert output.artifact.run_id == request.run_id
    assert runner.observed_payloads[0]["refused"] is False
    calls = ToolCallStore(
        ObjectStore(SqliteBackend(child_db))
    ).namespace_calls(config.store_namespace, config.identity_hash())
    assert [call.call_id for call in calls] == ["child-call"]


def _subprocess_boundary(tmp_path, mode: str):
    experiment = _experiment()
    store = ObjectStore(SqliteBackend(tmp_path / f"subprocess-{mode}.sqlite"))
    engine = EvaluationEngine(
        store=store,
        experiment=experiment,
        sampling=experiment.eval_configs.internal,
        execution_policy=execution_policy(),
        transport=runtime_transport(),
    )
    config = _tool_config(engine, f"codex-subprocess-{mode}")
    runtime = EvaluationRuntimeConfig(
        env_name="c18",
        model="openai/test",
        pool_n_per_stratum=2,
        split_sizes=(1, 1, 1),
        repeats=1,
        expected_eval_config_hash=engine.eval_config_ref.identity_hash,
        execution_policy=execution_policy(),
        transport_factory=(
            "tests.optimization.test_codex_adapter:runtime_transport"
        ),
    )
    executable = tmp_path / f"fake-codex-{mode}"
    executable.write_text(
        (
            f"#!{sys.executable}\n"
            "from tests.optimization.fake_codex_cli import app\n"
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
    request = _request(experiment.initial_candidate, config).model_copy(
        update={
            "output_contract": OutputContract(
                returned_proposal_count=1 if mode == "proposal" else 0
            ),
            "hyperparameters": {
                "adversarial_outside_path": str(outside_secret),
                "search_temperature": 0.7,
            },
        }
    )
    return store, config, runner, request, mcp_path


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
    store, config, runner, request, _mcp_path = _subprocess_boundary(
        tmp_path, "success"
    )
    adapter = CodexAdapter(
        runner,
        store=store,
        tool_store=ToolCallStore(store),
    )

    output = adapter.invoke(request, ())

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
    assert config.store_namespace == output.state_delta["tool_namespace"]


@pytest.mark.parametrize("mode", ["missing", "malformed"])
def test_subprocess_rejects_missing_or_malformed_artifact(
    tmp_path, mode: str, cross_platform_fake_codex_boundary
) -> None:
    _store, config, runner, request, _mcp_path = _subprocess_boundary(
        tmp_path, mode
    )

    with pytest.raises(OpaqueStepError, match="final output artifact"):
        runner.run(request, config)


def test_subprocess_nonzero_proposal_uses_mcp_evidence(
    tmp_path, cross_platform_fake_codex_boundary
) -> None:
    store, config, runner, request, mcp_path = _subprocess_boundary(
        tmp_path, "proposal"
    )
    mcp_store = ObjectStore(SqliteBackend(mcp_path))
    adapter = CodexAdapter(
        runner,
        store=store,
        tool_store=ToolCallStore(mcp_store),
    )

    output = adapter.invoke(request, ())

    assert output.proposed_status is StepStatus.COMPLETE
    assert len(output.accepted_candidates) == 1
    assert output.accepted_candidates[0].base_ref == (
        request.candidates[0].base_ref
    )
    assert len(output.tool_call_records) == 1
    record = output.tool_call_records[0]
    assert record.call.call_id == "subprocess-proposal-evaluation"
    assert record.result.refusal is None
    assert record.result.evaluation_evidence_refs
    calls = ToolCallStore(mcp_store).namespace_calls(
        config.store_namespace,
        config.identity_hash(),
    )
    assert [call.call_id for call in calls] == [
        "subprocess-proposal-evaluation"
    ]


@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="neighboring-secret enforcement requires macOS Seatbelt",
)
def test_macos_process_isolation_denies_neighboring_secret(tmp_path) -> None:
    _store, config, runner, request, _mcp_path = _subprocess_boundary(
        tmp_path, "success"
    )

    result = runner.run(request, config)

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


class _Registry:
    def __init__(self, adapter: CodexAdapter) -> None:
        self._adapter = adapter

    def resolve(self, adapter_key: str) -> CodexAdapter:
        assert adapter_key == "codex"
        return self._adapter
