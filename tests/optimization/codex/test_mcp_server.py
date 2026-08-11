from __future__ import annotations

import pytest
from dr_serialize import StrictJsonDecodeError
from dr_store import ObjectStore, SqliteBackend

from tests.envs.support import execution_policy
from tests.optimization.codex.support import (
    MODEL_ROUTE,
    ROW_JOB_ENTRYPOINT,
    FakeCodexRunner,
    ScriptedAgentCall,
    binding,
    engine,
    request,
    tool_config,
)
from whetstone.envs.factory import EnvExperiment
from whetstone.optimization.codex.mcp_environment import McpEnvironmentKey
from whetstone.optimization.codex.mcp_server import (
    _strict_env_json,
    build_server_from_env,
)
from whetstone.optimization.codex.runtime import EvaluationRuntimeConfig
from whetstone.optimization.tools.contracts import (
    ToolCapacityBinding,
    ToolConfig,
)


def _server_environment(
    *,
    sqlite_path: str,
    config: ToolConfig,
    capacity_binding: ToolCapacityBinding,
    runtime: EvaluationRuntimeConfig,
    experiment: EnvExperiment,
) -> dict[str, str]:
    return {
        McpEnvironmentKey.SQLITE_PATH.value: sqlite_path,
        McpEnvironmentKey.TOOL_CONFIG.value: config.model_dump_json(),
        McpEnvironmentKey.CAPACITY_BINDING.value: (
            capacity_binding.model_dump_json()
        ),
        McpEnvironmentKey.RUNTIME_CONFIG.value: runtime.model_dump_json(),
        McpEnvironmentKey.REWARD_POLICY.value: (
            experiment.reward_policy.model_dump_json()
        ),
    }


@pytest.mark.parametrize(
    "raw",
    ['{"value":1,"value":2}', '{"value":NaN}'],
)
def test_environment_json_rejects_non_strict_values(raw: str) -> None:
    with pytest.raises(StrictJsonDecodeError):
        _strict_env_json(raw)


def test_serialized_environment_reconstructs_evaluation_server(
    tmp_path, codex_experiment: EnvExperiment
) -> None:
    parent_store = ObjectStore(SqliteBackend(tmp_path / "parent.sqlite"))
    parent_engine = engine(parent_store, codex_experiment)
    config = tool_config(parent_engine, codex_experiment, "codex-child")
    runtime = EvaluationRuntimeConfig(
        env_name="c18",
        model=MODEL_ROUTE,
        pool_n_per_stratum=2,
        split_sizes=(1, 1, 1),
        num_samples=1,
        expected_eval_config_hash=parent_engine.eval_config_ref.config_hash,
        execution_policy=execution_policy(),
        row_job_entrypoint=ROW_JOB_ENTRYPOINT,
        partial_log_path=str(tmp_path / "child-partials"),
        prompt_cache_path=str(tmp_path / "child-cache"),
    )
    (tmp_path / "child-cache").mkdir()
    step_request = request(
        codex_experiment.initial_candidate,
        config,
        proposal_count=0,
    )
    server = build_server_from_env(
        _server_environment(
            sqlite_path=str(tmp_path / "child.sqlite"),
            config=config,
            capacity_binding=binding(step_request),
            runtime=runtime,
            experiment=codex_experiment,
        )
    )
    template = codex_experiment.initial_candidate.payload[
        "user_prompt_template"
    ]
    assert isinstance(template, str)
    runner = FakeCodexRunner(
        server=server,
        scripted_calls=(
            ScriptedAgentCall(
                call_id="child-call",
                base_ref=codex_experiment.initial_candidate.base_ref,
                model_route=MODEL_ROUTE,
                template=template,
            ),
        ),
        final_proposals=(),
    )

    output = runner.run(step_request, server.handle)

    assert output.artifact.run_id == step_request.run_id
    assert runner.observed_payloads[0]["refused"] is False
    assert server.handle.config == config


def test_server_rejects_reward_policy_identity_mismatch(
    tmp_path, codex_experiment: EnvExperiment
) -> None:
    store = ObjectStore(SqliteBackend(tmp_path / "parent.sqlite"))
    evaluation_engine = engine(store, codex_experiment)
    config = tool_config(
        evaluation_engine,
        codex_experiment,
        "codex-wrong-reward",
    ).model_copy(update={"reward_policy_hash": "0" * 64})
    runtime = EvaluationRuntimeConfig(
        env_name="c18",
        model=MODEL_ROUTE,
        pool_n_per_stratum=2,
        split_sizes=(1, 1, 1),
        num_samples=1,
        expected_eval_config_hash=evaluation_engine.eval_config_ref.config_hash,
        execution_policy=execution_policy(),
        row_job_entrypoint=ROW_JOB_ENTRYPOINT,
    )
    step_request = request(codex_experiment.initial_candidate, config)

    with pytest.raises(ValueError):
        build_server_from_env(
            _server_environment(
                sqlite_path=str(tmp_path / "child.sqlite"),
                config=config,
                capacity_binding=binding(step_request),
                runtime=runtime,
                experiment=codex_experiment,
            )
        )
