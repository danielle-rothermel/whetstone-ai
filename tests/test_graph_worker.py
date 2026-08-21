from __future__ import annotations

from collections.abc import Mapping

import pytest
from pydantic import ValidationError

from whetstone.eval.drivers.graph_row_request import (
    GraphRowRequest,
    import_path_for_callable,
    import_path_for_type,
    resolve_import_path,
)
from whetstone.eval.drivers.graph_worker import run_row
from whetstone.eval.drivers.subprocess_graph_rollout import (
    SubprocessGraphRolloutEvalDriver,
)
from whetstone.eval.protocol import EvalTaskView
from whetstone.provider.language_model import PlainPromptAdapter
from whetstone.provider.policy import ProviderExecutionPolicy, default_transport_policy
from whetstone.testing.fakes.eval_procedure import FakeEvalProcedureRunner
from whetstone.testing.fakes.transport import (
    FakeLlmTransport,
    fake_llm_transport_factory,
)
from whetstone.testing.toy.experiment import (
    TOY_MUTATION_FIELD,
    build_toy_experiment,
    toy_template_render_contract,
)

CUSTOM_GENERATION = "custom-worker-generation"


def custom_transport_factory(
    policy: ProviderExecutionPolicy,
) -> FakeLlmTransport:
    return FakeLlmTransport(
        transport_policy=policy.transport_policy,
        text_factory=lambda _request: CUSTOM_GENERATION,
    )


class CustomEvalProcedureRunner:
    def run_eval_node(
        self,
        *,
        node_id: str,
        node_inputs: Mapping[str, object],
        evaluation_procedure_config_hash: str,
        task: EvalTaskView,
    ) -> tuple[float | None, object | None, dict[str, object]]:
        _ = (node_id, node_inputs, evaluation_procedure_config_hash, task)
        return 0.42, {"text": CUSTOM_GENERATION}, {}


def _execution_policy() -> ProviderExecutionPolicy:
    return ProviderExecutionPolicy(
        transport_policy=default_transport_policy(
            api_key_env="WHETSTONE_TOY_API_KEY"
        )
    )


def _graph_row_request(**overrides: object) -> GraphRowRequest:
    experiment = build_toy_experiment()
    policy = _execution_policy()
    adapter = PlainPromptAdapter()
    payload: dict[str, object] = {
        "candidate_id": "cand-1",
        "task_id": "task-a",
        "task_index": 0,
        "seed_index": 0,
        "split_role": "internal_eval",
        "rendered_prompt": "Reply briefly to: hello A",
        "graph_config": experiment.rollout_graph.graph_config.model_dump(
            mode="json"
        ),
        "rollout_graph_hash": experiment.rollout_graph.graph_hash,
        "provider_call_config": (
            experiment.rollout_graph.provider_call_config.model_dump(mode="json")
        ),
        "rng_seed": 1,
        "mutation_field": TOY_MUTATION_FIELD,
        "eval_procedure_config_hash": (
            experiment.rollout_graph.procedure_config_hash
        ),
        "execution_policy": policy.model_dump(mode="json"),
        "execution_policy_hash": policy.identity_hash,
        "prompt_inputs": {"prompt": "hello A"},
        "gold": "A",
        "transport_factory": import_path_for_callable(fake_llm_transport_factory),
        "eval_runner": import_path_for_type(FakeEvalProcedureRunner),
        "prompt_adapter_type": import_path_for_type(PlainPromptAdapter),
        "prompt_adapter": adapter.model_dump(mode="json"),
    }
    payload.update(overrides)
    return GraphRowRequest.model_validate(payload)


def test_graph_row_request_reconstructs_non_default_prompt_adapter() -> None:
    adapter = PlainPromptAdapter(output_field="generation")
    request = _graph_row_request(prompt_adapter=adapter.model_dump(mode="json"))
    adapter_type = resolve_import_path(request.prompt_adapter_type)
    reconstructed = adapter_type.model_validate(request.prompt_adapter)
    assert reconstructed.output_field == "generation"


def test_run_row_uses_serialized_transport_and_eval_runner() -> None:
    request = _graph_row_request(
        transport_factory=import_path_for_callable(custom_transport_factory),
        eval_runner=import_path_for_type(CustomEvalProcedureRunner),
    )
    payload = run_row(request.model_dump(mode="json"))
    assert payload["output_text"] == CUSTOM_GENERATION
    assert payload["score"] == 0.42
    assert payload["row_state"] == "success"


def test_graph_row_request_rejects_missing_collaborator_fields() -> None:
    payload = _graph_row_request().model_dump(mode="json")
    del payload["transport_factory"]
    with pytest.raises(ValidationError):
        GraphRowRequest.model_validate(payload)


def test_subprocess_driver_rejects_lambda_transport_factory() -> None:
    with pytest.raises(ValueError, match="top-level"):
        SubprocessGraphRolloutEvalDriver(
            eval_runner=FakeEvalProcedureRunner(),
            mutation_field=TOY_MUTATION_FIELD,
            render_contract=toy_template_render_contract(),
            transport_factory=lambda policy: FakeLlmTransport(
                transport_policy=policy.transport_policy
            ),
        )


def test_subprocess_driver_rejects_nested_transport_factory() -> None:
    def nested_factory(policy: ProviderExecutionPolicy) -> FakeLlmTransport:
        return FakeLlmTransport(transport_policy=policy.transport_policy)

    with pytest.raises(ValueError, match="top-level"):
        SubprocessGraphRolloutEvalDriver(
            eval_runner=FakeEvalProcedureRunner(),
            mutation_field=TOY_MUTATION_FIELD,
            render_contract=toy_template_render_contract(),
            transport_factory=nested_factory,
        )


def test_subprocess_driver_accepts_importable_collaborators() -> None:
    driver = SubprocessGraphRolloutEvalDriver(
        eval_runner=FakeEvalProcedureRunner(),
        mutation_field=TOY_MUTATION_FIELD,
        render_contract=toy_template_render_contract(),
        transport_factory=fake_llm_transport_factory,
        prompt_adapter=PlainPromptAdapter(output_field="generation"),
    )
    assert driver._transport_factory_path == (  # noqa: SLF001
        "whetstone.testing.fakes.transport:fake_llm_transport_factory"
    )
    assert driver._eval_runner_path == (  # noqa: SLF001
        "whetstone.testing.fakes.eval_procedure:FakeEvalProcedureRunner"
    )
    assert driver._prompt_adapter_type_path == (  # noqa: SLF001
        "whetstone.provider.language_model:PlainPromptAdapter"
    )
    assert driver._prompt_adapter_payload == {"output_field": "generation"}  # noqa: SLF001


def test_reference_runtime_collaborator_paths_resolve() -> None:
    factory_path = import_path_for_callable(fake_llm_transport_factory)
    runner_path = import_path_for_type(FakeEvalProcedureRunner)
    adapter_path = import_path_for_type(PlainPromptAdapter)
    assert factory_path == (
        "whetstone.testing.fakes.transport:fake_llm_transport_factory"
    )
    assert runner_path == (
        "whetstone.testing.fakes.eval_procedure:FakeEvalProcedureRunner"
    )
    assert adapter_path == (
        "whetstone.provider.language_model:PlainPromptAdapter"
    )
    factory = resolve_import_path(factory_path)
    runner_type = resolve_import_path(runner_path)
    adapter_type = resolve_import_path(adapter_path)
    transport = factory(_execution_policy())
    runner = runner_type()
    adapter = adapter_type.model_validate({})
    assert isinstance(transport, FakeLlmTransport)
    assert isinstance(runner, FakeEvalProcedureRunner)
    assert isinstance(adapter, PlainPromptAdapter)
