from __future__ import annotations

from typing import Any, cast

from dr_store import ObjectStore, SqliteBackend

from tests.envs.support import (
    FakeTransport,
    constant_reply,
    execution_policy,
)
from whetstone.envs.factory import build_env_experiment
from whetstone.evaluation import (
    EngineToolEvaluator,
    EvaluationEngine,
    EvaluationEvidence,
)
from whetstone.optimization import (
    EvaluatingToolExecutor,
    FakeProposerTransport,
    GepaAdapter,
    MappingAdapterRegistry,
    OptimizationHarness,
    OptimizationStepRequest,
    OutputContract,
    ProposerConfig,
    StepKind,
    StepMode,
    ToolCallStore,
    ToolCapacity,
    ToolConfig,
    ToolDefinition,
    strict_pareto_accepts,
)
from whetstone.optimization.tool_eval import ToolEvaluation
from whetstone.optimization.tools import ToolCall

from .support import FULL_A


def _experiment():
    return build_env_experiment(
        "c18",
        model="openai/test",
        pool_n_per_stratum=2,
        split_sizes=(3, 1, 1),
        repeats=2,
    )


def _tool_config(engine: EvaluationEngine, name: str) -> ToolConfig:
    definition = ToolDefinition(
        tool_name=name,
        input_fields=("base_ref", "model_route", "template", "task_ids"),
        output_fields=("objective_values", "evaluation_evidence_ref"),
    )
    return ToolConfig(
        tool_name=name,
        tool_definition_ref=f"tooldef://{name}",
        tool_definition_identity_hash=definition.identity_hash(),
        endpoint=f"tool://{name}",
        eval_config_ref=engine.eval_config_ref.record_ref.content_hash,
        eval_config_identity_hash=engine.eval_config_ref.identity_hash,
        reward_policy_ref=engine.experiment.reward_policy.identity_hash(),
        capacity=ToolCapacity(max_accepted_calls=10),
        store_namespace=f"gepa-{name}",
    )


def _request(engine: EvaluationEngine) -> OptimizationStepRequest:
    minibatch = _tool_config(engine, "evaluate_minibatch")
    subset = _tool_config(engine, "evaluate_subset")
    return OptimizationStepRequest(
        run_id="gepa-run",
        step_id="gepa-0",
        optimizer_config_hash=FULL_A,
        adapter_key="gepa",
        mode=StepMode.TOOL_USING,
        kind=StepKind.TOOL,
        step_index=0,
        candidates=(engine.experiment.initial_candidate,),
        pools={"task_pool": list(engine.sampling.task_set.task_identities)},
        hyperparameters={
            "minibatch_size": 2,
            "max_reflection_attempts_per_step": 1,
            "max_reflection_lm_calls": 2,
            "returned_proposal_count": 1,
        },
        output_contract=OutputContract(returned_proposal_count=1),
        tool_configs=(minibatch, subset),
    )


def test_same_minibatch_reflection_evidence_and_restart(tmp_path) -> None:
    database = tmp_path / "gepa.sqlite"
    store = ObjectStore(SqliteBackend(database))
    transport = FakeTransport(reply=constant_reply("wrong"))
    experiment = _experiment()
    engine = EvaluationEngine(
        store=store,
        experiment=experiment,
        sampling=experiment.eval_configs.internal,
        execution_policy=execution_policy(),
        transport=transport,
    )
    reflection = FakeProposerTransport(
        {("gepa_reflection", 0): ("{question}\n{query}\nTrue or False.",)},
        execution_policy_hash=FULL_A,
        prompt_adapter_identity_hash=FULL_A,
    )
    adapter = GepaAdapter(
        reflection_config=ProposerConfig(
            provider_call_config_ref="provider://reflection",
            provider_call_config_hash=FULL_A,
        ),
        reflection_transport=reflection,
    )
    tool_store = ToolCallStore(store)
    request = _request(engine)

    class RecordingEvaluator:
        def __init__(self) -> None:
            self.calls: list[ToolCall] = []
            self.evaluations: list[ToolEvaluation] = []
            self.delegate = EngineToolEvaluator(engine)

        def evaluate(
            self, call: ToolCall, config: ToolConfig
        ) -> ToolEvaluation:
            self.calls.append(call)
            evaluation = self.delegate.evaluate(call, config)
            self.evaluations.append(evaluation)
            return evaluation

    evaluator = RecordingEvaluator()
    harness = OptimizationHarness(
        store=store,
        adapter_registry=MappingAdapterRegistry({"gepa": adapter}),
        tool_executor=EvaluatingToolExecutor(
            evaluator, experiment.reward_policy
        ),
        tool_store=tool_store,
    )

    result, result_ref = harness.run_step(request)

    assert len(result.tool_evidence) == 4
    assert result.state_ref is not None
    state = cast(dict[str, Any], store.get(result.state_ref.reference))
    same_minibatch = state["same_minibatch"]
    diagnostic = cast(dict[str, Any], state["diagnostic_evidence"])
    assert same_minibatch == diagnostic["task_ids"]
    assert diagnostic["parent_objectives"]
    prompt_request = reflection.calls[0][1]
    assert prompt_request.context["diagnostic_evidence"] == diagnostic
    assert "parent_objectives" in str(
        prompt_request.context["proposal_prompt"]
    )
    assert state["acceptance"]["policy"] == ("same_minibatch_strict_pareto/v1")
    assert state["acceptance"]["decision"] is True
    minibatch_hash = request.tool_configs[0].identity_hash()
    minibatch_calls = [
        call
        for call in evaluator.calls
        if call.tool_config_hash == minibatch_hash
    ]
    assert len(minibatch_calls) == 2
    assert (
        minibatch_calls[0].args["task_ids"]
        == minibatch_calls[1].args["task_ids"]
        == same_minibatch
    )
    minibatch_evaluations = [
        evaluation
        for call, evaluation in zip(
            evaluator.calls, evaluator.evaluations, strict=True
        )
        if call.tool_config_hash == minibatch_hash
    ]
    assert len(minibatch_evaluations) == 2
    derived_hashes = {
        evaluation.eval_config_hash for evaluation in minibatch_evaluations
    }
    assert len(derived_hashes) == 1
    derived_hash = derived_hashes.pop()
    assert derived_hash != engine.eval_config_ref.identity_hash
    assert {
        evaluation.source_eval_config_hash
        for evaluation in minibatch_evaluations
    } == {engine.eval_config_ref.identity_hash}
    assert diagnostic["tool_output"]["eval_config_hash"] == derived_hash
    for evaluation in minibatch_evaluations:
        evidence = EvaluationEvidence.model_validate(
            store.get(evaluation.rollout_refs[0].reference)
        )
        assert list(evidence.task_identities) == same_minibatch
        assert evidence.repeat_count == 2
        assert evidence.eval_config.identity_hash == derived_hash
        assert len(evidence.per_task_values) == 2

    class ExplodingRegistry:
        def resolve(self, adapter_key: str):
            raise AssertionError(f"restart resolved adapter {adapter_key}")

    fresh = OptimizationHarness(
        store=ObjectStore(SqliteBackend(database)),
        adapter_registry=ExplodingRegistry(),
    )
    replay, replay_ref = fresh.run_step(request)
    assert (replay, replay_ref) == (result, result_ref)
    assert len(transport.served) == 20


def test_strict_pareto_requires_no_regression_and_one_improvement() -> None:
    assert strict_pareto_accepts(
        parent={"correctness": 0.5, "compression": 10.0},
        child={"correctness": 0.5, "compression": 9.0},
    )
    assert not strict_pareto_accepts(
        parent={"correctness": 0.5, "compression": 10.0},
        child={"correctness": 0.6, "compression": 11.0},
    )
    assert not strict_pareto_accepts(
        parent={"correctness": 0.5, "compression": 10.0},
        child={"correctness": 0.5, "compression": 10.0},
    )


def test_registry_key_and_mode() -> None:
    adapter = GepaAdapter(
        reflection_config=ProposerConfig(
            provider_call_config_ref="provider://reflection",
            provider_call_config_hash=FULL_A,
        ),
        reflection_transport=FakeProposerTransport(
            {},
            execution_policy_hash=FULL_A,
            prompt_adapter_identity_hash=FULL_A,
        ),
    )
    registry = MappingAdapterRegistry({"gepa": adapter})

    assert registry.resolve("gepa") is adapter
    assert adapter.mode is StepMode.TOOL_USING
