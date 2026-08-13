from __future__ import annotations

from pydantic import BaseModel, ConfigDict, StrictStr

__all__ = ["GraphRunPreview", "run_toy_graph_preview"]


class GraphRunPreview(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    graph_hash: StrictStr
    llm_node_id: StrictStr
    eval_node_id: StrictStr
    prompt: StrictStr
    generation: StrictStr
    score: float | None
    submission: dict[str, object]


def run_toy_graph_preview(
    *,
    prompt: str = "hello sandbox",
) -> GraphRunPreview:
    from whetstone.evaluation.drivers.graph_row import run_rollout_row
    from whetstone.provider.policy import default_transport_policy
    from whetstone.experiment.graph.llm_call_run_node import (
        EvalRunNodeDeps,
        LlmCallRunNodeDeps,
    )
    from whetstone.experiment.graph.rollout_template import (
        EVAL_NODE_ID,
        LLM_NODE_ID,
    )
    from whetstone.experiment.graph.run_node_registry import build_run_node
    from whetstone.provider.language_model import PlainPromptAdapter
    from whetstone.provider.llm_call import LlmCallContext
    from whetstone.provider.policy import ProviderExecutionPolicy
    from whetstone.testing.fakes.eval_procedure import FakeEvalProcedureRunner
    from whetstone.testing.fakes.transport import FakeLlmTransport
    from whetstone.testing.toy.experiment import ToyTask, build_toy_experiment

    experiment = build_toy_experiment(num_samples=1)
    generation_graph = experiment.generation_graph
    provider_config = generation_graph.provider_call_config
    graph = generation_graph.graph_config
    graph_hash_value = generation_graph.graph_hash
    task = ToyTask(task_id="graph-task", prompt_inputs={"prompt": prompt}, gold=prompt)

    transport_policy = default_transport_policy(
        api_key_env="WHETSTONE_SANDBOX_KEY",
    )
    llm_context = LlmCallContext(
        execution_policy=ProviderExecutionPolicy(
            transport_policy=transport_policy
        ),
        transport=FakeLlmTransport(transport_policy=transport_policy),
        prompt_adapter=PlainPromptAdapter(),
    )
    run_node = build_run_node(
        llm_deps=LlmCallRunNodeDeps(
            context=llm_context,
            resolve_provider_call_config=lambda _ref: provider_config,
            graph_hash=graph_hash_value,
        ),
        eval_deps=EvalRunNodeDeps(
            runner=FakeEvalProcedureRunner(),
            task=task,
        ),
    )
    result = run_rollout_row(
        graph=graph,
        inputs={"prompt": prompt},
        run_node=run_node,
    )
    llm_outcome = result.outcomes[LLM_NODE_ID]
    eval_outcome = result.outcomes[EVAL_NODE_ID]
    generation = ""
    if llm_outcome.output is not None:
        generation = str(llm_outcome.output.values.get("provider_generation", ""))
    score = None
    submission: dict[str, object] = {}
    if eval_outcome.output is not None:
        evaluation = eval_outcome.output.values.get("evaluation")
        if isinstance(evaluation, (int, float)):
            score = float(evaluation)
        elif evaluation is not None:
            submission = {"value": evaluation}

    return GraphRunPreview(
        graph_hash=graph_hash_value,
        llm_node_id=LLM_NODE_ID,
        eval_node_id=EVAL_NODE_ID,
        prompt=prompt,
        generation=generation,
        score=score,
        submission=submission,
    )
