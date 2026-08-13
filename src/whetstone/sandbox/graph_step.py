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
    from whetstone.eval.drivers.graph_rollout import run_rollout_row
    from whetstone.experiment.graph.rollout_template import (
        EVAL_NODE_ID,
        LLM_NODE_ID,
    )
    from whetstone.provider.language_model import PlainPromptAdapter
    from whetstone.provider.llm_call import LlmCallContext
    from whetstone.provider.policy import ProviderExecutionPolicy, default_transport_policy
    from whetstone.testing.fakes.eval_procedure import FakeEvalProcedureRunner
    from whetstone.testing.fakes.transport import FakeLlmTransport
    from whetstone.testing.toy.experiment import (
        ToyTask,
        build_toy_experiment,
        toy_template_render_contract,
        TOY_MUTATION_FIELD,
    )

    experiment = build_toy_experiment(num_seeds=1)
    rollout_graph = experiment.rollout_graph
    provider_config = rollout_graph.provider_call_config
    graph_hash_value = rollout_graph.graph_hash
    task = ToyTask(task_id="graph-task", prompt_inputs={"prompt": prompt}, gold=prompt)
    candidate = experiment.initial_candidate

    transport_policy = default_transport_policy(
        api_key_env="WHETSTONE_SANDBOX_KEY",
    )
    execution_policy = ProviderExecutionPolicy(transport_policy=transport_policy)
    llm_context = LlmCallContext(
        execution_policy=execution_policy,
        transport=FakeLlmTransport(transport_policy=transport_policy),
        prompt_adapter=PlainPromptAdapter(),
    )
    row = run_rollout_row(
        experiment=experiment,
        candidate=candidate,
        task=task,
        task_index=0,
        seed_index=0,
        split_role="internal_eval",
        llm_context=llm_context,
        eval_runner=FakeEvalProcedureRunner(),
        render_contract=toy_template_render_contract(),
        mutation_field=TOY_MUTATION_FIELD,
        resolve_provider_call_config=lambda _ref: provider_config,
        graph_external_input_field="prompt",
    )
    generation = row.output_text or ""
    score = row.score
    submission: dict[str, object] = {}
    if row.submission_result is not None and score is None:
        submission = {"value": row.submission_result}

    return GraphRunPreview(
        graph_hash=graph_hash_value,
        llm_node_id=LLM_NODE_ID,
        eval_node_id=EVAL_NODE_ID,
        prompt=prompt,
        generation=generation,
        score=score,
        submission=submission,
    )
