from __future__ import annotations

from whetstone.eval.drivers.graph_row_request import GraphRowRequest
from whetstone.provider.llm_call import derive_rng_seed
from whetstone.provider.policy import ProviderExecutionPolicy, default_transport_policy
from whetstone.testing.toy.experiment import (
    TOY_MUTATION_FIELD,
    build_toy_experiment,
    toy_template_render_contract,
)


def sample_graph_row_request() -> GraphRowRequest:
    experiment = build_toy_experiment(num_seeds=1)
    sampling = experiment.eval_configs.internal
    task = sampling.tasks[0]
    rollout_graph = experiment.rollout_graph
    candidate = experiment.initial_candidate
    render_contract = toy_template_render_contract()
    rendered = render_contract.render(
        candidate.payload[TOY_MUTATION_FIELD],
        task.prompt_inputs,
    )
    execution_policy = ProviderExecutionPolicy(
        transport_policy=default_transport_policy(
            api_key_env="WHETSTONE_TOY_API_KEY"
        )
    )
    return GraphRowRequest(
        candidate_id=candidate.candidate_id,
        task_id=task.task_id,
        task_index=0,
        seed_index=0,
        split_role=sampling.split_role,
        rendered_prompt=rendered,
        graph_config=rollout_graph.graph_config.model_dump(mode="json"),
        rollout_graph_hash=rollout_graph.graph_hash,
        provider_call_config=rollout_graph.provider_call_config.model_dump(
            mode="json"
        ),
        rng_seed=derive_rng_seed(
            candidate.candidate_id,
            task.task_id,
            0,
        ),
        mutation_field=TOY_MUTATION_FIELD,
        eval_procedure_config_hash=rollout_graph.procedure_config_hash,
        execution_policy_hash=execution_policy.identity_hash,
        prompt_inputs=dict(task.prompt_inputs),
        gold=task.gold,
    )
