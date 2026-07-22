"""The transport-injected internal-eval loop over an env's internal split.

:func:`run_internal_eval` drives a candidate (the naive Initial Candidate in
the factory tests) through an injected transport over the internal split and
produces a provenance-bearing internal ``env_exact_match`` Rollout Aggregate
plus the Reward the Reward Policy maps it to.

The transport is injected -- a scripted fake in tests, the durable executor
in production -- so nothing here makes a live paid LLM call. Each deliberate
observation renders the candidate's prompt template against a task's public
external inputs, calls the provider driver, and (on an accepted Generation)
scores the text 0/1 with the env oracle via the whetstone metric-extraction
operator. A failed provider call is an explicit ``failed`` row, never a
silent zero.

The reduction is the two-stage mean the design mandates for a 0/1 exact-match
score: per-task mean over the task's repeats, then the unweighted mean across
the complete internal Task Set. The aggregate is named ``env_exact_match`` so
the Reward Policy term selects it, and its row accounting covers the whole
planned internal matrix (no row dropped).
"""

from __future__ import annotations

from dataclasses import dataclass

from dr_code.eval import (
    AggregationConfig,
    AggregationDefinition,
    AggregationInput,
    AggregationStatus,
    aggregate,
)
from dr_providers import (
    MessageRole,
    PromptMessage,
    ProviderCallConfig,
    ProviderCallRequest,
    Transcript,
)
from whetstone_envs.core import Instance

from whetstone.code_eval.aggregate import (
    RolloutAggregate,
    RowPolicy,
    RowValue,
    TaskRows,
)
from whetstone.envs.factory import EnvExperiment
from whetstone.envs.oracle_operator import (
    ENV_EXACT_MATCH_NAME,
    env_exact_match_score,
)
from whetstone.envs.registry import DEFAULT_REPEATS, EnvSpec, env_spec
from whetstone.envs.reward import reward_from_internal_aggregate
from whetstone.envs.rollout_definition import render_prompt
from whetstone.envs.task import EnvTask
from whetstone.optimization.reward import Reward
from whetstone.optimization.schema import Candidate
from whetstone.provider.driver import TransportCall, run_provider_call
from whetstone.provider.policy import ProviderExecutionPolicy


@dataclass(frozen=True, slots=True)
class InternalEvalResult:
    """One candidate's internal-eval outcome over the internal split.

    Carries the provenance-bearing internal ``env_exact_match`` Rollout
    Aggregate and the Reward the Reward Policy maps it to. Both are internal
    role by construction (the Reward refuses any official evidence).

    ``per_task_scores`` is the aligned per-task mean 0/1 oracle score (one
    entry per instance, in instance order) computed from the SAME driven rows
    that produced ``aggregate`` -- a failed or missing row contributes 0 to
    the mean so every task yields a comparable number. It exists so a paired
    bootstrap CI can consume these scores with zero additional provider calls;
    no second drive of the split is ever needed.
    """

    aggregate: RolloutAggregate
    reward: Reward
    per_task_scores: tuple[float, ...]
    per_task_counts: tuple[int, ...]


def _per_task_score(task: TaskRows) -> float:
    """Mean 0/1 score over a task's planned repeats (absent rows count 0)."""
    completed = task.completed_rows()
    if not completed:
        return 0.0
    total = sum(row.value if row.is_present else 0.0 for row in completed)
    return total / len(completed)


def _per_task_count(task: TaskRows) -> int:
    """Count of completed (scored) repeats behind this task's mean.

    This is the observation weight the paired/pooled bootstrap needs to combine
    a task's mean with additional-repeat means exactly (a weighted mean by
    counts), so escalation pools new observations rather than discarding them.
    """
    return len(task.completed_rows())


def _mean_aggregation_config(policy: RowPolicy) -> AggregationConfig:
    """A ``mean`` Aggregation Config with the row policy's missing-data rule.

    Kept local (public dr-code APIs only) so the internal-eval loop owns its
    completeness policy rather than reaching into a private helper.
    """
    missing_data = "propagate" if policy is RowPolicy.PROPAGATE else "skip"
    return AggregationDefinition(
        definition_id="whetstone.env.internal_eval.aggregation",
        version="1",
    ).materialize(
        {
            "reduction": "mean",
            "missing_data": missing_data,
            "zero_denominator": "not_applicable",
        }
    )


def _request(config: ProviderCallConfig, prompt: str) -> ProviderCallRequest:
    return ProviderCallRequest(
        config=config,
        transcript=Transcript(
            messages=(PromptMessage(role=MessageRole.USER, content=prompt),)
        ),
    )


def _generation_row(
    env: EnvSpec,
    *,
    candidate: Candidate,
    instance: Instance,
    provider_call_config: ProviderCallConfig,
    execution_policy: ProviderExecutionPolicy,
    transport: TransportCall,
    procedure_config_hash: str,
    logical_call_id: str,
) -> RowValue:
    """Run one repeat: render, call the transport, score via the env oracle."""
    prompt = render_prompt(env, candidate, instance)
    result = run_provider_call(
        request=_request(provider_call_config, prompt),
        policy=execution_policy,
        transport=transport,
        logical_call_id=logical_call_id,
    )
    if not result.succeeded or result.generation is None:
        return RowValue(failed=True)
    score = env_exact_match_score(
        env=env,
        generation=result.generation.text,
        gold=instance.gold,
        evaluation_procedure_config_hash=procedure_config_hash,
    )
    return RowValue(value=float(score.value))


def _env_exact_match_aggregate(
    *,
    graph_hash: str,
    eval_config_hash: str,
    evaluation_context_id: str,
    task_rows: tuple[TaskRows, ...],
    repeat_count: int,
    policy: RowPolicy,
) -> RolloutAggregate:
    """The ``env_exact_match`` internal Rollout Aggregate (two-stage mean)."""
    per_task_config = _mean_aggregation_config(policy)
    all_rows: list[RowValue] = []
    per_task_inputs: list[AggregationInput] = []
    for task in task_rows:
        completed = task.completed_rows()
        all_rows.extend(completed)
        task_output = aggregate(
            per_task_config,
            tuple(row.to_aggregation_input() for row in completed),
        )
        if task_output.status is AggregationStatus.OK:
            per_task_inputs.append(
                AggregationInput(value=task_output.value, applicable=True)
            )
        elif task_output.status is AggregationStatus.NOT_APPLICABLE:
            per_task_inputs.append(
                AggregationInput(value=None, applicable=False)
            )
        else:
            per_task_inputs.append(
                AggregationInput(value=None, applicable=True)
            )

    cross_task_config = _mean_aggregation_config(policy)
    output = aggregate(cross_task_config, tuple(per_task_inputs))
    present = sum(1 for r in all_rows if r.is_present)
    missing = sum(1 for r in all_rows if r.missing)
    failed = sum(1 for r in all_rows if r.failed)
    invalid = sum(1 for r in all_rows if r.invalid)
    return RolloutAggregate(
        name=ENV_EXACT_MATCH_NAME,
        graph_hash=graph_hash,
        eval_config_hash=eval_config_hash,
        evaluation_context_id=evaluation_context_id,
        task_count=len(task_rows),
        repeat_count=repeat_count,
        aggregation_output=output,
        rows_present=present,
        rows_missing=missing,
        rows_failed=failed,
        rows_invalid=invalid,
    )


def run_internal_eval(
    experiment: EnvExperiment,
    *,
    candidate: Candidate,
    instances: tuple[Instance, ...],
    execution_policy: ProviderExecutionPolicy,
    transport: TransportCall,
    repeats: int = DEFAULT_REPEATS,
    policy: RowPolicy = RowPolicy.PROPAGATE,
) -> InternalEvalResult:
    """Evaluate ``candidate`` over ``instances`` (the internal split).

    For each instance, ``repeats`` deliberate observations run through the
    injected transport; each accepted Generation is scored 0/1 by the env
    oracle. The per-task means reduce to a single internal ``env_exact_match``
    Rollout Aggregate, and the Reward Policy maps that aggregate's value to an
    internal-role Reward. No live paid call is made: the transport is
    injected.
    """
    env = env_spec(experiment.env_name)
    rd = experiment.rollout_definition
    procedure_hash = experiment.eval_configs.procedure_config_hash
    internal = experiment.eval_configs.internal
    eval_config_hash = internal.eval_config.config_identity_hash
    # The concrete internal Evaluation Context is minted by orchestration; the
    # helper stamps a stable internal id derived from the internal Eval Config
    # identity onto the aggregate provenance.
    evaluation_context_id = eval_config_hash

    task_rows: list[TaskRows] = []
    for instance in instances:
        task = EnvTask.from_instance(env.name, instance)
        rows = tuple(
            _generation_row(
                env,
                candidate=candidate,
                instance=instance,
                provider_call_config=rd.provider_call_config,
                execution_policy=execution_policy,
                transport=transport,
                procedure_config_hash=procedure_hash,
                logical_call_id=f"{task.task_identity()}#{index}",
            )
            for index in range(repeats)
        )
        task_rows.append(
            TaskRows(
                task_identity=task.task_identity(),
                expected_repeats=repeats,
                rows=rows,
            )
        )

    rollout_aggregate = _env_exact_match_aggregate(
        graph_hash=rd.graph_hash,
        eval_config_hash=eval_config_hash,
        evaluation_context_id=evaluation_context_id,
        task_rows=tuple(task_rows),
        repeat_count=repeats,
        policy=policy,
    )
    reward = reward_from_internal_aggregate(
        experiment.reward_policy,
        env_exact_match_value=rollout_aggregate.aggregation_output.value,
    )
    per_task_scores = tuple(_per_task_score(task) for task in task_rows)
    per_task_counts = tuple(_per_task_count(task) for task in task_rows)
    return InternalEvalResult(
        aggregate=rollout_aggregate,
        reward=reward,
        per_task_scores=per_task_scores,
        per_task_counts=per_task_counts,
    )


__all__ = [
    "InternalEvalResult",
    "run_internal_eval",
]
