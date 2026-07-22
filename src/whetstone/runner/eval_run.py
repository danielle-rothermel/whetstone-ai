"""The shared split-evaluation entry the runner uses for every measurement.

Every measurement the validation runner needs -- a pilot probe pass, a
baseline/ceiling/best official evaluation, or an optimizer internal-split
resolution -- is one candidate driven over one split's instances through the
stage-03 attempt driver, scored 0/1 by the env oracle, reduced to one
provenance-bearing ``env_exact_match`` Rollout Aggregate.

This module does NOT duplicate that loop: it wraps the already-built
:func:`whetstone.envs.internal_eval.run_internal_eval` (the transport-injected
loop that renders, calls the driver, scores, and reduces) and adds Result Store
persistence of the resulting aggregate so the same artifact lands whether the
driver ran in-process or (in the DBOS path) inside the durable executor.

The transport is injected -- a scripted fake in tests, a real dr-providers
``HttpProvider.invoke`` in a live run -- so nothing here makes a live paid call
by itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dr_providers import (
    MessageRole,
    PromptMessage,
    ProviderCallRequest,
    Transcript,
)
from dr_store import MemoryBackend, ObjectStore
from whetstone_envs.core import Instance

from whetstone.code_eval.aggregate import RolloutAggregate, RowPolicy
from whetstone.envs.factory import EnvExperiment
from whetstone.envs.internal_eval import run_internal_eval
from whetstone.envs.oracle_operator import env_exact_match_score
from whetstone.envs.registry import env_spec
from whetstone.envs.rollout_definition import render_prompt
from whetstone.envs.task import EnvTask
from whetstone.optimization.identity import TypedRef, typed_ref_for_record
from whetstone.optimization.schema import Candidate
from whetstone.provider.driver import TransportCall, run_provider_call
from whetstone.provider.policy import ProviderExecutionPolicy
from whetstone.runner.execution_mode import ExecutionMode

__all__ = [
    "AGGREGATE_ARTIFACT_SCHEMA",
    "SplitEvaluation",
    "evaluate_split",
    "internal_instances",
    "official_instances",
    "per_task_official_scores",
]

#: The Result Store schema for a persisted split-evaluation aggregate artifact.
AGGREGATE_ARTIFACT_SCHEMA = "whetstone.runner.split_aggregate"


@dataclass(frozen=True, slots=True)
class SplitEvaluation:
    """One candidate's evaluation over one split.

    ``score`` is the reduced two-stage mean of the ``env_exact_match``
    aggregate (``None`` when the aggregate is incomplete under PROPAGATE).
    ``aggregate`` is the provenance-bearing Rollout Aggregate; ``artifact_ref``
    is its persisted Result Store reference (identical content across execution
    modes). ``execution_mode`` records which path produced it.
    """

    split_role: str
    candidate_id: str
    score: float | None
    aggregate: RolloutAggregate
    artifact_ref: TypedRef
    execution_mode: ExecutionMode
    task_count: int
    repeat_count: int

    @property
    def is_complete(self) -> bool:
        return self.score is not None


def internal_instances(experiment: EnvExperiment) -> tuple[Instance, ...]:
    """The internal-split instances (optimizer feedback)."""
    return experiment.eval_configs.internal.instances


def official_instances(experiment: EnvExperiment) -> tuple[Instance, ...]:
    """The official-split instances (before/after comparison)."""
    return experiment.eval_configs.official.instances


def evaluate_split(
    experiment: EnvExperiment,
    *,
    candidate: Candidate,
    instances: tuple[Instance, ...],
    split_role: str,
    transport: TransportCall,
    execution_policy: ProviderExecutionPolicy,
    repeats: int,
    store: ObjectStore | None = None,
    execution_mode: ExecutionMode = ExecutionMode.IN_PROCESS,
    policy: RowPolicy = RowPolicy.PROPAGATE,
) -> SplitEvaluation:
    """Evaluate ``candidate`` over ``instances`` and persist the aggregate.

    Drives the stage-03 attempt loop through the injected ``transport`` (the
    same pure driver the durable executor wraps), reduces the per-task means to
    one ``env_exact_match`` Rollout Aggregate, then ``put``\\s that aggregate's
    content into the Result Store so the artifact is identical whether the
    caller ran in-process or under the DBOS orchestration path.
    """
    result = run_internal_eval(
        experiment,
        candidate=candidate,
        instances=instances,
        execution_policy=execution_policy,
        transport=transport,
        repeats=repeats,
        policy=policy,
    )
    aggregate = result.aggregate
    backing = store or ObjectStore(MemoryBackend())
    artifact: dict[str, Any] = {
        "schema": AGGREGATE_ARTIFACT_SCHEMA,
        "split_role": split_role,
        "candidate_id": candidate.candidate_id,
        "graph_hash": aggregate.graph_hash,
        "eval_config_hash": aggregate.eval_config_hash,
        "evaluation_context_id": aggregate.evaluation_context_id,
        "score": aggregate.aggregation_output.value,
        "task_count": aggregate.task_count,
        "repeat_count": aggregate.repeat_count,
        "rows_present": aggregate.rows_present,
        "rows_missing": aggregate.rows_missing,
        "rows_failed": aggregate.rows_failed,
        "rows_invalid": aggregate.rows_invalid,
        "execution_mode": execution_mode.value,
    }
    backing.put(AGGREGATE_ARTIFACT_SCHEMA, artifact)
    artifact_ref = typed_ref_for_record(AGGREGATE_ARTIFACT_SCHEMA, artifact)
    return SplitEvaluation(
        split_role=split_role,
        candidate_id=candidate.candidate_id,
        score=aggregate.aggregation_output.value,
        aggregate=aggregate,
        artifact_ref=artifact_ref,
        execution_mode=execution_mode,
        task_count=aggregate.task_count,
        repeat_count=aggregate.repeat_count,
    )


def per_task_official_scores(
    experiment: EnvExperiment,
    *,
    candidate: Candidate,
    instances: tuple[Instance, ...],
    transport: TransportCall,
    execution_policy: ProviderExecutionPolicy,
    repeats: int,
) -> tuple[float, ...]:
    """Per-task official mean scores for the paired bootstrap CI.

    Returns one score per instance -- the mean 0/1 oracle score over the task's
    ``repeats`` repeats (a failed provider call contributes 0 to the mean so a
    task always yields a comparable number). These aligned per-task vectors
    feed :func:`whetstone.runner.statistics.bootstrap_delta_ci`. Uses the same
    render + stage-03 driver + oracle path as :func:`evaluate_split`; the
    transport is injected (no live paid call by itself).
    """
    env = env_spec(experiment.env_name)
    rd = experiment.rollout_definition
    procedure_hash = experiment.eval_configs.procedure_config_hash
    scores: list[float] = []
    for instance in instances:
        task = EnvTask.from_instance(env.name, instance)
        prompt = render_prompt(env, candidate, instance)
        repeat_scores: list[float] = []
        for index in range(repeats):
            result = run_provider_call(
                request=ProviderCallRequest(
                    config=rd.provider_call_config,
                    transcript=Transcript(
                        messages=(
                            PromptMessage(
                                role=MessageRole.USER, content=prompt
                            ),
                        )
                    ),
                ),
                policy=execution_policy,
                transport=transport,
                logical_call_id=f"official::{task.task_identity()}::{index}",
            )
            if not result.succeeded or result.generation is None:
                repeat_scores.append(0.0)
                continue
            score = env_exact_match_score(
                env=env,
                generation=result.generation.text,
                gold=instance.gold,
                evaluation_procedure_config_hash=procedure_hash,
            )
            repeat_scores.append(float(score.value))
        scores.append(
            sum(repeat_scores) / len(repeat_scores) if repeat_scores else 0.0
        )
    return tuple(scores)
