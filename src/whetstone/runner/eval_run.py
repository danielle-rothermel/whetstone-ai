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

from dr_store import MemoryBackend, ObjectStore
from whetstone_envs.core import Instance

from whetstone.code_eval.aggregate import (
    CompletenessPolicy,
    RolloutAggregate,
    RowPolicy,
)
from whetstone.envs.factory import EnvExperiment
from whetstone.envs.internal_eval import RolloutOutput, run_internal_eval
from whetstone.execution.fanout import FanoutConfig
from whetstone.execution.partials import PartialLog
from whetstone.optimization.identity import TypedRef, typed_ref_for_record
from whetstone.optimization.mutation import MUTATION_FIELD
from whetstone.optimization.schema import Candidate
from whetstone.provider.driver import TransportCall
from whetstone.provider.policy import ProviderExecutionPolicy
from whetstone.runner.execution_mode import ExecutionMode

__all__ = [
    "AGGREGATE_ARTIFACT_SCHEMA",
    "SplitEvaluation",
    "evaluate_split",
    "internal_instances",
    "official_instances",
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

    ``per_task_scores`` is the aligned per-task mean 0/1 oracle score captured
    from the SAME evaluation pass that produced ``score`` (one entry per
    instance, in instance order). A paired bootstrap CI consumes these retained
    scores directly, so the CI reflects the exact calls behind the reported
    aggregate and no second drive of the split is ever made.
    """

    split_role: str
    candidate_id: str
    score: float | None
    aggregate: RolloutAggregate
    artifact_ref: TypedRef
    execution_mode: ExecutionMode
    task_count: int
    repeat_count: int
    per_task_scores: tuple[float, ...]
    per_task_counts: tuple[int, ...]
    #: Whether a rate-limit failure halved the shared effective concurrency.
    concurrency_halved: bool = False
    #: Whether the whole-phase wall deadline stopped dispatch (some units
    #: un-driven -> counted as missing rows).
    deadline_reached: bool = False
    #: Count of runner-level guard timeouts (belt-and-suspenders breaches).
    guard_timeouts: int = 0
    #: FULL model output text + score per DRIVEN row (additive logging for
    #: qualitative prompt->output analysis; empty on a resumed/restored pass).
    outputs: tuple[RolloutOutput, ...] = ()
    #: The ed1 SECOND objective reported alongside the primary ``score``: the
    #: Mean Compression Ratio (``None`` for QA envs, which have one objective).
    #: The Reward is derived from ``score`` (pass rate) ONLY; this is REPORTED,
    #: never rewarded (dual-objective / Pareto selection is a flagged
    #: follow-up).
    compression_score: float | None = None
    #: The aligned per-task Mean Compression Ratio (ed1 only), for the sidecar
    #: /
    #: dual-score reporting; empty for QA envs.
    per_task_compression: tuple[float | None, ...] = ()

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
    policy: RowPolicy | CompletenessPolicy = RowPolicy.PROPAGATE,
    fanout: FanoutConfig | None = None,
    partial_log: PartialLog | None = None,
    apply_reward: bool | None = None,
    render_guard: bool = False,
) -> SplitEvaluation:
    """Evaluate ``candidate`` over ``instances`` and persist the aggregate.

    Drives the stage-03 attempt loop through the injected ``transport`` (the
    same pure driver the durable executor wraps), reduces the per-task means to
    one ``env_exact_match`` Rollout Aggregate, then ``put``\\s that aggregate's
    content into the Result Store so the artifact is identical whether the
    caller ran in-process or under the DBOS orchestration path.

    The observations fan out through the bounded worker pool (``fanout``,
    default 5-way); a ``partial_log`` makes the drive resumable (an
    already-recorded ``(instance, candidate, repeat)`` observation is restored,
    and each new call is appended as it completes).

    Reward is derived ONLY on an internal-role evaluation that actually needs
    it. An OFFICIAL-role split (``split_role == "official"``) computes the
    aggregate + per-task vectors and derives NO Reward (the design's "official
    evaluation MUST derive no Reward"). An internal-role split derives a Reward
    by default, EXCEPT when the caller passes ``apply_reward=False`` -- the
    identity (eval) optimizer performs no search, so its internal-split
    measurement needs no Reward and must not crash on an incomplete internal
    aggregate under the FAIL policy. When ``apply_reward`` is ``None`` the
    default holds (reward iff the split is not official); an explicit bool
    overrides it. Either way, a timed-out observation that leaves an aggregate
    incomplete is then visible incompleteness in the aggregate/per-task rows,
    never a Reward-policy crash.

    ``render_guard`` is the belt-and-braces render guard for NON-canonical
    (proposed candidate) templates: when True, a render ``KeyError`` escaping
    the env probe surface fails THAT candidate's row as a typed failure rather
    than killing the cell. It defaults False so canonical naive/ceiling probe
    renders keep their loud template-drift crash. Intake validation already
    rejects bad candidate templates before eval; this guards any residual.
    """
    is_official = split_role == "official"
    apply_reward_resolved = (
        (not is_official) if apply_reward is None else apply_reward
    )
    # ed1 (enc-dec HumanEval) dispatch: a distinct 3-node
    # encoder->decoder->code
    # eval drive producing DUAL scores. The QA path below is untouched (byte-
    # identical) -- ed1 is a separate, self-contained branch.
    from whetstone.envs.ed1 import Ed1Experiment

    if isinstance(experiment, Ed1Experiment):
        return _evaluate_ed1_split(
            experiment,
            candidate=candidate,
            instances=instances,
            split_role=split_role,
            transport=transport,
            execution_policy=execution_policy,
            repeats=repeats,
            store=store,
            execution_mode=execution_mode,
            policy=policy,
            fanout=fanout,
            apply_reward=apply_reward_resolved,
        )
    result = run_internal_eval(
        experiment,
        candidate=candidate,
        instances=instances,
        execution_policy=execution_policy,
        transport=transport,
        repeats=repeats,
        policy=policy,
        fanout=fanout,
        partial_log=partial_log,
        partial_phase="cell",
        apply_reward=apply_reward_resolved,
        render_guard=render_guard,
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
        per_task_scores=result.per_task_scores,
        per_task_counts=result.per_task_counts,
        concurrency_halved=result.concurrency_halved,
        deadline_reached=result.deadline_reached,
        guard_timeouts=result.guard_timeouts,
        outputs=result.outputs,
    )


#: The Result Store schema for a persisted ed1 dual-aggregate artifact.
ED1_AGGREGATE_ARTIFACT_SCHEMA = "whetstone.runner.ed1_split_aggregate"


def _evaluate_ed1_split(
    experiment: Any,
    *,
    candidate: Candidate,
    instances: tuple[Instance, ...],
    split_role: str,
    transport: TransportCall,
    execution_policy: ProviderExecutionPolicy,
    repeats: int,
    store: ObjectStore | None,
    execution_mode: ExecutionMode,
    policy: RowPolicy | CompletenessPolicy,
    fanout: FanoutConfig | None,
    apply_reward: bool,
) -> SplitEvaluation:
    """Evaluate one ed1 candidate over a split via the enc-dec DUAL drive.

    Runs :func:`whetstone.envs.ed1_eval.run_ed1_eval` (encoder -> decoder ->
    code-eval), persists a DUAL aggregate artifact (pass rate + Mean
    Compression
    Ratio), and returns a :class:`SplitEvaluation` whose ``score`` is the pass
    rate (the reward-bearing metric) and whose ``compression_score`` /
    ``per_task_compression`` carry the reported compression, plus per-row
    outputs
    (encoder + decoder text) for the dual-score sidecar.
    """
    from whetstone.envs.ed1_eval import run_ed1_eval

    template = str(candidate.payload.get(MUTATION_FIELD, ""))
    ed = run_ed1_eval(
        experiment,
        candidate_template=template,
        candidate_id=candidate.candidate_id,
        instances=instances,
        execution_policy=execution_policy,
        transport=transport,
        scorer=experiment.scorer,
        repeats=repeats,
        policy=policy,
        fanout=fanout,
        apply_reward=apply_reward,
        store=store,
    )
    pass_agg = ed.pass_aggregate
    comp_agg = ed.compression_aggregate
    backing = store or ObjectStore(MemoryBackend())
    artifact: dict[str, Any] = {
        "schema": ED1_AGGREGATE_ARTIFACT_SCHEMA,
        "split_role": split_role,
        "candidate_id": candidate.candidate_id,
        "graph_hash": pass_agg.graph_hash,
        "eval_config_hash": pass_agg.eval_config_hash,
        "pass_rate": pass_agg.aggregation_output.value,
        "mean_compression_ratio": comp_agg.aggregation_output.value,
        "task_count": pass_agg.task_count,
        "repeat_count": pass_agg.repeat_count,
        "rows_present": pass_agg.rows_present,
        "rows_failed": pass_agg.rows_failed,
        "rows_invalid": pass_agg.rows_invalid,
        "execution_mode": execution_mode.value,
    }
    backing.put(ED1_AGGREGATE_ARTIFACT_SCHEMA, artifact)
    artifact_ref = typed_ref_for_record(
        ED1_AGGREGATE_ARTIFACT_SCHEMA, artifact
    )
    return SplitEvaluation(
        split_role=split_role,
        candidate_id=candidate.candidate_id,
        score=pass_agg.aggregation_output.value,
        aggregate=pass_agg,
        artifact_ref=artifact_ref,
        execution_mode=execution_mode,
        task_count=pass_agg.task_count,
        repeat_count=pass_agg.repeat_count,
        per_task_scores=ed.per_task_scores,
        per_task_counts=ed.per_task_counts,
        outputs=ed.outputs,
        compression_score=comp_agg.aggregation_output.value,
        per_task_compression=ed.per_task_compression,
    )
