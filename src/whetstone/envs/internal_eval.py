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

from collections.abc import Callable
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
from whetstone.execution.call_support import (
    guard_deadline_seconds,
    is_rate_limit_failure,
)
from whetstone.execution.fanout import (
    RUNNER_TIMEOUT_CODE,
    CallSpec,
    FanoutConfig,
    run_call_pool,
)
from whetstone.execution.partials import PartialCallRecord, PartialLog
from whetstone.optimization.reward import Reward
from whetstone.optimization.schema import Candidate
from whetstone.provider.attempt import ProviderCallResult
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

    ``concurrency_halved`` records whether a rate-limit failure halved the
    shared effective concurrency during this pass; ``deadline_reached`` records
    whether the whole-phase wall deadline stopped dispatch (leaving some units
    un-driven, counted as missing rows).
    """

    aggregate: RolloutAggregate
    reward: Reward
    per_task_scores: tuple[float, ...]
    per_task_counts: tuple[int, ...]
    concurrency_halved: bool = False
    deadline_reached: bool = False
    guard_timeouts: int = 0


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


@dataclass(frozen=True, slots=True)
class _RowOutcome:
    """One repeat's result plus its terminal Provider Call Result.

    ``row`` is the aggregate contribution; ``result`` is the terminal call
    Result (``None`` when this observation was RESTORED from a partial log, not
    re-driven -- a resumed cell never re-calls a recorded observation).
    ``score`` is the reconstructed 0/1 (``None`` on a failed row), used to
    append the partial record.
    """

    row: RowValue
    result: ProviderCallResult | None
    score: float | None
    failure_code: str = ""


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
) -> _RowOutcome:
    """Run one repeat: render, call the transport, score via the env oracle."""
    from whetstone.execution.call_support import failure_code_of

    prompt = render_prompt(env, candidate, instance)
    result = run_provider_call(
        request=_request(provider_call_config, prompt),
        policy=execution_policy,
        transport=transport,
        logical_call_id=logical_call_id,
    )
    if not result.succeeded or result.generation is None:
        return _RowOutcome(
            row=RowValue(failed=True),
            result=result,
            score=None,
            failure_code=failure_code_of(result),
        )
    score = env_exact_match_score(
        env=env,
        generation=result.generation.text,
        gold=instance.gold,
        evaluation_procedure_config_hash=procedure_config_hash,
    )
    return _RowOutcome(
        row=RowValue(value=float(score.value)),
        result=result,
        score=float(score.value),
    )


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
    fanout: FanoutConfig | None = None,
    partial_log: PartialLog | None = None,
    partial_phase: str = "cell",
) -> InternalEvalResult:
    """Evaluate ``candidate`` over ``instances`` (the internal split).

    For each instance, ``repeats`` deliberate observations run through the
    injected transport; each accepted Generation is scored 0/1 by the env
    oracle. The per-task means reduce to a single internal ``env_exact_match``
    Rollout Aggregate, and the Reward Policy maps that aggregate's value to an
    internal-role Reward. No live paid call is made: the transport is injected.

    The observations fan out through a bounded worker pool (``fanout``): at
    most ``concurrency`` calls run at once, each under a runner-level
    wall-clock guard, and the RECORDED per-task rows are assembled by their
    ``(candidate, instance, repeat)`` key in instance/repeat order -- so the
    aggregate is byte-identical regardless of completion order. When a
    ``partial_log`` is given, each completed call is appended as it finishes
    and any already-recorded ``(instance, candidate, repeat)`` observation is
    RESTORED from disk instead of re-driven (cell resume).
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
    fanout = fanout or FanoutConfig()
    unit = candidate.candidate_id

    recorded = _restore_recorded(
        partial_log, partial_phase, unit, env, procedure_hash
    )

    # Build one keyed CallSpec per (instance, repeat) NOT already on disk.
    tasks = [
        (instance, EnvTask.from_instance(env.name, instance))
        for instance in instances
    ]
    specs: list[CallSpec[tuple[str, str, int], _RowOutcome]] = []
    for instance, task in tasks:
        for index in range(repeats):
            key = (unit, str(instance.id), index)
            if key in recorded:
                continue
            specs.append(
                CallSpec(
                    key=key,
                    run=_row_thunk(
                        env,
                        candidate=candidate,
                        instance=instance,
                        provider_call_config=rd.provider_call_config,
                        execution_policy=execution_policy,
                        transport=transport,
                        procedure_config_hash=procedure_hash,
                        logical_call_id=f"{task.task_identity()}#{index}",
                        # The thunk persists its OWN partial record the instant
                        # the call completes, so a crash mid-drive keeps every
                        # already-finished call durably on disk (incremental
                        # persistence, not a post-hoc batch).
                        partial_log=partial_log,
                        partial_phase=partial_phase,
                        partial_instance_id=str(instance.id),
                        partial_unit=unit,
                        repeat_id=index,
                    ),
                    deadline_seconds=guard_deadline_seconds(execution_policy),
                )
            )

    outcome = run_call_pool(
        specs,
        concurrency=fanout.concurrency,
        is_rate_limited=_row_is_rate_limited,
        max_wall_seconds=fanout.max_wall_seconds,
    )

    driven: dict[tuple[str, str, int], _RowOutcome] = {}
    for res in outcome.results:
        if res.timed_out:
            driven[res.key] = _RowOutcome(
                row=RowValue(failed=True),
                result=None,
                score=None,
                failure_code=RUNNER_TIMEOUT_CODE,
            )
            # A guard timeout is a real (failed) observation: record it so a
            # resume does not re-drive the call that already blew the deadline.
            if partial_log is not None:
                _u, instance_id, index = res.key
                partial_log.append(
                    PartialCallRecord(
                        phase=partial_phase, instance_id=instance_id,
                        unit=unit, repeat_id=index, score=None,
                        failed=True, failure_code=RUNNER_TIMEOUT_CODE,
                    )
                )
        elif res.not_dispatched:
            # The whole-phase deadline stopped dispatch before this call: the
            # planned row is absent (missing), never a fabricated failure, and
            # nothing is recorded (a resume re-drives it).
            driven[res.key] = _RowOutcome(
                row=RowValue(missing=True), result=None, score=None
            )
        elif res.value is not None:
            driven[res.key] = res.value

    # Assemble per-task rows in instance/repeat order (restored + driven).
    task_rows: list[TaskRows] = []
    for instance, task in tasks:
        rows: list[RowValue] = []
        for index in range(repeats):
            key = (unit, str(instance.id), index)
            if key in recorded:
                rows.append(recorded[key])
            else:
                rows.append(driven[key].row)
        task_rows.append(
            TaskRows(
                task_identity=task.task_identity(),
                expected_repeats=repeats,
                rows=tuple(rows),
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
        concurrency_halved=outcome.concurrency_halved,
        deadline_reached=outcome.deadline_reached,
        guard_timeouts=outcome.guard_timeouts,
    )


def _row_thunk(
    env: EnvSpec,
    *,
    candidate: Candidate,
    instance: Instance,
    provider_call_config: ProviderCallConfig,
    execution_policy: ProviderExecutionPolicy,
    transport: TransportCall,
    procedure_config_hash: str,
    logical_call_id: str,
    partial_log: PartialLog | None,
    partial_phase: str,
    partial_instance_id: str,
    partial_unit: str,
    repeat_id: int,
) -> Callable[[], _RowOutcome]:
    """A zero-arg thunk running one repeat (the fan-out unit of work).

    On completion it appends its own :class:`PartialCallRecord` to the partial
    log (when one is given) BEFORE returning, so the observation is durable the
    instant the call finishes -- a crash between here and the phase's assembly
    keeps it.
    """

    def _run() -> _RowOutcome:
        outcome = _generation_row(
            env,
            candidate=candidate,
            instance=instance,
            provider_call_config=provider_call_config,
            execution_policy=execution_policy,
            transport=transport,
            procedure_config_hash=procedure_config_hash,
            logical_call_id=logical_call_id,
        )
        if partial_log is not None:
            partial_log.append(
                PartialCallRecord(
                    phase=partial_phase,
                    instance_id=partial_instance_id,
                    unit=partial_unit,
                    repeat_id=repeat_id,
                    score=outcome.score,
                    failed=outcome.row.failed,
                    failure_code=outcome.failure_code,
                )
            )
        return outcome

    return _run


def _row_is_rate_limited(outcome: _RowOutcome) -> bool:
    """Whether a driven row's terminal Result is a rate-limit failure."""
    return outcome.result is not None and is_rate_limit_failure(
        outcome.result
    )


def _restore_recorded(
    partial_log: PartialLog | None,
    phase: str,
    unit: str,
    env: EnvSpec,
    procedure_hash: str,
) -> dict[tuple[str, str, int], RowValue]:
    """Rebuild RowValues for observations already on disk (resume skip).

    A recorded failed observation restores a failed row; a recorded score
    restores a value row. Only records for THIS phase+unit are restored, keyed
    ``(unit, instance_id, repeat)`` to match the driven-call keys.
    """
    if partial_log is None:
        return {}
    restored: dict[tuple[str, str, int], RowValue] = {}
    for record in partial_log.load():
        if record.phase != phase or record.unit != unit:
            continue
        key = (unit, record.instance_id, record.repeat_id)
        if record.failed or record.score is None:
            restored[key] = RowValue(failed=True)
        else:
            restored[key] = RowValue(value=float(record.score))
    return restored


__all__ = [
    "InternalEvalResult",
    "run_internal_eval",
]
