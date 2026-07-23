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
    CompletenessPolicy,
    RolloutAggregate,
    RowPolicy,
    RowValue,
    TaskRows,
    aggregation_definition,
    as_completeness_policy,
    enforce_skip_tolerance,
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
    is_transient_transport_failure,
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

#: The typed failure code for a candidate row whose NON-canonical template
#: raised a render ``KeyError`` (an untrusted placeholder the render could not
#: fill). Belt-and-braces: intake validation rejects such templates before
#: eval, but if one still reaches render under the guarded (candidate) path it
#: fails THAT row as a typed failure instead of killing the cell. Canonical
#: naive/ceiling probe renders are NOT guarded and keep their loud crash.
RENDER_FAILURE_CODE = "render_key_error"


@dataclass(frozen=True, slots=True)
class RolloutOutput:
    """One driven rollout row's FULL model output + extracted score.

    Captured for qualitative prompt->output analysis: the candidate that was
    evaluated, the task instance + repeat index, the FULL untruncated model
    output text, and the 0/1 oracle score (``None`` on a failed/missing row,
    with the failure code). Restored (resumed) rows carry no fresh output text
    (``output_text=None``) since they were not re-driven.
    """

    candidate_id: str
    instance_id: str
    repeat: int
    output_text: str | None
    score: float | None
    failure_code: str = ""


@dataclass(frozen=True, slots=True)
class InternalEvalResult:
    """One candidate's evaluation outcome over a split.

    Carries the provenance-bearing ``env_exact_match`` Rollout Aggregate. On an
    internal/optimizer pass (``apply_reward=True``) it ALSO carries ``reward``,
    the internal-role Reward the Reward Policy maps the aggregate to (which
    refuses any official evidence). On an OFFICIAL pass
    (``apply_reward=False``) ``reward`` is ``None``: an official evaluation
    MUST derive no Reward -- it
    computes the aggregate + per-task vectors only, per the design vocabulary.

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
    reward: Reward | None
    per_task_scores: tuple[float, ...]
    per_task_counts: tuple[int, ...]
    concurrency_halved: bool = False
    deadline_reached: bool = False
    guard_timeouts: int = 0
    #: FULL model output text + score for every DRIVEN row this pass (in
    #: instance/repeat order). Additive logging for qualitative analysis;
    #: restored (resumed) rows are omitted (not re-driven).
    outputs: tuple[RolloutOutput, ...] = ()


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


def _mean_aggregation_config(policy: CompletenessPolicy) -> AggregationConfig:
    """A ``mean`` Aggregation Config with the declared completeness policy.

    Folds in the ``missing_data`` rule AND the identity-bearing bounded skip
    tolerance (``max_skip_fraction``) so a tolerant config has a distinct
    identity from an untolerant one. Kept local (public dr-code APIs only) so
    the internal-eval loop owns its completeness policy.
    """
    return aggregation_definition(
        "whetstone.env.internal_eval.aggregation"
    ).materialize(
        {
            "reduction": "mean",
            "missing_data": policy.missing_data,
            "zero_denominator": "not_applicable",
            "max_skip_fraction": policy.skip_fraction_token(),
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
    render_guard: bool = False,
) -> _RowOutcome:
    """Run one repeat: render, call the transport, score via the env oracle.

    When ``render_guard`` is True (a NON-canonical candidate template), a
    render ``KeyError`` from the env probe surface fails THIS row as a typed
    :data:`RENDER_FAILURE_CODE` failure -- never a cell-killing crash -- and no
    provider call is made. When False (canonical naive/ceiling probe), a
    render ``KeyError`` propagates loudly as the designed template-drift guard.
    """
    from whetstone.execution.call_support import failure_code_of

    if render_guard:
        try:
            prompt = render_prompt(env, candidate, instance)
        except KeyError:
            return _RowOutcome(
                row=RowValue(failed=True),
                result=None,
                score=None,
                failure_code=RENDER_FAILURE_CODE,
            )
    else:
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
    policy: RowPolicy | CompletenessPolicy,
) -> RolloutAggregate:
    """The ``env_exact_match`` internal Rollout Aggregate (two-stage mean)."""
    policy = as_completeness_policy(policy)
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
    output = enforce_skip_tolerance(
        output,
        policy=policy,
        skipped=missing + failed + invalid,
        planned=len(all_rows),
    )
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
    policy: RowPolicy | CompletenessPolicy = RowPolicy.PROPAGATE,
    fanout: FanoutConfig | None = None,
    partial_log: PartialLog | None = None,
    partial_phase: str = "cell",
    apply_reward: bool = True,
    render_guard: bool = False,
) -> InternalEvalResult:
    """Evaluate ``candidate`` over ``instances`` (internal or official split).

    For each instance, ``repeats`` deliberate observations run through the
    injected transport; each accepted Generation is scored 0/1 by the env
    oracle. The per-task means reduce to a single ``env_exact_match`` Rollout
    Aggregate. No live paid call is made: the transport is injected.

    **Reward application is caller-controlled.** When ``apply_reward`` is True
    (the internal/optimizer path, the default) the Reward Policy maps the
    aggregate value to an internal-role Reward; a missing aggregate under the
    FAIL missing-data policy surfaces as a typed ``CandidateEvaluationFailure``
    the optimizer loop handles (candidate marked failed), never a bare
    ``ValueError``. When ``apply_reward`` is False (the OFFICIAL path) NO
    Reward is derived: the result carries the aggregate + per-task vectors, so
    an
    incomplete official aggregate (timed-out observations) is visible
    incompleteness, never a process crash.

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

    def _spec(
        instance: Instance, task: EnvTask, index: int
    ) -> CallSpec[tuple[str, str, int], _RowOutcome]:
        key = (unit, str(instance.id), index)
        return CallSpec(
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
                # The thunk persists its OWN partial record the instant the
                # call completes, so a crash mid-drive keeps every finished
                # call durably on disk (incremental persistence, not a post-hoc
                # batch). Both the first drive AND a re-drive append a record.
                partial_log=partial_log,
                partial_phase=partial_phase,
                partial_instance_id=str(instance.id),
                partial_unit=unit,
                repeat_id=index,
                render_guard=render_guard,
            ),
            deadline_seconds=guard_deadline_seconds(execution_policy),
        )

    by_instance = {str(inst.id): (inst, tsk) for inst, tsk in tasks}
    specs = [
        _spec(instance, task, index)
        for instance, task in tasks
        for index in range(repeats)
        if (unit, str(instance.id), index) not in recorded
    ]

    def _drive(
        pending: list[CallSpec[tuple[str, str, int], _RowOutcome]],
    ) -> tuple[dict[tuple[str, str, int], _RowOutcome], bool, bool, int]:
        pool = run_call_pool(
            pending,
            concurrency=fanout.concurrency,
            is_rate_limited=_row_is_rate_limited,
            max_wall_seconds=fanout.max_wall_seconds,
        )
        driven: dict[tuple[str, str, int], _RowOutcome] = {}
        for res in pool.results:
            if res.timed_out:
                driven[res.key] = _RowOutcome(
                    row=RowValue(failed=True), result=None, score=None,
                    failure_code=RUNNER_TIMEOUT_CODE,
                )
            elif res.not_dispatched:
                # The whole-phase deadline stopped dispatch before this call:
                # the planned row is absent (missing), never a fabricated
                # failure, and nothing is recorded (a resume re-drives it).
                driven[res.key] = _RowOutcome(
                    row=RowValue(missing=True), result=None, score=None
                )
            elif res.value is not None:
                driven[res.key] = res.value
        return (
            driven,
            pool.concurrency_halved,
            pool.deadline_reached,
            pool.guard_timeouts,
        )

    driven, halved_1, deadline_1, guard_1 = _drive(specs)

    # --- ONE bounded re-drive of timed-out / transient-transport failures. ---
    # A runner-guard timeout or a TERMINAL transient transport failure (the
    # driver's own semantic retries were exhausted) is re-driven exactly once
    # through the same semantic-retry path before it lands as a failed row.
    # Both attempts are recorded in the partial log (the re-drive thunk appends
    # its own record); a re-drive that still fails/times-out lands as failed
    # row. A not-dispatched (deadline) row is NOT re-driven (a resume handles
    # it). This bounds a single flaky observation without re-driving the split.
    redrive_specs = [
        _spec(*by_instance[key[1]], key[2])
        for key, out in driven.items()
        if _should_redrive(out)
    ]
    halved_2 = deadline_2 = False
    guard_2 = 0
    if redrive_specs:
        redriven, halved_2, deadline_2, guard_2 = _drive(redrive_specs)
        driven.update(redriven)

    # A first-attempt timeout that was NOT re-driven (or a re-drive that also
    # timed out) is a real failed observation: record it so a resume does not
    # re-drive a call that already blew the deadline twice.
    if partial_log is not None:
        for key, out in driven.items():
            if out.failure_code == RUNNER_TIMEOUT_CODE:
                partial_log.append(
                    PartialCallRecord(
                        phase=partial_phase, instance_id=key[1],
                        unit=unit, repeat_id=key[2], score=None,
                        failed=True, failure_code=RUNNER_TIMEOUT_CODE,
                    )
                )

    concurrency_halved = halved_1 or halved_2
    deadline_reached = deadline_1 or deadline_2
    guard_timeouts = guard_1 + guard_2

    # Assemble per-task rows in instance/repeat order (restored + driven), and
    # collect the FULL model output text of every DRIVEN row (additive logging
    # for qualitative prompt->output analysis; restored rows carry no fresh
    # text since they were not re-driven).
    task_rows: list[TaskRows] = []
    outputs: list[RolloutOutput] = []
    for instance, task in tasks:
        rows: list[RowValue] = []
        for index in range(repeats):
            key = (unit, str(instance.id), index)
            if key in recorded:
                rows.append(recorded[key])
            else:
                outcome = driven[key]
                rows.append(outcome.row)
                outputs.append(
                    RolloutOutput(
                        candidate_id=unit,
                        instance_id=str(instance.id),
                        repeat=index,
                        output_text=_output_text_of(outcome.result),
                        score=outcome.score,
                        failure_code=outcome.failure_code,
                    )
                )
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
        policy=as_completeness_policy(policy),
    )
    # Reward is caller-controlled: internal/optimizer passes derive it; an
    # official pass MUST derive no Reward (aggregate + per-task vectors only).
    reward = (
        reward_from_internal_aggregate(
            experiment.reward_policy,
            env_exact_match_value=rollout_aggregate.aggregation_output.value,
        )
        if apply_reward
        else None
    )
    per_task_scores = tuple(_per_task_score(task) for task in task_rows)
    per_task_counts = tuple(_per_task_count(task) for task in task_rows)
    return InternalEvalResult(
        aggregate=rollout_aggregate,
        reward=reward,
        per_task_scores=per_task_scores,
        per_task_counts=per_task_counts,
        concurrency_halved=concurrency_halved,
        deadline_reached=deadline_reached,
        guard_timeouts=guard_timeouts,
        outputs=tuple(outputs),
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
    render_guard: bool = False,
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
            render_guard=render_guard,
        )
        if partial_log is not None:
            from whetstone.execution.call_support import call_telemetry
            tel = call_telemetry(outcome.result)
            partial_log.append(
                PartialCallRecord(
                    phase=partial_phase,
                    instance_id=partial_instance_id,
                    unit=partial_unit,
                    repeat_id=repeat_id,
                    score=outcome.score,
                    failed=outcome.row.failed,
                    failure_code=outcome.failure_code,
                    # FIX 6: retain the measured token counts on the cell path
                    # (they were null before) so spend reconciliation can sum
                    # them. Task 20: also the reasoning tokens + per-call
                    # latency. The raw_response is intentionally NOT persisted
                    # on the cell path (Rollout Results hold that evidence)
                    # so a
                    # cell's partial stays small.
                    prompt_tokens=tel.prompt_tokens,
                    completion_tokens=tel.completion_tokens,
                    total_tokens=tel.total_tokens,
                    reasoning_tokens=tel.reasoning_tokens,
                    latency_s=tel.latency_s,
                )
            )
        return outcome

    return _run


def _output_text_of(result: ProviderCallResult | None) -> str | None:
    """The FULL (untruncated) model output text of a driven call, else None.

    Returns the accepted Generation's text for a succeeded call; ``None`` for a
    failed / restored / generation-less call. Never truncated -- the sidecar
    keeps whole streams (c23 outputs are long).
    """
    if result is None or not result.succeeded or result.generation is None:
        return None
    return result.generation.text


def _row_is_rate_limited(outcome: _RowOutcome) -> bool:
    """Whether a driven row's terminal Result is a rate-limit failure."""
    return outcome.result is not None and is_rate_limit_failure(
        outcome.result
    )


def _should_redrive(outcome: _RowOutcome) -> bool:
    """Whether a first-attempt outcome earns ONE bounded re-drive.

    A runner-guard timeout (``runner_timeout``) or a terminal transient
    transport failure (the driver's own semantic retries exhausted) is
    re-driven once through the normal semantic-retry path before landing as a
    failed row. A clean provider rejection / blank / malformed response is NOT
    re-driven (re-driving the same request will not change a deterministic
    "no"); a missing (not-dispatched) row is left for a resume.
    """
    if outcome.failure_code == RUNNER_TIMEOUT_CODE:
        return True
    return outcome.result is not None and is_transient_transport_failure(
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
    "RENDER_FAILURE_CODE",
    "InternalEvalResult",
    "RolloutOutput",
    "run_internal_eval",
]
