"""The d1 direct-generation code-eval drive (single LLM call, pass-rate).

Drives one candidate over a d1 split through the injected transport, running a
SINGLE LLM Call per (task, repeat):

1. compose the mutable wrapper ``{body}`` (the candidate's Mutation-Surface
   payload) around the FROZEN input arm (the screen DIRECT-arm slice of the
   canonical HumanEval prompt; the ``renamed`` arm is the all-occurrence
   canonical-name scrub);
2. call the task model directly;
3. score the model output for correctness through the SAME dr-code HumanEval
   sandbox ed1 uses (the ``renamed`` arm scores against the RENAMED entry
   point -- the amendment-2 scoring trap, never the leaked canonical name).

It reduces to one aggregate -- the Average Binary Test Pass Rate -- using the
same two-stage mean as the QA and ED1 paths. The frozen input-arm construction
is owned by :mod:`whetstone.envs.input_transform`. Nothing here makes a live
paid call by itself: the transport and code-eval scorer are injected.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from dr_code.humaneval import HumanEvalTask
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
    RowValue,
)
from whetstone.envs.d1 import (
    D1_PASS_RATE_NAME,
    D1Experiment,
    render_d1_frame,
)
from whetstone.envs.ed1 import (
    ed1_reward_from_pass_rate,
    validate_ed1_body,
)
from whetstone.envs.ed1_eval import _aggregate_metric
from whetstone.envs.ed1_scoring import CodeScore, score_ed1_submission
from whetstone.envs.input_transform import (
    direct_body,
    renamed_task,
    split_prompt,
)
from whetstone.envs.internal_eval import RolloutOutput
from whetstone.envs.sampling import EnvSplitSampling
from whetstone.execution.call_support import (
    call_telemetry,
    failure_code_of,
    is_transient_transport_failure,
)
from whetstone.execution.fanout import CallSpec, FanoutConfig, run_call_pool
from whetstone.execution.partials import PartialCallRecord, PartialLog
from whetstone.execution.prompt_cache import (
    CallExecution,
    PartialCacheMarks,
    PromptResultCache,
    execute_call,
)
from whetstone.optimization.reward import Reward
from whetstone.provider.driver import TransportCall
from whetstone.provider.policy import ProviderExecutionPolicy


@dataclass(frozen=True, slots=True)
class D1EvalResult:
    """One candidate's d1 evaluation over a split (single pass-rate aggregate).

    ``pass_aggregate`` is the Average Binary Test Pass Rate (the plain Reward-
    bearing metric). ``reward`` is derived from it (when ``apply_reward``).
    Per-task vectors + outputs feed the CI / ledger / sidecar.
    """

    pass_aggregate: RolloutAggregate
    reward: Reward | None
    per_task_scores: tuple[float, ...]
    per_task_counts: tuple[int, ...]
    outputs: tuple[RolloutOutput, ...] = ()


@dataclass(frozen=True, slots=True)
class _D1RowOutcome:
    """One (task, repeat) direct rollout's result + provenance."""

    pass_value: float | None
    output_text: str | None
    failed: bool
    failure_code: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    latency_s: float | None = None
    #: Task-26 per-call provenance (``None`` when unknown): the provider stop
    #: reason of the accepted Generation + the FULL typed diagnostic of a
    #: failed call.
    finish_reason: str | None = None
    provider_error: dict[str, object] | None = None
    #: True when a TRANSIENT transport fault (timeout/stall/transport-error/
    #: rate-limit) exhausted its semantic retries -- eligible for ONE re-drive.
    redrivable: bool = False
    #: Task-31 prompt-cache execution marker (``None`` on a render-failed row
    #: with no call). Tells the persist step whether this row was served from
    #: cache (mark + null latency) or freshly driven.
    execution: CallExecution | None = None


def _request(config: ProviderCallConfig, prompt: str) -> ProviderCallRequest:
    return ProviderCallRequest(
        config=config,
        transcript=Transcript(
            messages=(PromptMessage(role=MessageRole.USER, content=prompt),)
        ),
    )


def _input_arm_text(
    experiment: D1Experiment, instance: Instance
) -> tuple[str, HumanEvalTask]:
    """The frozen input-arm text + the (possibly renamed) scoring task.

    REUSES the screen driver: ``split_prompt`` -> the arm slice via
    ``_direct_body``; the ``renamed`` arm additionally scrubs EVERY canonical-
    name occurrence and returns a RENAMED scoring task (the amendment-2 trap).
    """
    ht = experiment.humaneval_for(instance)
    parts = split_prompt(ht.prompt, ht.entry_point)
    arm = experiment.input_arm
    token = experiment.rename_token
    body = direct_body(f"direct_{arm}", parts, rename_token=token)
    score_task = (
        renamed_task(ht, old=ht.entry_point, new=token)
        if arm == "renamed"
        else ht
    )
    return body, score_task


def _drive_row(
    *,
    experiment: D1Experiment,
    candidate_body: str,
    instance: Instance,
    provider_call_config: ProviderCallConfig,
    execution_policy: ProviderExecutionPolicy,
    transport: TransportCall,
    scorer: Callable[..., CodeScore],
    logical_call_id: str,
    repeat_index: int,
    cache: PromptResultCache | None,
    cache_phase: str,
    cache_unit: str,
) -> _D1RowOutcome:
    """Run one direct generate->score rollout for one (task, repeat)."""
    input_arm, score_task = _input_arm_text(experiment, instance)
    try:
        prompt = render_d1_frame(candidate_body, input_arm=input_arm)
    except (KeyError, IndexError, ValueError):
        return _D1RowOutcome(
            pass_value=None,
            output_text=None,
            failed=True,
            failure_code="d1_wrapper_render_error",
        )
    execution = execute_call(
        request=_request(provider_call_config, prompt),
        policy=execution_policy,
        transport=transport,
        logical_call_id=logical_call_id,
        repeat_index=repeat_index,
        cache=cache,
        phase=cache_phase,
        unit=cache_unit,
    )
    result = execution.result
    if not result.succeeded or result.generation is None:
        return _D1RowOutcome(
            pass_value=None,
            output_text=None,
            failed=True,
            failure_code=failure_code_of(result),
            provider_error=call_telemetry(result).provider_error,
            redrivable=is_transient_transport_failure(result),
            execution=execution,
        )
    output_text = result.generation.text
    tel = call_telemetry(result)
    code_score = scorer(raw_submission=output_text, task=score_task)
    if code_score.infrastructure_unknown:
        return _D1RowOutcome(
            pass_value=None,
            output_text=output_text,
            failed=True,
            failure_code="code_eval_infrastructure_unknown",
            prompt_tokens=tel.prompt_tokens,
            completion_tokens=tel.completion_tokens,
            total_tokens=tel.total_tokens,
            reasoning_tokens=tel.reasoning_tokens,
            latency_s=tel.latency_s,
            finish_reason=tel.finish_reason,
            execution=execution,
        )
    return _D1RowOutcome(
        pass_value=code_score.row_value,
        output_text=output_text,
        failed=False,
        prompt_tokens=tel.prompt_tokens,
        completion_tokens=tel.completion_tokens,
        total_tokens=tel.total_tokens,
        reasoning_tokens=tel.reasoning_tokens,
        latency_s=tel.latency_s,
        finish_reason=tel.finish_reason,
        execution=execution,
    )


def _drive_and_persist(
    *,
    experiment: D1Experiment,
    candidate_body: str,
    candidate_id: str,
    instance: Instance,
    index: int,
    provider_call_config: ProviderCallConfig,
    execution_policy: ProviderExecutionPolicy,
    transport: TransportCall,
    scorer: Callable[..., CodeScore],
    partial_log: PartialLog | None,
    split_role: str,
    cache: PromptResultCache | None = None,
) -> _D1RowOutcome:
    """Drive one d1 row and append its partial record when it finishes."""
    outcome = _drive_row(
        experiment=experiment,
        candidate_body=candidate_body,
        instance=instance,
        provider_call_config=provider_call_config,
        execution_policy=execution_policy,
        transport=transport,
        scorer=scorer,
        logical_call_id=f"{candidate_id}:{instance.id}#{index}",
        repeat_index=index,
        cache=cache,
        cache_phase=split_role,
        cache_unit=candidate_id,
    )
    if partial_log is not None:
        # Task 31 honesty: null the latency of a cache-served row (no wire call
        # this time) and stamp the cache marker + original-entry provenance.
        marks = (
            outcome.execution.cache_marks()
            if outcome.execution is not None
            else PartialCacheMarks()
        )
        partial_log.append(
            PartialCallRecord(
                phase=split_role,
                instance_id=str(instance.id),
                unit=candidate_id,
                repeat_id=index,
                score=outcome.pass_value,
                failed=outcome.failed,
                failure_code=outcome.failure_code,
                split_role=split_role,
                prompt_tokens=outcome.prompt_tokens,
                completion_tokens=outcome.completion_tokens,
                total_tokens=outcome.total_tokens,
                reasoning_tokens=outcome.reasoning_tokens,
                latency_s=None if marks.cache_hit else outcome.latency_s,
                output_text=outcome.output_text,
                finish_reason=outcome.finish_reason,
                provider_error=outcome.provider_error,
                cache_hit=marks.cache_hit,
                cache_source_phase=marks.cache_source_phase,
                cache_source_unit=marks.cache_source_unit,
                cache_source_call_id=marks.cache_source_call_id,
                cache_source_at=marks.cache_source_at,
            )
        )
    return outcome


def _restore_recorded(
    partial_log: PartialLog | None,
    split_role: str,
    candidate_id: str,
) -> dict[tuple[str, int], _D1RowOutcome]:
    """Rebuild d1 row outcomes already durably recorded (resume skip)."""
    if partial_log is None:
        return {}
    restored: dict[tuple[str, int], _D1RowOutcome] = {}
    for record in partial_log.load():
        if record.phase != split_role or record.unit != candidate_id:
            continue
        restored[(record.instance_id, record.repeat_id)] = _D1RowOutcome(
            pass_value=record.score,
            output_text=None,
            failed=record.failed,
            failure_code=record.failure_code,
            prompt_tokens=record.prompt_tokens,
            completion_tokens=record.completion_tokens,
            total_tokens=record.total_tokens,
            reasoning_tokens=record.reasoning_tokens,
            latency_s=record.latency_s,
        )
    return restored


def _deadline(execution_policy: ProviderExecutionPolicy) -> float:
    from whetstone.execution.call_support import guard_deadline_seconds

    # A d1 row makes ONE wire call (direct generation), so the guard budgets a
    # single call's transport cap.
    return guard_deadline_seconds(execution_policy, wire_calls_per_unit=1)


def run_d1_eval(
    experiment: D1Experiment,
    *,
    candidate_body: str,
    candidate_id: str,
    sampling: EnvSplitSampling,
    execution_policy: ProviderExecutionPolicy,
    transport: TransportCall,
    scorer: Callable[..., CodeScore] | None = None,
    fanout: FanoutConfig | None = None,
    apply_reward: bool = True,
    partial_log: PartialLog | None = None,
    cache: PromptResultCache | None = None,
) -> D1EvalResult:
    """Drive ``candidate_body`` over a d1 split -> the pass-rate aggregate.

    Fans out one direct generate->score rollout per (task, repeat) through the
    injected transport + code scorer, reduces to the pass-rate aggregate,
    derives the plain pass-rate Reward (when ``apply_reward``), and collects
    per-row outputs. Incremental persistence + resume mirror the ed1 drive:
    each completed row appends its record when it finishes; a resumed drive
    restores already-recorded rows instead of re-paying.
    """
    validate_ed1_body(candidate_body)
    fanout = fanout or FanoutConfig()
    scorer = scorer or score_ed1_submission
    instances = sampling.instances
    repeats = sampling.repeat_plan.repeat_count
    split_role = sampling.split_role
    completeness = sampling.completeness_policy
    rd = experiment.rollout_definition
    graph_hash = rd.graph_hash
    if (
        sampling.eval_config.evaluation_procedure_config_hash
        != rd.procedure_config_hash
    ):
        raise ValueError(
            "sampling EvalConfig procedure does not match the experiment"
        )
    eval_config_hash = sampling.eval_config.config_identity_hash
    restored = _restore_recorded(partial_log, split_role, candidate_id)

    def _spec(
        instance: Instance, index: int
    ) -> CallSpec[tuple[str, int], _D1RowOutcome]:
        return CallSpec(
            key=(str(instance.id), index),
            run=lambda inst=instance, i=index: _drive_and_persist(
                experiment=experiment,
                candidate_body=candidate_body,
                candidate_id=candidate_id,
                instance=inst,
                index=i,
                provider_call_config=rd.provider_call_config,
                execution_policy=execution_policy,
                transport=transport,
                scorer=scorer,
                partial_log=partial_log,
                split_role=split_role,
                cache=cache,
            ),
            deadline_seconds=_deadline(execution_policy),
        )

    by_instance = {str(inst.id): inst for inst in instances}

    def _drive(
        pending: list[CallSpec[tuple[str, int], _D1RowOutcome]],
    ) -> dict[tuple[str, int], _D1RowOutcome]:
        pool = run_call_pool(
            pending,
            concurrency=fanout.concurrency,
            is_rate_limited=lambda _o: False,
            max_wall_seconds=fanout.max_wall_seconds,
        )
        out: dict[tuple[str, int], _D1RowOutcome] = {}
        for res in pool.results:
            if res.value is not None:
                out[res.key] = res.value
            else:
                out[res.key] = _D1RowOutcome(
                    pass_value=None,
                    output_text=None,
                    failed=True,
                    failure_code="runner_timeout",
                    redrivable=True,
                )
        return out

    specs = [
        _spec(instance, index)
        for instance in instances
        for index in range(repeats)
        if (str(instance.id), index) not in restored
    ]
    driven: dict[tuple[str, int], _D1RowOutcome] = dict(restored)
    driven.update(_drive(specs))

    # ONE bounded re-drive of timed-out / transient-transport rows (a single
    # flaky observation must not fail the whole d1 arm under FAIL policy).
    redrive_specs = [
        _spec(by_instance[key[0]], key[1])
        for key, out in driven.items()
        if out.redrivable
    ]
    if redrive_specs:
        driven.update(_drive(redrive_specs))

    pass_rows: list[tuple[str, list[RowValue]]] = []
    outputs: list[RolloutOutput] = []
    per_task_scores: list[float] = []
    per_task_counts: list[int] = []
    for instance in instances:
        task_id = str(instance.id)
        p_rows: list[RowValue] = []
        for index in range(repeats):
            outcome = driven[(task_id, index)]
            if outcome.failed or outcome.pass_value is None:
                p_rows.append(RowValue(failed=True))
            else:
                p_rows.append(RowValue(value=float(outcome.pass_value)))
            outputs.append(
                RolloutOutput(
                    candidate_id=candidate_id,
                    instance_id=task_id,
                    repeat=index,
                    output_text=outcome.output_text,
                    score=(
                        None
                        if outcome.pass_value is None
                        else float(outcome.pass_value)
                    ),
                    failure_code=outcome.failure_code,
                    finish_reason=outcome.finish_reason,
                    provider_error=outcome.provider_error,
                )
            )
        pass_rows.append((task_id, p_rows))
        # Per-task pass mean + planned-repeat weight (IDENTICAL to ed1/QA): an
        # absent/failed row counts 0, the weight is the planned repeat count.
        total = sum(
            float(r.value or 0.0) if r.is_present else 0.0 for r in p_rows
        )
        per_task_scores.append(total / len(p_rows) if p_rows else 0.0)
        per_task_counts.append(len(p_rows))

    pass_aggregate = _aggregate_metric(
        name=D1_PASS_RATE_NAME,
        graph_hash=graph_hash,
        eval_config_hash=eval_config_hash,
        per_task_rows=pass_rows,
        repeats=repeats,
        policy=completeness,
    )
    reward: Reward | None = None
    if apply_reward:
        reward = ed1_reward_from_pass_rate(
            experiment.reward_policy,
            pass_rate=pass_aggregate.aggregation_output.value,
        )
    return D1EvalResult(
        pass_aggregate=pass_aggregate,
        reward=reward,
        per_task_scores=tuple(per_task_scores),
        per_task_counts=tuple(per_task_counts),
        outputs=tuple(outputs),
    )


__all__ = [
    "D1EvalResult",
    "run_d1_eval",
]
