"""The process-isolated internal-eval loop over an env's internal split.

:func:`run_internal_eval` drives a candidate (the naive Initial Candidate in
the factory tests) through serializable row jobs over the internal split and
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

import hashlib
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

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
from pydantic import BaseModel, ConfigDict, JsonValue, model_validator
from whetstone_envs.core import Instance

from whetstone.code_eval.aggregate import (
    CompletenessPolicy,
    RolloutAggregate,
    RowValue,
    TaskRows,
    aggregation_definition,
    enforce_skip_tolerance,
)
from whetstone.envs.factory import EnvExperiment
from whetstone.envs.oracle_operator import (
    ENV_EXACT_MATCH_NAME,
    env_exact_match_score,
)
from whetstone.envs.registry import EnvSpec, env_spec
from whetstone.envs.reward import reward_from_internal_aggregate
from whetstone.envs.rollout_definition import (
    render_prompt,
    validate_candidate_prompt,
)
from whetstone.envs.sampling import EnvSplitSampling
from whetstone.envs.task import EnvTask
from whetstone.execution.call_support import (
    guard_deadline_seconds,
    is_rate_limit_failure,
    is_transient_transport_failure,
)
from whetstone.execution.fanout import (
    DEFAULT_CONCURRENCY,
    CallSpec,
    FanoutStatus,
    ProcessJob,
    run_call_pool,
)
from whetstone.execution.partials import PartialCallRecord, PartialLog
from whetstone.execution.prompt_cache import (
    PromptResultCache,
    execute_call,
)
from whetstone.optimization.reward import Reward
from whetstone.optimization.schema import Candidate
from whetstone.provider.driver import TransportCall
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
    #: Task-26 per-call provenance (coverage-honest -- ``None`` when unknown):
    #: ``finish_reason`` is the provider stop reason of the accepted Generation
    #: (a truncated ``length`` is distinguishable from a clean ``stop``);
    #: ``provider_error`` is the FULL typed provider-failure diagnostic for a
    #: FAILED row (not just the short ``failure_code``). Both ``None`` on a
    #: restored (resumed) row (not re-driven).
    finish_reason: str | None = None
    provider_error: dict[str, object] | None = None
    #: ed1 enc-dec budget diagnostics on EVERY rollout row (task 26 item 6):
    #: the per-task ``max_budget`` (chars) the encoder was told to respect and
    #: the derived ``over_budget`` flag, so a consumer never re-derives
    #: ``round(budget_ratio * len)`` off-row. ``None`` for QA/d1 (no budget)
    #: and for a no-budget ed1 frame.
    max_budget: int | None = None
    over_budget: bool | None = None


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


class InternalRowOutcome(BaseModel):
    """Serializable result of one internal-evaluation process job."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
        strict=True,
    )

    score: float | None
    failed: bool = False
    missing: bool = False
    failure_code: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    latency_s: float | None = None
    output_text: str | None = None
    finish_reason: str | None = None
    provider_error: dict[str, object] | None = None
    rate_limited: bool = False
    redrivable: bool = False
    cache_hit: bool = False
    cache_source_phase: str | None = None
    cache_source_unit: str | None = None
    cache_source_call_id: str | None = None
    cache_source_at: str | None = None

    @model_validator(mode="after")
    def _valid_row_state(self) -> InternalRowOutcome:
        if self.failed and self.missing:
            raise ValueError("a row cannot be both failed and missing")
        if (self.failed or self.missing) == (self.score is not None):
            raise ValueError(
                "a present row requires a score and an absent row forbids one"
            )
        if self.cache_hit != (self.cache_source_call_id is not None):
            raise ValueError(
                "cache_hit and original-entry provenance must be paired"
            )
        return self

    @property
    def row(self) -> RowValue:
        if self.failed:
            return RowValue(failed=True)
        if self.missing:
            return RowValue(missing=True)
        assert self.score is not None
        return RowValue(value=self.score)


_INTERNAL_ROW_REQUEST_SCHEMA = "whetstone.envs.internal_row_request/v1"
_INTERNAL_ROW_RESULT_SCHEMA = "whetstone.envs.internal_row_result/v1"


class ProcessInstance(BaseModel):
    """JSON-safe form of the frozen environment Instance value object."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    id: str
    seed: int
    strata: tuple[str, ...]
    prompt_inputs: dict[str, str]
    gold: str

    @classmethod
    def from_instance(cls, instance: Instance) -> ProcessInstance:
        return cls(
            id=str(instance.id),
            seed=instance.seed,
            strata=instance.strata,
            prompt_inputs=dict(instance.prompt_inputs),
            gold=instance.gold,
        )

    def to_instance(self) -> Instance:
        return Instance(
            id=self.id,
            seed=self.seed,
            strata=self.strata,
            prompt_inputs=self.prompt_inputs,
            gold=self.gold,
        )


def process_request_identity(model: BaseModel) -> str:
    """Hash one strict row request's canonical JSON representation."""
    payload = json.dumps(
        model.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def start_phase_deadline(max_wall_seconds: float | None) -> float | None:
    """Validate one phase wall and convert it to an absolute deadline."""
    if max_wall_seconds is None:
        return None
    if type(max_wall_seconds) not in (int, float):
        raise ValueError(
            "max_wall_seconds must be a finite nonnegative real number"
        )
    try:
        seconds = float(max_wall_seconds)
    except OverflowError:
        raise ValueError(
            "max_wall_seconds must be a finite nonnegative real number "
            "representable as seconds"
        ) from None
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(
            "max_wall_seconds must be a finite nonnegative real number"
        )
    return time.monotonic() + seconds


def remaining_phase_wall_seconds(deadline: float | None) -> float | None:
    """Return the nonnegative remainder of one shared phase wall."""
    if deadline is None:
        return None
    return max(0.0, deadline - time.monotonic())


class InternalRowRequest(BaseModel):
    """Complete serializable request and provenance for one internal row."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_name: Literal["whetstone.envs.internal_row_request/v1"] = (
        _INTERNAL_ROW_REQUEST_SCHEMA
    )
    env_name: str
    candidate: Candidate
    instance: ProcessInstance
    provider_call_config: ProviderCallConfig
    execution_policy: ProviderExecutionPolicy
    procedure_config_hash: str
    logical_call_id: str
    repeat_index: int
    drive_ordinal: int
    cache_phase: str
    cache_unit: str
    cache_root: str | None
    render_guard: bool

    @property
    def request_identity(self) -> str:
        return process_request_identity(self)

    @classmethod
    def from_process_payload(cls, payload: JsonValue) -> InternalRowRequest:
        """Validate a decoded JSON payload using Pydantic's JSON semantics."""
        return cls.model_validate_json(json.dumps(payload))


class InternalRowResult(BaseModel):
    """A row outcome cryptographically bound to its submitted request."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_name: Literal["whetstone.envs.internal_row_result/v1"] = (
        _INTERNAL_ROW_RESULT_SCHEMA
    )
    request_identity: str
    outcome: InternalRowOutcome

    @classmethod
    def from_process_payload(cls, payload: JsonValue) -> InternalRowResult:
        """Validate a decoded worker result using JSON semantics."""
        return cls.model_validate_json(json.dumps(payload))


type InternalRowJobFactory = Callable[[InternalRowRequest], ProcessJob]


RUNNER_TIMEOUT_CODE = "runner_timeout"


def drive_internal_row(
    env: EnvSpec,
    *,
    candidate: Candidate,
    instance: Instance,
    provider_call_config: ProviderCallConfig,
    execution_policy: ProviderExecutionPolicy,
    transport: TransportCall,
    procedure_config_hash: str,
    logical_call_id: str,
    repeat_index: int,
    drive_ordinal: int,
    cache: PromptResultCache | None,
    cache_phase: str,
    cache_unit: str,
    render_guard: bool = False,
) -> InternalRowOutcome:
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
            return InternalRowOutcome(
                score=None,
                failed=True,
                failure_code=RENDER_FAILURE_CODE,
            )
    else:
        prompt = render_prompt(env, candidate, instance)
    execution = execute_call(
        request=_request(provider_call_config, prompt),
        policy=execution_policy,
        transport=transport,
        logical_call_id=logical_call_id,
        repeat_index=repeat_index,
        drive_ordinal=drive_ordinal,
        cache=cache,
        phase=cache_phase,
        unit=cache_unit,
    )
    result = execution.result
    telemetry = execution.telemetry()
    marks = execution.cache_marks()
    if not result.succeeded or result.generation is None:
        return InternalRowOutcome(
            score=None,
            failed=True,
            failure_code=failure_code_of(result),
            latency_s=telemetry.latency_s,
            provider_error=telemetry.provider_error,
            rate_limited=is_rate_limit_failure(result),
            redrivable=is_transient_transport_failure(result),
            cache_hit=marks.cache_hit,
            cache_source_phase=marks.cache_source_phase,
            cache_source_unit=marks.cache_source_unit,
            cache_source_call_id=marks.cache_source_call_id,
            cache_source_at=marks.cache_source_at,
        )
    score = env_exact_match_score(
        env=env,
        generation=result.generation.text,
        gold=instance.gold,
        evaluation_procedure_config_hash=procedure_config_hash,
    )
    return InternalRowOutcome(
        score=float(score.value),
        prompt_tokens=telemetry.prompt_tokens,
        completion_tokens=telemetry.completion_tokens,
        total_tokens=telemetry.total_tokens,
        reasoning_tokens=telemetry.reasoning_tokens,
        latency_s=telemetry.latency_s,
        output_text=result.generation.text,
        finish_reason=telemetry.finish_reason,
        cache_hit=marks.cache_hit,
        cache_source_phase=marks.cache_source_phase,
        cache_source_unit=marks.cache_source_unit,
        cache_source_call_id=marks.cache_source_call_id,
        cache_source_at=marks.cache_source_at,
    )


def _env_exact_match_aggregate(
    *,
    graph_hash: str,
    eval_config_hash: str,
    evaluation_context_id: str,
    task_rows: tuple[TaskRows, ...],
    repeat_count: int,
    policy: CompletenessPolicy,
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
    sampling: EnvSplitSampling,
    execution_policy: ProviderExecutionPolicy,
    row_job_factory: InternalRowJobFactory,
    concurrency: int = DEFAULT_CONCURRENCY,
    max_wall_seconds: float | None = None,
    partial_log: PartialLog | None = None,
    apply_reward: bool = True,
    render_guard: bool = False,
    cache: PromptResultCache | None = None,
) -> InternalEvalResult:
    """Evaluate ``candidate`` over one exact derived sampling contract.

    For each bound instance, ``row_job_factory`` supplies one serializable
    process job whose result decodes to :class:`InternalRowOutcome`. The
    per-task means reduce to a single ``env_exact_match`` Rollout Aggregate.
    The factory is the trusted execution authority: it must give its worker
    the exact :class:`InternalRowRequest` and use the declared evaluation
    procedure. The parent verifies the returned request identity before any
    persistence, which prevents cross-row attribution but does not attest an
    arbitrary worker's scoring implementation.

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

    The observations fan out through a bounded worker pool: at
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
    instances = sampling.instances
    validate_candidate_prompt(env, candidate, instances)
    repeats = sampling.repeat_plan.repeat_count
    partial_phase = sampling.split_role
    policy = sampling.completeness_policy
    eval_config_hash = sampling.eval_config.config_identity_hash
    if sampling.eval_config.evaluation_procedure_config_hash != procedure_hash:
        raise ValueError(
            "sampling EvalConfig procedure does not match the experiment"
        )
    # The concrete internal Evaluation Context is minted by orchestration; the
    # helper stamps a stable internal id derived from the internal Eval Config
    # identity onto the aggregate provenance.
    evaluation_context_id = eval_config_hash
    unit = candidate.candidate_id

    recorded = _restore_recorded(
        partial_log, partial_phase, unit, env, procedure_hash
    )

    # Build one keyed CallSpec per (instance, repeat) NOT already on disk.
    tasks = [
        (instance, EnvTask.from_instance(env.name, instance))
        for instance in instances
    ]

    def _persist(
        instance: Instance,
        index: int,
        outcome: InternalRowOutcome,
    ) -> None:
        if partial_log is None:
            return
        partial_log.append(
            PartialCallRecord(
                phase=partial_phase,
                instance_id=str(instance.id),
                unit=unit,
                repeat_id=index,
                score=outcome.score,
                failed=outcome.failed,
                failure_code=outcome.failure_code,
                split_role=sampling.split_role,
                prompt_tokens=outcome.prompt_tokens,
                completion_tokens=outcome.completion_tokens,
                total_tokens=outcome.total_tokens,
                reasoning_tokens=outcome.reasoning_tokens,
                latency_s=outcome.latency_s,
                output_text=outcome.output_text,
                finish_reason=outcome.finish_reason,
                provider_error=outcome.provider_error,
                cache_hit=outcome.cache_hit,
                cache_source_phase=outcome.cache_source_phase,
                cache_source_unit=outcome.cache_source_unit,
                cache_source_call_id=outcome.cache_source_call_id,
                cache_source_at=outcome.cache_source_at,
            )
        )

    def _spec(
        instance: Instance,
        index: int,
        *,
        drive_ordinal: int,
    ) -> CallSpec[tuple[str, str, int], InternalRowOutcome]:
        key = (unit, str(instance.id), index)
        request = InternalRowRequest(
            env_name=env.name,
            candidate=candidate,
            instance=ProcessInstance.from_instance(instance),
            provider_call_config=rd.provider_call_config,
            execution_policy=execution_policy,
            procedure_config_hash=procedure_hash,
            logical_call_id=(
                f"{EnvTask.from_instance(env.name, instance).task_identity()}"
                f"#{index}"
            ),
            repeat_index=index,
            drive_ordinal=drive_ordinal,
            cache_phase=partial_phase,
            cache_unit=unit,
            cache_root=None if cache is None else str(cache.root),
            render_guard=render_guard,
        )

        def _decode(value: JsonValue) -> InternalRowOutcome:
            result = InternalRowResult.from_process_payload(value)
            if result.request_identity != request.request_identity:
                raise ValueError(
                    "internal row result does not match its submitted request"
                )
            return result.outcome

        return CallSpec(
            key=key,
            job=row_job_factory(request),
            decode=_decode,
            deadline_seconds=guard_deadline_seconds(execution_policy),
            commit=lambda outcome, inst=instance, i=index: _persist(
                inst, i, outcome
            ),
        )

    by_instance = {str(inst.id): (inst, tsk) for inst, tsk in tasks}
    phase_deadline = start_phase_deadline(max_wall_seconds)
    specs = [
        _spec(instance, index, drive_ordinal=0)
        for instance, task in tasks
        for index in range(repeats)
        if (unit, str(instance.id), index) not in recorded
    ]

    effective_concurrency = concurrency

    def _drive(
        pending: list[CallSpec[tuple[str, str, int], InternalRowOutcome]],
    ) -> tuple[
        dict[tuple[str, str, int], InternalRowOutcome], bool, bool, int
    ]:
        nonlocal effective_concurrency
        pool = run_call_pool(
            pending,
            concurrency=effective_concurrency,
            is_rate_limited=_row_is_rate_limited,
            max_wall_seconds=remaining_phase_wall_seconds(phase_deadline),
        )
        effective_concurrency = pool.effective_concurrency
        driven: dict[tuple[str, str, int], InternalRowOutcome] = {}
        for res in pool.results:
            if res.status is FanoutStatus.UNIT_TIMEOUT:
                driven[res.key] = InternalRowOutcome(
                    score=None,
                    failed=True,
                    failure_code=RUNNER_TIMEOUT_CODE,
                    redrivable=True,
                )
            elif res.status in {
                FanoutStatus.NOT_DISPATCHED,
                FanoutStatus.OPERATION_DEADLINE,
            }:
                # The whole-phase deadline stopped dispatch before this call:
                # the planned row is absent (missing), never a fabricated
                # failure, and nothing is recorded (a resume re-drives it).
                driven[res.key] = InternalRowOutcome(score=None, missing=True)
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
    # Both attempts are recorded in the partial log (the parent commit appends
    # each accepted record); a re-drive that still fails/times-out lands as a
    # failed row. A not-dispatched (deadline) row is NOT re-driven (a resume
    # handles it). This bounds one flaky observation without re-driving the
    # split.
    redrive_specs = [
        _spec(by_instance[key[1]][0], key[2], drive_ordinal=1)
        for key, out in driven.items()
        if _should_redrive(out)
    ]
    halved_2 = deadline_2 = False
    guard_2 = 0
    if redrive_specs:
        redriven, halved_2, deadline_2, guard_2 = _drive(redrive_specs)
        driven.update(
            (key, outcome)
            for key, outcome in redriven.items()
            if not outcome.missing
        )

    # A first-attempt timeout that was NOT re-driven (or a re-drive that also
    # timed out) is a real failed observation: record it so a resume does not
    # re-drive a call that already blew the deadline twice.
    if partial_log is not None:
        for key, out in driven.items():
            if out.failure_code == RUNNER_TIMEOUT_CODE:
                _persist(by_instance[key[1]][0], key[2], out)

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
                        output_text=outcome.output_text,
                        score=outcome.score,
                        failure_code=outcome.failure_code,
                        finish_reason=outcome.finish_reason,
                        provider_error=outcome.provider_error,
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
        policy=policy,
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


def _row_is_rate_limited(outcome: InternalRowOutcome) -> bool:
    """Whether a driven row's terminal Result is a rate-limit failure."""
    return outcome.rate_limited


def _should_redrive(outcome: InternalRowOutcome) -> bool:
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
    return outcome.redrivable


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
    "InternalRowJobFactory",
    "InternalRowOutcome",
    "RolloutOutput",
    "drive_internal_row",
    "run_internal_eval",
]
