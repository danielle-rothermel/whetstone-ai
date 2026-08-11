from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from dr_providers import (
    MessageRole,
    PromptMessage,
    ProviderCallConfig,
    ProviderCallRequest,
    Transcript,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    JsonValue,
    PrivateAttr,
    model_validator,
)
from whetstone_envs.core import Instance

from whetstone.core.identity import IdentityHash
from whetstone.core.roles import EvaluationRole
from whetstone.envs.factory import EnvExperiment
from whetstone.envs.oracle_operator import (
    ENV_EXACT_MATCH_NAME,
    env_exact_match_score,
)
from whetstone.envs.registry import EnvSpec, env_spec
from whetstone.envs.reward import reward_from_internal_aggregate
from whetstone.envs.rollout_definition import (
    LLM_NODE_ID,
    render_prompt,
    validate_candidate_prompt,
)
from whetstone.envs.sampling import (
    EnvSplitSampling,
    validate_evaluation_role_for_split,
)
from whetstone.envs.task import EnvTask
from whetstone.evaluation.code.aggregate import (
    RolloutAggregate,
    RowValue,
    TaskRows,
    unweighted_task_mean,
)
from whetstone.evaluation.traces import (
    RENDER_FAILURE_CODE,
    ExecutedComponentStep,
    ExecutedComponentTracePayload,
    ExecutedRowState,
    _llm_component_step,
    _llm_component_values,
    validate_executed_component_trace,
)
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
from whetstone.execution.resume import (
    index_partial_records,
    resolve_exact_resume,
)
from whetstone.experiment.binding import (
    EvaluationBinding,
    eval_config_reference,
)
from whetstone.experiment.candidate import Candidate
from whetstone.experiment.reward import Reward
from whetstone.provider.driver import TransportCall
from whetstone.provider.policy import ProviderExecutionPolicy


@dataclass(frozen=True, slots=True)
class RolloutOutput:
    """One rollout row's exact execution trace, display output, and score.

    The row state and executed components are authoritative. ``output_text``
    is the full untruncated sidecar/display output and ``score`` is ``None`` on
    a failed or missing row. Exact partial resume restores the same values.
    """

    candidate_id: str
    instance_id: str
    repeat: int
    row_state: ExecutedRowState
    executed_component_steps: tuple[ExecutedComponentStep, ...]
    output_text: str | None
    score: float | None
    failure_code: str = ""
    #: Per-call provenance (coverage-honest -- ``None`` when unknown):
    #: ``finish_reason`` is the provider stop reason of the accepted Generation
    #: (a truncated ``length`` is distinguishable from a clean ``stop``);
    #: ``provider_error`` is the FULL typed provider-failure diagnostic for a
    #: FAILED row (not just the short ``failure_code``). Both are ``None`` when
    #: unknown.
    finish_reason: str | None = None
    provider_error: dict[str, object] | None = None
    #: ed1 enc-dec budget diagnostics on every rollout row:
    #: the per-task ``max_budget`` (chars) the encoder was told to respect and
    #: the derived ``over_budget`` flag, so a consumer never re-derives
    #: ``round(budget_ratio * len)`` off-row. ``None`` for QA/d1 (no budget)
    #: and for a no-budget ed1 frame.
    max_budget: int | None = None
    over_budget: bool | None = None

    @property
    def failed(self) -> bool:
        return self.row_state is ExecutedRowState.FAILED

    @property
    def missing(self) -> bool:
        return self.row_state is ExecutedRowState.MISSING

    @property
    def invalid(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class InternalEvalResult:
    """One candidate's evaluation outcome over a split.

    An official-role binding produces no Reward. Failed or missing rows
    contribute 0 to the aligned per-task means.
    """

    aggregate: RolloutAggregate
    reward: Reward | None
    per_task_scores: tuple[float, ...]
    per_task_counts: tuple[int, ...]
    #: Exact row state, trace, display output, and score for every planned row
    #: in instance/repeat order, including exact partial restores.
    outputs: tuple[RolloutOutput, ...]
    #: Additional aggregates cited by a reward but not used as the evidence's
    #: primary aggregate. ED1 uses this for its compression aggregate.
    supplemental_aggregates: tuple[RolloutAggregate, ...] = ()
    #: Every planned row-request identity for this exact Evaluation Binding,
    #: both drive ordinals. Restoration is strictly scoped to this set, so it
    #: is also the only set a caller may attribute partial rows to.
    request_identities: frozenset[str] = frozenset()
    concurrency_halved: bool = False
    deadline_reached: bool = False
    guard_timeouts: int = 0


def _per_task_score(task: TaskRows, repeat_count: int) -> float:
    completed = task.completed_rows(repeat_count)
    if not completed:
        return 0.0
    total = sum(
        float(row.value or 0.0) if row.is_present else 0.0 for row in completed
    )
    return total / len(completed)


def _per_task_count(task: TaskRows, repeat_count: int) -> int:
    """Count of completed (scored) repeats behind this task's mean.

    This is the observation weight the paired/pooled bootstrap needs to combine
    a task's mean with additional-repeat means exactly (a weighted mean by
    counts), so escalation pools new observations rather than discarding them.
    """
    return len(task.completed_rows(repeat_count))


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
    row_state: ExecutedRowState
    executed_component_steps: tuple[ExecutedComponentStep, ...]
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
        validate_executed_component_trace(self.executed_component_steps)
        if (self.row_state is ExecutedRowState.SUCCESS) != (
            self.score is not None
        ):
            raise ValueError(
                "a present row requires a score and an absent row forbids one"
            )
        if self.row_state is ExecutedRowState.MISSING:
            if self.executed_component_steps or self.output_text is not None:
                raise ValueError(
                    "a missing row cannot contain execution output"
                )
        elif self.executed_component_steps:
            if len(self.executed_component_steps) != 1:
                raise ValueError(
                    "an internal row executes exactly one component"
                )
            step = self.executed_component_steps[0]
            _prompt, generation = _llm_component_values(
                step, component_id=LLM_NODE_ID
            )
            if generation != self.output_text:
                raise ValueError(
                    "internal trace generation must match output_text"
                )
        elif self.output_text is not None:
            raise ValueError(
                "internal output_text requires its executed component"
            )
        if (
            self.row_state is ExecutedRowState.SUCCESS
            and not self.executed_component_steps
        ):
            raise ValueError("a successful internal row requires its trace")
        if self.cache_hit != (self.cache_source_call_id is not None):
            raise ValueError(
                "cache_hit and original-entry provenance must be paired"
            )
        return self

    @property
    def failed(self) -> bool:
        return self.row_state is ExecutedRowState.FAILED

    @property
    def missing(self) -> bool:
        return self.row_state is ExecutedRowState.MISSING

    @property
    def row(self) -> RowValue:
        if self.failed:
            return RowValue(failed=True)
        if self.missing:
            return RowValue(missing=True)
        assert self.score is not None
        return RowValue(value=self.score)


_INTERNAL_ROW_REQUEST_SCHEMA = "whetstone.envs.internal_row_request/v2"
_INTERNAL_ROW_RESULT_SCHEMA = "whetstone.envs.internal_row_result/v3"


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


def _process_payload_identity(payload: JsonValue) -> str:
    """Hash the exact finite JSON payload submitted to a process worker."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def process_request_identity(model: BaseModel) -> str:
    """Hash one strict row request's submitted JSON representation."""
    return _process_payload_identity(model.model_dump(mode="json"))


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

    _submitted_request_identity: str | None = PrivateAttr(default=None)

    schema_name: Literal["whetstone.envs.internal_row_request/v2"] = (
        _INTERNAL_ROW_REQUEST_SCHEMA
    )
    env_name: str
    candidate: Candidate
    instance: ProcessInstance
    provider_call_config: ProviderCallConfig
    execution_policy: ProviderExecutionPolicy
    procedure_config_hash: str
    evaluation_binding_hash: IdentityHash
    logical_call_id: str
    repeat_index: int
    drive_ordinal: int
    cache_phase: str
    cache_unit: str
    cache_root: str | None
    render_guard: bool

    @property
    def request_identity(self) -> str:
        return self._submitted_request_identity or process_request_identity(
            self
        )

    @classmethod
    def from_process_payload(cls, payload: JsonValue) -> InternalRowRequest:
        """Validate a decoded JSON payload using Pydantic's JSON semantics."""
        request = cls.model_validate_json(json.dumps(payload))
        request._submitted_request_identity = _process_payload_identity(
            payload
        )
        return request


class InternalRowResult(BaseModel):
    """A row outcome cryptographically bound to its submitted request."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_name: Literal["whetstone.envs.internal_row_result/v3"] = (
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
                row_state=ExecutedRowState.FAILED,
                executed_component_steps=(),
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
            row_state=ExecutedRowState.FAILED,
            executed_component_steps=(),
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
        row_state=ExecutedRowState.SUCCESS,
        executed_component_steps=(
            _llm_component_step(
                trace_index=0,
                component_id=LLM_NODE_ID,
                prompt=prompt,
                generation=result.generation.text,
            ),
        ),
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


def run_internal_eval(
    experiment: EnvExperiment,
    *,
    candidate: Candidate,
    sampling: EnvSplitSampling,
    execution_policy: ProviderExecutionPolicy,
    row_job_factory: InternalRowJobFactory,
    evaluation_binding: EvaluationBinding,
    concurrency: int = DEFAULT_CONCURRENCY,
    max_wall_seconds: float | None = None,
    partial_log: PartialLog | None = None,
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

    Reward application follows the exact ``evaluation_binding`` role. The
    internal role maps the aggregate through the Reward Policy; a missing
    aggregate under FAIL surfaces as a typed ``CandidateEvaluationFailure``.
    The official role derives no Reward: the result carries the aggregate and
    per-task vectors, so an
    incomplete official aggregate (timed-out observations) is visible
    incompleteness, never a process crash.

    The observations fan out through a bounded worker pool: at
    most ``concurrency`` calls run at once, each under a runner-level
    wall-clock guard, and the RECORDED per-task rows are assembled by their
    ``(candidate, instance, repeat)`` key in instance/repeat order -- so the
    aggregate is byte-identical regardless of completion order. When a
    ``partial_log`` is given, each completed call is appended as it finishes
    and only an observation for the exact row-request identity is RESTORED
    from disk instead of re-driven. A pending ordinal-0 observation resumes
    at the exact ordinal-1 request.
    """
    env = env_spec(experiment.env_name)
    rd = experiment.rollout_definition
    procedure_hash = experiment.eval_configs.procedure_config_hash
    instances = sampling.instances
    validate_candidate_prompt(env, candidate, instances)
    repeats = sampling.repeat_plan.repeat_count
    partial_phase = sampling.split_role
    if evaluation_binding.eval_config != eval_config_reference(
        sampling.eval_config
    ):
        raise ValueError(
            "evaluation binding must name the exact sampling Eval Config"
        )
    validate_evaluation_role_for_split(
        split_role=partial_phase,
        evaluation_role=evaluation_binding.role,
    )
    if sampling.eval_config.evaluation_procedure_config_hash != procedure_hash:
        raise ValueError(
            "sampling EvalConfig procedure does not match the experiment"
        )
    evaluation_binding_id = evaluation_binding.identity_hash()
    unit = str(candidate.candidate_id)

    tasks = [
        (instance, EnvTask.from_instance(env.name, instance))
        for instance in instances
    ]
    by_instance = {str(inst.id): (inst, tsk) for inst, tsk in tasks}

    def _persist(
        instance: Instance,
        index: int,
        outcome: InternalRowOutcome,
        *,
        request_identity: str,
        redrive_pending: bool,
    ) -> None:
        if partial_log is None:
            return
        partial_log.append(
            PartialCallRecord(
                phase=partial_phase,
                instance_id=str(instance.id),
                unit=unit,
                repeat_id=index,
                request_identity=request_identity,
                redrive_pending=redrive_pending,
                score=outcome.score,
                failed=outcome.failed,
                failure_code=outcome.failure_code,
                split_role=sampling.split_role,
                prompt_tokens=outcome.prompt_tokens,
                completion_tokens=outcome.completion_tokens,
                total_tokens=outcome.total_tokens,
                reasoning_tokens=outcome.reasoning_tokens,
                latency_s=None if outcome.cache_hit else outcome.latency_s,
                output_text=outcome.output_text,
                observation_payload=ExecutedComponentTracePayload(
                    row_state=outcome.row_state,
                    executed_component_steps=outcome.executed_component_steps,
                ).model_dump(mode="json"),
                finish_reason=outcome.finish_reason,
                provider_error=outcome.provider_error,
                cache_hit=outcome.cache_hit,
                cache_source_phase=outcome.cache_source_phase,
                cache_source_unit=outcome.cache_source_unit,
                cache_source_call_id=outcome.cache_source_call_id,
                cache_source_at=outcome.cache_source_at,
            )
        )

    def _row_request(
        instance: Instance,
        index: int,
        *,
        drive_ordinal: int,
    ) -> InternalRowRequest:
        return InternalRowRequest(
            env_name=env.name,
            candidate=candidate,
            instance=ProcessInstance.from_instance(instance),
            provider_call_config=rd.provider_call_config,
            execution_policy=execution_policy,
            procedure_config_hash=procedure_hash,
            evaluation_binding_hash=evaluation_binding_id,
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

    requests_by_key = {
        (unit, str(instance.id), index): (
            _row_request(instance, index, drive_ordinal=0),
            _row_request(instance, index, drive_ordinal=1),
        )
        for instance, _task in tasks
        for index in range(repeats)
    }
    planned_request_identities = frozenset(
        request.request_identity
        for ordinals in requests_by_key.values()
        for request in ordinals
    )
    partial_records = index_partial_records(
        () if partial_log is None else partial_log.load(),
        phase=partial_phase,
        unit=unit,
    )
    recorded: dict[tuple[str, str, int], InternalRowOutcome] = {}
    driven: dict[tuple[str, str, int], InternalRowOutcome] = {}
    initial_requests: list[InternalRowRequest] = []
    resumed_redrive_requests: list[InternalRowRequest] = []
    for key, (ordinal_0, ordinal_1) in requests_by_key.items():
        decision = resolve_exact_resume(
            partial_records,
            instance_id=key[1],
            repeat_id=key[2],
            ordinal_0_request_identity=ordinal_0.request_identity,
            ordinal_1_request_identity=ordinal_1.request_identity,
        )
        if decision.record is not None:
            restored = _internal_outcome_from_record(decision.record)
            if decision.drive_ordinal is None:
                recorded[key] = restored
            else:
                driven[key] = restored
        if decision.drive_ordinal == 0:
            initial_requests.append(ordinal_0)
        elif decision.drive_ordinal == 1:
            resumed_redrive_requests.append(ordinal_1)

    def _spec(
        request: InternalRowRequest,
    ) -> CallSpec[tuple[str, str, int], InternalRowOutcome]:
        key = (unit, request.instance.id, request.repeat_index)
        instance = by_instance[request.instance.id][0]

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
            commit=lambda outcome: _persist(
                instance,
                request.repeat_index,
                outcome,
                request_identity=request.request_identity,
                redrive_pending=(
                    request.drive_ordinal == 0 and _should_redrive(outcome)
                ),
            ),
        )

    phase_deadline = start_phase_deadline(max_wall_seconds)
    effective_concurrency = concurrency

    def _drive(
        requests: list[InternalRowRequest],
    ) -> tuple[
        dict[tuple[str, str, int], InternalRowOutcome], bool, bool, int
    ]:
        nonlocal effective_concurrency
        specs = [_spec(request) for request in requests]
        request_by_key = {
            (unit, request.instance.id, request.repeat_index): request
            for request in requests
        }
        pool = run_call_pool(
            specs,
            concurrency=effective_concurrency,
            is_rate_limited=_row_is_rate_limited,
            max_wall_seconds=remaining_phase_wall_seconds(phase_deadline),
        )
        effective_concurrency = pool.effective_concurrency
        driven: dict[tuple[str, str, int], InternalRowOutcome] = {}
        for res in pool.results:
            request = request_by_key[res.key]
            if res.status is FanoutStatus.UNIT_TIMEOUT:
                outcome = InternalRowOutcome(
                    score=None,
                    row_state=ExecutedRowState.FAILED,
                    executed_component_steps=(),
                    failure_code=RUNNER_TIMEOUT_CODE,
                    redrivable=True,
                )
                driven[res.key] = outcome
                _persist(
                    by_instance[res.key[1]][0],
                    res.key[2],
                    outcome,
                    request_identity=request.request_identity,
                    redrive_pending=request.drive_ordinal == 0,
                )
            elif res.status in {
                FanoutStatus.NOT_DISPATCHED,
                FanoutStatus.OPERATION_DEADLINE,
            }:
                # The whole-phase deadline stopped dispatch before this call:
                # the planned row is absent (missing), never a fabricated
                # failure, and nothing is recorded (a resume re-drives it).
                driven[res.key] = InternalRowOutcome(
                    score=None,
                    row_state=ExecutedRowState.MISSING,
                    executed_component_steps=(),
                )
            elif res.value is not None:
                driven[res.key] = res.value
        return (
            driven,
            pool.concurrency_halved,
            pool.deadline_reached,
            pool.guard_timeouts,
        )

    first_driven, halved_1, deadline_1, guard_1 = _drive(initial_requests)
    driven.update(first_driven)

    # --- ONE bounded re-drive of timed-out / transient-transport failures. ---
    # A runner-guard timeout or a TERMINAL transient transport failure (the
    # driver's own semantic retries were exhausted) is re-driven exactly once
    # through the same semantic-retry path before it lands as a failed row.
    # Both attempts are recorded in the partial log (the parent commit appends
    # each accepted record); a re-drive that still fails/times-out lands as a
    # failed row. A not-dispatched (deadline) row is NOT re-driven (a resume
    # handles it). This bounds one flaky observation without re-driving the
    # split.
    redrive_requests = resumed_redrive_requests + [
        requests_by_key[key][1]
        for key, out in first_driven.items()
        if _should_redrive(out)
    ]
    halved_2 = deadline_2 = False
    guard_2 = 0
    if redrive_requests:
        redriven, halved_2, deadline_2, guard_2 = _drive(redrive_requests)
        driven.update(
            (key, outcome)
            for key, outcome in redriven.items()
            if not outcome.missing
        )

    concurrency_halved = halved_1 or halved_2
    deadline_reached = deadline_1 or deadline_2
    guard_timeouts = guard_1 + guard_2

    # Assemble one complete per-task matrix with exact traces in
    # instance/repeat order. Restored rows retain their persisted output and
    # trace; deadline-stopped rows remain explicit missing outcomes.
    task_rows: list[TaskRows] = []
    outputs: list[RolloutOutput] = []
    for instance, task in tasks:
        rows: list[RowValue] = []
        for index in range(repeats):
            key = (unit, str(instance.id), index)
            outcome = recorded.get(key, driven.get(key))
            if outcome is None:
                raise RuntimeError("internal row assembly is incomplete")
            rows.append(outcome.row)
            outputs.append(
                RolloutOutput(
                    candidate_id=unit,
                    instance_id=str(instance.id),
                    repeat=index,
                    row_state=outcome.row_state,
                    executed_component_steps=outcome.executed_component_steps,
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
                rows=tuple(rows),
            )
        )

    rollout_aggregate = unweighted_task_mean(
        aggregate_name=ENV_EXACT_MATCH_NAME,
        graph_hash=rd.graph_hash,
        evaluation_binding_hash=evaluation_binding_id,
        task_rows=tuple(task_rows),
        plan=sampling.evaluation_matrix_plan,
    )
    # The exact Evaluation Binding role controls Reward derivation. Official
    # evaluation carries only aggregate and per-task evidence.
    reward = (
        reward_from_internal_aggregate(
            experiment.reward_policy,
            env_exact_match_value=rollout_aggregate.aggregation_output.value,
            evidence_refs=(rollout_aggregate.record_ref(),),
        )
        if evaluation_binding.role is EvaluationRole.INTERNAL
        else None
    )
    per_task_scores = tuple(
        _per_task_score(task, repeats) for task in task_rows
    )
    per_task_counts = tuple(
        _per_task_count(task, repeats) for task in task_rows
    )
    return InternalEvalResult(
        aggregate=rollout_aggregate,
        reward=reward,
        per_task_scores=per_task_scores,
        per_task_counts=per_task_counts,
        concurrency_halved=concurrency_halved,
        deadline_reached=deadline_reached,
        guard_timeouts=guard_timeouts,
        outputs=tuple(outputs),
        request_identities=planned_request_identities,
    )


def _row_is_rate_limited(outcome: InternalRowOutcome) -> bool:
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


def _internal_outcome_from_record(
    record: PartialCallRecord,
) -> InternalRowOutcome:
    payload = ExecutedComponentTracePayload.from_json_value(
        record.observation_payload
    )
    if record.failed != (payload.row_state is ExecutedRowState.FAILED):
        raise ValueError(
            "internal partial row state conflicts with failed flag"
        )
    return InternalRowOutcome(
        score=None if record.score is None else float(record.score),
        row_state=payload.row_state,
        executed_component_steps=payload.executed_component_steps,
        failure_code=record.failure_code,
        prompt_tokens=record.prompt_tokens,
        completion_tokens=record.completion_tokens,
        total_tokens=record.total_tokens,
        reasoning_tokens=record.reasoning_tokens,
        latency_s=record.latency_s,
        output_text=record.output_text,
        finish_reason=record.finish_reason,
        provider_error=record.provider_error,
        redrivable=record.redrive_pending,
        cache_hit=record.cache_hit,
        cache_source_phase=record.cache_source_phase,
        cache_source_unit=record.cache_source_unit,
        cache_source_call_id=record.cache_source_call_id,
        cache_source_at=record.cache_source_at,
    )


__all__ = [
    "InternalEvalResult",
    "InternalRowJobFactory",
    "InternalRowOutcome",
    "RolloutOutput",
    "drive_internal_row",
    "run_internal_eval",
]
