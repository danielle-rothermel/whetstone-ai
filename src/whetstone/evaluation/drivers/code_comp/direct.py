from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from dr_code.humaneval import HumanEvalTask
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
from whetstone.envs.d1 import (
    D1_SUBMISSION_SCORE_NAME,
    D1Experiment,
    render_d1_frame,
)
from whetstone.envs.ed1 import (
    reward_from_primary_score,
    validate_ed1_body,
)
from whetstone.envs.ed1_scoring import (
    BatchScoringDeadlineExceeded,
    CodeBatchScorer,
    CodeScore,
    CodeScoringInput,
)
from whetstone.envs.input_transform import (
    direct_body,
    renamed_task,
    split_prompt,
)
from whetstone.envs.rollout_definition import LLM_NODE_ID
from whetstone.envs.sampling import (
    EnvSplitSampling,
    validate_evaluation_role_for_split,
)
from whetstone.evaluation.aggregate import (
    RolloutAggregate,
    RowValue,
    TaskRows,
    unweighted_task_mean,
)
from whetstone.evaluation.drivers.internal import (
    ProcessInstance,
    RolloutOutput,
    _llm_component_step,
    _llm_component_values,
    _process_payload_identity,
    process_request_identity,
    start_phase_deadline,
)
from whetstone.evaluation.traces import (
    ExecutedComponentStep,
    ExecutedComponentTracePayload,
    ExecutedRowState,
    validate_executed_component_trace,
)
from whetstone.execution.call_support import (
    failure_code_of,
    is_transient_transport_failure,
)
from whetstone.execution.fanout import (
    DEFAULT_CONCURRENCY,
    CallSpec,
    FanoutStatus,
    ProcessJob,
)
from whetstone.execution.partials import PartialCallRecord, PartialLog
from whetstone.execution.prompt_cache import (
    PromptResultCache,
    execute_call,
)
from whetstone.execution.resume import (
    resolve_exact_resume,
)
from whetstone.experiment.binding import (
    EvaluationBinding,
    eval_config_reference,
)
from whetstone.experiment.reward import Reward
from whetstone.provider.driver import TransportCall
from whetstone.provider.policy import ProviderExecutionPolicy


def _d1_compat():
    import whetstone.evaluation.drivers.d1 as compat

    return compat


@dataclass(frozen=True, slots=True)
class D1EvalResult:
    """One candidate's D1 evaluation over a split.

    ``submission_score_aggregate`` is the reward-bearing HumanEval Submission
    Score. ``reward`` is derived for an internal Evaluation Binding. Per-task
    vectors and
    outputs feed the CI, ledger, and sidecar.
    """

    submission_score_aggregate: RolloutAggregate
    reward: Reward | None
    per_task_scores: tuple[float, ...]
    per_task_counts: tuple[int, ...]
    outputs: tuple[RolloutOutput, ...]


class D1RowOutcome(BaseModel):
    """One (task, repeat) direct rollout's result + provenance."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
        strict=True,
    )

    submission_score: float | None
    output_text: str | None
    row_state: ExecutedRowState
    executed_component_steps: tuple[ExecutedComponentStep, ...]
    failure_code: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    latency_s: float | None = None
    #: The accepted Generation's stop reason and full typed failure diagnostic.
    finish_reason: str | None = None
    provider_error: dict[str, object] | None = None
    #: True when a TRANSIENT transport fault (timeout/stall/transport-error/
    #: rate-limit) exhausted its semantic retries -- eligible for ONE re-drive.
    redrivable: bool = False
    cache_hit: bool = False
    cache_source_phase: str | None = None
    cache_source_unit: str | None = None
    cache_source_call_id: str | None = None
    cache_source_at: str | None = None

    @model_validator(mode="after")
    def _valid_outcome(self) -> D1RowOutcome:
        validate_executed_component_trace(self.executed_component_steps)
        if (self.row_state is ExecutedRowState.SUCCESS) != (
            self.submission_score is not None
        ):
            raise ValueError(
                "a successful row requires a score and an absent row "
                "forbids one"
            )
        if self.row_state is ExecutedRowState.MISSING:
            if self.executed_component_steps or self.output_text is not None:
                raise ValueError(
                    "a missing row cannot contain execution output"
                )
        elif self.executed_component_steps:
            if len(self.executed_component_steps) != 1:
                raise ValueError("a D1 row executes exactly one component")
            step = self.executed_component_steps[0]
            _prompt, generation = _llm_component_values(
                step, component_id=LLM_NODE_ID
            )
            if generation != self.output_text:
                raise ValueError("D1 trace generation must match output_text")
        elif self.output_text is not None:
            raise ValueError("D1 output_text requires its executed component")
        if (
            self.row_state is ExecutedRowState.SUCCESS
            and not self.executed_component_steps
        ):
            raise ValueError("a successful D1 row requires its trace")
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


class D1GeneratedRowOutcome(BaseModel):
    """A completed provider row awaiting coordinator-side code scoring."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
        strict=True,
    )

    output_text: str
    executed_component_steps: tuple[ExecutedComponentStep, ...]
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    latency_s: float | None = None
    finish_reason: str | None = None
    cache_hit: bool = False
    cache_source_phase: str | None = None
    cache_source_unit: str | None = None
    cache_source_call_id: str | None = None
    cache_source_at: str | None = None

    @model_validator(mode="after")
    def _valid_generated_outcome(self) -> D1GeneratedRowOutcome:
        validate_executed_component_trace(self.executed_component_steps)
        if len(self.executed_component_steps) != 1:
            raise ValueError("a generated D1 row requires one component")
        _prompt, generation = _llm_component_values(
            self.executed_component_steps[0], component_id=LLM_NODE_ID
        )
        if generation != self.output_text:
            raise ValueError("D1 trace generation must match output_text")
        if self.cache_hit != (self.cache_source_call_id is not None):
            raise ValueError(
                "cache_hit and original-entry provenance must be paired"
            )
        return self


_D1_ROW_REQUEST_SCHEMA = "whetstone.envs.d1_row_request/v2"
_D1_ROW_RESULT_SCHEMA = "whetstone.envs.d1_row_result/v3"


class HumanEvalTaskPayload(BaseModel):
    """Stable JSON fields needed to reconstruct one HumanEval task."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    task_id: str
    prompt: str
    canonical_solution: str
    entry_point: str
    test: str

    @classmethod
    def from_task(cls, task: HumanEvalTask) -> HumanEvalTaskPayload:
        return cls(
            task_id=task.task_id,
            prompt=task.prompt,
            canonical_solution=task.canonical_solution,
            entry_point=task.entry_point,
            test=task.test,
        )

    def to_task(self) -> HumanEvalTask:
        return HumanEvalTask(
            task_id=self.task_id,
            prompt=self.prompt,
            canonical_solution=self.canonical_solution,
            entry_point=self.entry_point,
            test=self.test,
        )


class D1RowRequest(BaseModel):
    """Complete serializable request and provenance for one D1 row."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    _submitted_request_identity: str | None = PrivateAttr(default=None)

    schema_name: Literal["whetstone.envs.d1_row_request/v2"] = (
        _D1_ROW_REQUEST_SCHEMA
    )
    candidate_body: str
    candidate_id: str
    instance: ProcessInstance
    humaneval_task: HumanEvalTaskPayload
    input_arm: str
    rename_token: str
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

    @property
    def request_identity(self) -> str:
        return self._submitted_request_identity or process_request_identity(
            self
        )

    @classmethod
    def from_process_payload(cls, payload: JsonValue) -> D1RowRequest:
        """Validate a decoded JSON payload using Pydantic's JSON semantics."""
        request = cls.model_validate_json(json.dumps(payload))
        request._submitted_request_identity = _process_payload_identity(
            payload
        )
        return request


class D1RowResult(BaseModel):
    """A D1 outcome cryptographically bound to its submitted request."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_name: Literal["whetstone.envs.d1_row_result/v3"] = (
        _D1_ROW_RESULT_SCHEMA
    )
    request_identity: str
    outcome: D1RowOutcome | D1GeneratedRowOutcome

    @classmethod
    def from_process_payload(cls, payload: JsonValue) -> D1RowResult:
        """Validate a decoded worker result using JSON semantics."""
        return cls.model_validate_json(json.dumps(payload))


type D1RowJobFactory = Callable[[D1RowRequest], ProcessJob]


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

    ``split_prompt`` and ``direct_body`` derive the selected arm. The
    ``renamed`` arm also scrubs every canonical-name occurrence and returns a
    scoring task with the renamed entry point.
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


def drive_d1_row(
    *,
    experiment: D1Experiment,
    candidate_body: str,
    instance: Instance,
    provider_call_config: ProviderCallConfig,
    execution_policy: ProviderExecutionPolicy,
    transport: TransportCall,
    scorer: Callable[..., CodeScore] | None,
    logical_call_id: str,
    repeat_index: int,
    drive_ordinal: int,
    cache: PromptResultCache | None,
    cache_phase: str,
    cache_unit: str,
) -> D1RowOutcome | D1GeneratedRowOutcome:
    """Run one direct generation, optionally scoring inside the worker."""
    input_arm, score_task = _input_arm_text(experiment, instance)
    try:
        prompt = render_d1_frame(candidate_body, input_arm=input_arm)
    except (KeyError, IndexError, ValueError):
        return D1RowOutcome(
            submission_score=None,
            output_text=None,
            row_state=ExecutedRowState.FAILED,
            executed_component_steps=(),
            failure_code="d1_wrapper_render_error",
        )
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
        return D1RowOutcome(
            submission_score=None,
            output_text=None,
            row_state=ExecutedRowState.FAILED,
            executed_component_steps=(),
            failure_code=failure_code_of(result),
            latency_s=telemetry.latency_s,
            provider_error=telemetry.provider_error,
            redrivable=is_transient_transport_failure(result),
            cache_hit=marks.cache_hit,
            cache_source_phase=marks.cache_source_phase,
            cache_source_unit=marks.cache_source_unit,
            cache_source_call_id=marks.cache_source_call_id,
            cache_source_at=marks.cache_source_at,
        )
    output_text = result.generation.text
    executed_component_steps = (
        _llm_component_step(
            trace_index=0,
            component_id=LLM_NODE_ID,
            prompt=prompt,
            generation=output_text,
        ),
    )
    if scorer is None:
        return D1GeneratedRowOutcome(
            output_text=output_text,
            executed_component_steps=executed_component_steps,
            prompt_tokens=telemetry.prompt_tokens,
            completion_tokens=telemetry.completion_tokens,
            total_tokens=telemetry.total_tokens,
            reasoning_tokens=telemetry.reasoning_tokens,
            latency_s=telemetry.latency_s,
            finish_reason=telemetry.finish_reason,
            cache_hit=marks.cache_hit,
            cache_source_phase=marks.cache_source_phase,
            cache_source_unit=marks.cache_source_unit,
            cache_source_call_id=marks.cache_source_call_id,
            cache_source_at=marks.cache_source_at,
        )
    code_score = scorer(raw_submission=output_text, task=score_task)
    if code_score.infrastructure_unknown:
        return D1RowOutcome(
            submission_score=None,
            output_text=output_text,
            row_state=ExecutedRowState.FAILED,
            executed_component_steps=executed_component_steps,
            failure_code="code_eval_infrastructure_unknown",
            prompt_tokens=telemetry.prompt_tokens,
            completion_tokens=telemetry.completion_tokens,
            total_tokens=telemetry.total_tokens,
            reasoning_tokens=telemetry.reasoning_tokens,
            latency_s=telemetry.latency_s,
            finish_reason=telemetry.finish_reason,
            cache_hit=marks.cache_hit,
            cache_source_phase=marks.cache_source_phase,
            cache_source_unit=marks.cache_source_unit,
            cache_source_call_id=marks.cache_source_call_id,
            cache_source_at=marks.cache_source_at,
        )
    return D1RowOutcome(
        submission_score=code_score.row_value,
        output_text=output_text,
        row_state=ExecutedRowState.SUCCESS,
        executed_component_steps=executed_component_steps,
        prompt_tokens=telemetry.prompt_tokens,
        completion_tokens=telemetry.completion_tokens,
        total_tokens=telemetry.total_tokens,
        reasoning_tokens=telemetry.reasoning_tokens,
        latency_s=telemetry.latency_s,
        finish_reason=telemetry.finish_reason,
        cache_hit=marks.cache_hit,
        cache_source_phase=marks.cache_source_phase,
        cache_source_unit=marks.cache_source_unit,
        cache_source_call_id=marks.cache_source_call_id,
        cache_source_at=marks.cache_source_at,
    )


def _d1_outcome_from_record(record: PartialCallRecord) -> D1RowOutcome:
    """Rebuild the accepted outcome stored for one exact D1 request."""
    payload = ExecutedComponentTracePayload.from_json_value(
        record.observation_payload
    )
    if record.failed != (payload.row_state is ExecutedRowState.FAILED):
        raise ValueError("D1 partial row state conflicts with failed flag")
    return D1RowOutcome(
        submission_score=(
            None if record.score is None else float(record.score)
        ),
        output_text=record.output_text,
        row_state=payload.row_state,
        executed_component_steps=payload.executed_component_steps,
        failure_code=record.failure_code,
        prompt_tokens=record.prompt_tokens,
        completion_tokens=record.completion_tokens,
        total_tokens=record.total_tokens,
        reasoning_tokens=record.reasoning_tokens,
        latency_s=record.latency_s,
        finish_reason=record.finish_reason,
        provider_error=record.provider_error,
        redrivable=record.redrive_pending,
        cache_hit=record.cache_hit,
        cache_source_phase=record.cache_source_phase,
        cache_source_unit=record.cache_source_unit,
        cache_source_call_id=record.cache_source_call_id,
        cache_source_at=record.cache_source_at,
    )


def _should_redrive(outcome: D1RowOutcome) -> bool:
    """Whether an ordinal-0 D1 result requires the bounded second attempt."""
    return outcome.failure_code == "runner_timeout" or outcome.redrivable


def _finish_generated_row(
    outcome: D1GeneratedRowOutcome,
    score: CodeScore,
) -> D1RowOutcome:
    if score.infrastructure_unknown:
        return D1RowOutcome(
            submission_score=None,
            output_text=outcome.output_text,
            row_state=ExecutedRowState.FAILED,
            executed_component_steps=outcome.executed_component_steps,
            failure_code="code_eval_infrastructure_unknown",
            prompt_tokens=outcome.prompt_tokens,
            completion_tokens=outcome.completion_tokens,
            total_tokens=outcome.total_tokens,
            reasoning_tokens=outcome.reasoning_tokens,
            latency_s=outcome.latency_s,
            finish_reason=outcome.finish_reason,
            cache_hit=outcome.cache_hit,
            cache_source_phase=outcome.cache_source_phase,
            cache_source_unit=outcome.cache_source_unit,
            cache_source_call_id=outcome.cache_source_call_id,
            cache_source_at=outcome.cache_source_at,
        )
    return D1RowOutcome(
        submission_score=score.row_value,
        output_text=outcome.output_text,
        row_state=ExecutedRowState.SUCCESS,
        executed_component_steps=outcome.executed_component_steps,
        prompt_tokens=outcome.prompt_tokens,
        completion_tokens=outcome.completion_tokens,
        total_tokens=outcome.total_tokens,
        reasoning_tokens=outcome.reasoning_tokens,
        latency_s=outcome.latency_s,
        finish_reason=outcome.finish_reason,
        cache_hit=outcome.cache_hit,
        cache_source_phase=outcome.cache_source_phase,
        cache_source_unit=outcome.cache_source_unit,
        cache_source_call_id=outcome.cache_source_call_id,
        cache_source_at=outcome.cache_source_at,
    )


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
    row_job_factory: D1RowJobFactory,
    evaluation_binding: EvaluationBinding,
    concurrency: int = DEFAULT_CONCURRENCY,
    max_wall_seconds: float | None = None,
    partial_log: PartialLog | None = None,
    cache: PromptResultCache | None = None,
    batch_scorer: CodeBatchScorer | None = None,
) -> D1EvalResult:
    """Drive ``candidate_body`` over a D1 split.

    Fans out one serializable direct-generation process job per (task, repeat),
    batch-scores generated submissions in the coordinator when requested,
    reduces to the HumanEval Submission Score aggregate, derives its Reward for
    an internal Evaluation Binding, and collects per-row outputs.
    Incremental
    persistence + resume mirror the ED1 drive: each completed row appends its
    record when it finishes; a resumed drive restores only an exact request
    identity instead of re-paying. A pending ordinal-0 record resumes at the
    exact ordinal-1 request.

    A worker may still return a fully scored row during the dependency cutover.
    A generated row is not written to the partial log until coordinator scoring
    completes, so a crash in that interval may repeat generation; the prompt
    cache remains the available no-wire replay path.
    """
    validate_ed1_body(candidate_body)
    instances = sampling.instances
    repeats = sampling.repeat_plan.repeat_count
    split_role = sampling.split_role
    rd = experiment.rollout_definition
    graph_hash = rd.graph_hash
    if (
        sampling.eval_config.evaluation_procedure_config_hash
        != rd.procedure_config_hash
    ):
        raise ValueError(
            "sampling EvalConfig procedure does not match the experiment"
        )
    if evaluation_binding.eval_config != eval_config_reference(
        sampling.eval_config
    ):
        raise ValueError(
            "evaluation binding must name the exact sampling Eval Config"
        )
    validate_evaluation_role_for_split(
        split_role=split_role,
        evaluation_role=evaluation_binding.role,
    )
    evaluation_binding_id = evaluation_binding.identity_hash()
    by_instance = {str(inst.id): inst for inst in instances}

    def _persist(
        instance: Instance,
        index: int,
        outcome: D1RowOutcome,
        *,
        request_identity: str,
        redrive_pending: bool,
    ) -> None:
        if partial_log is None:
            return
        partial_log.append(
            PartialCallRecord(
                phase=split_role,
                instance_id=str(instance.id),
                unit=candidate_id,
                repeat_id=index,
                request_identity=request_identity,
                redrive_pending=redrive_pending,
                score=outcome.submission_score,
                failed=outcome.failed,
                failure_code=outcome.failure_code,
                split_role=split_role,
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
    ) -> D1RowRequest:
        return D1RowRequest(
            candidate_body=candidate_body,
            candidate_id=candidate_id,
            instance=ProcessInstance.from_instance(instance),
            humaneval_task=HumanEvalTaskPayload.from_task(
                experiment.humaneval_for(instance)
            ),
            input_arm=experiment.input_arm,
            rename_token=experiment.rename_token,
            provider_call_config=rd.provider_call_config,
            execution_policy=execution_policy,
            procedure_config_hash=rd.procedure_config_hash,
            evaluation_binding_hash=evaluation_binding_id,
            logical_call_id=f"{candidate_id}:{instance.id}#{index}",
            repeat_index=index,
            drive_ordinal=drive_ordinal,
            cache_phase=split_role,
            cache_unit=candidate_id,
            cache_root=None if cache is None else str(cache.root),
        )

    requests_by_key = {
        (str(instance.id), index): (
            _row_request(instance, index, drive_ordinal=0),
            _row_request(instance, index, drive_ordinal=1),
        )
        for instance in instances
        for index in range(repeats)
    }
    partial_records = _d1_compat().index_partial_records(
        () if partial_log is None else partial_log.load(),
        phase=split_role,
        unit=candidate_id,
    )
    driven: dict[tuple[str, int], D1RowOutcome | D1GeneratedRowOutcome] = {}
    completed_requests: dict[tuple[str, int], D1RowRequest] = {}
    initial_requests: list[D1RowRequest] = []
    resumed_redrive_requests: list[D1RowRequest] = []
    for key, (ordinal_0, ordinal_1) in requests_by_key.items():
        decision = resolve_exact_resume(
            partial_records,
            instance_id=key[0],
            repeat_id=key[1],
            ordinal_0_request_identity=ordinal_0.request_identity,
            ordinal_1_request_identity=ordinal_1.request_identity,
        )
        if decision.record is not None:
            driven[key] = _d1_outcome_from_record(decision.record)
        if decision.drive_ordinal == 0:
            initial_requests.append(ordinal_0)
        elif decision.drive_ordinal == 1:
            resumed_redrive_requests.append(ordinal_1)

    def _spec(
        request: D1RowRequest,
    ) -> CallSpec[tuple[str, int], D1RowOutcome | D1GeneratedRowOutcome]:
        instance = by_instance[request.instance.id]

        def _decode(
            value: JsonValue,
        ) -> D1RowOutcome | D1GeneratedRowOutcome:
            result = D1RowResult.from_process_payload(value)
            if result.request_identity != request.request_identity:
                raise ValueError(
                    "D1 row result does not match its submitted request"
                )
            return result.outcome

        return CallSpec(
            key=(request.instance.id, request.repeat_index),
            job=row_job_factory(request),
            decode=_decode,
            deadline_seconds=_deadline(execution_policy),
            commit=lambda outcome: (
                _persist(
                    instance,
                    request.repeat_index,
                    outcome,
                    request_identity=request.request_identity,
                    redrive_pending=(
                        request.drive_ordinal == 0 and _should_redrive(outcome)
                    ),
                )
                if isinstance(outcome, D1RowOutcome)
                else None
            ),
        )

    phase_deadline = start_phase_deadline(max_wall_seconds)
    effective_concurrency = concurrency

    def _drive(
        requests: list[D1RowRequest],
    ) -> dict[tuple[str, int], D1RowOutcome | D1GeneratedRowOutcome]:
        nonlocal effective_concurrency
        specs = [_spec(request) for request in requests]
        request_by_key = {
            (request.instance.id, request.repeat_index): request
            for request in requests
        }
        pool = _d1_compat().run_call_pool(
            specs,
            concurrency=effective_concurrency,
            is_rate_limited=lambda _o: False,
            max_wall_seconds=_d1_compat().remaining_phase_wall_seconds(
                phase_deadline
            ),
        )
        effective_concurrency = pool.effective_concurrency
        out: dict[tuple[str, int], D1RowOutcome | D1GeneratedRowOutcome] = {}
        for res in pool.results:
            if res.status is FanoutStatus.COMPLETED and res.value is not None:
                out[res.key] = res.value
                completed_requests[res.key] = request_by_key[res.key]
            elif res.status is FanoutStatus.UNIT_TIMEOUT:
                request = request_by_key[res.key]
                outcome = D1RowOutcome(
                    submission_score=None,
                    output_text=None,
                    row_state=ExecutedRowState.FAILED,
                    executed_component_steps=(),
                    failure_code="runner_timeout",
                    redrivable=True,
                )
                out[res.key] = outcome
                _persist(
                    by_instance[res.key[0]],
                    res.key[1],
                    outcome,
                    request_identity=request.request_identity,
                    redrive_pending=request.drive_ordinal == 0,
                )
            else:
                out[res.key] = D1RowOutcome(
                    submission_score=None,
                    output_text=None,
                    row_state=ExecutedRowState.MISSING,
                    executed_component_steps=(),
                )
        return out

    first_driven = _drive(initial_requests)
    driven.update(first_driven)

    # ONE bounded re-drive of timed-out / transient-transport rows (a single
    # flaky observation must not fail the whole d1 arm under FAIL policy).
    redrive_requests = resumed_redrive_requests + [
        requests_by_key[key][1]
        for key, outcome in first_driven.items()
        if isinstance(outcome, D1RowOutcome) and _should_redrive(outcome)
    ]
    if redrive_requests:
        redriven = _drive(redrive_requests)
        driven.update(
            (key, outcome)
            for key, outcome in redriven.items()
            if not isinstance(outcome, D1RowOutcome) or not outcome.missing
        )

    generated_keys = [
        (str(instance.id), index)
        for instance in instances
        for index in range(repeats)
        if isinstance(driven[(str(instance.id), index)], D1GeneratedRowOutcome)
    ]
    if generated_keys:
        if batch_scorer is None:
            raise ValueError(
                "coordinator-side D1 scoring requires a batch_scorer"
            )
        scoring_inputs = tuple(
            CodeScoringInput(
                raw_submission=generated.output_text,
                task=_input_arm_text(experiment, by_instance[instance_id])[1],
            )
            for instance_id, _index in generated_keys
            for generated in (driven[(instance_id, _index)],)
            if isinstance(generated, D1GeneratedRowOutcome)
        )
        remaining_wall_seconds = _d1_compat().remaining_phase_wall_seconds(
            phase_deadline
        )
        if remaining_wall_seconds == 0.0:
            scores = None
        else:
            try:
                scores = tuple(
                    batch_scorer(
                        scoring_inputs,
                        max_wall_seconds=remaining_wall_seconds,
                    )
                )
            except BatchScoringDeadlineExceeded:
                scores = None
        if scores is None:
            for key in generated_keys:
                driven[key] = D1RowOutcome(
                    submission_score=None,
                    output_text=None,
                    row_state=ExecutedRowState.MISSING,
                    executed_component_steps=(),
                )
        else:
            if len(scores) != len(generated_keys):
                raise ValueError(
                    "D1 batch scorer returned the wrong result count"
                )
            for key, score in zip(generated_keys, scores, strict=True):
                generated = driven[key]
                if not isinstance(generated, D1GeneratedRowOutcome):
                    raise AssertionError(
                        "generated D1 row changed before scoring"
                    )
                outcome = _finish_generated_row(generated, score)
                driven[key] = outcome
                request = completed_requests[key]
                _persist(
                    by_instance[key[0]],
                    key[1],
                    outcome,
                    request_identity=request.request_identity,
                    redrive_pending=False,
                )

    submission_rows: list[tuple[str, list[RowValue]]] = []
    outputs: list[RolloutOutput] = []
    per_task_scores: list[float] = []
    per_task_counts: list[int] = []
    for instance in instances:
        task_id = str(instance.id)
        task_submission_rows: list[RowValue] = []
        for index in range(repeats):
            outcome = driven[(task_id, index)]
            if not isinstance(outcome, D1RowOutcome):
                raise AssertionError("D1 row was not scored")
            if outcome.missing:
                task_submission_rows.append(RowValue(missing=True))
            elif outcome.failed or outcome.submission_score is None:
                task_submission_rows.append(RowValue(failed=True))
            else:
                task_submission_rows.append(
                    RowValue(value=float(outcome.submission_score))
                )
            outputs.append(
                RolloutOutput(
                    candidate_id=candidate_id,
                    instance_id=task_id,
                    repeat=index,
                    row_state=outcome.row_state,
                    executed_component_steps=outcome.executed_component_steps,
                    output_text=outcome.output_text,
                    score=(
                        None
                        if outcome.submission_score is None
                        else float(outcome.submission_score)
                    ),
                    failure_code=outcome.failure_code,
                    finish_reason=outcome.finish_reason,
                    provider_error=outcome.provider_error,
                )
            )
        submission_rows.append((task_id, task_submission_rows))
        # Per-task submission-score mean + planned-repeat weight. As in ED1/QA,
        # an absent/failed row counts 0 and weight is the repeat count.
        total = sum(
            float(r.value or 0.0) if r.is_present else 0.0
            for r in task_submission_rows
        )
        per_task_scores.append(
            total / len(task_submission_rows) if task_submission_rows else 0.0
        )
        per_task_counts.append(len(task_submission_rows))

    submission_score_aggregate = unweighted_task_mean(
        aggregate_name=D1_SUBMISSION_SCORE_NAME,
        graph_hash=graph_hash,
        evaluation_binding_hash=evaluation_binding_id,
        task_rows=tuple(
            TaskRows(
                task_identity=task_identity,
                rows=tuple(rows),
            )
            for task_identity, rows in submission_rows
        ),
        plan=sampling.evaluation_matrix_plan,
    )
    reward: Reward | None = None
    if evaluation_binding.role is EvaluationRole.INTERNAL:
        reward = reward_from_primary_score(
            experiment.reward_policy,
            primary_score=(
                submission_score_aggregate.aggregation_output.value
            ),
            evidence_refs=(submission_score_aggregate.record_ref(),),
        )
    return D1EvalResult(
        submission_score_aggregate=submission_score_aggregate,
        reward=reward,
        per_task_scores=tuple(per_task_scores),
        per_task_counts=tuple(per_task_counts),
        outputs=tuple(outputs),
    )


__all__ = [
    "D1EvalResult",
    "D1GeneratedRowOutcome",
    "D1RowJobFactory",
    "D1RowOutcome",
    "drive_d1_row",
    "run_d1_eval",
]
