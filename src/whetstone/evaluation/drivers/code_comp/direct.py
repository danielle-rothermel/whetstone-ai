from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from dr_code.humaneval import HumanEvalTask
from dr_graph import (
    GraphRunResult,
    NodeConfig,
    NodeOutcomeStatus,
    NodeOutput,
)
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
    field_serializer,
    field_validator,
    model_validator,
)
from whetstone_envs.core import Instance

from whetstone.core.identity import IdentityHash
from whetstone.core.roles import EvaluationRole
from whetstone.envs.code_comp.constants import CODE_COMP_SUBMISSION_SCORE_NAME
from whetstone.envs.code_comp.dataset import code_comp_task_hash
from whetstone.envs.code_comp.generation_graph.direct import (
    render_direct_frame,
)
from whetstone.envs.code_comp.input_arms import (
    direct_body,
    renamed_task,
    split_prompt,
)
from whetstone.envs.code_comp.modes.direct import DirectExperiment
from whetstone.envs.code_comp.mutation_surface import validate_instruction_body
from whetstone.envs.code_comp.reward.blended import reward_from_primary_score
from whetstone.envs.code_comp.scoring import (
    BatchScoringDeadlineExceeded,
    CodeBatchScorer,
    CodeScoringInput,
)
from whetstone.envs.code_comp.submission_result import (
    CodeSubmissionResult,
    submission_result_from_record,
    submission_result_to_record,
)
from whetstone.envs.generation_graph import (
    EVAL_NODE_ID,
    LLM_NODE_ID,
    PROMPT_EXTERNAL_INPUT,
)
from whetstone.envs.sampling import (
    EnvSplitSampling,
    validate_evaluation_role_for_split,
)
from whetstone.evaluation.aggregate import (
    Aggregate,
    RowValue,
    TaskRows,
    unweighted_task_mean,
)
from whetstone.evaluation.attribution import attribute_generated_row
from whetstone.evaluation.drivers.graph_execution import (
    METADATA_FAILURE_CODE_KEY,
    METADATA_PROMPT_KEY,
    METADATA_REDRIVABLE_KEY,
    METADATA_SUBMISSION_RESULT_KEY,
    GenerationNodeError,
    cache_marks_from_metadata,
    cache_marks_metadata,
    cancelled_row_state,
    external_input_field,
    graph_run_cancelled,
    metadata_prompt,
    node_error_failure_code,
    node_error_redrivable,
    node_error_row_state,
    node_text,
    require_node_error,
    require_node_success,
    run_generation_graph,
    single_node_input,
    telemetry_from_metadata,
    telemetry_metadata,
)
from whetstone.evaluation.drivers.row_common import (
    ProcessTask,
    RolloutOutput,
    _process_payload_hash,
    process_request_hash,
    remaining_phase_wall_seconds,
    start_phase_deadline,
)
from whetstone.evaluation.traces import (
    ExecutedComponentStep,
    ExecutedComponentTracePayload,
    ExecutedRowState,
    _llm_component_step,
    _llm_component_values,
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
from whetstone.experiment.graph.nodes import (
    EVAL_OUTPUT_FIELD,
    PROVIDER_GENERATION_OUTPUT_FIELD,
)
from whetstone.experiment.reward import Reward
from whetstone.provider.driver import TransportCall
from whetstone.provider.policy import ProviderExecutionPolicy


@dataclass(frozen=True, slots=True)
class DirectEvalResult:
    """One candidate's D1 evaluation over a split.

    ``submission_score_aggregate`` is the reward-bearing HumanEval Submission
    Score. ``reward`` is derived for an internal Evaluation Binding. Per-task
    vectors and
    outputs feed the CI, ledger, and sidecar.
    """

    submission_score_aggregate: Aggregate
    reward: Reward | None
    per_task_scores: tuple[float, ...]
    per_task_counts: tuple[int, ...]
    outputs: tuple[RolloutOutput, ...]
    request_identities: frozenset[str] = frozenset()


class DirectRowOutcome(BaseModel):
    """One (task, sample_index) direct generation's result + provenance."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
        strict=True,
        arbitrary_types_allowed=True,
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
    #: The accepted provider generation's stop reason and full failure
    #: diagnostic.
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
    code_submission_result: CodeSubmissionResult | None = None

    @field_validator("code_submission_result", mode="before")
    @classmethod
    def _coerce_code_submission_result(
        cls, value: object
    ) -> CodeSubmissionResult | None:
        if value is None or isinstance(value, CodeSubmissionResult):
            return value
        return submission_result_from_record(value)

    @field_serializer("code_submission_result", when_used="json")
    def _serialize_code_submission_result(
        self, value: CodeSubmissionResult | None
    ) -> dict[str, object] | None:
        record = submission_result_to_record(value)
        return None if record is None else record.model_dump(mode="json")

    @model_validator(mode="after")
    def _valid_outcome(self) -> DirectRowOutcome:
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


class DirectGeneratedRowOutcome(BaseModel):
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
    def _valid_generated_outcome(self) -> DirectGeneratedRowOutcome:
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


_D1_ROW_REQUEST_SCHEMA = "whetstone.envs.code_comp_direct_row_request/v2"
_D1_ROW_RESULT_SCHEMA = "whetstone.envs.code_comp_direct_row_result/v3"


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


class DirectRowRequest(BaseModel):
    """Complete serializable request and provenance for one D1 row."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    _submitted_request_hash: str | None = PrivateAttr(default=None)

    schema_name: Literal["whetstone.envs.code_comp_direct_row_request/v2"] = (
        _D1_ROW_REQUEST_SCHEMA
    )
    candidate_body: str
    candidate_id: str
    instance: ProcessTask
    humaneval_task: HumanEvalTaskPayload
    input_arm: str
    rename_token: str
    provider_call_config: ProviderCallConfig
    execution_policy: ProviderExecutionPolicy
    procedure_config_hash: str
    evaluation_binding_hash: IdentityHash
    logical_call_id: str
    sample_index: int
    drive_ordinal: int
    cache_phase: str
    cache_unit: str
    cache_root: str | None

    @property
    def request_hash(self) -> str:
        return self._submitted_request_hash or process_request_hash(self)

    @classmethod
    def from_process_payload(cls, payload: JsonValue) -> DirectRowRequest:
        """Validate a decoded JSON payload using Pydantic's JSON semantics."""
        request = cls.model_validate_json(json.dumps(payload))
        request._submitted_request_hash = _process_payload_hash(payload)
        return request


class DirectRowResult(BaseModel):
    """A D1 outcome cryptographically bound to its submitted request."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_name: Literal["whetstone.envs.code_comp_direct_row_result/v3"] = (
        _D1_ROW_RESULT_SCHEMA
    )
    request_hash: str
    outcome: DirectRowOutcome | DirectGeneratedRowOutcome

    @classmethod
    def from_process_payload(cls, payload: JsonValue) -> DirectRowResult:
        """Validate a decoded worker result using JSON semantics."""
        return cls.model_validate_json(json.dumps(payload))


type DirectRowJobFactory = Callable[[DirectRowRequest], ProcessJob]


def _request(config: ProviderCallConfig, prompt: str) -> ProviderCallRequest:
    return ProviderCallRequest(
        config=config,
        transcript=Transcript(
            messages=(PromptMessage(role=MessageRole.USER, content=prompt),)
        ),
    )


def _input_arm_text(
    experiment: DirectExperiment, instance: Instance
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


def drive_direct_row(
    *,
    experiment: DirectExperiment,
    candidate_body: str,
    instance: Instance,
    provider_call_config: ProviderCallConfig,
    execution_policy: ProviderExecutionPolicy,
    transport: TransportCall,
    scorer: Callable[..., CodeSubmissionResult] | None,
    logical_call_id: str,
    sample_index: int,
    drive_ordinal: int,
    cache: PromptResultCache | None,
    cache_phase: str,
    cache_unit: str,
) -> DirectRowOutcome | DirectGeneratedRowOutcome:
    """Run one row by interpreting the d1 generation graph via dr-graph.

    The generation executor delegates to ``execute_graph`` over the
    experiment's ``graph_config``; the direct provider call and the optional
    in-worker scoring run as node behaviors, carrying the wire prompt,
    telemetry, and cache marks on ``NodeOutput.metadata``. The graph-complete
    run result is then mapped onto the unchanged wire outcome models
    (``DirectRowOutcome`` / ``DirectGeneratedRowOutcome``).
    """
    input_arm, score_task = _input_arm_text(experiment, instance)
    try:
        prompt = render_direct_frame(candidate_body, input_arm=input_arm)
    except (KeyError, IndexError, ValueError):
        # The rendered prompt IS the graph external input, so a render
        # failure precedes the graph run.
        return DirectRowOutcome(
            submission_score=None,
            output_text=None,
            row_state=ExecutedRowState.FAILED,
            executed_component_steps=(),
            failure_code="d1_wrapper_render_error",
        )
    rd = experiment.generation_graph

    def _llm_node(
        node: NodeConfig, node_inputs: Mapping[str, Any]
    ) -> NodeOutput:
        wire_prompt = single_node_input(node, node_inputs)
        execution = execute_call(
            request=_request(provider_call_config, wire_prompt),
            policy=execution_policy,
            transport=transport,
            logical_call_id=logical_call_id,
            sample_index=sample_index,
            drive_ordinal=drive_ordinal,
            cache=cache,
            phase=cache_phase,
            unit=cache_unit,
        )
        result = execution.result
        telemetry = execution.telemetry()
        marks = execution.cache_marks()
        if not result.succeeded or result.provider_generation is None:
            raise GenerationNodeError(
                "D1 provider call failed",
                metadata={
                    **telemetry_metadata(telemetry),
                    **cache_marks_metadata(marks),
                    METADATA_FAILURE_CODE_KEY: failure_code_of(result),
                    METADATA_REDRIVABLE_KEY: (
                        is_transient_transport_failure(result)
                    ),
                },
            )
        return NodeOutput(
            values={
                PROVIDER_GENERATION_OUTPUT_FIELD: (
                    result.provider_generation.text
                )
            },
            metadata={
                METADATA_PROMPT_KEY: wire_prompt,
                **telemetry_metadata(telemetry),
                **cache_marks_metadata(marks),
            },
        )

    def _eval_node(
        node: NodeConfig, node_inputs: Mapping[str, Any]
    ) -> NodeOutput:
        output_text = single_node_input(node, node_inputs)
        if scorer is None:
            # Coordinator-side batch scoring: the graph run completes with an
            # unscored terminal output; the generated row is scored later.
            return NodeOutput(values={EVAL_OUTPUT_FIELD: None})
        submission = scorer(raw_submission=output_text, task=score_task)
        if not isinstance(submission, CodeSubmissionResult):
            raise TypeError("D1 scorer returned an unsupported result")
        record = submission_result_to_record(submission)
        record_json = (
            None if record is None else record.model_dump(mode="json")
        )
        if submission.score.infrastructure_unknown:
            raise GenerationNodeError(
                "D1 code eval infrastructure unknown",
                metadata={
                    METADATA_FAILURE_CODE_KEY: (
                        "code_eval_infrastructure_unknown"
                    ),
                    METADATA_SUBMISSION_RESULT_KEY: record_json,
                },
            )
        return NodeOutput(
            values={EVAL_OUTPUT_FIELD: submission.score.row_value},
            metadata={METADATA_SUBMISSION_RESULT_KEY: record_json},
        )

    def _run_node(
        node: NodeConfig, node_inputs: Mapping[str, Any]
    ) -> NodeOutput:
        if node.node_id == LLM_NODE_ID:
            return _llm_node(node, node_inputs)
        if node.node_id == EVAL_NODE_ID:
            return _eval_node(node, node_inputs)
        raise ValueError(f"unexpected D1 graph node {node.node_id!r}")

    run = run_generation_graph(
        graph=rd.graph_config,
        inputs={external_input_field(PROMPT_EXTERNAL_INPUT): prompt},
        run_node=_run_node,
    )
    return _direct_outcome_from_graph_run(run, scorer_deferred=scorer is None)


def _direct_outcome_from_graph_run(
    run: GraphRunResult,
    *,
    scorer_deferred: bool,
) -> DirectRowOutcome | DirectGeneratedRowOutcome:
    """Map one graph-complete D1 run onto its wire row outcome."""
    if graph_run_cancelled(run):
        # Cancellation attributes to the missing cell (pinned F4 table);
        # downstream nodes are BLOCKED, so the row carries no execution
        # output.
        return DirectRowOutcome(
            submission_score=None,
            output_text=None,
            row_state=cancelled_row_state(),
            executed_component_steps=(),
        )
    llm = run.outcomes[LLM_NODE_ID]
    if llm.status is NodeOutcomeStatus.ERROR:
        error = require_node_error(llm)
        telemetry = telemetry_from_metadata(error.metadata)
        marks = cache_marks_from_metadata(error.metadata)
        return DirectRowOutcome(
            submission_score=None,
            output_text=None,
            row_state=node_error_row_state(error),
            executed_component_steps=(),
            failure_code=node_error_failure_code(error),
            latency_s=telemetry.latency_s,
            provider_error=telemetry.provider_error,
            redrivable=node_error_redrivable(error),
            cache_hit=marks.cache_hit,
            cache_source_phase=marks.cache_source_phase,
            cache_source_unit=marks.cache_source_unit,
            cache_source_call_id=marks.cache_source_call_id,
            cache_source_at=marks.cache_source_at,
        )
    llm_output = require_node_success(llm)
    output_text = node_text(llm_output, field=PROVIDER_GENERATION_OUTPUT_FIELD)
    telemetry = telemetry_from_metadata(llm_output.metadata)
    marks = cache_marks_from_metadata(llm_output.metadata)
    executed_component_steps = (
        _llm_component_step(
            trace_index=0,
            component_id=LLM_NODE_ID,
            prompt=metadata_prompt(llm_output.metadata),
            generation=output_text,
        ),
    )
    ev = run.outcomes[EVAL_NODE_ID]
    if ev.status is NodeOutcomeStatus.ERROR:
        error = require_node_error(ev)
        submission_record = error.metadata.get(METADATA_SUBMISSION_RESULT_KEY)
        return DirectRowOutcome(
            submission_score=None,
            output_text=output_text,
            row_state=node_error_row_state(error),
            executed_component_steps=executed_component_steps,
            failure_code=node_error_failure_code(error),
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
            code_submission_result=(
                None
                if submission_record is None
                else submission_result_from_record(submission_record)
            ),
        )
    ev_output = require_node_success(ev)
    if scorer_deferred:
        return DirectGeneratedRowOutcome(
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
    submission_record = ev_output.metadata.get(METADATA_SUBMISSION_RESULT_KEY)
    if submission_record is None:
        raise AssertionError(
            "a scored D1 eval node must carry its submission result"
        )
    submission = submission_result_from_record(submission_record)
    if submission is None:
        raise AssertionError(
            "a recorded submission result must restore to a submission"
        )
    return DirectRowOutcome(
        submission_score=submission.score.row_value,
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
        code_submission_result=submission,
    )


def _direct_outcome_from_record(record: PartialCallRecord) -> DirectRowOutcome:
    """Rebuild the accepted outcome stored for one exact D1 request."""
    payload = ExecutedComponentTracePayload.from_json_value(
        record.observation_payload
    )
    if record.failed != (payload.row_state is ExecutedRowState.FAILED):
        raise ValueError("D1 partial row state conflicts with failed flag")
    return DirectRowOutcome(
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


def _should_redrive(outcome: DirectRowOutcome) -> bool:
    """Whether an ordinal-0 D1 result requires the bounded second attempt."""
    return outcome.failure_code == "runner_timeout" or outcome.redrivable


def _finish_generated_row(
    outcome: DirectGeneratedRowOutcome,
    submission: CodeSubmissionResult,
) -> DirectRowOutcome:
    if submission.score.infrastructure_unknown:
        return DirectRowOutcome(
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
            code_submission_result=submission,
        )
    return DirectRowOutcome(
        submission_score=submission.score.row_value,
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
        code_submission_result=submission,
    )


def _deadline(execution_policy: ProviderExecutionPolicy) -> float:
    from whetstone.execution.call_support import guard_deadline_seconds

    # A d1 row makes ONE wire call (direct generation), so the guard budgets a
    # single call's transport cap.
    return guard_deadline_seconds(execution_policy, wire_calls_per_unit=1)


def run_direct_eval(
    experiment: DirectExperiment,
    *,
    candidate_body: str,
    candidate_id: str,
    sampling: EnvSplitSampling,
    execution_policy: ProviderExecutionPolicy,
    row_job_factory: DirectRowJobFactory,
    evaluation_binding: EvaluationBinding,
    concurrency: int = DEFAULT_CONCURRENCY,
    max_wall_seconds: float | None = None,
    partial_log: PartialLog | None = None,
    cache: PromptResultCache | None = None,
    batch_scorer: CodeBatchScorer | None = None,
) -> DirectEvalResult:
    """Drive ``candidate_body`` over a D1 split.

    Fans out one serializable direct-generation process job per
    (task, sample_index),
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
    validate_instruction_body(candidate_body)
    instances = sampling.tasks
    num_samples = sampling.sample_plan.num_samples
    split_role = sampling.split_role
    rd = experiment.generation_graph
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
        outcome: DirectRowOutcome,
        *,
        request_hash: str,
        redrive_pending: bool,
    ) -> None:
        if partial_log is None:
            return
        partial_log.append(
            PartialCallRecord(
                phase=split_role,
                task_id=str(instance.id),
                unit=candidate_id,
                sample_index=index,
                request_hash=request_hash,
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
    ) -> DirectRowRequest:
        return DirectRowRequest(
            candidate_body=candidate_body,
            candidate_id=candidate_id,
            instance=ProcessTask.from_instance(instance),
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
            sample_index=index,
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
        for index in range(num_samples)
    }
    planned_request_identities = frozenset(
        request.request_hash
        for requests in requests_by_key.values()
        for request in requests
    )
    partial_records = index_partial_records(
        () if partial_log is None else partial_log.load(),
        phase=split_role,
        unit=candidate_id,
    )
    driven: dict[
        tuple[str, int], DirectRowOutcome | DirectGeneratedRowOutcome
    ] = {}
    completed_requests: dict[tuple[str, int], DirectRowRequest] = {}
    initial_requests: list[DirectRowRequest] = []
    resumed_redrive_requests: list[DirectRowRequest] = []
    for key, (ordinal_0, ordinal_1) in requests_by_key.items():
        decision = resolve_exact_resume(
            partial_records,
            task_id=key[0],
            sample_index=key[1],
            ordinal_0_request_hash=ordinal_0.request_hash,
            ordinal_1_request_hash=ordinal_1.request_hash,
        )
        if decision.record is not None:
            driven[key] = _direct_outcome_from_record(decision.record)
        if decision.drive_ordinal == 0:
            initial_requests.append(ordinal_0)
        elif decision.drive_ordinal == 1:
            resumed_redrive_requests.append(ordinal_1)

    def _spec(
        request: DirectRowRequest,
    ) -> CallSpec[
        tuple[str, int], DirectRowOutcome | DirectGeneratedRowOutcome
    ]:
        instance = by_instance[request.instance.id]

        def _decode(
            value: JsonValue,
        ) -> DirectRowOutcome | DirectGeneratedRowOutcome:
            result = DirectRowResult.from_process_payload(value)
            if result.request_hash != request.request_hash:
                raise ValueError(
                    "D1 row result does not match its submitted request"
                )
            return result.outcome

        return CallSpec(
            key=(request.instance.id, request.sample_index),
            job=row_job_factory(request),
            decode=_decode,
            deadline_seconds=_deadline(execution_policy),
            commit=lambda outcome: (
                _persist(
                    instance,
                    request.sample_index,
                    outcome,
                    request_hash=request.request_hash,
                    redrive_pending=(
                        request.drive_ordinal == 0 and _should_redrive(outcome)
                    ),
                )
                if isinstance(outcome, DirectRowOutcome)
                else None
            ),
        )

    phase_deadline = start_phase_deadline(max_wall_seconds)
    effective_concurrency = concurrency

    def _drive(
        requests: list[DirectRowRequest],
    ) -> dict[tuple[str, int], DirectRowOutcome | DirectGeneratedRowOutcome]:
        nonlocal effective_concurrency
        specs = [_spec(request) for request in requests]
        request_by_key = {
            (request.instance.id, request.sample_index): request
            for request in requests
        }
        pool = run_call_pool(
            specs,
            concurrency=effective_concurrency,
            is_rate_limited=lambda _o: False,
            max_wall_seconds=remaining_phase_wall_seconds(phase_deadline),
        )
        effective_concurrency = pool.effective_concurrency
        out: dict[
            tuple[str, int], DirectRowOutcome | DirectGeneratedRowOutcome
        ] = {}
        for res in pool.results:
            if res.status is FanoutStatus.COMPLETED and res.value is not None:
                out[res.key] = res.value
                completed_requests[res.key] = request_by_key[res.key]
            elif res.status is FanoutStatus.UNIT_TIMEOUT:
                request = request_by_key[res.key]
                outcome = DirectRowOutcome(
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
                    request_hash=request.request_hash,
                    redrive_pending=request.drive_ordinal == 0,
                )
            else:
                out[res.key] = DirectRowOutcome(
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
        if isinstance(outcome, DirectRowOutcome) and _should_redrive(outcome)
    ]
    if redrive_requests:
        redriven = _drive(redrive_requests)
        driven.update(
            (key, outcome)
            for key, outcome in redriven.items()
            if not isinstance(outcome, DirectRowOutcome) or not outcome.missing
        )

    generated_keys = [
        (str(instance.id), index)
        for instance in instances
        for index in range(num_samples)
        if isinstance(
            driven[(str(instance.id), index)], DirectGeneratedRowOutcome
        )
    ]
    if generated_keys:
        if batch_scorer is None:
            raise ValueError(
                "coordinator-side D1 scoring requires a batch_scorer"
            )
        scoring_inputs = tuple(
            CodeScoringInput(
                raw_submission=generated.output_text,
                task=_input_arm_text(experiment, by_instance[task_id])[1],
            )
            for task_id, _index in generated_keys
            for generated in (driven[(task_id, _index)],)
            if isinstance(generated, DirectGeneratedRowOutcome)
        )
        remaining_wall_seconds = remaining_phase_wall_seconds(phase_deadline)
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
                driven[key] = DirectRowOutcome(
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
            for key, submission in zip(generated_keys, scores, strict=True):
                generated = driven[key]
                if not isinstance(generated, DirectGeneratedRowOutcome):
                    raise AssertionError(
                        "generated D1 row changed before scoring"
                    )
                outcome = _finish_generated_row(generated, submission)
                driven[key] = outcome
                request = completed_requests[key]
                _persist(
                    by_instance[key[0]],
                    key[1],
                    outcome,
                    request_hash=request.request_hash,
                    redrive_pending=False,
                )

    submission_rows: list[tuple[str, list[RowValue]]] = []
    outputs: list[RolloutOutput] = []
    per_task_scores: list[float] = []
    per_task_counts: list[int] = []
    for task_index, instance in enumerate(instances):
        task_id = str(instance.id)
        task_hash = code_comp_task_hash(instance)
        task_submission_rows: list[RowValue] = []
        for index in range(num_samples):
            outcome = driven[(task_id, index)]
            if not isinstance(outcome, DirectRowOutcome):
                raise AssertionError("D1 row was not scored")
            task_submission_rows.append(
                attribute_generated_row(
                    row_state=outcome.row_state,
                    score=outcome.submission_score,
                    failure_code=outcome.failure_code,
                )
            )
            outputs.append(
                RolloutOutput(
                    candidate_id=candidate_id,
                    task_id=task_id,
                    task_index=task_index,
                    sample_index=index,
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
                    code_submission_result=outcome.code_submission_result,
                )
            )
        submission_rows.append((task_hash, task_submission_rows))
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
        aggregate_name=CODE_COMP_SUBMISSION_SCORE_NAME,
        graph_hash=graph_hash,
        evaluation_binding_hash=evaluation_binding_id,
        task_rows=tuple(
            TaskRows(
                task_hash=task_hash,
                rows=tuple(rows),
            )
            for task_hash, rows in submission_rows
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
    return DirectEvalResult(
        submission_score_aggregate=submission_score_aggregate,
        reward=reward,
        per_task_scores=tuple(per_task_scores),
        per_task_counts=tuple(per_task_counts),
        outputs=tuple(outputs),
        request_identities=planned_request_identities,
    )


__all__ = [
    "DirectEvalResult",
    "DirectGeneratedRowOutcome",
    "DirectRowJobFactory",
    "DirectRowOutcome",
    "drive_direct_row",
    "run_direct_eval",
]
