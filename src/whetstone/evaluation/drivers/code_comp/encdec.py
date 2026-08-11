from __future__ import annotations

import json
from collections import Counter
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
    field_serializer,
    field_validator,
    model_validator,
)
from whetstone_envs.core import Instance

from whetstone.core.identity import IdentityHash
from whetstone.core.roles import EvaluationRole
from whetstone.envs.code_comp.constants import (
    CODE_COMP_COMPRESSION_NAME,
    CODE_COMP_SUBMISSION_SCORE_NAME,
    DECODER_TEMPLATE,
)
from whetstone.envs.code_comp.dataset import humaneval_task_from_instance
from whetstone.envs.code_comp.generation_graph.encdec import (
    DECODER_NODE_ID,
    ENCODER_NODE_ID,
)
from whetstone.envs.code_comp.modes.encdec import EncDecExperiment
from whetstone.envs.code_comp.mutant.dataset import MutantRecord
from whetstone.envs.code_comp.mutation_surface import (
    render_encoder_frame,
    validate_instruction_body,
)
from whetstone.envs.code_comp.reward.blended import (
    code_comp_reward_from_blended,
    reward_from_primary_score,
)
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
from whetstone.evaluation.code.compression_selection import (
    select_compression_reference,
)
from whetstone.evaluation.compression import zstd_compressed_utf8_byte_length
from whetstone.evaluation.drivers.internal import (
    ProcessTask,
    RolloutOutput,
    _llm_component_step,
    _llm_component_values,
    _process_payload_hash,
    process_request_hash,
    remaining_phase_wall_seconds,
    start_phase_deadline,
)
from whetstone.evaluation.metrics.blended import blend_per_task
from whetstone.evaluation.metrics.compression_measurements import (
    compression_ratio_from_bytes,
)
from whetstone.evaluation.traces import (
    ExecutedComponentStep,
    ExecutedRowState,
    validate_executed_component_trace,
)
from whetstone.execution.call_support import CallTelemetry, call_telemetry
from whetstone.execution.fanout import (
    DEFAULT_CONCURRENCY,
    CallSpec,
    FanoutStatus,
    ProcessJob,
    run_call_pool,
)
from whetstone.execution.partials import PartialCallRecord, PartialLog
from whetstone.execution.prompt_cache import (
    CacheProvenance,
    PromptResultCache,
    execute_call,
    partial_cache_marks,
)
from whetstone.execution.resume import (
    index_partial_records,
    resolve_exact_resume,
)
from whetstone.experiment.binding import (
    EvaluationBinding,
    eval_config_reference,
)
from whetstone.experiment.graph.character_budget import (
    derive_character_bound,
)
from whetstone.experiment.reward import Reward
from whetstone.provider.driver import TransportCall
from whetstone.provider.policy import ProviderExecutionPolicy


@dataclass(frozen=True, slots=True)
class EncDecRowDiag:
    """Per-row diagnostics; exceeding the budget never clips or fails a row."""

    task_id: str
    sample_index: int
    metric_name: str
    metric_value: float | None
    compression: float | None
    failed: bool
    failure_code: str
    max_budget: int | None
    encoder_len: int | None
    over_budget: bool | None

    def as_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "sample_index": self.sample_index,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "compression": self.compression,
            "failed": self.failed,
            "failure_code": self.failure_code,
            "max_budget": self.max_budget,
            "encoder_len": self.encoder_len,
            "over_budget": self.over_budget,
        }


@dataclass(frozen=True, slots=True)
class EncDecEvalDiagnostics:
    """Aggregate health summary derived from typed row diagnostics."""

    present_rows: int
    failed_rows: int
    none_reason: str | None


@dataclass(frozen=True, slots=True)
class EncDecEvalResult:
    """One candidate's ED1-family evaluation over a split (dual aggregates).

    ``primary_aggregate`` is HumanEval Submission Score for ED1 and Fidelity to
    Mutant for ED1M. ``compression_aggregate`` is Mean Compression Ratio
    (reported, never the Reward). ``reward`` is derived from the primary
    aggregate when unblended. Per-task vectors + outputs feed the CI / ledger /
    sidecar; ``row_diags`` explains arm-level Nones.
    """

    primary_aggregate: Aggregate
    compression_aggregate: Aggregate
    reward: Reward | None
    #: The CI vector: the PER-TASK BLENDED reward when a blend config is set;
    #: otherwise the per-task primary mean. The paired bootstrap uses this.
    per_task_scores: tuple[float, ...]
    per_task_counts: tuple[int, ...]
    per_task_compression: tuple[float | None, ...]
    outputs: tuple[RolloutOutput, ...]
    #: The raw per-task primary mean, always reported separately even when
    #: ``per_task_scores`` carries the blend.
    per_task_primary: tuple[float, ...] = ()
    #: ed1m only: the per-task REPORTED attractor pull (fraction of
    #: discriminating inputs that snapped to canonical); ``None`` per task with
    #: no attractor sample; empty for ed1/QA.
    per_task_attractor: tuple[float | None, ...] = ()
    row_diags: tuple[EncDecRowDiag, ...] = ()
    request_identities: frozenset[str] = frozenset()
    concurrency_halved: bool = False
    deadline_reached: bool = False
    guard_timeouts: int = 0

    @property
    def diagnostics(self) -> EncDecEvalDiagnostics:
        """Summarize whether the evaluation produced usable rows."""
        present = self.primary_aggregate.rows_present
        failed = self.primary_aggregate.rows_failed
        none_reason: str | None = None
        if present == 0:
            codes = Counter(
                row.failure_code or "unknown_failure"
                for row in self.row_diags
                if row.failed
            )
            dominant, count = (
                codes.most_common(1)[0] if codes else ("no_present_rows", 0)
            )
            none_reason = f"0 present rows; {dominant} affected {count} row(s)"
        return EncDecEvalDiagnostics(
            present_rows=present,
            failed_rows=failed,
            none_reason=none_reason,
        )


def _validate_encdec_component_steps(
    steps: tuple[ExecutedComponentStep, ...],
    *,
    encoder_text: str | None,
    decoder_text: str | None,
) -> None:
    """Bind an ED1 execution prefix to its exact accepted generations."""
    validate_executed_component_trace(steps)
    expected_components = (ENCODER_NODE_ID, DECODER_NODE_ID)
    actual_components = tuple(step.component_id for step in steps)
    if actual_components != expected_components[: len(actual_components)]:
        raise ValueError("ED1 trace must be an encode/decode execution prefix")
    if encoder_text is None:
        if steps:
            raise ValueError("ED1 trace requires its encoder_text")
    else:
        if not steps:
            raise ValueError("ED1 encoder_text requires its executed step")
        _encoder_prompt, encoder_generation = _llm_component_values(
            steps[0], component_id=ENCODER_NODE_ID
        )
        if encoder_generation != encoder_text:
            raise ValueError("ED1 encoder trace must match encoder_text")
    if decoder_text is None:
        if len(steps) > 1:
            raise ValueError("ED1 decoder trace requires its decoder_text")
    else:
        if len(steps) != 2:
            raise ValueError("ED1 decoder_text requires its executed step")
        decoder_step = steps[1]
        decoder_prompt, decoder_generation = _llm_component_values(
            decoder_step, component_id=DECODER_NODE_ID
        )
        if (
            decoder_prompt
            != DECODER_TEMPLATE.format(encoder_output=encoder_text)
            or decoder_generation != decoder_text
        ):
            raise ValueError("ED1 decoder trace must match decoder execution")


class EncDecRowOutcome(BaseModel):
    """One (task, sample_index) generation's dual result + provenance.

    ``primary_value`` is ED1's HumanEval Submission Score or ED1M's fractional
    Fidelity to Mutant. ``attractor_pull`` is the ED1M reported contamination
    measurement (``None`` for ED1).
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
        strict=True,
        arbitrary_types_allowed=True,
    )

    primary_value: float | None
    compression_value: float | None
    encoder_text: str | None
    decoder_text: str | None
    row_state: ExecutedRowState
    executed_component_steps: tuple[ExecutedComponentStep, ...]
    failure_code: str = ""
    #: ed1m only: the reported attractor-pull for this row (``None`` for ed1).
    attractor_pull: float | None = None
    #: Summed encoder+decoder token usage (for spend reconciliation on the
    #: partial log); ``None`` when a call carried no usage block.
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    #: Summed encoder+decoder reasoning tokens and wall-clock latency.
    #: wall-clock latency for the row. ``None`` when the provider exposed no
    #: reasoning detail (never 0-conflated).
    reasoning_tokens: int | None = None
    latency_s: float | None = None
    #: Exceeding MAX_BUDGET is diagnostic and never clips or fails a row.
    max_budget: int | None = None
    encoder_len: int | None = None
    #: ``finish_reason`` is the decoder call's
    #: provider stop reason (the terminal output-bearing call -- a truncated
    #: ``length`` decode is distinguishable from a clean ``stop``);
    #: ``provider_error`` is the FULL typed diagnostic of whichever call failed
    #: (encoder or decoder). Both ``None`` when unknown.
    finish_reason: str | None = None
    provider_error: dict[str, object] | None = None
    #: True when this row failed on a TRANSIENT transport fault (timeout /
    #: stalled response / transport error / rate limit) whose driver-level
    #: semantic retries were exhausted -- eligible for ONE bounded re-drive.
    #: A deterministic failure (render error, provider rejection, infra-unknown
    #: scoring) is NOT redrivable (re-driving the same input will not change a
    #: deterministic "no").
    redrivable: bool = False
    #: A dual ed1 row is ``cache_hit`` True only
    #: when BOTH the encoder AND decoder calls were served from cache (no wire
    #: call at all this time -> latency nulled). If either leg was freshly
    #: driven the row's latency is genuine and ``cache_hit`` is False.
    #: ``cache_provenance`` refs the ENCODER entry's original source on a full
    #: hit (the row's primary provenance), else ``None``.
    cache_hit: bool = False
    cache_provenance: CacheProvenance | None = None
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
    def _valid_outcome(self) -> EncDecRowOutcome:
        _validate_encdec_component_steps(
            self.executed_component_steps,
            encoder_text=self.encoder_text,
            decoder_text=self.decoder_text,
        )
        if (self.row_state is ExecutedRowState.SUCCESS) != (
            self.primary_value is not None
        ):
            raise ValueError(
                "a successful row requires a primary value and an absent row "
                "forbids one"
            )
        if self.row_state is ExecutedRowState.MISSING:
            if (
                self.executed_component_steps
                or self.encoder_text is not None
                or self.decoder_text is not None
            ):
                raise ValueError(
                    "a missing ED1 row cannot contain execution output"
                )
        if (
            self.row_state is ExecutedRowState.SUCCESS
            and len(self.executed_component_steps) != 2
        ):
            raise ValueError("a successful ED1 row requires encode and decode")
        if self.cache_hit != (self.cache_provenance is not None):
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
    def over_budget(self) -> bool | None:
        if self.max_budget is None or self.encoder_len is None:
            return None
        return self.encoder_len > self.max_budget


class EncDecGeneratedRowOutcome(BaseModel):
    """A completed encode/decode row awaiting coordinator-side scoring."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
        strict=True,
    )

    compression_value: float | None
    encoder_text: str
    decoder_text: str
    executed_component_steps: tuple[ExecutedComponentStep, ...]
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    latency_s: float | None = None
    max_budget: int | None = None
    encoder_len: int
    finish_reason: str | None = None
    cache_hit: bool = False
    cache_provenance: CacheProvenance | None = None

    @model_validator(mode="after")
    def _valid_generated_outcome(self) -> EncDecGeneratedRowOutcome:
        _validate_encdec_component_steps(
            self.executed_component_steps,
            encoder_text=self.encoder_text,
            decoder_text=self.decoder_text,
        )
        if len(self.executed_component_steps) != 2:
            raise ValueError(
                "a generated ED1 row requires encode and decode steps"
            )
        if self.cache_hit != (self.cache_provenance is not None):
            raise ValueError(
                "cache_hit and original-entry provenance must be paired"
            )
        return self


class EncDecPartialPayload(BaseModel):
    """Strict ED1-family state stored beside generic partial-call fields."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
        strict=True,
    )

    compression_value: float | None
    encoder_text: str | None
    decoder_text: str | None
    attractor_pull: float | None
    max_budget: int | None
    encoder_len: int | None
    row_state: ExecutedRowState
    executed_component_steps: tuple[ExecutedComponentStep, ...]

    @model_validator(mode="after")
    def _valid_trace(self) -> EncDecPartialPayload:
        _validate_encdec_component_steps(
            self.executed_component_steps,
            encoder_text=self.encoder_text,
            decoder_text=self.decoder_text,
        )
        if (
            self.row_state is ExecutedRowState.MISSING
            and self.executed_component_steps
        ):
            raise ValueError("a missing ED1 partial cannot contain a trace")
        if (
            self.row_state is ExecutedRowState.SUCCESS
            and len(self.executed_component_steps) != 2
        ):
            raise ValueError("a successful ED1 partial requires both steps")
        return self

    @classmethod
    def from_json_value(
        cls, payload: JsonValue | None
    ) -> EncDecPartialPayload:
        """Validate a decoded partial payload using strict JSON semantics."""
        return cls.model_validate_json(json.dumps(payload))


_ED1_ROW_REQUEST_SCHEMA = "whetstone.envs.code_comp_encdec_row_request/v2"
_ED1_ROW_RESULT_SCHEMA = "whetstone.envs.code_comp_encdec_row_result/v3"


class EncDecRowRequest(BaseModel):
    """Complete serializable request and provenance for one ED1-family row."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    _submitted_request_hash: str | None = PrivateAttr(default=None)

    schema_name: Literal["whetstone.envs.code_comp_encdec_row_request/v2"] = (
        _ED1_ROW_REQUEST_SCHEMA
    )
    env_name: str
    dataset_revision: str
    primary_metric_name: str
    graph_hash: str
    candidate_template: str
    candidate_id: str
    instance: ProcessTask
    provider_call_config: ProviderCallConfig
    execution_policy: ProviderExecutionPolicy
    procedure_config_hash: str
    evaluation_binding_hash: IdentityHash
    budget_ratio: float | None
    logical_call_id: str
    sample_index: int
    drive_ordinal: int
    cache_phase: str
    cache_unit: str
    cache_root: str | None
    mutant_record: MutantRecord | None = None

    @model_validator(mode="after")
    def _valid_mutant_binding(self) -> EncDecRowRequest:
        from whetstone.envs.code_comp.modes.mutant import (
            CODE_COMP_MUTANT_FIDELITY_NAME,
        )

        if self.primary_metric_name == CODE_COMP_MUTANT_FIDELITY_NAME:
            if self.mutant_record is None:
                raise ValueError(
                    "an encdec_mutant row requires its authenticated mutant"
                )
            if self.mutant_record.content_hash != self.instance.id:
                raise ValueError(
                    "encdec_mutant identity does not match the row instance"
                )
        elif self.mutant_record is not None:
            raise ValueError("a non-mutant row forbids mutant oracle data")
        return self

    @property
    def request_hash(self) -> str:
        return self._submitted_request_hash or process_request_hash(self)

    @classmethod
    def from_process_payload(cls, payload: JsonValue) -> EncDecRowRequest:
        """Validate a decoded JSON payload using Pydantic's JSON semantics."""
        request = cls.model_validate_json(json.dumps(payload))
        request._submitted_request_hash = _process_payload_hash(payload)
        return request


class EncDecRowResult(BaseModel):
    """An ED1 outcome cryptographically bound to its submitted request."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_name: Literal["whetstone.envs.code_comp_encdec_row_result/v3"] = (
        _ED1_ROW_RESULT_SCHEMA
    )
    request_hash: str
    outcome: EncDecRowOutcome | EncDecGeneratedRowOutcome

    @classmethod
    def from_process_payload(cls, payload: JsonValue) -> EncDecRowResult:
        """Validate a decoded worker result using JSON semantics."""
        return cls.model_validate_json(json.dumps(payload))


type EncDecRowJobFactory = Callable[[EncDecRowRequest], ProcessJob]


def _request(config: ProviderCallConfig, prompt: str) -> ProviderCallRequest:
    return ProviderCallRequest(
        config=config,
        transcript=Transcript(
            messages=(PromptMessage(role=MessageRole.USER, content=prompt),)
        ),
    )


def _none_add(x: float | None, y: float | None) -> float | None:
    """Sum two optional numbers, None-preserving (None iff BOTH are None)."""
    if x is None and y is None:
        return None
    return (x or 0) + (y or 0)


def _sum_telemetry(a: CallTelemetry, b: CallTelemetry) -> CallTelemetry:
    """Sum two per-call telemetries into one row telemetry (enc + dec).

    Each field is ``None`` only if BOTH calls lacked it; otherwise the present
    values sum (a missing side counts as 0) so the row carries the enc+dec
    token/reasoning spend + total latency. Coverage-honest: a reasoning-free
    provider keeps ``reasoning_tokens=None``, never 0.
    """

    def _int(v: float | None) -> int | None:
        return None if v is None else int(v)

    return CallTelemetry(
        prompt_tokens=_int(_none_add(a.prompt_tokens, b.prompt_tokens)),
        completion_tokens=_int(
            _none_add(a.completion_tokens, b.completion_tokens)
        ),
        total_tokens=_int(_none_add(a.total_tokens, b.total_tokens)),
        reasoning_tokens=_int(
            _none_add(a.reasoning_tokens, b.reasoning_tokens)
        ),
        latency_s=_none_add(a.latency_s, b.latency_s),
    )


def _render_encoder(
    body: str, *, input_code: str, max_budget: int | None
) -> str:
    """Render the encoder prompt: the immutable frame around an instruction.

    ``body`` is the Mutation-Surface payload (the instruction ONLY); the
    input code, optional budget suffix, and punctuation come from
    ``ENCODER_FRAME``, so every candidate keeps them by construction. A body
    carrying a ``{placeholder}`` would raise here
    (KeyError/IndexError/ValueError) -> a per-row failure, but intake
    validation rejects such bodies first.
    """
    return render_encoder_frame(
        body, input_code=input_code, max_budget=max_budget
    )


def drive_encdec_row(
    *,
    experiment: EncDecExperiment,
    candidate_template: str,
    instance: Instance,
    provider_call_config: ProviderCallConfig,
    execution_policy: ProviderExecutionPolicy,
    transport: TransportCall,
    scorer: Callable[..., object] | None,
    logical_call_id: str,
    sample_index: int,
    drive_ordinal: int,
    cache: PromptResultCache | None,
    cache_phase: str,
    cache_unit: str,
) -> EncDecRowOutcome | EncDecGeneratedRowOutcome:
    """Run one encode/decode row, optionally scoring inside the worker."""
    input_code = instance.prompt_inputs["input_code"]
    rd = experiment.encdec_generation_graph
    assert rd is not None
    # A None budget rule omits MAX_BUDGET and the rendered budget suffix.
    rule = rd.budget_rule
    max_budget = (
        None
        if rule is None
        else derive_character_bound(rule, task_length=len(input_code))
    )
    try:
        encoder_prompt = _render_encoder(
            candidate_template, input_code=input_code, max_budget=max_budget
        )
    except (KeyError, IndexError, ValueError):
        return EncDecRowOutcome(
            primary_value=None,
            compression_value=None,
            encoder_text=None,
            decoder_text=None,
            row_state=ExecutedRowState.FAILED,
            executed_component_steps=(),
            failure_code="encoder_render_error",
            max_budget=max_budget,
            encoder_len=None,
        )
    enc_exec = execute_call(
        request=_request(provider_call_config, encoder_prompt),
        policy=execution_policy,
        transport=transport,
        logical_call_id=f"{logical_call_id}:enc",
        sample_index=sample_index,
        drive_ordinal=drive_ordinal,
        cache=cache,
        phase=cache_phase,
        unit=cache_unit,
    )
    enc = enc_exec.result
    if not enc.succeeded or enc.provider_generation is None:
        from whetstone.execution.call_support import (
            failure_code_of,
            is_transient_transport_failure,
        )

        # A failed call still carries whatever the transport measured (a
        # failed call has no usage, so tokens stay None -- coverage-honest --
        # but its accepted latency is real spend and is recorded).
        enc_tel = call_telemetry(enc)
        return EncDecRowOutcome(
            primary_value=None,
            compression_value=None,
            encoder_text=None,
            decoder_text=None,
            row_state=ExecutedRowState.FAILED,
            executed_component_steps=(),
            failure_code=failure_code_of(enc),
            prompt_tokens=enc_tel.prompt_tokens,
            completion_tokens=enc_tel.completion_tokens,
            total_tokens=enc_tel.total_tokens,
            reasoning_tokens=enc_tel.reasoning_tokens,
            latency_s=enc_tel.latency_s,
            max_budget=max_budget,
            encoder_len=None,
            provider_error=enc_tel.provider_error,
            redrivable=is_transient_transport_failure(enc),
        )
    encoder_text = enc.provider_generation.text
    encoder_len = len(encoder_text)
    executed_component_steps = (
        _llm_component_step(
            trace_index=0,
            component_id=ENCODER_NODE_ID,
            prompt=encoder_prompt,
            generation=encoder_text,
        ),
    )
    decoder_prompt = DECODER_TEMPLATE.format(encoder_output=encoder_text)
    dec_exec = execute_call(
        request=_request(provider_call_config, decoder_prompt),
        policy=execution_policy,
        transport=transport,
        logical_call_id=f"{logical_call_id}:dec",
        sample_index=sample_index,
        drive_ordinal=drive_ordinal,
        cache=cache,
        phase=cache_phase,
        unit=cache_unit,
    )
    dec = dec_exec.result
    # A DUAL row is a full cache hit only when BOTH legs were served (no wire
    # call this time); the encoder entry is the row's primary provenance.
    row_cache_hit = enc_exec.cache_hit and dec_exec.cache_hit
    row_cache_prov = enc_exec.provenance if row_cache_hit else None
    if not dec.succeeded or dec.provider_generation is None:
        from whetstone.execution.call_support import (
            failure_code_of,
            is_transient_transport_failure,
        )

        # The ENCODER leg succeeded, so its token spend is fully known and is
        # real spend regardless of the decoder's failure; summing in the failed
        # decoder's telemetry adds its accepted latency (it has no usage).
        dec_tel = call_telemetry(dec)
        fail_tel = _sum_telemetry(call_telemetry(enc), dec_tel)
        return EncDecRowOutcome(
            primary_value=None,
            compression_value=None,
            encoder_text=encoder_text,
            decoder_text=None,
            row_state=ExecutedRowState.FAILED,
            executed_component_steps=executed_component_steps,
            failure_code=failure_code_of(dec),
            prompt_tokens=fail_tel.prompt_tokens,
            completion_tokens=fail_tel.completion_tokens,
            total_tokens=fail_tel.total_tokens,
            reasoning_tokens=fail_tel.reasoning_tokens,
            latency_s=fail_tel.latency_s,
            max_budget=max_budget,
            encoder_len=encoder_len,
            provider_error=dec_tel.provider_error,
            redrivable=is_transient_transport_failure(dec),
        )
    decoder_text = dec.provider_generation.text
    executed_component_steps = (
        *executed_component_steps,
        _llm_component_step(
            trace_index=1,
            component_id=DECODER_NODE_ID,
            prompt=decoder_prompt,
            generation=decoder_text,
        ),
    )
    dec_tel = call_telemetry(dec)
    tel = _sum_telemetry(call_telemetry(enc), dec_tel)
    compression = _compression_ratio(encoder_text, input_code)

    if scorer is None:
        return EncDecGeneratedRowOutcome(
            compression_value=compression,
            encoder_text=encoder_text,
            decoder_text=decoder_text,
            executed_component_steps=executed_component_steps,
            prompt_tokens=tel.prompt_tokens,
            completion_tokens=tel.completion_tokens,
            total_tokens=tel.total_tokens,
            reasoning_tokens=tel.reasoning_tokens,
            latency_s=tel.latency_s,
            max_budget=max_budget,
            encoder_len=encoder_len,
            finish_reason=dec_tel.finish_reason,
            cache_hit=row_cache_hit,
            cache_provenance=row_cache_prov,
        )

    # Correctness (decoder output) -- may be an infrastructure-unknown, which
    # fails the row (never scored 0). ed1 scores the HumanEval test suite; ed1m
    # scores the mutant's per-input oracle (fidelity + attractor). Compression
    # (encoder output) is always computed (it does not depend on the sandbox).
    submission = _score_row(experiment, instance, decoder_text, scorer)
    if submission.score.infrastructure_unknown:
        return EncDecRowOutcome(
            primary_value=None,
            compression_value=None,
            encoder_text=encoder_text,
            decoder_text=decoder_text,
            row_state=ExecutedRowState.FAILED,
            executed_component_steps=executed_component_steps,
            failure_code="code_eval_infrastructure_unknown",
            prompt_tokens=tel.prompt_tokens,
            completion_tokens=tel.completion_tokens,
            total_tokens=tel.total_tokens,
            reasoning_tokens=tel.reasoning_tokens,
            latency_s=tel.latency_s,
            max_budget=max_budget,
            encoder_len=encoder_len,
            finish_reason=dec_tel.finish_reason,
            cache_hit=row_cache_hit,
            cache_provenance=row_cache_prov,
            code_submission_result=submission,
        )
    return EncDecRowOutcome(
        primary_value=submission.score.row_value,
        compression_value=compression,
        encoder_text=encoder_text,
        decoder_text=decoder_text,
        row_state=ExecutedRowState.SUCCESS,
        executed_component_steps=executed_component_steps,
        attractor_pull=submission.score.attractor_pull,
        prompt_tokens=tel.prompt_tokens,
        completion_tokens=tel.completion_tokens,
        total_tokens=tel.total_tokens,
        reasoning_tokens=tel.reasoning_tokens,
        latency_s=tel.latency_s,
        max_budget=max_budget,
        encoder_len=encoder_len,
        finish_reason=dec_tel.finish_reason,
        cache_hit=row_cache_hit,
        cache_provenance=row_cache_prov,
        code_submission_result=submission,
    )


def _score_row(
    experiment: EncDecExperiment,
    instance: Instance,
    decoder_text: str,
    scorer: Callable[..., object],
) -> CodeSubmissionResult:
    """Score one reconstruction: ed1 HumanEval suite OR ed1m mutant oracle.

    ed1m (an ``MutantExperiment``) scores the decoder output against the
    instance's mutant per-input oracle (fractional fidelity + reported
    attractor pull). Every other ed1 experiment scores the HumanEval test suite
    via the injected HumanEval submission scorer.
    """
    from whetstone.envs.code_comp.modes.mutant import (
        MutantExperiment,
        score_mutant_row,
    )

    if isinstance(experiment, MutantExperiment):
        return score_mutant_row(
            experiment,
            instance,
            decoder_text,
            scorer,
        )
    task = humaneval_task_from_instance(instance)
    submission = scorer(raw_submission=decoder_text, task=task)
    if not isinstance(submission, CodeSubmissionResult):
        raise TypeError("ED1 scorer returned an unsupported result")
    return submission


def _encdec_partial_payload(outcome: EncDecRowOutcome) -> EncDecPartialPayload:
    """Extract the strict ED1-family portion of a partial observation."""
    return EncDecPartialPayload(
        compression_value=outcome.compression_value,
        encoder_text=outcome.encoder_text,
        decoder_text=outcome.decoder_text,
        attractor_pull=outcome.attractor_pull,
        max_budget=outcome.max_budget,
        encoder_len=outcome.encoder_len,
        row_state=outcome.row_state,
        executed_component_steps=outcome.executed_component_steps,
    )


def _encdec_outcome_from_record(record: PartialCallRecord) -> EncDecRowOutcome:
    """Rebuild the accepted outcome stored for one exact ED1 request."""
    payload = EncDecPartialPayload.from_json_value(record.observation_payload)
    if record.output_text != payload.decoder_text:
        raise ValueError(
            "ED1 partial output_text does not match its observation payload"
        )
    if record.failed != (payload.row_state is ExecutedRowState.FAILED):
        raise ValueError("ED1 partial row state conflicts with failed flag")
    return EncDecRowOutcome(
        primary_value=None if record.score is None else float(record.score),
        compression_value=payload.compression_value,
        encoder_text=payload.encoder_text,
        decoder_text=payload.decoder_text,
        row_state=payload.row_state,
        executed_component_steps=payload.executed_component_steps,
        failure_code=record.failure_code,
        attractor_pull=payload.attractor_pull,
        prompt_tokens=record.prompt_tokens,
        completion_tokens=record.completion_tokens,
        total_tokens=record.total_tokens,
        reasoning_tokens=record.reasoning_tokens,
        latency_s=record.latency_s,
        max_budget=payload.max_budget,
        encoder_len=payload.encoder_len,
        finish_reason=record.finish_reason,
        provider_error=record.provider_error,
        redrivable=record.redrive_pending,
    )


def _should_redrive(outcome: EncDecRowOutcome) -> bool:
    """Whether an ordinal-0 ED1 result requires the bounded second attempt."""
    return outcome.failure_code == "runner_timeout" or outcome.redrivable


def _finish_generated_row(
    outcome: EncDecGeneratedRowOutcome,
    submission: CodeSubmissionResult,
) -> EncDecRowOutcome:
    if submission.score.infrastructure_unknown:
        return EncDecRowOutcome(
            primary_value=None,
            compression_value=outcome.compression_value,
            encoder_text=outcome.encoder_text,
            decoder_text=outcome.decoder_text,
            row_state=ExecutedRowState.FAILED,
            executed_component_steps=outcome.executed_component_steps,
            failure_code="code_eval_infrastructure_unknown",
            prompt_tokens=outcome.prompt_tokens,
            completion_tokens=outcome.completion_tokens,
            total_tokens=outcome.total_tokens,
            reasoning_tokens=outcome.reasoning_tokens,
            latency_s=outcome.latency_s,
            max_budget=outcome.max_budget,
            encoder_len=outcome.encoder_len,
            finish_reason=outcome.finish_reason,
            cache_hit=outcome.cache_hit,
            cache_provenance=outcome.cache_provenance,
            code_submission_result=submission,
        )
    return EncDecRowOutcome(
        primary_value=submission.score.row_value,
        compression_value=outcome.compression_value,
        encoder_text=outcome.encoder_text,
        decoder_text=outcome.decoder_text,
        row_state=ExecutedRowState.SUCCESS,
        executed_component_steps=outcome.executed_component_steps,
        attractor_pull=submission.score.attractor_pull,
        prompt_tokens=outcome.prompt_tokens,
        completion_tokens=outcome.completion_tokens,
        total_tokens=outcome.total_tokens,
        reasoning_tokens=outcome.reasoning_tokens,
        latency_s=outcome.latency_s,
        max_budget=outcome.max_budget,
        encoder_len=outcome.encoder_len,
        finish_reason=outcome.finish_reason,
        cache_hit=outcome.cache_hit,
        cache_provenance=outcome.cache_provenance,
        code_submission_result=submission,
    )


def _compression_ratio(encoder_text: str, input_code: str) -> float | None:
    """The zstd-19 Compression Ratio of the ENCODER output vs the reference.

    Reference = the exact ``gt_code_wo_comments`` bytes (= ``input_code``
    here).
    ``None`` for a zero-length reference (explicit zero-denominator, never
    coerced).
    """
    reference = select_compression_reference(_RefView(input_code))
    length = zstd_compressed_utf8_byte_length(encoder_text)
    return compression_ratio_from_bytes(
        numerator_bytes=length, reference=reference
    )


@dataclass(frozen=True, slots=True)
class _RefView:
    """A minimal ``ExperimentTaskView`` (only ``gt_code_wo_comments``)."""

    gt_code_wo_comments: str


def _primary_metric_name(experiment: EncDecExperiment) -> str:
    """Return the concrete primary metric identity for this ED1-family env."""
    from whetstone.envs.code_comp.modes.mutant import (
        CODE_COMP_MUTANT_FIDELITY_NAME,
        MutantExperiment,
    )

    if isinstance(experiment, MutantExperiment):
        return CODE_COMP_MUTANT_FIDELITY_NAME
    return CODE_COMP_SUBMISSION_SCORE_NAME


def run_encdec_eval(
    experiment: EncDecExperiment,
    *,
    candidate_template: str,
    candidate_id: str,
    sampling: EnvSplitSampling,
    execution_policy: ProviderExecutionPolicy,
    row_job_factory: EncDecRowJobFactory,
    evaluation_binding: EvaluationBinding,
    concurrency: int = DEFAULT_CONCURRENCY,
    max_wall_seconds: float | None = None,
    partial_log: PartialLog | None = None,
    cache: PromptResultCache | None = None,
    batch_scorer: CodeBatchScorer | None = None,
) -> EncDecEvalResult:
    """Drive ``candidate_template`` over an ed1 split -> dual aggregates.

    Fans out one serializable enc->dec process job per (task, sample_index),
    batch-scores generated ED1 submissions in the coordinator when requested,
    reduces to the primary + compression aggregates, derives Reward from the
    primary aggregate when unblended, and collects per-row outputs (encoder +
    decoder text) for the dual-score sidecar.

    Incremental persistence: when a ``partial_log`` is given, each (task,
    repeat) row is appended by its parent-owned commit the instant it
    completes, so a crash/interrupt mid-drive keeps every finished row on
    disk. A resumed drive restores only an exact row-request identity instead
    of re-driving and re-paying; a pending ordinal-0 record resumes at the
    exact ordinal-1 request.

    ED1M retains its mutant-specific in-worker scorer. A generated ED1 row is
    not written to the partial log until coordinator scoring completes, so a
    crash in that interval may repeat generation; the prompt cache remains the
    available no-wire replay path.
    """
    validate_instruction_body(candidate_template)
    instances = sampling.tasks
    num_samples = sampling.sample_plan.num_samples
    split_role = sampling.split_role
    rd = experiment.encdec_generation_graph
    assert rd is not None
    graph_hash = rd.graph_hash
    primary_metric_name = _primary_metric_name(experiment)
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
        outcome: EncDecRowOutcome,
        *,
        request_hash: str,
        redrive_pending: bool,
    ) -> None:
        if partial_log is None:
            return
        marks = partial_cache_marks(
            outcome.cache_hit, outcome.cache_provenance
        )
        partial_log.append(
            PartialCallRecord(
                phase=split_role,
                task_id=str(instance.id),
                unit=candidate_id,
                sample_index=index,
                request_hash=request_hash,
                redrive_pending=redrive_pending,
                score=outcome.primary_value,
                failed=outcome.failed,
                failure_code=outcome.failure_code,
                split_role=split_role,
                prompt_tokens=outcome.prompt_tokens,
                completion_tokens=outcome.completion_tokens,
                total_tokens=outcome.total_tokens,
                reasoning_tokens=outcome.reasoning_tokens,
                latency_s=None if marks.cache_hit else outcome.latency_s,
                output_text=outcome.decoder_text,
                observation_payload=_encdec_partial_payload(
                    outcome
                ).model_dump(mode="json"),
                finish_reason=outcome.finish_reason,
                provider_error=outcome.provider_error,
                cache_hit=marks.cache_hit,
                cache_source_phase=marks.cache_source_phase,
                cache_source_unit=marks.cache_source_unit,
                cache_source_call_id=marks.cache_source_call_id,
                cache_source_at=marks.cache_source_at,
            )
        )

    def _row_request(
        instance: Instance,
        index: int,
        *,
        drive_ordinal: int,
    ) -> EncDecRowRequest:
        from whetstone.envs.code_comp.modes.mutant import MutantExperiment

        mutant_record = (
            experiment.mutants[str(instance.id)]
            if isinstance(experiment, MutantExperiment)
            else None
        )
        return EncDecRowRequest(
            env_name=rd.env_name,
            dataset_revision=experiment.dataset_revision,
            primary_metric_name=primary_metric_name,
            graph_hash=graph_hash,
            candidate_template=candidate_template,
            candidate_id=candidate_id,
            instance=ProcessTask.from_instance(instance),
            provider_call_config=rd.provider_call_config,
            execution_policy=execution_policy,
            procedure_config_hash=rd.procedure_config_hash,
            evaluation_binding_hash=evaluation_binding_id,
            budget_ratio=rd.budget_ratio,
            logical_call_id=f"{candidate_id}:{instance.id}#{index}",
            sample_index=index,
            drive_ordinal=drive_ordinal,
            cache_phase=split_role,
            cache_unit=candidate_id,
            cache_root=None if cache is None else str(cache.root),
            mutant_record=mutant_record,
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
        tuple[str, int], EncDecRowOutcome | EncDecGeneratedRowOutcome
    ] = {}
    completed_requests: dict[tuple[str, int], EncDecRowRequest] = {}
    initial_requests: list[EncDecRowRequest] = []
    resumed_redrive_requests: list[EncDecRowRequest] = []
    for key, (ordinal_0, ordinal_1) in requests_by_key.items():
        decision = resolve_exact_resume(
            partial_records,
            task_id=key[0],
            sample_index=key[1],
            ordinal_0_request_hash=ordinal_0.request_hash,
            ordinal_1_request_hash=ordinal_1.request_hash,
        )
        if decision.record is not None:
            driven[key] = _encdec_outcome_from_record(decision.record)
        if decision.drive_ordinal == 0:
            initial_requests.append(ordinal_0)
        elif decision.drive_ordinal == 1:
            resumed_redrive_requests.append(ordinal_1)

    def _spec(
        request: EncDecRowRequest,
    ) -> CallSpec[
        tuple[str, int], EncDecRowOutcome | EncDecGeneratedRowOutcome
    ]:
        instance = by_instance[request.instance.id]

        def _decode(
            value: JsonValue,
        ) -> EncDecRowOutcome | EncDecGeneratedRowOutcome:
            result = EncDecRowResult.from_process_payload(value)
            if result.request_hash != request.request_hash:
                raise ValueError(
                    "ED1 row result does not match its submitted request"
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
                if isinstance(outcome, EncDecRowOutcome)
                else None
            ),
        )

    phase_deadline = start_phase_deadline(max_wall_seconds)
    effective_concurrency = concurrency

    def _drive(
        requests: list[EncDecRowRequest],
    ) -> tuple[
        dict[tuple[str, int], EncDecRowOutcome | EncDecGeneratedRowOutcome],
        bool,
        bool,
        int,
    ]:
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
            tuple[str, int], EncDecRowOutcome | EncDecGeneratedRowOutcome
        ] = {}
        for res in pool.results:
            if res.status is FanoutStatus.COMPLETED and res.value is not None:
                out[res.key] = res.value
                completed_requests[res.key] = request_by_key[res.key]
            elif res.status is FanoutStatus.UNIT_TIMEOUT:
                # A runner-guard timeout: the row hung past its (2-call) guard.
                # Marked redrivable so ONE bounded re-drive gets a fresh try
                # before it lands as a failed row (a single hung row must not
                # kill an anchor arm under the FAIL policy).
                request = request_by_key[res.key]
                outcome = EncDecRowOutcome(
                    primary_value=None,
                    compression_value=None,
                    encoder_text=None,
                    decoder_text=None,
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
                out[res.key] = EncDecRowOutcome(
                    primary_value=None,
                    compression_value=None,
                    encoder_text=None,
                    decoder_text=None,
                    row_state=ExecutedRowState.MISSING,
                    executed_component_steps=(),
                )
        return (
            out,
            pool.concurrency_halved,
            pool.deadline_reached,
            pool.guard_timeouts,
        )

    first_driven, halved_1, deadline_1, guard_1 = _drive(initial_requests)
    driven.update(first_driven)

    # --- ONE bounded re-drive of timed-out / transient-transport rows. ---
    # A runner-guard timeout or a TERMINAL transient transport failure (enc or
    # dec) is re-driven exactly once before it lands as a failed row, so a
    # single flaky observation never fails the whole ed1 arm under FAIL policy
    # (the eval:ed1:a1 kill). A deterministic failure (render error, provider
    # rejection, infra-unknown scoring) is NOT re-driven. Mirrors the QA arm's
    # bounded re-drive; the re-drive persists its own partial record.
    redrive_requests = resumed_redrive_requests + [
        requests_by_key[key][1]
        for key, outcome in first_driven.items()
        if isinstance(outcome, EncDecRowOutcome) and _should_redrive(outcome)
    ]
    halved_2 = deadline_2 = False
    guard_2 = 0
    if redrive_requests:
        redriven, halved_2, deadline_2, guard_2 = _drive(redrive_requests)
        driven.update(
            (key, outcome)
            for key, outcome in redriven.items()
            if not isinstance(outcome, EncDecRowOutcome) or not outcome.missing
        )
    concurrency_halved = halved_1 or halved_2
    deadline_reached = deadline_1 or deadline_2
    guard_timeouts = guard_1 + guard_2

    generated_keys = [
        (str(instance.id), index)
        for instance in instances
        for index in range(num_samples)
        if isinstance(
            driven[(str(instance.id), index)], EncDecGeneratedRowOutcome
        )
    ]
    if generated_keys:
        from whetstone.envs.code_comp.modes.mutant import MutantExperiment

        if isinstance(experiment, MutantExperiment):
            raise ValueError("ED1M rows require their mutant scorer in-worker")
        if batch_scorer is None:
            raise ValueError(
                "coordinator-side ED1 scoring requires a batch_scorer"
            )
        scoring_inputs = tuple(
            CodeScoringInput(
                raw_submission=generated.decoder_text,
                task=humaneval_task_from_instance(by_instance[task_id]),
            )
            for task_id, index in generated_keys
            for generated in (driven[(task_id, index)],)
            if isinstance(generated, EncDecGeneratedRowOutcome)
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
                driven[key] = EncDecRowOutcome(
                    primary_value=None,
                    compression_value=None,
                    encoder_text=None,
                    decoder_text=None,
                    row_state=ExecutedRowState.MISSING,
                    executed_component_steps=(),
                    max_budget=None,
                    encoder_len=None,
                )
        else:
            if len(scores) != len(generated_keys):
                raise ValueError(
                    "ED1 batch scorer returned the wrong result count"
                )
            for key, submission in zip(generated_keys, scores, strict=True):
                generated = driven[key]
                if not isinstance(generated, EncDecGeneratedRowOutcome):
                    raise AssertionError(
                        "generated ED1 row changed before scoring"
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

    # Assemble per-task rows (primary + compression) + outputs, instance/repeat
    # order.
    primary_rows: list[tuple[str, list[RowValue]]] = []
    comp_rows: list[tuple[str, list[RowValue]]] = []
    outputs: list[RolloutOutput] = []
    row_diags: list[EncDecRowDiag] = []
    per_task_scores: list[float] = []
    per_task_counts: list[int] = []
    per_task_compression: list[float | None] = []
    per_task_attractor: list[float | None] = []
    for task_index, instance in enumerate(instances):
        task_id = str(instance.id)
        task_primary_rows: list[RowValue] = []
        c_rows: list[RowValue] = []
        comp_vals: list[float] = []
        attr_vals: list[float] = []
        for index in range(num_samples):
            outcome = driven[(task_id, index)]
            if not isinstance(outcome, EncDecRowOutcome):
                raise AssertionError("ED1 row was not scored")
            if outcome.attractor_pull is not None:
                attr_vals.append(outcome.attractor_pull)
            row_diags.append(
                EncDecRowDiag(
                    task_id=task_id,
                    sample_index=index,
                    metric_name=primary_metric_name,
                    metric_value=outcome.primary_value,
                    compression=outcome.compression_value,
                    failed=outcome.failed,
                    failure_code=outcome.failure_code,
                    max_budget=outcome.max_budget,
                    encoder_len=outcome.encoder_len,
                    over_budget=outcome.over_budget,
                )
            )
            if outcome.missing:
                task_primary_rows.append(RowValue(missing=True))
            elif outcome.failed or outcome.primary_value is None:
                task_primary_rows.append(RowValue(failed=True))
            else:
                task_primary_rows.append(
                    RowValue(value=float(outcome.primary_value))
                )
            if outcome.missing:
                c_rows.append(RowValue(missing=True))
            elif outcome.compression_value is None:
                c_rows.append(
                    RowValue(failed=True)
                    if outcome.failed
                    else RowValue(invalid=True)
                )
            else:
                c_rows.append(RowValue(value=float(outcome.compression_value)))
                comp_vals.append(float(outcome.compression_value))
            outputs.append(
                RolloutOutput(
                    candidate_id=candidate_id,
                    task_id=task_id,
                    task_index=task_index,
                    sample_index=index,
                    row_state=outcome.row_state,
                    executed_component_steps=outcome.executed_component_steps,
                    output_text=_row_output_text(outcome),
                    score=(
                        None
                        if outcome.primary_value is None
                        else float(outcome.primary_value)
                    ),
                    failure_code=outcome.failure_code,
                    finish_reason=outcome.finish_reason,
                    provider_error=outcome.provider_error,
                    max_budget=outcome.max_budget,
                    over_budget=outcome.over_budget,
                    code_submission_result=outcome.code_submission_result,
                )
            )
        primary_rows.append((task_id, task_primary_rows))
        comp_rows.append((task_id, c_rows))
        # Per-task primary mean + observation weight for the paired CI,
        # computed identically to the QA lane
        # (``internal_eval._per_task_score`` / ``_per_task_count``), so ED1
        # skipped rows feed the paired/pooled
        # bootstrap exactly as c18's SKIP lane does: the mean divides by the
        # planned repeats (an absent/failed row counts 0), and the weight is
        # the planned repeat count -- not the present-only count, which would
        # mis-weight a task with skipped rows when escalation pools repeats.
        total = sum(
            float(r.value or 0.0) if r.is_present else 0.0
            for r in task_primary_rows
        )
        per_task_scores.append(
            total / len(task_primary_rows) if task_primary_rows else 0.0
        )
        per_task_counts.append(len(task_primary_rows))
        per_task_compression.append(
            sum(comp_vals) / len(comp_vals) if comp_vals else None
        )
        per_task_attractor.append(
            sum(attr_vals) / len(attr_vals) if attr_vals else None
        )

    primary_aggregate = unweighted_task_mean(
        aggregate_name=primary_metric_name,
        graph_hash=graph_hash,
        evaluation_binding_hash=evaluation_binding_id,
        task_rows=tuple(
            TaskRows(
                task_hash=task_hash,
                rows=tuple(rows),
            )
            for task_hash, rows in primary_rows
        ),
        plan=sampling.evaluation_matrix_plan,
    )
    compression_aggregate = unweighted_task_mean(
        aggregate_name=CODE_COMP_COMPRESSION_NAME,
        graph_hash=graph_hash,
        evaluation_binding_hash=evaluation_binding_id,
        task_rows=tuple(
            TaskRows(
                task_hash=task_hash,
                rows=tuple(rows),
            )
            for task_hash, rows in comp_rows
        ),
        plan=sampling.evaluation_matrix_plan,
    )

    # ED1 always uses the PER-TASK blended reward for internal selection and
    # the official comparison vector (primary score + compression are always
    # reported separately). ED1M shares this driver and may retain its
    # primary-only behavior. The blend is composed per task, so the paired
    # bootstrap operates on blended rewards exactly as env_exact_match does for
    # QA.
    blend_config = experiment.blend_config
    primary_scores = tuple(per_task_scores)
    if blend_config is not None:
        reward_scores = blend_per_task(
            primary_scores, tuple(per_task_compression), blend_config
        )
    else:
        reward_scores = primary_scores

    if evaluation_binding.role is EvaluationRole.INTERNAL:
        if blend_config is not None:
            # The aggregate blended reward = MEAN over tasks of the per-task
            # blended rewards. GATED on the primary aggregate's completeness:
            # ``per_task_scores`` folds an absent/failed row in as 0.0, so a
            # raw mean would silently certify an INCOMPLETE evaluation that the
            # primary aggregate refuses (value None under PROPAGATE). No
            # primary value -> no blended value -> the FAIL policy raises.
            mean_blended = (
                sum(reward_scores) / len(reward_scores)
                if reward_scores
                and primary_aggregate.aggregation_output.value is not None
                else None
            )
            reward = code_comp_reward_from_blended(
                blend_config,
                env_name=experiment.env_name,
                blended=mean_blended,
                evidence_refs=(
                    primary_aggregate.record_ref(),
                    compression_aggregate.record_ref(),
                ),
            )
        else:
            reward = reward_from_primary_score(
                experiment.reward_policy,
                primary_score=primary_aggregate.aggregation_output.value,
                evidence_refs=(primary_aggregate.record_ref(),),
            )
    else:
        reward = None
    return EncDecEvalResult(
        primary_aggregate=primary_aggregate,
        compression_aggregate=compression_aggregate,
        reward=reward,
        # The CI vector: blended reward per task when blending, else primary.
        per_task_scores=reward_scores,
        per_task_counts=tuple(per_task_counts),
        per_task_compression=tuple(per_task_compression),
        per_task_primary=primary_scores,
        per_task_attractor=tuple(per_task_attractor),
        outputs=tuple(outputs),
        row_diags=tuple(row_diags),
        request_identities=planned_request_identities,
        concurrency_halved=concurrency_halved,
        deadline_reached=deadline_reached,
        guard_timeouts=guard_timeouts,
    )


def _row_output_text(outcome: EncDecRowOutcome) -> str | None:
    """The sidecar output text: the encoder + decoder outputs (both kept)."""
    if outcome.encoder_text is None and outcome.decoder_text is None:
        return None
    return (
        f"ENCODER:\n{outcome.encoder_text or ''}\n\n"
        f"DECODER:\n{outcome.decoder_text or ''}"
    )


#: An ed1 row makes TWO sequential wire calls (encoder THEN decoder), so its
#: runner guard must budget both calls' transport caps -- otherwise the guard
#: (sized for one call) trips mid-decoder the instant the encoder used any
#: time, masquerading as a transport-bound regression (the eval:ed1:a1 hang).
_ED1_WIRE_CALLS_PER_ROW = 2


def _deadline(execution_policy: ProviderExecutionPolicy) -> float:
    from whetstone.execution.call_support import guard_deadline_seconds

    return guard_deadline_seconds(
        execution_policy, wire_calls_per_unit=_ED1_WIRE_CALLS_PER_ROW
    )


__all__ = [
    "EncDecEvalDiagnostics",
    "EncDecEvalResult",
    "EncDecGeneratedRowOutcome",
    "EncDecPartialPayload",
    "EncDecRowDiag",
    "EncDecRowJobFactory",
    "EncDecRowOutcome",
    "drive_encdec_row",
    "run_encdec_eval",
]
