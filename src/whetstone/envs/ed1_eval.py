"""The ed1 encoder->decoder->code-eval drive (dual scores).

Drives one candidate over an ed1 split through serializable process jobs,
running the three-node rollout per (task, repeat):

1. render the candidate's Mutation-Surface ENCODER template against the task's
   ``INPUT_CODE`` (= ``gt_code_wo_comments``) and the per-task character budget
   ``MAX_BUDGET = round(budget_ratio * chars(input_code))`` (guidance, not a
   hard clip -- the design: "the budget steers, nothing clips");
2. call the encoder (the shared enc/dec route);
3. render the fixed DECODER template against the ENCODER output and call the
   decoder;
4. score the DECODER output for the environment's primary metric (ED1:
   HumanEval Submission Score; ED1M: Fidelity to Mutant) and the ENCODER output
   for compression (whetstone zstd-19 Compression Ratio vs
   ``gt_code_wo_comments``).

It reduces to the environment's primary aggregate plus Mean Compression Ratio
using the shared two-stage unweighted task mean, and returns both with per-row
outputs. Nothing here makes a live paid call by itself: the transport and
code-eval scorer are injected.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from dr_code.mutants.dataset import MutantRecord
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
    model_validator,
)
from whetstone_envs.core import Instance

from whetstone.code_eval.aggregate import (
    RolloutAggregate,
    RowValue,
    TaskRows,
    unweighted_task_mean,
)
from whetstone.code_eval.compression_selection import (
    select_compression_reference,
)
from whetstone.code_eval.scoring import (
    compressed_description_length_bytes,
    compression_ratio_value,
)
from whetstone.envs.ed1 import (
    DECODER_TEMPLATE,
    ED1_COMPRESSION_NAME,
    ED1_SUBMISSION_SCORE_NAME,
    Ed1Experiment,
    ed1_reward_from_blended,
    humaneval_task_from_instance,
    render_encoder_frame,
    reward_from_primary_score,
    validate_ed1_body,
)
from whetstone.envs.ed1_blended import blend_per_task
from whetstone.envs.ed1_scoring import CodeScore
from whetstone.envs.encdec_rollout import DECODER_NODE_ID, ENCODER_NODE_ID
from whetstone.envs.internal_eval import (
    ExecutedComponentStep,
    ExecutedRowState,
    ProcessInstance,
    RolloutOutput,
    _canonical_provider_call_config_payload,
    _llm_component_step,
    _llm_component_values,
    _process_payload_identity,
    process_request_identity,
    remaining_phase_wall_seconds,
    start_phase_deadline,
    validate_executed_component_trace,
)
from whetstone.envs.partial_resume import (
    index_partial_records,
    resolve_exact_resume,
)
from whetstone.envs.sampling import (
    EnvSplitSampling,
    validate_evaluation_role_for_split,
)
from whetstone.evaluation_role import EvaluationRole
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
from whetstone.graph.character_budget import CharacterBudgetRule
from whetstone.optimization.identity import IdentityHash
from whetstone.optimization.reward import Reward
from whetstone.optimization.schema import (
    EvaluationBinding,
    eval_config_reference,
)
from whetstone.provider.driver import TransportCall
from whetstone.provider.policy import ProviderExecutionPolicy


@dataclass(frozen=True, slots=True)
class Ed1RowDiag:
    """One (task, repeat) row's diagnostic record for the pilot artifact.

    Explains an arm-level ``None`` from disk: the typed ``failure_code`` (empty
    when the row succeeded), the primary/compression scalars, the per-task
    ``max_budget`` the encoder was told to respect, the actual encoder-output
    length, and the derived ``over_budget`` flag (an over-budget row is NEVER
    clipped or failed -- the budget only steers, so this is diagnostic only).
    """

    instance_id: str
    repeat: int
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
            "instance_id": self.instance_id,
            "repeat": self.repeat,
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
class Ed1EvalDiagnostics:
    """Aggregate health summary derived from typed row diagnostics."""

    present_rows: int
    failed_rows: int
    none_reason: str | None


@dataclass(frozen=True, slots=True)
class Ed1EvalResult:
    """One candidate's ED1-family evaluation over a split (dual aggregates).

    ``primary_aggregate`` is HumanEval Submission Score for ED1 and Fidelity to
    Mutant for ED1M. ``compression_aggregate`` is Mean Compression Ratio
    (reported, never the Reward). ``reward`` is derived from the primary
    aggregate when unblended. Per-task vectors + outputs feed the CI / ledger /
    sidecar; ``row_diags`` explains arm-level Nones.
    """

    primary_aggregate: RolloutAggregate
    compression_aggregate: RolloutAggregate
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
    row_diags: tuple[Ed1RowDiag, ...] = ()

    @property
    def diagnostics(self) -> Ed1EvalDiagnostics:
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
        return Ed1EvalDiagnostics(
            present_rows=present,
            failed_rows=failed,
            none_reason=none_reason,
        )


def _validate_ed1_component_steps(
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


class Ed1RowOutcome(BaseModel):
    """One (task, repeat) rollout's dual result + provenance.

    ``primary_value`` is ED1's HumanEval Submission Score or ED1M's fractional
    Fidelity to Mutant. ``attractor_pull`` is the ED1M reported contamination
    measurement (``None`` for ED1).
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
        strict=True,
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
    #: Task-20 telemetry: summed encoder+decoder reasoning tokens + summed
    #: wall-clock latency for the row. ``None`` when the provider exposed no
    #: reasoning detail (never 0-conflated).
    reasoning_tokens: int | None = None
    latency_s: float | None = None
    #: Budget diagnostics: the per-task MAX_BUDGET (chars) the encoder was told
    #: to respect and the actual encoder-output length. ``over_budget`` is a
    #: derived flag (encoder_len > max_budget) -- an over-budget row is NOT
    #: clipped or failed (the budget only steers), so this is diagnostic only.
    max_budget: int | None = None
    encoder_len: int | None = None
    #: Task-26 per-call provenance. ``finish_reason`` is the DECODER call's
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
    #: Task-31 prompt-cache honesty. A DUAL ed1 row is ``cache_hit`` True only
    #: when BOTH the encoder AND decoder calls were served from cache (no wire
    #: call at all this time -> latency nulled). If either leg was freshly
    #: driven the row's latency is genuine and ``cache_hit`` is False.
    #: ``cache_provenance`` refs the ENCODER entry's original source on a full
    #: hit (the row's primary provenance), else ``None``.
    cache_hit: bool = False
    cache_provenance: CacheProvenance | None = None

    @model_validator(mode="after")
    def _valid_outcome(self) -> Ed1RowOutcome:
        _validate_ed1_component_steps(
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
        """True when the encoder output exceeded MAX_BUDGET (diagnostic only).

        ``None`` when either the budget or the encoder length is unknown (a
        pre-encoder failure, e.g. a render error), so a reader distinguishes
        "measured, within budget" from "never measured".
        """
        if self.max_budget is None or self.encoder_len is None:
            return None
        return self.encoder_len > self.max_budget


class Ed1PartialPayload(BaseModel):
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
    def _valid_trace(self) -> Ed1PartialPayload:
        _validate_ed1_component_steps(
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
    def from_json_value(cls, payload: JsonValue | None) -> Ed1PartialPayload:
        """Validate a decoded partial payload using strict JSON semantics."""
        return cls.model_validate_json(json.dumps(payload))


_ED1_ROW_REQUEST_SCHEMA = "whetstone.envs.ed1_row_request/v2"
_ED1_ROW_RESULT_SCHEMA = "whetstone.envs.ed1_row_result/v3"


class Ed1RowRequest(BaseModel):
    """Complete serializable request and provenance for one ED1-family row."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    _submitted_request_identity: str | None = PrivateAttr(default=None)

    schema_name: Literal["whetstone.envs.ed1_row_request/v2"] = (
        _ED1_ROW_REQUEST_SCHEMA
    )
    env_name: str
    dataset_revision: str
    primary_metric_name: str
    graph_hash: str
    candidate_template: str
    candidate_id: str
    instance: ProcessInstance
    provider_call_config: ProviderCallConfig
    execution_policy: ProviderExecutionPolicy
    procedure_config_hash: str
    evaluation_binding_hash: IdentityHash
    budget_ratio: float | None
    logical_call_id: str
    repeat_index: int
    drive_ordinal: int
    cache_phase: str
    cache_unit: str
    cache_root: str | None
    mutant_record: MutantRecord | None = None

    @field_serializer("provider_call_config")
    def _serialize_provider_call_config(
        self, config: ProviderCallConfig
    ) -> dict[str, object]:
        return _canonical_provider_call_config_payload(config)

    @model_validator(mode="after")
    def _valid_mutant_binding(self) -> Ed1RowRequest:
        from whetstone.envs.ed1m import ED1M_ENV_NAME

        if self.env_name == ED1M_ENV_NAME:
            if self.mutant_record is None:
                raise ValueError(
                    "an ED1M row requires its authenticated mutant"
                )
            if self.mutant_record.content_identity != self.instance.id:
                raise ValueError(
                    "ED1M mutant identity does not match the row instance"
                )
        elif self.mutant_record is not None:
            raise ValueError("a non-ED1M row forbids mutant oracle data")
        return self

    @property
    def request_identity(self) -> str:
        return self._submitted_request_identity or process_request_identity(
            self
        )

    @classmethod
    def from_process_payload(cls, payload: JsonValue) -> Ed1RowRequest:
        """Validate a decoded JSON payload using Pydantic's JSON semantics."""
        request = cls.model_validate_json(json.dumps(payload))
        request._submitted_request_identity = _process_payload_identity(
            payload
        )
        return request


class Ed1RowResult(BaseModel):
    """An ED1 outcome cryptographically bound to its submitted request."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_name: Literal["whetstone.envs.ed1_row_result/v3"] = (
        _ED1_ROW_RESULT_SCHEMA
    )
    request_identity: str
    outcome: Ed1RowOutcome

    @classmethod
    def from_process_payload(cls, payload: JsonValue) -> Ed1RowResult:
        """Validate a decoded worker result using JSON semantics."""
        return cls.model_validate_json(json.dumps(payload))


type Ed1RowJobFactory = Callable[[Ed1RowRequest], ProcessJob]


def _request(config: ProviderCallConfig, prompt: str) -> ProviderCallRequest:
    return ProviderCallRequest(
        config=config,
        transcript=Transcript(
            messages=(PromptMessage(role=MessageRole.USER, content=prompt),)
        ),
    )


def _max_budget(input_code: str, rule: CharacterBudgetRule) -> int:
    """``MAX_BUDGET = round(ratio * chars(input_code))`` (design rule)."""
    return round(rule.ratio * len(input_code))


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
    """Render the encoder prompt: the immutable frame around a strategy body.

    ``body`` is the Mutation-Surface payload (the strategy sentence ONLY); the
    budget clause + fenced code block come from ``ENCODER_FRAME``, so every
    candidate keeps them by construction. A body carrying a ``{placeholder}``
    would raise here (KeyError/IndexError/ValueError) -> a per-row failure, but
    intake validation rejects such bodies first.
    """
    return render_encoder_frame(
        body, input_code=input_code, max_budget=max_budget
    )


def drive_ed1_row(
    *,
    experiment: Ed1Experiment,
    candidate_template: str,
    instance: Instance,
    provider_call_config: ProviderCallConfig,
    execution_policy: ProviderExecutionPolicy,
    transport: TransportCall,
    scorer: Callable[..., CodeScore],
    logical_call_id: str,
    repeat_index: int,
    drive_ordinal: int,
    cache: PromptResultCache | None,
    cache_phase: str,
    cache_unit: str,
) -> Ed1RowOutcome:
    """Run one enc->dec->score rollout for one (task, repeat)."""
    input_code = instance.prompt_inputs["input_code"]
    rd = experiment.encdec_rollout
    assert rd is not None
    # NO-BUDGET frame (task 22.4): budget_rule None -> no MAX_BUDGET, no budget
    # sentence rendered (render_encoder_frame drops the clause on None).
    rule = rd.budget_rule
    max_budget = None if rule is None else _max_budget(input_code, rule)
    try:
        encoder_prompt = _render_encoder(
            candidate_template, input_code=input_code, max_budget=max_budget
        )
    except (KeyError, IndexError, ValueError):
        return Ed1RowOutcome(
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
        repeat_index=repeat_index,
        drive_ordinal=drive_ordinal,
        cache=cache,
        phase=cache_phase,
        unit=cache_unit,
    )
    enc = enc_exec.result
    if not enc.succeeded or enc.generation is None:
        from whetstone.execution.call_support import (
            failure_code_of,
            is_transient_transport_failure,
        )

        return Ed1RowOutcome(
            primary_value=None,
            compression_value=None,
            encoder_text=None,
            decoder_text=None,
            row_state=ExecutedRowState.FAILED,
            executed_component_steps=(),
            failure_code=failure_code_of(enc),
            max_budget=max_budget,
            encoder_len=None,
            provider_error=call_telemetry(enc).provider_error,
            redrivable=is_transient_transport_failure(enc),
        )
    encoder_text = enc.generation.text
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
        repeat_index=repeat_index,
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
    if not dec.succeeded or dec.generation is None:
        from whetstone.execution.call_support import (
            failure_code_of,
            is_transient_transport_failure,
        )

        return Ed1RowOutcome(
            primary_value=None,
            compression_value=None,
            encoder_text=encoder_text,
            decoder_text=None,
            row_state=ExecutedRowState.FAILED,
            executed_component_steps=executed_component_steps,
            failure_code=failure_code_of(dec),
            max_budget=max_budget,
            encoder_len=encoder_len,
            provider_error=call_telemetry(dec).provider_error,
            redrivable=is_transient_transport_failure(dec),
        )
    decoder_text = dec.generation.text
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

    # Correctness (decoder output) -- may be an infrastructure-unknown, which
    # fails the row (never scored 0). ed1 scores the HumanEval test suite; ed1m
    # scores the mutant's per-input oracle (fidelity + attractor). Compression
    # (encoder output) is always computed (it does not depend on the sandbox).
    code_score = _score_row(experiment, instance, decoder_text, scorer)
    if code_score.infrastructure_unknown:
        return Ed1RowOutcome(
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
        )
    compression = _compression_ratio(encoder_text, input_code)
    return Ed1RowOutcome(
        primary_value=code_score.row_value,
        compression_value=compression,
        encoder_text=encoder_text,
        decoder_text=decoder_text,
        row_state=ExecutedRowState.SUCCESS,
        executed_component_steps=executed_component_steps,
        attractor_pull=code_score.attractor_pull,
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


def _score_row(
    experiment: Ed1Experiment,
    instance: Instance,
    decoder_text: str,
    scorer: Callable[..., CodeScore],
) -> CodeScore:
    """Score one reconstruction: ed1 HumanEval suite OR ed1m mutant oracle.

    ed1m (an ``Ed1mExperiment``) scores the decoder output against the
    instance's mutant per-input oracle (fractional fidelity + reported
    attractor pull). Every other ed1 experiment scores the HumanEval test suite
    via the injected HumanEval submission scorer.
    """
    from whetstone.envs.ed1m import Ed1mExperiment, score_ed1m_row

    if isinstance(experiment, Ed1mExperiment):
        return score_ed1m_row(experiment, instance, decoder_text)
    task = humaneval_task_from_instance(instance)
    return scorer(raw_submission=decoder_text, task=task)


def _ed1_partial_payload(outcome: Ed1RowOutcome) -> Ed1PartialPayload:
    """Extract the strict ED1-family portion of a partial observation."""
    return Ed1PartialPayload(
        compression_value=outcome.compression_value,
        encoder_text=outcome.encoder_text,
        decoder_text=outcome.decoder_text,
        attractor_pull=outcome.attractor_pull,
        max_budget=outcome.max_budget,
        encoder_len=outcome.encoder_len,
        row_state=outcome.row_state,
        executed_component_steps=outcome.executed_component_steps,
    )


def _ed1_outcome_from_record(record: PartialCallRecord) -> Ed1RowOutcome:
    """Rebuild the accepted outcome stored for one exact ED1 request."""
    payload = Ed1PartialPayload.from_json_value(record.observation_payload)
    if record.output_text != payload.decoder_text:
        raise ValueError(
            "ED1 partial output_text does not match its observation payload"
        )
    if record.failed != (payload.row_state is ExecutedRowState.FAILED):
        raise ValueError("ED1 partial row state conflicts with failed flag")
    return Ed1RowOutcome(
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


def _should_redrive(outcome: Ed1RowOutcome) -> bool:
    """Whether an ordinal-0 ED1 result requires the bounded second attempt."""
    return outcome.failure_code == "runner_timeout" or outcome.redrivable


def _compression_ratio(encoder_text: str, input_code: str) -> float | None:
    """The zstd-19 Compression Ratio of the ENCODER output vs the reference.

    Reference = the exact ``gt_code_wo_comments`` bytes (= ``input_code``
    here).
    ``None`` for a zero-length reference (explicit zero-denominator, never
    coerced).
    """
    reference = select_compression_reference(_RefView(input_code))
    length = compressed_description_length_bytes(encoder_text)
    return compression_ratio_value(
        compressed_description_length=length, reference=reference
    )


@dataclass(frozen=True, slots=True)
class _RefView:
    """A minimal ``ExperimentTaskView`` (only ``gt_code_wo_comments``)."""

    gt_code_wo_comments: str


def _primary_metric_name(experiment: Ed1Experiment) -> str:
    """Return the concrete primary metric identity for this ED1-family env."""
    from whetstone.envs.ed1m import (
        ED1M_FIDELITY_NAME,
        Ed1mExperiment,
    )

    if isinstance(experiment, Ed1mExperiment):
        return ED1M_FIDELITY_NAME
    return ED1_SUBMISSION_SCORE_NAME


def run_ed1_eval(
    experiment: Ed1Experiment,
    *,
    candidate_template: str,
    candidate_id: str,
    sampling: EnvSplitSampling,
    execution_policy: ProviderExecutionPolicy,
    row_job_factory: Ed1RowJobFactory,
    evaluation_binding: EvaluationBinding,
    concurrency: int = DEFAULT_CONCURRENCY,
    max_wall_seconds: float | None = None,
    partial_log: PartialLog | None = None,
    cache: PromptResultCache | None = None,
) -> Ed1EvalResult:
    """Drive ``candidate_template`` over an ed1 split -> dual aggregates.

    Fans out one serializable enc->dec->score process job per (task, repeat),
    reduces to the primary + compression aggregates, derives Reward from the
    primary aggregate when unblended, and collects per-row outputs (encoder +
    decoder text) for the dual-score sidecar.

    Incremental persistence: when a ``partial_log`` is given, each (task,
    repeat) row is appended by its parent-owned commit the instant it
    completes, so a crash/interrupt mid-drive keeps every finished row on
    disk. A resumed drive restores only an exact row-request identity instead
    of re-driving and re-paying; a pending ordinal-0 record resumes at the
    exact ordinal-1 request.

    ``row_job_factory`` is the trusted scoring authority. It must execute the
    exact :class:`Ed1RowRequest` under its declared procedure; the parent binds
    result identity before persistence but cannot attest arbitrary worker code.
    """
    validate_ed1_body(candidate_template)
    instances = sampling.instances
    repeats = sampling.repeat_plan.repeat_count
    split_role = sampling.split_role
    rd = experiment.encdec_rollout
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
        outcome: Ed1RowOutcome,
        *,
        request_identity: str,
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
                instance_id=str(instance.id),
                unit=candidate_id,
                repeat_id=index,
                request_identity=request_identity,
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
                observation_payload=_ed1_partial_payload(outcome).model_dump(
                    mode="json"
                ),
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
    ) -> Ed1RowRequest:
        from whetstone.envs.ed1m import Ed1mExperiment

        mutant_record = (
            experiment.mutants[str(instance.id)]
            if isinstance(experiment, Ed1mExperiment)
            else None
        )
        return Ed1RowRequest(
            env_name=rd.env_name,
            dataset_revision=experiment.dataset_revision,
            primary_metric_name=primary_metric_name,
            graph_hash=graph_hash,
            candidate_template=candidate_template,
            candidate_id=candidate_id,
            instance=ProcessInstance.from_instance(instance),
            provider_call_config=rd.provider_call_config,
            execution_policy=execution_policy,
            procedure_config_hash=rd.procedure_config_hash,
            evaluation_binding_hash=evaluation_binding_id,
            budget_ratio=rd.budget_ratio,
            logical_call_id=f"{candidate_id}:{instance.id}#{index}",
            repeat_index=index,
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
        for index in range(repeats)
    }
    partial_records = index_partial_records(
        () if partial_log is None else partial_log.load(),
        phase=split_role,
        unit=candidate_id,
    )
    driven: dict[tuple[str, int], Ed1RowOutcome] = {}
    initial_requests: list[Ed1RowRequest] = []
    resumed_redrive_requests: list[Ed1RowRequest] = []
    for key, (ordinal_0, ordinal_1) in requests_by_key.items():
        decision = resolve_exact_resume(
            partial_records,
            instance_id=key[0],
            repeat_id=key[1],
            ordinal_0_request_identity=ordinal_0.request_identity,
            ordinal_1_request_identity=ordinal_1.request_identity,
        )
        if decision.record is not None:
            driven[key] = _ed1_outcome_from_record(decision.record)
        if decision.drive_ordinal == 0:
            initial_requests.append(ordinal_0)
        elif decision.drive_ordinal == 1:
            resumed_redrive_requests.append(ordinal_1)

    def _spec(
        request: Ed1RowRequest,
    ) -> CallSpec[tuple[str, int], Ed1RowOutcome]:
        instance = by_instance[request.instance.id]

        def _decode(value: JsonValue) -> Ed1RowOutcome:
            result = Ed1RowResult.from_process_payload(value)
            if result.request_identity != request.request_identity:
                raise ValueError(
                    "ED1 row result does not match its submitted request"
                )
            return result.outcome

        return CallSpec(
            key=(request.instance.id, request.repeat_index),
            job=row_job_factory(request),
            decode=_decode,
            deadline_seconds=_deadline(execution_policy),
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
        requests: list[Ed1RowRequest],
    ) -> dict[tuple[str, int], Ed1RowOutcome]:
        nonlocal effective_concurrency
        specs = [_spec(request) for request in requests]
        request_by_key = {
            (request.instance.id, request.repeat_index): request
            for request in requests
        }
        pool = run_call_pool(
            specs,
            concurrency=effective_concurrency,
            is_rate_limited=lambda _o: False,
            max_wall_seconds=remaining_phase_wall_seconds(phase_deadline),
        )
        effective_concurrency = pool.effective_concurrency
        out: dict[tuple[str, int], Ed1RowOutcome] = {}
        for res in pool.results:
            if res.status is FanoutStatus.COMPLETED and res.value is not None:
                out[res.key] = res.value
            elif res.status is FanoutStatus.UNIT_TIMEOUT:
                # A runner-guard timeout: the row hung past its (2-call) guard.
                # Marked redrivable so ONE bounded re-drive gets a fresh try
                # before it lands as a failed row (a single hung row must not
                # kill an anchor arm under the FAIL policy).
                request = request_by_key[res.key]
                outcome = Ed1RowOutcome(
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
                    request_identity=request.request_identity,
                    redrive_pending=request.drive_ordinal == 0,
                )
            else:
                out[res.key] = Ed1RowOutcome(
                    primary_value=None,
                    compression_value=None,
                    encoder_text=None,
                    decoder_text=None,
                    row_state=ExecutedRowState.MISSING,
                    executed_component_steps=(),
                )
        return out

    first_driven = _drive(initial_requests)
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
        if _should_redrive(outcome)
    ]
    if redrive_requests:
        redriven = _drive(redrive_requests)
        driven.update(
            (key, outcome)
            for key, outcome in redriven.items()
            if not outcome.missing
        )

    # Assemble per-task rows (primary + compression) + outputs, instance/repeat
    # order.
    primary_rows: list[tuple[str, list[RowValue]]] = []
    comp_rows: list[tuple[str, list[RowValue]]] = []
    outputs: list[RolloutOutput] = []
    row_diags: list[Ed1RowDiag] = []
    per_task_scores: list[float] = []
    per_task_counts: list[int] = []
    per_task_compression: list[float | None] = []
    per_task_attractor: list[float | None] = []
    for instance in instances:
        task_id = str(instance.id)
        task_primary_rows: list[RowValue] = []
        c_rows: list[RowValue] = []
        comp_vals: list[float] = []
        attr_vals: list[float] = []
        for index in range(repeats):
            outcome = driven[(task_id, index)]
            if outcome.attractor_pull is not None:
                attr_vals.append(outcome.attractor_pull)
            row_diags.append(
                Ed1RowDiag(
                    instance_id=task_id,
                    repeat=index,
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
                    instance_id=task_id,
                    repeat=index,
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
                task_identity=task_identity,
                rows=tuple(rows),
            )
            for task_identity, rows in primary_rows
        ),
        plan=sampling.evaluation_matrix_plan,
    )
    compression_aggregate = unweighted_task_mean(
        aggregate_name=ED1_COMPRESSION_NAME,
        graph_hash=graph_hash,
        evaluation_binding_hash=evaluation_binding_id,
        task_rows=tuple(
            TaskRows(
                task_identity=task_identity,
                rows=tuple(rows),
            )
            for task_identity, rows in comp_rows
        ),
        plan=sampling.evaluation_matrix_plan,
    )

    # Task 22: the weighted-blend reward. When a blend config is set, the
    # CERTIFICATION metric + the per-task CI vector are the PER-TASK blended
    # reward (primary score + compression are always reported separately). The
    # blend is composed per task, so the paired bootstrap operates on blended
    # rewards exactly as env_exact_match does for QA. With no blend,
    # ``per_task_scores`` is the per-task primary mean.
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
            # blended rewards (unweighted mean over the tasks with a present
            # primary mean; a fully-failed task with mean 0 still counts,
            # matching the primary aggregate's completeness handling).
            mean_blended = (
                sum(reward_scores) / len(reward_scores)
                if reward_scores
                else None
            )
            reward = ed1_reward_from_blended(
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
    return Ed1EvalResult(
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
    )


def _row_output_text(outcome: Ed1RowOutcome) -> str | None:
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
    "Ed1EvalDiagnostics",
    "Ed1EvalResult",
    "Ed1PartialPayload",
    "Ed1RowDiag",
    "Ed1RowJobFactory",
    "Ed1RowOutcome",
    "drive_ed1_row",
    "run_ed1_eval",
]
