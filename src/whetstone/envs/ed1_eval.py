"""The ed1 encoder->decoder->code-eval drive (dual scores).

Drives one candidate over an ed1 split through the injected transport, running
the three-node rollout per (task, repeat):

1. render the candidate's Mutation-Surface ENCODER template against the task's
   ``INPUT_CODE`` (= ``gt_code_wo_comments``) and the per-task character budget
   ``MAX_BUDGET = round(budget_ratio * chars(input_code))`` (guidance, not a
   hard clip -- the design: "the budget steers, nothing clips");
2. call the encoder (the shared enc/dec route);
3. render the fixed DECODER template against the ENCODER output and call the
   decoder;
4. score the DECODER output for correctness (dr-code HumanEval sandbox ->
Binary
   Test Pass Score) and the ENCODER output for compression (whetstone zstd-19
   Compression Ratio vs ``gt_code_wo_comments``).

It reduces to TWO aggregates -- the Average Binary Test Pass Rate (the
reward-bearing metric) and the Mean Compression Ratio (reported alongside) --
using the same two-stage mean the QA path uses, and returns a
:class:`~whetstone.runner.eval_run.SplitEvaluation` whose ``score`` is the pass
rate, carrying the compression scalar + per-row outputs for the dual-score
ledger/trace/sidecar. Nothing here makes a live paid call by itself: the
transport and the code-eval scorer are injected.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from dr_code.eval import (
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
from dr_store import MemoryBackend, ObjectStore
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
    ED1_PASS_RATE_NAME,
    Ed1Experiment,
    ed1_reward_from_blended,
    ed1_reward_from_pass_rate,
    humaneval_task_from_instance,
    render_encoder_frame,
)
from whetstone.envs.ed1_blended import blend_per_task
from whetstone.envs.ed1_scoring import CodeScore, score_ed1_submission
from whetstone.envs.internal_eval import RolloutOutput
from whetstone.execution.call_support import CallTelemetry, call_telemetry
from whetstone.execution.fanout import CallSpec, FanoutConfig, run_call_pool
from whetstone.execution.partials import PartialCallRecord, PartialLog
from whetstone.graph.character_budget import CharacterBudgetRule
from whetstone.optimization.reward import Reward
from whetstone.provider.driver import TransportCall, run_provider_call
from whetstone.provider.policy import ProviderExecutionPolicy


@dataclass(frozen=True, slots=True)
class Ed1RowDiag:
    """One (task, repeat) row's diagnostic record for the pilot artifact.

    Explains an arm-level ``None`` from disk: the typed ``failure_code`` (empty
    when the row succeeded), the pass/compression scalars, the per-task
    ``max_budget`` the encoder was told to respect, the actual encoder-output
    length, and the derived ``over_budget`` flag (an over-budget row is NEVER
    clipped or failed -- the budget only steers, so this is diagnostic only).
    """

    instance_id: str
    repeat: int
    passed: float | None
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
            "passed": self.passed,
            "compression": self.compression,
            "failed": self.failed,
            "failure_code": self.failure_code,
            "max_budget": self.max_budget,
            "encoder_len": self.encoder_len,
            "over_budget": self.over_budget,
        }


@dataclass(frozen=True, slots=True)
class Ed1EvalResult:
    """One candidate's ed1 evaluation over a split (dual aggregates).

    ``pass_aggregate`` is the Average Binary Test Pass Rate (the reward-bearing
    metric); ``compression_aggregate`` is the Mean Compression Ratio (reported,
    never the Reward). ``reward`` is derived from the pass aggregate only (when
    ``apply_reward``). Per-task vectors + outputs feed the CI / ledger /
    sidecar; ``row_diags`` explains arm-level Nones (the pilot artifact).
    """

    pass_aggregate: RolloutAggregate
    compression_aggregate: RolloutAggregate
    reward: Reward | None
    #: The CI vector: the PER-TASK BLENDED reward when a blend config is set,
    #: else the per-task pass mean (task 22). The paired bootstrap uses this.
    per_task_scores: tuple[float, ...]
    per_task_counts: tuple[int, ...]
    per_task_compression: tuple[float | None, ...]
    #: The raw per-task pass mean, ALWAYS reported separately (even when
    #: per_task_scores carries the blend), so pass rate + compression stay
    #: visible components in traces/sidecars/cells.
    per_task_pass: tuple[float, ...] = ()
    outputs: tuple[RolloutOutput, ...] = ()
    row_diags: tuple[Ed1RowDiag, ...] = ()


@dataclass(frozen=True, slots=True)
class _Ed1RowOutcome:
    """One (task, repeat) rollout's dual result + provenance."""

    pass_value: float | None
    compression_value: float | None
    encoder_text: str | None
    decoder_text: str | None
    failed: bool
    failure_code: str = ""
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
    #: True when this row failed on a TRANSIENT transport fault (timeout /
    #: stalled response / transport error / rate limit) whose driver-level
    #: semantic retries were exhausted -- eligible for ONE bounded re-drive.
    #: A deterministic failure (render error, provider rejection, infra-unknown
    #: scoring) is NOT redrivable (re-driving the same input will not change a
    #: deterministic "no").
    redrivable: bool = False

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


def _drive_row(
    *,
    experiment: Ed1Experiment,
    candidate_template: str,
    instance: Instance,
    provider_call_config: ProviderCallConfig,
    execution_policy: ProviderExecutionPolicy,
    transport: TransportCall,
    scorer: Callable[..., CodeScore],
    logical_call_id: str,
) -> _Ed1RowOutcome:
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
        return _Ed1RowOutcome(
            pass_value=None, compression_value=None,
            encoder_text=None, decoder_text=None,
            failed=True, failure_code="encoder_render_error",
            max_budget=max_budget, encoder_len=None,
        )
    enc = run_provider_call(
        request=_request(provider_call_config, encoder_prompt),
        policy=execution_policy, transport=transport,
        logical_call_id=f"{logical_call_id}:enc",
    )
    if not enc.succeeded or enc.generation is None:
        from whetstone.execution.call_support import (
            failure_code_of,
            is_transient_transport_failure,
        )
        return _Ed1RowOutcome(
            pass_value=None, compression_value=None,
            encoder_text=None, decoder_text=None,
            failed=True, failure_code=failure_code_of(enc),
            max_budget=max_budget, encoder_len=None,
            redrivable=is_transient_transport_failure(enc),
        )
    encoder_text = enc.generation.text
    encoder_len = len(encoder_text)
    decoder_prompt = DECODER_TEMPLATE.format(encoder_output=encoder_text)
    dec = run_provider_call(
        request=_request(provider_call_config, decoder_prompt),
        policy=execution_policy, transport=transport,
        logical_call_id=f"{logical_call_id}:dec",
    )
    if not dec.succeeded or dec.generation is None:
        from whetstone.execution.call_support import (
            failure_code_of,
            is_transient_transport_failure,
        )
        return _Ed1RowOutcome(
            pass_value=None, compression_value=None,
            encoder_text=encoder_text, decoder_text=None,
            failed=True, failure_code=failure_code_of(dec),
            max_budget=max_budget, encoder_len=encoder_len,
            redrivable=is_transient_transport_failure(dec),
        )
    decoder_text = dec.generation.text
    tel = _sum_telemetry(call_telemetry(enc), call_telemetry(dec))

    # Correctness (decoder output) -- may be an infrastructure-unknown, which
    # fails the row (never scored 0). Compression (encoder output) is always
    # computed (it does not depend on the sandbox).
    task = humaneval_task_from_instance(instance)
    code_score = scorer(raw_submission=decoder_text, task=task)
    if code_score.infrastructure_unknown:
        return _Ed1RowOutcome(
            pass_value=None, compression_value=None,
            encoder_text=encoder_text, decoder_text=decoder_text,
            failed=True, failure_code="code_eval_infrastructure_unknown",
            prompt_tokens=tel.prompt_tokens,
            completion_tokens=tel.completion_tokens,
            total_tokens=tel.total_tokens,
            reasoning_tokens=tel.reasoning_tokens, latency_s=tel.latency_s,
            max_budget=max_budget, encoder_len=encoder_len,
        )
    compression = _compression_ratio(encoder_text, input_code)
    return _Ed1RowOutcome(
        pass_value=float(code_score.passed),
        compression_value=compression,
        encoder_text=encoder_text, decoder_text=decoder_text,
        failed=False,
        prompt_tokens=tel.prompt_tokens,
        completion_tokens=tel.completion_tokens,
        total_tokens=tel.total_tokens,
        reasoning_tokens=tel.reasoning_tokens, latency_s=tel.latency_s,
        max_budget=max_budget, encoder_len=encoder_len,
    )


def _drive_and_persist(
    *,
    experiment: Ed1Experiment,
    candidate_template: str,
    candidate_id: str,
    instance: Instance,
    index: int,
    provider_call_config: ProviderCallConfig,
    execution_policy: ProviderExecutionPolicy,
    transport: TransportCall,
    scorer: Callable[..., CodeScore],
    partial_log: PartialLog | None,
    split_role: str,
) -> _Ed1RowOutcome:
    """Drive one ed1 row and append its partial record the instant it finishes.

    The dual result (compression + encoder/decoder text) that the reducer
    needs but ``PartialCallRecord`` has no native field for is carried in the
    record's ``raw_response`` as a compact JSON blob, so a resumed drive can
    fully reconstruct the row without re-paying. Mirrors the QA row thunk's
    persist-on-completion contract.
    """
    outcome = _drive_row(
        experiment=experiment,
        candidate_template=candidate_template,
        instance=instance,
        provider_call_config=provider_call_config,
        execution_policy=execution_policy,
        transport=transport,
        scorer=scorer,
        logical_call_id=f"{candidate_id}:{instance.id}#{index}",
    )
    if partial_log is not None:
        partial_log.append(
            PartialCallRecord(
                phase=split_role,
                instance_id=str(instance.id),
                unit=candidate_id,
                repeat_id=index,
                score=outcome.pass_value,
                failed=outcome.failed,
                failure_code=outcome.failure_code,
                prompt_tokens=outcome.prompt_tokens,
                completion_tokens=outcome.completion_tokens,
                total_tokens=outcome.total_tokens,
                # ed1 dual payload (compression + texts) the reducer needs on
                # resume; QA leaves raw_response empty on the cell path.
                raw_response=_encode_ed1_payload(outcome),
            )
        )
    return outcome


def _encode_ed1_payload(outcome: _Ed1RowOutcome) -> str:
    """Compact JSON of the ed1 dual extras for a partial record's resume."""
    return json.dumps({
        "compression_value": outcome.compression_value,
        "encoder_text": outcome.encoder_text,
        "decoder_text": outcome.decoder_text,
    })


def _restore_ed1_recorded(
    partial_log: PartialLog | None,
    split_role: str,
    candidate_id: str,
) -> dict[tuple[str, int], _Ed1RowOutcome]:
    """Rebuild ed1 row outcomes already durably recorded (resume skip).

    Restores only records for THIS phase (``split_role``) + unit
    (``candidate_id``), keyed ``(instance_id, repeat)`` to match the driven
    keys. The dual extras are decoded from ``raw_response``; a record missing
    that blob (e.g. a QA-shaped record) restores a pass-only failed/value row.
    """
    if partial_log is None:
        return {}
    restored: dict[tuple[str, int], _Ed1RowOutcome] = {}
    for record in partial_log.load():
        if record.phase != split_role or record.unit != candidate_id:
            continue
        extras: dict[str, object] = {}
        if record.raw_response:
            try:
                loaded = json.loads(record.raw_response)
                if isinstance(loaded, dict):
                    extras = loaded
            except json.JSONDecodeError:
                extras = {}
        comp = extras.get("compression_value")
        comp_value = (
            float(comp) if isinstance(comp, int | float) else None
        )
        restored[(record.instance_id, record.repeat_id)] = _Ed1RowOutcome(
            pass_value=record.score,
            compression_value=comp_value,
            encoder_text=_opt_str(extras.get("encoder_text")),
            decoder_text=_opt_str(extras.get("decoder_text")),
            failed=record.failed,
            failure_code=record.failure_code,
            prompt_tokens=record.prompt_tokens,
            completion_tokens=record.completion_tokens,
            total_tokens=record.total_tokens,
        )
    return restored


def _opt_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


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


def _mean_aggregation_config(policy: CompletenessPolicy):
    return aggregation_definition(
        "whetstone.ed1.eval.aggregation"
    ).materialize(
        {
            "reduction": "mean",
            "missing_data": policy.missing_data,
            "zero_denominator": "not_applicable",
            "max_skip_fraction": policy.skip_fraction_token(),
        }
    )


def _aggregate_metric(
    *,
    name: str,
    graph_hash: str,
    eval_config_hash: str,
    per_task_rows: list[tuple[str, list[RowValue]]],
    repeats: int,
    policy: CompletenessPolicy,
) -> RolloutAggregate:
    """Two-stage-mean aggregate over per-task RowValue lists (QA-identical)."""
    per_task_config = _mean_aggregation_config(policy)
    all_rows: list[RowValue] = []
    per_task_inputs: list[AggregationInput] = []
    task_rows_objs: list[TaskRows] = []
    for task_identity, rows in per_task_rows:
        completed = [r for r in rows]
        all_rows.extend(completed)
        task_rows_objs.append(
            TaskRows(
                task_identity=task_identity,
                expected_repeats=repeats,
                rows=tuple(rows),
            )
        )
        task_output = aggregate(
            per_task_config,
            tuple(r.to_aggregation_input() for r in completed),
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
        output, policy=policy, skipped=missing + failed + invalid,
        planned=len(all_rows),
    )
    return RolloutAggregate(
        name=name,
        graph_hash=graph_hash,
        eval_config_hash=eval_config_hash,
        evaluation_context_id=eval_config_hash,
        task_count=len(per_task_rows),
        repeat_count=repeats,
        aggregation_output=output,
        rows_present=present,
        rows_missing=missing,
        rows_failed=failed,
        rows_invalid=invalid,
    )


def run_ed1_eval(
    experiment: Ed1Experiment,
    *,
    candidate_template: str,
    candidate_id: str,
    instances: tuple[Instance, ...],
    execution_policy: ProviderExecutionPolicy,
    transport: TransportCall,
    scorer: Callable[..., CodeScore] | None = None,
    repeats: int = 3,
    policy: RowPolicy | CompletenessPolicy | None = None,
    fanout: FanoutConfig | None = None,
    apply_reward: bool = True,
    store: ObjectStore | None = None,
    partial_log: PartialLog | None = None,
    split_role: str = "cell",
) -> Ed1EvalResult:
    """Drive ``candidate_template`` over an ed1 split -> dual aggregates.

    Fans out one enc->dec->score rollout per (task, repeat) through the
    injected
    transport + code scorer, reduces to the pass-rate + compression aggregates,
    derives the pass-rate Reward (when ``apply_reward``), and collects per-row
    outputs (encoder + decoder text) for the dual-score sidecar.

    Incremental persistence: when a ``partial_log`` is given, each (task,
    repeat) row appends its OWN dual-result record the instant it completes
    (thread-safe), so a crash/interrupt mid-drive keeps every finished row on
    disk. A resumed drive restores already-recorded rows (keyed by the
    candidate ``unit`` = ``candidate_id`` and ``split_role`` phase) instead of
    re-driving+re-paying for them.
    """
    _ = store or ObjectStore(MemoryBackend())
    fanout = fanout or FanoutConfig()
    scorer = scorer or score_ed1_submission
    completeness = as_completeness_policy(
        policy if policy is not None else experiment.completeness_policy
    )
    rd = experiment.encdec_rollout
    assert rd is not None
    graph_hash = rd.graph_hash
    eval_config_hash = (
        experiment.eval_configs.internal.eval_config.config_identity_hash
    )
    restored = _restore_ed1_recorded(partial_log, split_role, candidate_id)

    def _spec(
        instance: Instance, index: int
    ) -> CallSpec[tuple[str, int], _Ed1RowOutcome]:
        return CallSpec(
            key=(str(instance.id), index),
            run=lambda inst=instance, i=index: _drive_and_persist(
                experiment=experiment,
                candidate_template=candidate_template,
                candidate_id=candidate_id,
                instance=inst,
                index=i,
                provider_call_config=rd.provider_call_config,
                execution_policy=execution_policy,
                transport=transport,
                scorer=scorer,
                partial_log=partial_log,
                split_role=split_role,
            ),
            deadline_seconds=_deadline(execution_policy),
        )

    by_instance = {str(inst.id): inst for inst in instances}

    def _drive(
        pending: list[CallSpec[tuple[str, int], _Ed1RowOutcome]],
    ) -> dict[tuple[str, int], _Ed1RowOutcome]:
        pool = run_call_pool(
            pending, concurrency=fanout.concurrency,
            is_rate_limited=lambda _o: False,
            max_wall_seconds=fanout.max_wall_seconds,
        )
        out: dict[tuple[str, int], _Ed1RowOutcome] = {}
        for res in pool.results:
            if res.value is not None:
                out[res.key] = res.value
            else:
                # A runner-guard timeout: the row hung past its (2-call) guard.
                # Marked redrivable so ONE bounded re-drive gets a fresh try
                # before it lands as a failed row (a single hung row must not
                # kill an anchor arm under the FAIL policy).
                out[res.key] = _Ed1RowOutcome(
                    pass_value=None, compression_value=None,
                    encoder_text=None, decoder_text=None,
                    failed=True, failure_code="runner_timeout",
                    redrivable=True,
                )
        return out

    # Only drive rows NOT already durably recorded (resume skip).
    specs = [
        _spec(instance, index)
        for instance in instances
        for index in range(repeats)
        if (str(instance.id), index) not in restored
    ]
    driven: dict[tuple[str, int], _Ed1RowOutcome] = dict(restored)
    driven.update(_drive(specs))

    # --- ONE bounded re-drive of timed-out / transient-transport rows. ---
    # A runner-guard timeout or a TERMINAL transient transport failure (enc or
    # dec) is re-driven exactly once before it lands as a failed row, so a
    # single flaky observation never fails the whole ed1 arm under FAIL policy
    # (the eval:ed1:a1 kill). A deterministic failure (render error, provider
    # rejection, infra-unknown scoring) is NOT re-driven. Mirrors the QA arm's
    # bounded re-drive; the re-drive persists its own partial record.
    redrive_specs = [
        _spec(by_instance[key[0]], key[1])
        for key, out in driven.items()
        if out.redrivable
    ]
    if redrive_specs:
        driven.update(_drive(redrive_specs))

    # Assemble per-task rows (pass + compression) + outputs, instance/repeat
    # order.
    pass_rows: list[tuple[str, list[RowValue]]] = []
    comp_rows: list[tuple[str, list[RowValue]]] = []
    outputs: list[RolloutOutput] = []
    row_diags: list[Ed1RowDiag] = []
    per_task_scores: list[float] = []
    per_task_counts: list[int] = []
    per_task_compression: list[float | None] = []
    for instance in instances:
        task_id = str(instance.id)
        p_rows: list[RowValue] = []
        c_rows: list[RowValue] = []
        comp_vals: list[float] = []
        for index in range(repeats):
            outcome = driven[(task_id, index)]
            row_diags.append(
                Ed1RowDiag(
                    instance_id=task_id,
                    repeat=index,
                    passed=outcome.pass_value,
                    compression=outcome.compression_value,
                    failed=outcome.failed,
                    failure_code=outcome.failure_code,
                    max_budget=outcome.max_budget,
                    encoder_len=outcome.encoder_len,
                    over_budget=outcome.over_budget,
                )
            )
            if outcome.failed or outcome.pass_value is None:
                p_rows.append(RowValue(failed=True))
            else:
                p_rows.append(RowValue(value=float(outcome.pass_value)))
            if outcome.compression_value is None:
                c_rows.append(
                    RowValue(failed=True) if outcome.failed
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
                    output_text=_row_output_text(outcome),
                    score=(
                        None if outcome.pass_value is None
                        else float(outcome.pass_value)
                    ),
                    failure_code=outcome.failure_code,
                )
            )
        pass_rows.append((task_id, p_rows))
        comp_rows.append((task_id, c_rows))
        # Per-task pass mean + observation weight for the paired CI. Computed
        # IDENTICALLY to the QA lane (``internal_eval._per_task_score`` /
        # ``_per_task_count``) so ed1 skipped rows feed the paired/pooled
        # bootstrap exactly as c18's SKIP lane does: the mean divides by the
        # planned repeats (an absent/failed row counts 0), and the weight is
        # the planned repeat count -- not the present-only count, which would
        # mis-weight a task with skipped rows when escalation pools repeats.
        total = sum(
            float(r.value or 0.0) if r.is_present else 0.0 for r in p_rows
        )
        per_task_scores.append(total / len(p_rows) if p_rows else 0.0)
        per_task_counts.append(len(p_rows))
        per_task_compression.append(
            sum(comp_vals) / len(comp_vals) if comp_vals else None
        )

    pass_aggregate = _aggregate_metric(
        name=ED1_PASS_RATE_NAME, graph_hash=graph_hash,
        eval_config_hash=eval_config_hash, per_task_rows=pass_rows,
        repeats=repeats, policy=completeness,
    )
    compression_aggregate = _aggregate_metric(
        name=ED1_COMPRESSION_NAME, graph_hash=graph_hash,
        eval_config_hash=eval_config_hash, per_task_rows=comp_rows,
        repeats=repeats, policy=completeness,
    )

    # Task 22: the weighted-blend reward. When a blend config is set, the
    # CERTIFICATION metric + the per-task CI vector are the PER-TASK blended
    # reward (pass rate + compression ALWAYS also reported separately). The
    # blend is composed PER TASK, so the paired bootstrap operates on blended
    # rewards exactly as env_exact_match does for QA. Pass-only (blend None)
    # keeps the historical per_task_scores = per-task pass mean.
    blend_config = experiment.blend_config
    pass_scores = tuple(per_task_scores)
    if blend_config is not None:
        reward_scores = blend_per_task(
            pass_scores, tuple(per_task_compression), blend_config
        )
    else:
        reward_scores = pass_scores

    if apply_reward:
        if blend_config is not None:
            # The aggregate blended reward = MEAN over tasks of the per-task
            # blended rewards (unweighted mean over the tasks with a present
            # pass mean; a fully-failed task (pass mean 0) still counts,
            # matching the pass-aggregate's completeness handling).
            mean_blended = (
                sum(reward_scores) / len(reward_scores)
                if reward_scores else None
            )
            reward = ed1_reward_from_blended(
                blend_config, blended=mean_blended
            )
        else:
            reward = ed1_reward_from_pass_rate(
                experiment.reward_policy,
                pass_rate=pass_aggregate.aggregation_output.value,
            )
    else:
        reward = None
    return Ed1EvalResult(
        pass_aggregate=pass_aggregate,
        compression_aggregate=compression_aggregate,
        reward=reward,
        # The CI vector: blended reward per task when blending, else pass mean.
        per_task_scores=reward_scores,
        per_task_counts=tuple(per_task_counts),
        per_task_compression=tuple(per_task_compression),
        per_task_pass=pass_scores,
        outputs=tuple(outputs),
        row_diags=tuple(row_diags),
    )


def _row_output_text(outcome: _Ed1RowOutcome) -> str | None:
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
    "Ed1EvalResult",
    "Ed1RowDiag",
    "run_ed1_eval",
]
