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
    ed1_reward_from_pass_rate,
    humaneval_task_from_instance,
)
from whetstone.envs.ed1_scoring import CodeScore, score_ed1_submission
from whetstone.envs.internal_eval import RolloutOutput
from whetstone.execution.fanout import CallSpec, FanoutConfig, run_call_pool
from whetstone.graph.character_budget import CharacterBudgetRule
from whetstone.optimization.reward import Reward
from whetstone.provider.driver import TransportCall, run_provider_call
from whetstone.provider.policy import ProviderExecutionPolicy


@dataclass(frozen=True, slots=True)
class Ed1EvalResult:
    """One candidate's ed1 evaluation over a split (dual aggregates).

    ``pass_aggregate`` is the Average Binary Test Pass Rate (the reward-bearing
    metric); ``compression_aggregate`` is the Mean Compression Ratio (reported,
    never the Reward). ``reward`` is derived from the pass aggregate only (when
    ``apply_reward``). Per-task vectors + outputs feed the CI / ledger /
    sidecar.
    """

    pass_aggregate: RolloutAggregate
    compression_aggregate: RolloutAggregate
    reward: Reward | None
    per_task_scores: tuple[float, ...]
    per_task_counts: tuple[int, ...]
    per_task_compression: tuple[float | None, ...]
    outputs: tuple[RolloutOutput, ...]


@dataclass(frozen=True, slots=True)
class _Ed1RowOutcome:
    """One (task, repeat) rollout's dual result + provenance."""

    pass_value: float | None
    compression_value: float | None
    encoder_text: str | None
    decoder_text: str | None
    failed: bool
    failure_code: str = ""


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


def _render_encoder(template: str, *, input_code: str, max_budget: int) -> str:
    """Render the encoder Mutation-Surface template.

    Fills ``{input_code}`` and ``{max_budget}``. Unknown placeholders are the
    intake validator's concern (upstream); here a KeyError is a per-row
    failure.
    """
    return template.format(input_code=input_code, max_budget=max_budget)


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
    max_budget = _max_budget(input_code, rd.budget_rule)
    try:
        encoder_prompt = _render_encoder(
            candidate_template, input_code=input_code, max_budget=max_budget
        )
    except (KeyError, IndexError, ValueError):
        return _Ed1RowOutcome(
            pass_value=None, compression_value=None,
            encoder_text=None, decoder_text=None,
            failed=True, failure_code="encoder_render_error",
        )
    enc = run_provider_call(
        request=_request(provider_call_config, encoder_prompt),
        policy=execution_policy, transport=transport,
        logical_call_id=f"{logical_call_id}:enc",
    )
    if not enc.succeeded or enc.generation is None:
        from whetstone.execution.call_support import failure_code_of
        return _Ed1RowOutcome(
            pass_value=None, compression_value=None,
            encoder_text=None, decoder_text=None,
            failed=True, failure_code=failure_code_of(enc),
        )
    encoder_text = enc.generation.text
    decoder_prompt = DECODER_TEMPLATE.format(encoder_output=encoder_text)
    dec = run_provider_call(
        request=_request(provider_call_config, decoder_prompt),
        policy=execution_policy, transport=transport,
        logical_call_id=f"{logical_call_id}:dec",
    )
    if not dec.succeeded or dec.generation is None:
        from whetstone.execution.call_support import failure_code_of
        return _Ed1RowOutcome(
            pass_value=None, compression_value=None,
            encoder_text=encoder_text, decoder_text=None,
            failed=True, failure_code=failure_code_of(dec),
        )
    decoder_text = dec.generation.text

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
        )
    compression = _compression_ratio(encoder_text, input_code)
    return _Ed1RowOutcome(
        pass_value=float(code_score.passed),
        compression_value=compression,
        encoder_text=encoder_text, decoder_text=decoder_text,
        failed=False,
    )


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
) -> Ed1EvalResult:
    """Drive ``candidate_template`` over an ed1 split -> dual aggregates.

    Fans out one enc->dec->score rollout per (task, repeat) through the
    injected
    transport + code scorer, reduces to the pass-rate + compression aggregates,
    derives the pass-rate Reward (when ``apply_reward``), and collects per-row
    outputs (encoder + decoder text) for the dual-score sidecar.
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

    def _spec(
        instance: Instance, index: int
    ) -> CallSpec[tuple[str, int], _Ed1RowOutcome]:
        return CallSpec(
            key=(str(instance.id), index),
            run=lambda inst=instance, i=index: _drive_row(
                experiment=experiment,
                candidate_template=candidate_template,
                instance=inst,
                provider_call_config=rd.provider_call_config,
                execution_policy=execution_policy,
                transport=transport,
                scorer=scorer,
                logical_call_id=f"{candidate_id}:{inst.id}#{i}",
            ),
            deadline_seconds=_deadline(execution_policy),
        )

    specs = [
        _spec(instance, index)
        for instance in instances
        for index in range(repeats)
    ]
    pool = run_call_pool(
        specs, concurrency=fanout.concurrency,
        is_rate_limited=lambda _o: False,
        max_wall_seconds=fanout.max_wall_seconds,
    )
    driven: dict[tuple[str, int], _Ed1RowOutcome] = {}
    for res in pool.results:
        if res.value is not None:
            driven[res.key] = res.value
        else:
            driven[res.key] = _Ed1RowOutcome(
                pass_value=None, compression_value=None,
                encoder_text=None, decoder_text=None,
                failed=True, failure_code="runner_timeout",
            )

    # Assemble per-task rows (pass + compression) + outputs, instance/repeat
    # order.
    pass_rows: list[tuple[str, list[RowValue]]] = []
    comp_rows: list[tuple[str, list[RowValue]]] = []
    outputs: list[RolloutOutput] = []
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
        # Per-task pass mean (failed rows count 0) for the paired CI.
        total = sum(
            float(r.value or 0.0) if r.is_present else 0.0 for r in p_rows
        )
        per_task_scores.append(total / len(p_rows) if p_rows else 0.0)
        per_task_counts.append(sum(1 for r in p_rows if r.is_present))
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
    reward = (
        ed1_reward_from_pass_rate(
            experiment.reward_policy,
            pass_rate=pass_aggregate.aggregation_output.value,
        )
        if apply_reward
        else None
    )
    return Ed1EvalResult(
        pass_aggregate=pass_aggregate,
        compression_aggregate=compression_aggregate,
        reward=reward,
        per_task_scores=tuple(per_task_scores),
        per_task_counts=tuple(per_task_counts),
        per_task_compression=tuple(per_task_compression),
        outputs=tuple(outputs),
    )


def _row_output_text(outcome: _Ed1RowOutcome) -> str | None:
    """The sidecar output text: the encoder + decoder outputs (both kept)."""
    if outcome.encoder_text is None and outcome.decoder_text is None:
        return None
    return (
        f"ENCODER:\n{outcome.encoder_text or ''}\n\n"
        f"DECODER:\n{outcome.decoder_text or ''}"
    )


def _deadline(execution_policy: ProviderExecutionPolicy) -> float:
    from whetstone.execution.call_support import guard_deadline_seconds

    return guard_deadline_seconds(execution_policy)


__all__ = [
    "Ed1EvalResult",
    "run_ed1_eval",
]
