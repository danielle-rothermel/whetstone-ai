"""The cross-env factory + process-isolated internal-eval loop.

The load-bearing proof: a full internal-eval of the naive candidate on a
tiny pool, driven by a FAKE transport, produces a valid Rollout Aggregate +
Reward -- per env, with no live paid call.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from dr_code.eval import AggregationStatus
from dr_providers import (
    ControlConstraints,
    GenerationControls,
    ModelRoute,
    Protocol,
    ProviderCallConfig,
    ProviderCallDefinition,
    ProviderKind,
    ReasoningRequestShape,
    RequestControl,
    TokenLimitParameter,
)

from tests.envs.support import (
    ReplyFn,
    constant_reply,
    execution_policy,
    process_row_job_factory,
    row_job_factory,
)
from whetstone.core.identity import IdentityHash, ImmutableJsonObject, TypedRef
from whetstone.core.roles import EvaluationRole
from whetstone.envs.factory import EnvExperiment, build_env_experiment
from whetstone.envs.oracle_operator import env_exact_match_score
from whetstone.envs.registry import ENV_NAMES, env_spec
from whetstone.envs.reward import CandidateEvaluationFailure
from whetstone.envs.rollout_definition import LLM_NODE_ID, PromptInputError
from whetstone.evaluation.drivers.internal import (
    InternalRowOutcome,
    InternalRowRequest,
    InternalRowResult,
    ProcessInstance,
    process_request_identity,
    run_internal_eval,
    start_phase_deadline,
)
from whetstone.evaluation.traces import (
    MAX_EXECUTED_COMPONENT_FIELDS,
    MAX_EXECUTED_COMPONENT_JSON_BYTES,
    MAX_EXECUTED_COMPONENT_STEPS,
    ExecutedComponentStep,
    ExecutedComponentTracePayload,
    ExecutedRowState,
    _bounded_trace_json_size,
    _llm_component_step,
    validate_executed_component_trace,
)
from whetstone.execution.fanout import (
    FanoutResult,
    FanoutStatus,
    PoolOutcome,
    ProcessJob,
)
from whetstone.execution.partials import PartialCallRecord, PartialLog
from whetstone.experiment.binding import (
    EVALUATION_BINDING_SCHEMA_VERSION,
    EvaluationBinding,
    eval_config_reference,
)
from whetstone.experiment.candidate import Candidate
from whetstone.experiment.reward import Reward
from whetstone.optimization.proposal.mutation import MUTATION_FIELD

_MODEL = "openai/gpt-5-nano"
_SPLIT = (2, 2, 2)
# At n_per_stratum=sum(_SPLIT), even one stratum supplies the complete split;
# additional strata can only increase capacity. This bound is independent of
# the generated pool sizes observed by the fit loop.
_SPLIT_FIT_CEILING = sum(_SPLIT)


def _semantic_trace_bytes(
    steps: tuple[ExecutedComponentStep, ...],
) -> bytes:
    return json.dumps(
        [step.model_dump(mode="json") for step in steps],
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()


def _attempt_mapping_mutation(value: Any, key: str, replacement: Any) -> None:
    value[key] = replacement


def _successful_internal_outcome(
    *, prompt: str = "test prompt", output_text: str = "test output"
) -> InternalRowOutcome:
    return InternalRowOutcome(
        score=1.0,
        row_state=ExecutedRowState.SUCCESS,
        executed_component_steps=(
            _llm_component_step(
                trace_index=0,
                component_id=LLM_NODE_ID,
                prompt=prompt,
                generation=output_text,
            ),
        ),
        output_text=output_text,
    )


def test_executed_component_step_pins_wire_fields_and_order() -> None:
    step = _llm_component_step(
        trace_index=0,
        component_id="generate",
        prompt="exact prompt",
        generation="exact generation",
    )
    assert step.model_dump(mode="json") == {
        "trace_index": 0,
        "component_id": "generate",
        "input_field_names": ["prompt"],
        "output_field_names": ["generation"],
        "inputs": {"prompt": "exact prompt"},
        "outputs": {"generation": "exact generation"},
    }

    ordered = ExecutedComponentStep.model_validate(
        {
            "trace_index": 0,
            "component_id": "generate",
            "input_field_names": ("second", "first"),
            "output_field_names": ("generation",),
            "inputs": {"first": 1, "second": 2},
            "outputs": {"generation": "ok"},
        },
        strict=True,
    )
    assert tuple(ordered.inputs) == ("second", "first")
    assert ordered.inputs.to_json() == {"second": 2, "first": 1}

    with pytest.raises(ValueError, match="unique and non-overlapping"):
        ExecutedComponentStep.model_validate(
            {
                "trace_index": 0,
                "component_id": "generate",
                "input_field_names": ("value",),
                "output_field_names": ("value",),
                "inputs": {"value": 1},
                "outputs": {"value": 2},
            },
            strict=True,
        )


@pytest.mark.parametrize("malformed", [("python tuple",), float("nan")])
def test_executed_component_step_rejects_non_strict_json(malformed) -> None:
    with pytest.raises(ValueError, match=r"strict JSON|finite numbers"):
        ExecutedComponentStep.model_validate(
            {
                "trace_index": 0,
                "component_id": "generate",
                "input_field_names": ("prompt",),
                "output_field_names": ("generation",),
                "inputs": {"prompt": malformed},
                "outputs": {"generation": "ok"},
            },
            strict=True,
        )


def test_executed_component_step_is_deeply_mutation_isolated() -> None:
    source: dict[str, Any] = {
        "payload": {"messages": ["public"], "metadata": {"safe": True}}
    }
    step = ExecutedComponentStep.model_validate(
        {
            "trace_index": 0,
            "component_id": "generate",
            "input_field_names": ("payload",),
            "output_field_names": ("generation",),
            "inputs": source,
            "outputs": {"generation": "accepted"},
        },
        strict=True,
    )
    original_bytes = _semantic_trace_bytes((step,))

    _attempt_mapping_mutation(source["payload"], "api_key", "source-secret")
    source["payload"]["messages"].append("source-secret")
    assert _semantic_trace_bytes((step,)) == original_bytes

    with pytest.raises(TypeError):
        _attempt_mapping_mutation(
            step.inputs, "payload", {"api_key": "model-secret"}
        )
    nested = step.inputs["payload"]
    assert isinstance(nested, ImmutableJsonObject)
    with pytest.raises(TypeError):
        _attempt_mapping_mutation(nested, "api_key", "model-secret")

    dumped = step.model_dump(mode="json")
    dumped["inputs"]["payload"]["api_key"] = "dump-secret"
    dumped["inputs"]["payload"]["messages"].append("dump-secret")
    assert _semantic_trace_bytes((step,)) == original_bytes
    assert b"secret" not in _semantic_trace_bytes((step,))


def test_executed_component_trace_enforces_all_fixed_bounds() -> None:
    names = tuple(
        f"field_{index}" for index in range(MAX_EXECUTED_COMPONENT_FIELDS + 1)
    )
    with pytest.raises(ValueError, match="field count"):
        ExecutedComponentStep.model_validate(
            {
                "trace_index": 0,
                "component_id": "generate",
                "input_field_names": names,
                "output_field_names": (),
                "inputs": {name: None for name in names},
                "outputs": {},
            },
            strict=True,
        )
    with pytest.raises(ValueError, match="byte bound"):
        _llm_component_step(
            trace_index=0,
            component_id="generate",
            prompt="x" * MAX_EXECUTED_COMPONENT_JSON_BYTES,
            generation="ok",
        )

    repeated = tuple(
        _llm_component_step(
            trace_index=index,
            component_id="generate",
            prompt="prompt",
            generation="generation",
        )
        for index in range(MAX_EXECUTED_COMPONENT_STEPS)
    )
    assert validate_executed_component_trace(repeated) == repeated
    with pytest.raises(ValueError, match="step count"):
        validate_executed_component_trace(
            (
                *repeated,
                _llm_component_step(
                    trace_index=MAX_EXECUTED_COMPONENT_STEPS,
                    component_id="generate",
                    prompt="prompt",
                    generation="generation",
                ),
            )
        )
    with pytest.raises(ValueError, match="contiguous from zero"):
        validate_executed_component_trace(
            (repeated[0], repeated[1].model_copy(update={"trace_index": 2}))
        )


def test_executed_component_trace_aborts_aggregate_accounting_early() -> None:
    empty = _llm_component_step(
        trace_index=0,
        component_id="generate",
        prompt="",
        generation="",
    )
    target_size = MAX_EXECUTED_COMPONENT_JSON_BYTES - 8
    large = _llm_component_step(
        trace_index=0,
        component_id="generate",
        prompt="x" * (target_size - empty.canonical_json_bytes),
        generation="",
    )
    small = _llm_component_step(
        trace_index=1,
        component_id="generate",
        prompt="small",
        generation="small",
    )
    assert large.canonical_json_bytes == target_size
    assert (
        large.canonical_json_bytes == len(_semantic_trace_bytes((large,))) - 2
    )
    assert validate_executed_component_trace((large,)) == (large,)
    with pytest.raises(ValueError, match="byte bound"):
        validate_executed_component_trace((large, small))

    consumed = 0

    def sizes():
        nonlocal consumed
        step_sizes = (
            large.canonical_json_bytes,
            small.canonical_json_bytes,
            *(1 for _ in range(MAX_EXECUTED_COMPONENT_STEPS - 2)),
        )
        for size in step_sizes:
            consumed += 1
            yield size

    with pytest.raises(ValueError, match="byte bound"):
        _bounded_trace_json_size(sizes())
    assert consumed == 2


def test_executed_component_trace_partial_round_trip_preserves_order(
    tmp_path: Path,
) -> None:
    step = ExecutedComponentStep.model_validate(
        {
            "trace_index": 0,
            "component_id": "nonlexical",
            "input_field_names": ("zeta", "alpha"),
            "output_field_names": ("omega", "beta"),
            "inputs": {"alpha": {"position": 2}, "zeta": [1, 2]},
            "outputs": {"beta": False, "omega": "first"},
        },
        strict=True,
    )
    payload = ExecutedComponentTracePayload(
        row_state=ExecutedRowState.SUCCESS,
        executed_component_steps=(step,),
    )
    before = _semantic_trace_bytes((step,))
    log = PartialLog(path=tmp_path / "ordered-trace.partial")
    log.append(
        PartialCallRecord(
            phase="internal",
            instance_id="instance",
            unit="unit",
            repeat_id=0,
            request_identity="0" * 64,
            redrive_pending=False,
            observation_payload=payload.model_dump(mode="json"),
        )
    )

    restored_payload = ExecutedComponentTracePayload.from_json_value(
        log.load()[0].observation_payload
    )
    restored = restored_payload.executed_component_steps[0]
    assert restored.input_field_names == ("zeta", "alpha")
    assert tuple(restored.inputs) == ("zeta", "alpha")
    assert restored.inputs["zeta"] == (1, 2)
    assert restored.inputs["alpha"] == {"position": 2}
    assert restored.output_field_names == ("omega", "beta")
    assert tuple(restored.outputs) == ("omega", "beta")
    assert restored.outputs["omega"] == "first"
    assert restored.outputs["beta"] is False
    assert _semantic_trace_bytes((restored,)) == before


def test_successful_row_cannot_omit_its_declared_trace() -> None:
    with pytest.raises(ValueError, match="requires its trace"):
        InternalRowOutcome(
            score=1.0,
            row_state=ExecutedRowState.SUCCESS,
            executed_component_steps=(),
            output_text=None,
        )


def _tiny_experiment(env_name: str) -> EnvExperiment:
    # n_per_stratum=1 gives >= 4 instances (all envs have >= 4 strata except
    # c18 which has 4); a (2,2,2) split needs >= 6, so grow the pool until it
    # is large enough. For a stratified-split env (c22, whose pool is blocked)
    # each stratum must independently hold its per-stratum quota, so grow until
    # the stratified split is satisfiable rather than only until the total
    # instance count clears sum(_SPLIT).
    env = env_spec(env_name)
    attempted_sizes: list[int] = []
    for n in range(1, _SPLIT_FIT_CEILING + 1):
        attempted_sizes.append(n)
        if _split_fits(env, n):
            break
    else:
        raise AssertionError(
            f"{env_name} could not fit split {_SPLIT} by independently "
            f"derived n_per_stratum ceiling {_SPLIT_FIT_CEILING}; "
            f"attempted_sizes={attempted_sizes}; "
            f"final_attempted_size={attempted_sizes[-1]}"
        )
    return build_env_experiment(
        env_name,
        model=_MODEL,
        pool_n_per_stratum=n,
        split_sizes=_SPLIT,
        repeats=2,
    )


def _binding(
    exp: EnvExperiment,
    *,
    role: EvaluationRole = EvaluationRole.INTERNAL,
) -> EvaluationBinding:
    sampling = (
        exp.eval_configs.internal
        if role is EvaluationRole.INTERNAL
        else exp.eval_configs.official
    )
    return EvaluationBinding(
        schema_version=EVALUATION_BINDING_SCHEMA_VERSION,
        eval_config=eval_config_reference(sampling.eval_config),
        role=role,
        authority_principal=(
            "test-authority" if role is EvaluationRole.OFFICIAL else None
        ),
        campaign="env-test",
    )


def _split_fits(env, n: int) -> bool:
    """True once a pool at ``n_per_stratum=n`` can serve the ``_SPLIT`` totals.

    For a contiguous-split env the whole pool need only exceed ``sum(_SPLIT)``;
    for a stratified-split env each stratum must hold its per-stratum quota, so
    ``n`` must clear the largest single-stratum draw.
    """
    pool = env.generate_pool(n_per_stratum=n)
    if not env.stratified_split:
        return len(pool) >= sum(_SPLIT)
    n_strata = len(pool.strata)
    per_stratum_max = sum(
        -(-part // n_strata)
        for part in _SPLIT  # ceil division per split part
    )
    return n >= per_stratum_max


def test_tiny_experiment_split_fit_search_is_bounded(monkeypatch) -> None:
    attempted_sizes: list[int] = []

    def never_fits(_env, n: int) -> bool:
        attempted_sizes.append(n)
        return False

    monkeypatch.setattr(sys.modules[__name__], "_split_fits", never_fits)
    with pytest.raises(AssertionError) as error:
        _tiny_experiment("c11")

    assert attempted_sizes == list(range(1, _SPLIT_FIT_CEILING + 1))
    assert f"attempted_sizes={attempted_sizes}" in str(error.value)
    assert f"final_attempted_size={_SPLIT_FIT_CEILING}" in str(error.value)


def _correct_reply(env_name: str, instances) -> ReplyFn:
    """A reply fn that returns the correct answer for the matching task.

    The env oracle grades the generation against each task's gold; the fake
    returns each task's own correct answer keyed off its rendered prompt so
    the internal eval scores a clean 1.0.
    """
    env = env_spec(env_name)
    from whetstone.envs.rollout_definition import (
        initial_candidate,
        render_prompt,
    )

    naive = initial_candidate(env)
    # Map rendered-naive-prompt -> the correct generation for that instance.
    correct_by_prompt: dict[str, str] = {}
    for inst in instances:
        prompt = render_prompt(env, naive, inst)
        correct_by_prompt[prompt] = _correct_generation(env, inst)

    def reply(prompt: str) -> str:
        return correct_by_prompt.get(prompt, "")

    return reply


def _correct_generation(env, instance) -> str:
    """The known-correct generation for an instance (per env)."""
    if env.name == "c22":
        # A response satisfying whatever stack the instance carries is
        # instance-specific; the c22 internal-eval test uses a hand-built
        # single-instance fixture instead (see the c22-specific test below).
        return instance.gold
    # For the re-derive envs the gold IS the correct answer.
    return instance.gold


def _internal_jobs(
    experiment: EnvExperiment,
    reply: ReplyFn,
    *,
    candidate: Candidate | None = None,
    served: list[str] | None = None,
):
    env = env_spec(experiment.env_name)
    active_candidate = candidate or experiment.initial_candidate
    procedure_hash = experiment.eval_configs.procedure_config_hash

    def outcome(instance, _repeat: int, _drive_ordinal: int):
        from whetstone.envs.rollout_definition import render_prompt

        prompt = render_prompt(env, active_candidate, instance)
        if served is not None:
            served.append(prompt)
        text = reply(prompt)
        if not text.strip():
            return InternalRowOutcome(
                score=None,
                row_state=ExecutedRowState.FAILED,
                executed_component_steps=(),
                failure_code="blank_generation",
            )
        score = env_exact_match_score(
            env=env,
            generation=text,
            gold=instance.gold,
            evaluation_procedure_config_hash=procedure_hash,
        )
        outcome = _successful_internal_outcome(prompt=prompt, output_text=text)
        return outcome.model_copy(update={"score": float(score.value)})

    return row_job_factory(outcome)


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_build_env_experiment_returns_all_five_deliverables(
    env_name: str,
) -> None:
    # The factory contract (five deliverables + shared Procedure identity) is
    # N-independent, so build over a tiny pool -- avoids the full-N c18/c18h
    # PrOntoQA regeneration this parametrization would otherwise pay twice.
    exp = _tiny_experiment(env_name)
    d = exp.as_dict()
    assert set(d) == {
        "rollout_definition",
        "initial_candidate",
        "ceiling_candidate",
        "eval_configs",
        "reward_policy",
    }
    # The Rollout Definition and both Eval Configs share one Procedure id.
    assert (
        exp.rollout_definition.procedure_config_hash
        == exp.eval_configs.procedure_config_hash
    )


def test_process_row_wire_schemas_are_pinned() -> None:
    from whetstone.evaluation.drivers.d1 import D1RowRequest, D1RowResult
    from whetstone.evaluation.drivers.ed1 import Ed1RowRequest, Ed1RowResult

    assert InternalRowRequest.model_fields["schema_name"].default == (
        "whetstone.envs.internal_row_request/v2"
    )
    assert InternalRowResult.model_fields["schema_name"].default == (
        "whetstone.envs.internal_row_result/v3"
    )
    assert D1RowRequest.model_fields["schema_name"].default == (
        "whetstone.envs.d1_row_request/v2"
    )
    assert D1RowResult.model_fields["schema_name"].default == (
        "whetstone.envs.d1_row_result/v3"
    )
    assert Ed1RowRequest.model_fields["schema_name"].default == (
        "whetstone.envs.ed1_row_request/v2"
    )
    assert Ed1RowResult.model_fields["schema_name"].default == (
        "whetstone.envs.ed1_row_result/v3"
    )
    assert tuple(InternalRowRequest.model_fields) == (
        "schema_name",
        "env_name",
        "candidate",
        "instance",
        "provider_call_config",
        "execution_policy",
        "procedure_config_hash",
        "evaluation_binding_hash",
        "logical_call_id",
        "repeat_index",
        "drive_ordinal",
        "cache_phase",
        "cache_unit",
        "cache_root",
        "render_guard",
    )
    assert tuple(D1RowRequest.model_fields) == (
        "schema_name",
        "candidate_body",
        "candidate_id",
        "instance",
        "humaneval_task",
        "input_arm",
        "rename_token",
        "provider_call_config",
        "execution_policy",
        "procedure_config_hash",
        "evaluation_binding_hash",
        "logical_call_id",
        "repeat_index",
        "drive_ordinal",
        "cache_phase",
        "cache_unit",
        "cache_root",
    )
    assert tuple(Ed1RowRequest.model_fields) == (
        "schema_name",
        "env_name",
        "dataset_revision",
        "primary_metric_name",
        "graph_hash",
        "candidate_template",
        "candidate_id",
        "instance",
        "provider_call_config",
        "execution_policy",
        "procedure_config_hash",
        "evaluation_binding_hash",
        "budget_ratio",
        "logical_call_id",
        "repeat_index",
        "drive_ordinal",
        "cache_phase",
        "cache_unit",
        "cache_root",
        "mutant_record",
    )
    assert tuple(InternalRowResult.model_fields) == (
        "schema_name",
        "request_identity",
        "outcome",
    )
    assert tuple(D1RowResult.model_fields) == (
        "schema_name",
        "request_identity",
        "outcome",
    )
    assert tuple(Ed1RowResult.model_fields) == (
        "schema_name",
        "request_identity",
        "outcome",
    )
    identity_fixture = ProcessInstance(
        id="row-1",
        seed=7,
        strata=("a", "b"),
        prompt_inputs={"z": "last", "a": "first"},
        gold="answer",
    )
    assert process_request_identity(identity_fixture) == (
        "7b01df9bfff5d96a10f35c4e1c5d473f6b3bb7dd72b2e5d49062f788a640c619"
    )


@pytest.mark.parametrize(
    "value",
    [True, -1.0, float("nan"), "1", 10**10_000],
    ids=("bool", "negative", "nan", "string", "huge-int"),
)
def test_phase_wall_uses_strict_fanout_duration_contract(value) -> None:
    with pytest.raises(ValueError, match="finite nonnegative real number"):
        start_phase_deadline(value)


@pytest.mark.parametrize("env_name", ["c11", "c19", "c18", "c23"])
def test_internal_eval_naive_candidate_clean_pass(env_name: str) -> None:
    exp = _tiny_experiment(env_name)
    internal_insts = exp.eval_configs.internal.instances
    result = run_internal_eval(
        exp,
        candidate=exp.initial_candidate,
        sampling=exp.eval_configs.internal,
        execution_policy=execution_policy(),
        row_job_factory=_internal_jobs(
            exp, _correct_reply(env_name, internal_insts)
        ),
        evaluation_binding=_binding(exp),
    )
    agg = result.aggregate
    assert agg.name == "env_exact_match"
    assert agg.aggregation_output.status is AggregationStatus.OK
    assert agg.aggregation_output.value == pytest.approx(1.0)
    # Complete matrix accounting: every planned row is present, none dropped.
    planned = agg.task_count * agg.repeat_count
    assert agg.rows_present == planned
    assert agg.rows_missing == agg.rows_failed == agg.rows_invalid == 0
    # A valid internal-role Reward maps the aggregate.
    assert isinstance(result.reward, Reward)
    assert result.reward.evidence_role is EvaluationRole.INTERNAL
    assert result.reward.value == pytest.approx(1.0)


def test_c22_internal_eval_produces_valid_aggregate_and_reward() -> None:
    # c22 correct responses are constraint-stack-specific (proven at score 1
    # against a hand-built fixture in test_oracle_operator); here the full
    # internal-eval loop is exercised end to end through the c22 gold-first
    # oracle, producing a VALID Rollout Aggregate + Reward. A response that
    # satisfies no stack scores 0 across the split.
    exp = _tiny_experiment("c22")
    result = run_internal_eval(
        exp,
        candidate=exp.initial_candidate,
        sampling=exp.eval_configs.internal,
        execution_policy=execution_policy(),
        row_job_factory=_internal_jobs(
            exp, constant_reply("plain, comma-laden text")
        ),
        evaluation_binding=_binding(exp),
    )
    agg = result.aggregate
    assert agg.name == "env_exact_match"
    assert agg.aggregation_output.status is AggregationStatus.OK
    assert agg.aggregation_output.value == pytest.approx(0.0)
    planned = agg.task_count * agg.repeat_count
    assert agg.rows_present == planned
    assert isinstance(result.reward, Reward)
    assert result.reward.evidence_role is EvaluationRole.INTERNAL
    assert result.reward.value == pytest.approx(0.0)


def test_internal_eval_wrong_answers_score_zero() -> None:
    exp = _tiny_experiment("c18")
    # Always answer the opposite label so every task scores 0.
    result = run_internal_eval(
        exp,
        candidate=exp.initial_candidate,
        sampling=exp.eval_configs.internal,
        execution_policy=execution_policy(),
        row_job_factory=_internal_jobs(
            exp, constant_reply("definitely-not-a-label")
        ),
        evaluation_binding=_binding(exp),
    )
    assert result.aggregate.aggregation_output.value == pytest.approx(0.0)
    assert isinstance(result.reward, Reward)
    assert result.reward.value == pytest.approx(0.0)


def test_internal_process_job_runs_real_row_driver() -> None:
    exp = _tiny_experiment("c18")
    result = run_internal_eval(
        exp,
        candidate=exp.initial_candidate,
        sampling=exp.eval_configs.internal,
        execution_policy=execution_policy(),
        row_job_factory=process_row_job_factory(
            "tests.envs.process_workers:drive_internal_success"
        ),
        evaluation_binding=_binding(exp),
    )

    assert result.aggregate.rows_failed == 0
    assert result.aggregate.aggregation_output.value == pytest.approx(1.0)
    output = result.outputs[0]
    step = output.executed_component_steps[0]
    assert output.row_state is ExecutedRowState.SUCCESS
    assert step.component_id == "generate"
    assert step.outputs == {"generation": output.output_text}
    assert step.input_field_names == ("prompt",)


def test_internal_result_for_different_request_is_rejected() -> None:
    exp = _tiny_experiment("c18")

    def mismatched(request: InternalRowRequest) -> ProcessJob:
        result = InternalRowResult(
            request_identity="0" * 64,
            outcome=_successful_internal_outcome(),
        )
        return ProcessJob(
            entrypoint="tests.envs.process_workers:return_payload",
            payload=result.model_dump(mode="json"),
        )

    with pytest.raises(ValueError, match="does not match"):
        run_internal_eval(
            exp,
            candidate=exp.initial_candidate,
            sampling=exp.eval_configs.internal,
            execution_policy=execution_policy(),
            row_job_factory=mismatched,
            evaluation_binding=_binding(exp),
        )


def test_internal_v2_request_hash_is_pinned() -> None:
    exp = _tiny_experiment("c18")
    sampling = exp.eval_configs.internal
    base = _internal_jobs(exp, _correct_reply("c18", sampling.instances))
    requests: list[InternalRowRequest] = []

    def capture(request: InternalRowRequest) -> ProcessJob:
        requests.append(request)
        return base(request)

    run_internal_eval(
        exp,
        candidate=exp.initial_candidate,
        sampling=sampling,
        execution_policy=execution_policy(),
        row_job_factory=capture,
        evaluation_binding=_binding(exp),
    )

    assert requests[0].request_identity == (
        "f40bb1bb5488c3165fa714f87498291ab2a6abc487434a7d6b2967cc978ff583"
    )


_CROSS_SEED_REQUEST_SCRIPT = """
import json
import sys

sys.path.insert(0, {repo_root!r})

from tests.envs.test_factory import _fixed_unordered_provider_request

request = _fixed_unordered_provider_request()
sys.stdout.write(
    json.dumps(
        request.model_dump(mode="json"),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
)
sys.stdout.write("\\n")
sys.stdout.write(request.request_identity)
"""


def _fixed_unordered_provider_request() -> InternalRowRequest:
    controls = frozenset(
        {
            RequestControl.REASONING,
            RequestControl.TEMPERATURE,
            RequestControl.TOKEN_LIMIT,
            RequestControl.TOP_P,
        }
    )
    provider_call_config = ProviderCallConfig(
        definition=ProviderCallDefinition(
            definition_id="test.chat_completions",
            route=ModelRoute(
                provider=ProviderKind.OPENROUTER,
                protocol=Protocol.CHAT_COMPLETIONS,
                model="test/model",
            ),
            constraints=ControlConstraints(
                supported_controls=controls,
                token_limit_parameter=(
                    TokenLimitParameter.MAX_COMPLETION_TOKENS
                ),
                reasoning_shape=ReasoningRequestShape.REASONING_OBJECT,
            ),
            required_controls=frozenset(
                {
                    RequestControl.TEMPERATURE,
                    RequestControl.TOKEN_LIMIT,
                }
            ),
            extension_keys=frozenset({"alpha", "omega"}),
        ),
        controls=GenerationControls(temperature=0.0, token_limit=1),
    )
    return InternalRowRequest(
        env_name="c18",
        candidate=Candidate(
            candidate_id="fixed-candidate",
            base_ref=TypedRef(
                schema_name="fixed-candidate-base",
                content_hash="1" * 64,
            ),
        ),
        instance=ProcessInstance(
            id="fixed-instance",
            seed=1,
            strata=("fixed",),
            prompt_inputs={"question": "Fixed question?", "query": "True"},
            gold="True",
        ),
        provider_call_config=provider_call_config,
        execution_policy=execution_policy(),
        procedure_config_hash="2" * 64,
        evaluation_binding_hash=IdentityHash("3" * 64),
        logical_call_id="fixed-call#0",
        repeat_index=0,
        drive_ordinal=0,
        cache_phase="internal_eval",
        cache_unit="fixed-candidate",
        cache_root=None,
        render_guard=False,
    )


def test_internal_row_request_json_is_stable_across_python_hash_seeds() -> (
    None
):
    """The submitted row JSON and its identity must not vary with hash seed.

    ``dr_providers`` models three provider-definition fields as frozensets.
    This fixed request independently populates all three and asserts the
    consumer serialization boundary without generating or evaluating a C18
    pool: the submitted row is byte-identical, and hashes identically, under
    fresh interpreters with different hash seeds.
    """
    repo_root = str(Path(__file__).resolve().parents[2])
    script = _CROSS_SEED_REQUEST_SCRIPT.format(repo_root=repo_root)
    outputs: list[str] = []
    for seed in ("0", "1", "42"):
        environment = dict(os.environ)
        environment["PYTHONHASHSEED"] = seed
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert completed.returncode == 0, completed.stderr
        outputs.append(completed.stdout)

    assert len(set(outputs)) == 1, outputs


@pytest.mark.parametrize(
    ("split_name", "evaluation_role"),
    [
        ("official", EvaluationRole.INTERNAL),
        ("internal", EvaluationRole.OFFICIAL),
    ],
)
def test_internal_eval_rejects_binding_role_mismatch_before_restore(
    monkeypatch,
    split_name: str,
    evaluation_role: EvaluationRole,
) -> None:
    exp = _tiny_experiment("c18")
    sampling = getattr(exp.eval_configs, split_name)
    binding = EvaluationBinding(
        schema_version=EVALUATION_BINDING_SCHEMA_VERSION,
        eval_config=eval_config_reference(sampling.eval_config),
        role=evaluation_role,
        authority_principal=(
            "test-authority"
            if evaluation_role is EvaluationRole.OFFICIAL
            else None
        ),
        campaign="env-test",
    )

    def should_not_restore(*_args, **_kwargs):
        raise AssertionError("role mismatch must fail before partial restore")

    def should_not_build(_request: InternalRowRequest) -> ProcessJob:
        raise AssertionError("role mismatch must fail before job construction")

    monkeypatch.setattr(
        "whetstone.evaluation.drivers.internal.index_partial_records",
        should_not_restore,
    )
    with pytest.raises(ValueError, match="does not match split role"):
        run_internal_eval(
            exp,
            candidate=exp.initial_candidate,
            sampling=sampling,
            execution_policy=execution_policy(),
            row_job_factory=should_not_build,
            evaluation_binding=binding,
        )


def test_internal_redrive_preserves_phase_bounds(monkeypatch) -> None:
    exp = _tiny_experiment("c18")
    calls: list[tuple[int, float | None]] = []

    def pool(specs, *, concurrency, is_rate_limited, max_wall_seconds):
        del is_rate_limited
        calls.append((concurrency, max_wall_seconds))
        first = len(calls) == 1
        outcome = InternalRowOutcome(
            score=None if first else 1.0,
            row_state=(
                ExecutedRowState.FAILED if first else ExecutedRowState.SUCCESS
            ),
            executed_component_steps=(
                ()
                if first
                else _successful_internal_outcome().executed_component_steps
            ),
            output_text=None if first else "test output",
            failure_code="rate_limit" if first else "",
            rate_limited=first,
            redrivable=first,
        )
        return PoolOutcome(
            results=tuple(
                FanoutResult(
                    key=spec.key,
                    status=FanoutStatus.COMPLETED,
                    value=outcome,
                )
                for spec in specs
            ),
            effective_concurrency=2 if first else concurrency,
            concurrency_halved=first,
            deadline_reached=False,
            guard_timeouts=0,
        )

    monkeypatch.setattr(
        "whetstone.evaluation.drivers.internal.run_call_pool", pool
    )
    result = run_internal_eval(
        exp,
        candidate=exp.initial_candidate,
        sampling=exp.eval_configs.internal,
        execution_policy=execution_policy(),
        row_job_factory=row_job_factory(
            lambda _instance, _repeat, _drive: _successful_internal_outcome()
        ),
        evaluation_binding=_binding(exp),
        concurrency=4,
        max_wall_seconds=10.0,
    )

    assert calls[0][0] == 4
    assert calls[1][0] == 2
    assert calls[1][1] is not None
    assert calls[0][1] is not None
    assert calls[1][1] <= calls[0][1]
    assert result.aggregate.rows_failed == 0


def test_internal_resume_requires_exact_evaluation_binding(
    tmp_path: Path,
) -> None:
    from whetstone.execution.partials import PartialLog

    exp = _tiny_experiment("c18")
    sampling = exp.eval_configs.internal
    binding_a = _binding(exp)
    binding_b = binding_a.model_copy(update={"campaign": "other-campaign"})
    log = PartialLog(path=tmp_path / "internal-binding.partial")
    reply = _correct_reply("c18", sampling.instances)

    run_internal_eval(
        exp,
        candidate=exp.initial_candidate,
        sampling=sampling,
        execution_policy=execution_policy(),
        row_job_factory=_internal_jobs(exp, reply),
        evaluation_binding=binding_a,
        partial_log=log,
    )
    identities_a = {record.request_identity for record in log.load()}

    served_b: list[str] = []
    run_internal_eval(
        exp,
        candidate=exp.initial_candidate,
        sampling=sampling,
        execution_policy=execution_policy(),
        row_job_factory=_internal_jobs(exp, reply, served=served_b),
        evaluation_binding=binding_b,
        partial_log=log,
    )

    assert len(served_b) == (
        len(sampling.instances) * sampling.repeat_plan.repeat_count
    )
    identities_b = {
        record.request_identity
        for record in log.load()
        if record.request_identity not in identities_a
    }
    assert len(identities_b) == len(identities_a)


def test_internal_partial_resume_restores_exact_trace(tmp_path: Path) -> None:
    from whetstone.execution.partials import PartialLog

    exp = _tiny_experiment("c18")
    sampling = exp.eval_configs.internal
    log = PartialLog(path=tmp_path / "internal-trace.partial")
    reply = _correct_reply("c18", sampling.instances)
    first = run_internal_eval(
        exp,
        candidate=exp.initial_candidate,
        sampling=sampling,
        execution_policy=execution_policy(),
        row_job_factory=_internal_jobs(exp, reply),
        evaluation_binding=_binding(exp),
        partial_log=log,
    )

    def boom(_request: InternalRowRequest) -> ProcessJob:
        raise AssertionError("restored internal rows must not execute")

    resumed = run_internal_eval(
        exp,
        candidate=exp.initial_candidate,
        sampling=sampling,
        execution_policy=execution_policy(),
        row_job_factory=boom,
        evaluation_binding=_binding(exp),
        partial_log=log,
    )

    assert resumed.outputs == first.outputs


def test_internal_pending_ordinal_zero_resumes_at_ordinal_one(
    tmp_path: Path, monkeypatch
) -> None:
    from whetstone.execution.partials import PartialLog

    exp = _tiny_experiment("c18")
    sampling = exp.eval_configs.internal
    log = PartialLog(path=tmp_path / "internal-redrive.partial")
    pending = InternalRowOutcome(
        score=None,
        row_state=ExecutedRowState.FAILED,
        executed_component_steps=(),
        failure_code="transport_error",
        redrivable=True,
    )
    pool_calls = 0

    def crash_after_ordinal_zero(
        specs, *, concurrency, is_rate_limited, max_wall_seconds
    ):
        nonlocal pool_calls
        del is_rate_limited, max_wall_seconds
        pool_calls += 1
        if pool_calls == 2:
            raise RuntimeError("simulated crash before ordinal one")
        for spec in specs:
            spec.commit(pending)
        return PoolOutcome(
            results=tuple(
                FanoutResult(
                    key=spec.key,
                    status=FanoutStatus.COMPLETED,
                    value=pending,
                )
                for spec in specs
            ),
            effective_concurrency=concurrency,
            concurrency_halved=False,
            deadline_reached=False,
            guard_timeouts=0,
        )

    monkeypatch.setattr(
        "whetstone.evaluation.drivers.internal.run_call_pool",
        crash_after_ordinal_zero,
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        run_internal_eval(
            exp,
            candidate=exp.initial_candidate,
            sampling=sampling,
            execution_policy=execution_policy(),
            row_job_factory=row_job_factory(lambda *_args: pending),
            evaluation_binding=_binding(exp),
            partial_log=log,
        )
    assert {record.redrive_pending for record in log.load()} == {True}

    monkeypatch.undo()
    resumed_ordinals: list[int] = []

    def success(_instance, _repeat: int, drive_ordinal: int):
        resumed_ordinals.append(drive_ordinal)
        return _successful_internal_outcome()

    run_internal_eval(
        exp,
        candidate=exp.initial_candidate,
        sampling=sampling,
        execution_policy=execution_policy(),
        row_job_factory=row_job_factory(success),
        evaluation_binding=_binding(exp),
        partial_log=log,
    )
    assert resumed_ordinals
    assert set(resumed_ordinals) == {1}
    assert {record.redrive_pending for record in log.load()} == {False, True}


def test_internal_terminal_timeout_persists_both_exact_requests(
    tmp_path: Path, monkeypatch
) -> None:
    from whetstone.execution.partials import PartialLog

    exp = _tiny_experiment("c18")
    sampling = exp.eval_configs.official
    log = PartialLog(path=tmp_path / "internal-timeout.partial")

    def timeout_pool(specs, *, concurrency, is_rate_limited, max_wall_seconds):
        del is_rate_limited, max_wall_seconds
        return PoolOutcome(
            results=tuple(
                FanoutResult(key=spec.key, status=FanoutStatus.UNIT_TIMEOUT)
                for spec in specs
            ),
            effective_concurrency=concurrency,
            concurrency_halved=False,
            deadline_reached=False,
            guard_timeouts=len(specs),
        )

    monkeypatch.setattr(
        "whetstone.evaluation.drivers.internal.run_call_pool", timeout_pool
    )
    run_internal_eval(
        exp,
        candidate=exp.initial_candidate,
        sampling=sampling,
        execution_policy=execution_policy(),
        row_job_factory=row_job_factory(
            lambda *_args: _successful_internal_outcome()
        ),
        evaluation_binding=_binding(exp, role=EvaluationRole.OFFICIAL),
        partial_log=log,
    )

    records = log.load()
    expected_rows = len(sampling.instances) * sampling.repeat_plan.repeat_count
    assert len(records) == expected_rows * 2
    assert {record.failure_code for record in records} == {"runner_timeout"}
    assert {record.redrive_pending for record in records} == {False, True}
    assert len({record.request_identity for record in records}) == len(records)

    monkeypatch.undo()

    def boom(_request: InternalRowRequest) -> ProcessJob:
        raise AssertionError(
            "ordinal-one timeout must restore without repayment"
        )

    resumed = run_internal_eval(
        exp,
        candidate=exp.initial_candidate,
        sampling=sampling,
        execution_policy=execution_policy(),
        row_job_factory=boom,
        evaluation_binding=_binding(exp, role=EvaluationRole.OFFICIAL),
        partial_log=log,
    )
    assert resumed.aggregate.rows_failed == expected_rows


def test_invalid_prompt_is_rejected_before_transport() -> None:
    exp = _tiny_experiment("c18")
    invalid = Candidate(
        candidate_id="invalid-input",
        base_ref=exp.initial_candidate.base_ref,
        payload={MUTATION_FIELD: "Answer {unavailable_gold}."},
    )
    served: list[str] = []

    with pytest.raises(PromptInputError) as error:
        run_internal_eval(
            exp,
            candidate=invalid,
            sampling=exp.eval_configs.internal,
            execution_policy=execution_policy(),
            row_job_factory=_internal_jobs(
                exp,
                constant_reply("unused"),
                candidate=invalid,
                served=served,
            ),
            evaluation_binding=_binding(exp),
        )

    assert error.value.offending == ("unavailable_gold",)
    assert served == []


def test_internal_eval_is_deterministic() -> None:
    exp = _tiny_experiment("c18")
    internal_insts = exp.eval_configs.internal.instances
    reply = _correct_reply("c18", internal_insts)
    a = run_internal_eval(
        exp,
        candidate=exp.initial_candidate,
        sampling=exp.eval_configs.internal,
        execution_policy=execution_policy(),
        row_job_factory=_internal_jobs(exp, reply),
        evaluation_binding=_binding(exp),
    )
    b = run_internal_eval(
        exp,
        candidate=exp.initial_candidate,
        sampling=exp.eval_configs.internal,
        execution_policy=execution_policy(),
        row_job_factory=_internal_jobs(exp, reply),
        evaluation_binding=_binding(exp),
    )
    assert a.aggregate.aggregation_output.value == (
        b.aggregate.aggregation_output.value
    )
    assert isinstance(a.reward, Reward)
    assert isinstance(b.reward, Reward)
    assert a.reward.value == b.reward.value
    assert a.aggregate.graph_hash == b.aggregate.graph_hash


def test_blank_generation_is_a_failed_row_not_a_silent_zero() -> None:
    # A blank generation is not an accepted Generation (a provider semantic
    # failure); the internal-eval marks it a FAILED row. Under the default
    # PROPAGATE policy that makes the aggregate visibly incomplete (value
    # None) -- never a silent 0 -- so on the internal/optimizer path (reward
    # applied) the FAIL Reward Policy surfaces the TYPED
    # CandidateEvaluationFailure the optimizer loop handles, not a bare crash.
    exp = _tiny_experiment("c18")
    with pytest.raises(CandidateEvaluationFailure):
        run_internal_eval(
            exp,
            candidate=exp.initial_candidate,
            sampling=exp.eval_configs.internal,
            execution_policy=execution_policy(),
            row_job_factory=_internal_jobs(exp, constant_reply("   ")),
            evaluation_binding=_binding(exp),
        )


def test_official_eval_incomplete_aggregate_derives_no_reward() -> None:
    # An official-role binding with incomplete
    # evidence (all-blank -> failed rows -> aggregate None, PROPAGATE) must
    # NOT crash and must derive NO Reward -- visible incompleteness only.
    exp = _tiny_experiment("c18")
    result = run_internal_eval(
        exp,
        candidate=exp.initial_candidate,
        sampling=exp.eval_configs.official,
        execution_policy=execution_policy(),
        row_job_factory=_internal_jobs(exp, constant_reply("   ")),
        evaluation_binding=_binding(exp, role=EvaluationRole.OFFICIAL),
    )
    assert result.reward is None
    assert result.aggregate.aggregation_output.value is None
    assert result.aggregate.rows_failed > 0


def test_failed_rows_still_visible_in_provenance() -> None:
    # All-failed rows leave the reduction non-OK (never a fabricated zero),
    # while remaining counted in the aggregate provenance.
    from whetstone.evaluation.code.aggregate import (
        RowValue,
        TaskRows,
        unweighted_task_mean,
    )

    experiment = _tiny_experiment("c18")
    sampling = experiment.eval_configs.internal
    task_rows = tuple(
        TaskRows(
            task_identity=task_identity,
            rows=(RowValue(failed=True), RowValue(failed=True)),
        )
        for task_identity in sampling.task_set.task_identities
    )
    agg = unweighted_task_mean(
        aggregate_name="env_exact_match",
        graph_hash=experiment.rollout_definition.graph_hash,
        evaluation_binding_hash="c" * 64,
        task_rows=task_rows,
        plan=sampling.evaluation_matrix_plan,
    )
    assert agg.rows_failed == len(task_rows) * 2
    assert agg.rows_present == 0
    assert agg.aggregation_output.status is not AggregationStatus.OK


def test_unknown_env_rejected() -> None:
    from whetstone.envs.registry import UnknownEnvError

    with pytest.raises(UnknownEnvError):
        build_env_experiment("c99", model=_MODEL)
