"""The cross-env factory + process-isolated internal-eval loop.

The load-bearing proof: a full internal-eval of the naive candidate on a
tiny pool, driven by a FAKE transport, produces a valid Rollout Aggregate +
Reward -- per env, with no live paid call.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from dr_code.eval import AggregationStatus

from tests.envs.support import (
    ReplyFn,
    constant_reply,
    execution_policy,
    process_row_job_factory,
    row_job_factory,
)
from whetstone.envs.factory import EnvExperiment, build_env_experiment
from whetstone.envs.internal_eval import (
    InternalRowOutcome,
    InternalRowRequest,
    InternalRowResult,
    ProcessInstance,
    process_request_identity,
    run_internal_eval,
    start_phase_deadline,
)
from whetstone.envs.oracle_operator import env_exact_match_score
from whetstone.envs.registry import ENV_NAMES, env_spec
from whetstone.envs.reward import CandidateEvaluationFailure
from whetstone.envs.rollout_definition import PromptInputError
from whetstone.evaluation_role import EvaluationRole
from whetstone.execution.fanout import (
    FanoutResult,
    FanoutStatus,
    PoolOutcome,
    ProcessJob,
)
from whetstone.optimization.mutation import MUTATION_FIELD
from whetstone.optimization.reward import Reward
from whetstone.optimization.schema import (
    EVALUATION_BINDING_SCHEMA_VERSION,
    Candidate,
    EvaluationBinding,
    eval_config_reference,
)

_MODEL = "openai/gpt-5-nano"
_SPLIT = (2, 2, 2)


def _tiny_experiment(env_name: str) -> EnvExperiment:
    # n_per_stratum=1 gives >= 4 instances (all envs have >= 4 strata except
    # c18 which has 4); a (2,2,2) split needs >= 6, so grow the pool until it
    # is large enough. For a stratified-split env (c22, whose pool is blocked)
    # each stratum must independently hold its per-stratum quota, so grow until
    # the stratified split is satisfiable rather than only until the total
    # instance count clears sum(_SPLIT).
    env = env_spec(env_name)
    n = 1
    while not _split_fits(env, n):
        n += 1
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
                failed=True,
                failure_code="blank_generation",
            )
        score = env_exact_match_score(
            env=env,
            generation=text,
            gold=instance.gold,
            evaluation_procedure_config_hash=procedure_hash,
        )
        return InternalRowOutcome(
            score=float(score.value),
            output_text=text,
        )

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
    from whetstone.envs.d1_eval import D1RowRequest, D1RowResult
    from whetstone.envs.ed1_eval import Ed1RowRequest, Ed1RowResult

    assert InternalRowRequest.model_fields["schema_name"].default == (
        "whetstone.envs.internal_row_request/v2"
    )
    assert InternalRowResult.model_fields["schema_name"].default == (
        "whetstone.envs.internal_row_result/v2"
    )
    assert D1RowRequest.model_fields["schema_name"].default == (
        "whetstone.envs.d1_row_request/v2"
    )
    assert D1RowResult.model_fields["schema_name"].default == (
        "whetstone.envs.d1_row_result/v2"
    )
    assert Ed1RowRequest.model_fields["schema_name"].default == (
        "whetstone.envs.ed1_row_request/v2"
    )
    assert Ed1RowResult.model_fields["schema_name"].default == (
        "whetstone.envs.ed1_row_result/v2"
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


def test_internal_result_for_different_request_is_rejected() -> None:
    exp = _tiny_experiment("c18")

    def mismatched(request: InternalRowRequest) -> ProcessJob:
        result = InternalRowResult(
            request_identity="0" * 64,
            outcome=InternalRowOutcome(score=1.0),
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
    provider = requests[0].model_dump(mode="json")["provider_call_config"]
    definition = provider["definition"]
    constraints = definition["constraints"]
    assert definition["required_controls"] == sorted(
        definition["required_controls"]
    )
    assert definition["extension_keys"] == sorted(definition["extension_keys"])
    assert constraints["supported_controls"] == sorted(
        constraints["supported_controls"]
    )


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
        "whetstone.envs.internal_eval.index_partial_records",
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
            failed=first,
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

    monkeypatch.setattr("whetstone.envs.internal_eval.run_call_pool", pool)
    result = run_internal_eval(
        exp,
        candidate=exp.initial_candidate,
        sampling=exp.eval_configs.internal,
        execution_policy=execution_policy(),
        row_job_factory=row_job_factory(
            lambda _instance, _repeat, _drive: InternalRowOutcome(score=1.0)
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


def test_internal_pending_ordinal_zero_resumes_at_ordinal_one(
    tmp_path: Path, monkeypatch
) -> None:
    from whetstone.execution.partials import PartialLog

    exp = _tiny_experiment("c18")
    sampling = exp.eval_configs.internal
    log = PartialLog(path=tmp_path / "internal-redrive.partial")
    pending = InternalRowOutcome(
        score=None,
        failed=True,
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
        "whetstone.envs.internal_eval.run_call_pool",
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
        return InternalRowOutcome(score=1.0)

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
        "whetstone.envs.internal_eval.run_call_pool", timeout_pool
    )
    run_internal_eval(
        exp,
        candidate=exp.initial_candidate,
        sampling=sampling,
        execution_policy=execution_policy(),
        row_job_factory=row_job_factory(
            lambda *_args: InternalRowOutcome(score=1.0)
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
    from whetstone.code_eval.aggregate import (
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
