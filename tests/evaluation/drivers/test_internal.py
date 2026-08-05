"""Internal evaluation driver and process-wire contracts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from dr_code.eval import AggregationStatus

from tests.envs.support import (
    constant_reply,
    execution_policy,
    process_row_job_factory,
    row_job_factory,
)
from tests.evaluation.drivers import support as driver_support
from tests.evaluation.drivers.support import (
    _SPLIT_FIT_CEILING,
    _binding,
    _correct_reply,
    _internal_jobs,
    _successful_internal_outcome,
    _tiny_experiment,
)
from whetstone.core.roles import EvaluationRole
from whetstone.envs.reward import CandidateEvaluationFailure
from whetstone.envs.rollout_definition import PromptInputError
from whetstone.evaluation.drivers.internal import (
    InternalRowOutcome,
    InternalRowRequest,
    InternalRowResult,
    ProcessInstance,
    process_request_identity,
    run_internal_eval,
    start_phase_deadline,
)
from whetstone.evaluation.traces import ExecutedRowState
from whetstone.execution.fanout import (
    FanoutResult,
    FanoutStatus,
    PoolOutcome,
    ProcessJob,
)
from whetstone.experiment.binding import (
    EVALUATION_BINDING_SCHEMA_VERSION,
    EvaluationBinding,
    eval_config_reference,
)
from whetstone.experiment.candidate import Candidate
from whetstone.experiment.reward import Reward
from whetstone.optimization.proposal.mutation import MUTATION_FIELD


def test_tiny_experiment_split_fit_search_is_bounded(monkeypatch) -> None:
    attempted_sizes: list[int] = []

    def never_fits(_env, n: int) -> bool:
        attempted_sizes.append(n)
        return False

    monkeypatch.setattr(driver_support, "_split_fits", never_fits)
    with pytest.raises(AssertionError) as error:
        _tiny_experiment("c11")

    assert attempted_sizes == list(range(1, _SPLIT_FIT_CEILING + 1))
    assert f"attempted_sizes={attempted_sizes}" in str(error.value)
    assert f"final_attempted_size={_SPLIT_FIT_CEILING}" in str(error.value)


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
@pytest.mark.process_integration
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


@pytest.mark.process_integration
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


@pytest.mark.process_integration
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


@pytest.mark.process_integration
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


@pytest.mark.process_integration
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


@pytest.mark.process_integration
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

from tests.evaluation.drivers.support import _fixed_unordered_provider_request

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


@pytest.mark.process_integration
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
    repo_root = str(Path(__file__).resolve().parents[3])
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


@pytest.mark.process_integration
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


@pytest.mark.process_integration
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


@pytest.mark.process_integration
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


@pytest.mark.process_integration
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


@pytest.mark.process_integration
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


@pytest.mark.process_integration
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
