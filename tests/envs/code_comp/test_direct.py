from __future__ import annotations

from pathlib import Path

import pytest

from tests.envs.support import (
    evaluation_binding,
    execution_policy,
    process_row_job_factory,
    row_job_factory,
    synthetic_code_comp_tasks,
)
from whetstone.envs.code_comp.constants import (
    CODE_COMP_DATASET_REVISION,
    CODE_COMP_INVALID_BODY,
    CODE_COMP_SUBMISSION_SCORE_NAME,
)
from whetstone.envs.code_comp.modes.direct import (
    DIRECT_INPUT_ARMS,
    DIRECT_RENAMED_ARM,
    DIRECT_WRAPPER_BODY_CEILING,
    DIRECT_WRAPPER_BODY_NAIVE,
    build_direct_experiment,
    direct_ceiling_candidate,
    direct_initial_candidate,
    render_direct_frame,
)
from whetstone.envs.code_comp.mutation_surface import (
    InstructionBodyError,
    instruction_body_rejection,
)
from whetstone.envs.code_comp.scoring import (
    BatchScoringDeadlineExceeded,
    CodeScore,
)
from whetstone.envs.input_transform import (
    direct_prompt,
    rename_identifier,
    split_prompt,
)
from whetstone.envs.rollout_definition import LLM_NODE_ID
from whetstone.evaluation.drivers.code_comp.direct import (
    DirectGeneratedRowOutcome,
    DirectRowOutcome,
    DirectRowRequest,
    _input_arm_text,
    run_direct_eval,
)
from whetstone.evaluation.drivers.internal import _llm_component_step
from whetstone.evaluation.traces import ExecutedRowState
from whetstone.execution.fanout import (
    FanoutResult,
    FanoutStatus,
    PoolOutcome,
)
from whetstone.execution.partials import PartialLog
from whetstone.optimization.proposal.mutation import MUTATION_FIELD


def _tasks(limit: int = 3):
    return synthetic_code_comp_tasks(limit)


def _successful_outcome() -> DirectRowOutcome:
    output_text = "def rebuilt():\n    return 1\n"
    return DirectRowOutcome(
        submission_score=1.0,
        output_text=output_text,
        row_state=ExecutedRowState.SUCCESS,
        executed_component_steps=(
            _llm_component_step(
                trace_index=0,
                component_id=LLM_NODE_ID,
                prompt="test prompt",
                generation=output_text,
            ),
        ),
    )


def _generated_outcome(output_text: str) -> DirectGeneratedRowOutcome:
    return DirectGeneratedRowOutcome(
        output_text=output_text,
        executed_component_steps=(
            _llm_component_step(
                trace_index=0,
                component_id=LLM_NODE_ID,
                prompt="test prompt",
                generation=output_text,
            ),
        ),
    )


def test_coordinator_scores_generated_d1_rows_in_one_batch() -> None:
    tasks = _tasks(2)
    experiment = build_direct_experiment(
        tasks=tasks,
        num_samples=2,
        internal_n=2,
        official_n=2,
    )
    batch_sizes: list[int] = []

    def generated(instance, sample_index: int, _drive_ordinal: int):
        return _generated_outcome(f"# {instance.id} sample {sample_index}")

    def score_batch(inputs, *, max_wall_seconds: float | None = None):
        assert max_wall_seconds is None
        batch_sizes.append(len(inputs))
        return tuple(
            CodeScore(
                passed=True,
                infrastructure_unknown=False,
                outcome="passed",
            )
            for _item in inputs
        )

    result = run_direct_eval(
        experiment,
        candidate_body=DIRECT_WRAPPER_BODY_NAIVE,
        candidate_id="d1-batch",
        sampling=experiment.eval_configs.internal,
        execution_policy=execution_policy(),
        row_job_factory=row_job_factory(generated),
        evaluation_binding=evaluation_binding(
            experiment.eval_configs.internal
        ),
        batch_scorer=score_batch,
    )

    assert batch_sizes == [4]
    assert [output.score for output in result.outputs] == [1.0] * 4


def test_d1_batch_deadline_leaves_generated_rows_missing(
    monkeypatch,
) -> None:
    experiment = build_direct_experiment(
        tasks=_tasks(1),
        num_samples=1,
        internal_n=1,
        official_n=1,
    )

    def completed_pool(
        specs, *, concurrency, is_rate_limited, max_wall_seconds
    ):
        del is_rate_limited, max_wall_seconds
        return PoolOutcome(
            results=tuple(
                FanoutResult(
                    key=spec.key,
                    status=FanoutStatus.COMPLETED,
                    value=_generated_outcome("generated"),
                )
                for spec in specs
            ),
            effective_concurrency=concurrency,
            concurrency_halved=False,
            deadline_reached=False,
            guard_timeouts=0,
        )

    remaining_walls = iter((1.0, 0.0))
    monkeypatch.setattr(
        "whetstone.evaluation.drivers.code_comp.direct.run_call_pool",
        completed_pool,
    )
    monkeypatch.setattr(
        "whetstone.evaluation.drivers.code_comp.direct.remaining_phase_wall_seconds",
        lambda _deadline: next(remaining_walls),
    )

    def must_not_score(inputs, *, max_wall_seconds: float | None = None):
        del inputs
        raise AssertionError(
            f"expired phase started scoring with {max_wall_seconds=}"
        )

    result = run_direct_eval(
        experiment,
        candidate_body=DIRECT_WRAPPER_BODY_NAIVE,
        candidate_id="d1-batch-deadline",
        sampling=experiment.eval_configs.official,
        execution_policy=execution_policy(),
        row_job_factory=_passing_jobs(),
        evaluation_binding=evaluation_binding(
            experiment.eval_configs.official,
            official=True,
        ),
        max_wall_seconds=10.0,
        batch_scorer=must_not_score,
    )

    assert result.submission_score_aggregate.rows_missing == 1
    assert result.outputs[0].missing is True


def test_d1_batch_deadline_exception_leaves_generated_rows_missing(
    monkeypatch,
) -> None:
    experiment = build_direct_experiment(
        tasks=_tasks(1),
        num_samples=1,
        internal_n=1,
        official_n=1,
    )

    def completed_pool(
        specs, *, concurrency, is_rate_limited, max_wall_seconds
    ):
        del is_rate_limited, max_wall_seconds
        return PoolOutcome(
            results=tuple(
                FanoutResult(
                    key=spec.key,
                    status=FanoutStatus.COMPLETED,
                    value=_generated_outcome("generated"),
                )
                for spec in specs
            ),
            effective_concurrency=concurrency,
            concurrency_halved=False,
            deadline_reached=False,
            guard_timeouts=0,
        )

    monkeypatch.setattr(
        "whetstone.evaluation.drivers.code_comp.direct.run_call_pool",
        completed_pool,
    )

    def deadline(inputs, *, max_wall_seconds: float | None = None):
        del inputs
        assert max_wall_seconds is not None
        raise BatchScoringDeadlineExceeded

    result = run_direct_eval(
        experiment,
        candidate_body=DIRECT_WRAPPER_BODY_NAIVE,
        candidate_id="d1-mid-batch-deadline",
        sampling=experiment.eval_configs.official,
        execution_policy=execution_policy(),
        row_job_factory=_passing_jobs(),
        evaluation_binding=evaluation_binding(
            experiment.eval_configs.official,
            official=True,
        ),
        max_wall_seconds=10.0,
        batch_scorer=deadline,
    )

    assert result.submission_score_aggregate.rows_missing == 1


def _passing_jobs(*, served: list[str] | None = None):
    def outcome(instance, _repeat: int, _drive_ordinal: int):
        if served is not None:
            served.append(str(instance.id))
        return _successful_outcome()

    return row_job_factory(outcome)


def test_each_input_arm_has_distinct_graph_and_eval_identity() -> None:
    tasks = _tasks()
    graphs: set[str] = set()
    evals: set[str] = set()
    for arm in DIRECT_INPUT_ARMS:
        experiment = build_direct_experiment(input_arm=arm, tasks=tasks)
        graphs.add(experiment.rollout_definition.graph_hash)
        evals.add(experiment.eval_configs.official.eval_config.config_hash)
    assert len(graphs) == len(DIRECT_INPUT_ARMS)
    assert len(evals) == len(DIRECT_INPUT_ARMS)

    renamed = [
        build_direct_experiment(
            input_arm=DIRECT_RENAMED_ARM, tasks=tasks, rename_token=token
        )
        for token in ("target_fxn", "other_fxn")
    ]
    assert (
        renamed[0].rollout_definition.graph_hash
        != renamed[1].rollout_definition.graph_hash
    )
    assert (
        renamed[0].eval_configs.official.eval_config.config_hash
        != renamed[1].eval_configs.official.eval_config.config_hash
    )


def test_rename_token_does_not_churn_identity_on_arms_that_ignore_it() -> None:
    tasks = _tasks()
    for arm in DIRECT_INPUT_ARMS:
        if arm == DIRECT_RENAMED_ARM:
            continue
        a = build_direct_experiment(
            input_arm=arm, tasks=tasks, rename_token="target_fxn"
        )
        b = build_direct_experiment(
            input_arm=arm, tasks=tasks, rename_token="other_fxn"
        )
        assert (
            a.rollout_definition.graph_hash == b.rollout_definition.graph_hash
        )
        assert (
            a.eval_configs.official.eval_config.config_hash
            == b.eval_configs.official.eval_config.config_hash
        )


@pytest.mark.parametrize(
    "bad", ["not a token", "2fxn", "", "def", "class", "target-fxn", "a.b"]
)
def test_invalid_rename_token_is_rejected_at_build_time(bad: str) -> None:
    with pytest.raises(ValueError, match="rename_token"):
        build_direct_experiment(
            input_arm=DIRECT_RENAMED_ARM, tasks=_tasks(1), rename_token=bad
        )


def test_valid_rename_token_is_accepted() -> None:
    experiment = build_direct_experiment(
        input_arm=DIRECT_RENAMED_ARM, tasks=_tasks(1), rename_token="solve_it"
    )
    assert experiment.rename_token == "solve_it"


def test_naive_prompt_matches_canonical_direct_prompt() -> None:
    tasks = _tasks(1)
    for arm in DIRECT_INPUT_ARMS:
        experiment = build_direct_experiment(input_arm=arm, tasks=tasks)
        instance = experiment.eval_configs.internal.tasks[0]
        body, _ = _input_arm_text(experiment, instance)
        actual = render_direct_frame(DIRECT_WRAPPER_BODY_NAIVE, input_arm=body)
        task = experiment.humaneval_for(instance)
        expected = direct_prompt(
            f"direct_{arm}",
            split_prompt(task.prompt, task.entry_point),
            rename_token=experiment.rename_token,
        )
        assert actual == expected


def test_renamed_arm_scrubs_and_scores_renamed_entry_point() -> None:
    experiment = build_direct_experiment(input_arm="renamed", tasks=_tasks(1))
    instance = experiment.eval_configs.internal.tasks[0]
    body, score_task = _input_arm_text(experiment, instance)
    original = experiment.humaneval_for(instance)
    assert original.entry_point not in body
    assert experiment.rename_token in body
    assert score_task.entry_point == experiment.rename_token
    assert original.entry_point not in score_task.test


def test_candidates_and_pass_only_reward_are_explicit() -> None:
    naive = direct_initial_candidate()
    ceiling = direct_ceiling_candidate()
    assert naive.payload[MUTATION_FIELD] == DIRECT_WRAPPER_BODY_NAIVE
    assert ceiling.payload[MUTATION_FIELD] == DIRECT_WRAPPER_BODY_CEILING
    assert naive.payload != ceiling.payload
    experiment = build_direct_experiment(tasks=_tasks())
    assert [term.name for term in experiment.reward_policy.terms] == [
        CODE_COMP_SUBMISSION_SCORE_NAME
    ]
    assert experiment.dataset_revision == CODE_COMP_DATASET_REVISION


def test_body_restrictions_are_preflight_safe() -> None:
    rejection = instruction_body_rejection("Solve {input_arm} now.")
    assert rejection == ("{input_arm}",)
    assert CODE_COMP_INVALID_BODY
    assert instruction_body_rejection("Solve it carefully.") == ()
    tasks = _tasks(1)
    experiment = build_direct_experiment(tasks=tasks)
    served: list[str] = []

    with pytest.raises(InstructionBodyError) as error:
        run_direct_eval(
            experiment,
            candidate_body="Solve {input_arm} now.",
            candidate_id="invalid-body",
            sampling=experiment.eval_configs.internal,
            execution_policy=execution_policy(max_attempts=1),
            row_job_factory=_passing_jobs(served=served),
            evaluation_binding=evaluation_binding(
                experiment.eval_configs.internal
            ),
        )

    assert error.value.code == CODE_COMP_INVALID_BODY
    assert served == []


@pytest.mark.parametrize("arm", ["original", "docstring", "renamed"])
def test_direct_evaluator_records_exact_pass_rate(arm: str) -> None:
    tasks = _tasks(2)
    experiment = build_direct_experiment(
        input_arm=arm,
        tasks=tasks,
        num_samples=2,
        internal_n=2,
        official_n=2,
    )
    result = run_direct_eval(
        experiment,
        candidate_body=DIRECT_WRAPPER_BODY_NAIVE,
        candidate_id="d1-naive",
        sampling=experiment.eval_configs.internal,
        execution_policy=execution_policy(max_attempts=1),
        row_job_factory=_passing_jobs(),
        evaluation_binding=evaluation_binding(
            experiment.eval_configs.internal
        ),
    )
    assert (
        result.submission_score_aggregate.name
        == CODE_COMP_SUBMISSION_SCORE_NAME
    )
    assert (
        result.submission_score_aggregate.aggregation_output.value
        == pytest.approx(1)
    )
    assert result.submission_score_aggregate.num_samples == 2
    assert result.submission_score_aggregate.eval_config_hash == (
        experiment.eval_configs.internal.eval_config.config_hash
    )
    assert result.per_task_counts == (2, 2)
    assert len(result.outputs) == 4
    assert result.reward is not None
    assert (
        result.reward.input_citations[0].name
        == CODE_COMP_SUBMISSION_SCORE_NAME
    )


def test_d1_process_job_runs_real_row_driver() -> None:
    tasks = _tasks(1)
    experiment = build_direct_experiment(
        tasks=tasks,
        num_samples=1,
        internal_n=1,
        official_n=1,
    )
    result = run_direct_eval(
        experiment,
        candidate_body=DIRECT_WRAPPER_BODY_NAIVE,
        candidate_id="d1-process-job",
        sampling=experiment.eval_configs.internal,
        execution_policy=execution_policy(max_attempts=1),
        row_job_factory=process_row_job_factory(
            "tests.envs.process_workers:drive_d1_success"
        ),
        evaluation_binding=evaluation_binding(
            experiment.eval_configs.internal
        ),
    )

    assert result.submission_score_aggregate.rows_failed == 0
    assert result.submission_score_aggregate.aggregation_output.value == 1.0
    output = result.outputs[0]
    input_arm, _ = _input_arm_text(
        experiment, experiment.eval_configs.internal.tasks[0]
    )
    assert output.row_state is ExecutedRowState.SUCCESS
    assert output.executed_component_steps[0].model_dump(mode="json") == {
        "trace_index": 0,
        "component_id": "generate",
        "input_field_names": ["prompt"],
        "output_field_names": ["generation"],
        "inputs": {
            "prompt": render_direct_frame(
                DIRECT_WRAPPER_BODY_NAIVE, input_arm=input_arm
            )
        },
        "outputs": {"generation": output.output_text},
    }


def test_d1_v2_request_hash_is_pinned() -> None:
    experiment = build_direct_experiment(
        tasks=_tasks(1), num_samples=1, internal_n=1, official_n=1
    )
    requests: list[DirectRowRequest] = []
    base = _passing_jobs()

    def capture(request: DirectRowRequest):
        requests.append(request)
        return base(request)

    run_direct_eval(
        experiment,
        candidate_body=DIRECT_WRAPPER_BODY_NAIVE,
        candidate_id="code_comp-direct-golden",
        sampling=experiment.eval_configs.internal,
        execution_policy=execution_policy(max_attempts=1),
        row_job_factory=capture,
        evaluation_binding=evaluation_binding(
            experiment.eval_configs.internal
        ),
    )

    assert requests[0].request_hash == (
        "999cf939758541a9fc07cc1f1d0866028f652ad4ebf7bc3964529b6d3e4addca"
    )


@pytest.mark.parametrize(
    ("split_name", "official_binding"),
    [("official", False), ("internal", True)],
)
def test_d1_rejects_binding_role_mismatch_before_restore(
    monkeypatch, split_name: str, official_binding: bool
) -> None:
    experiment = build_direct_experiment(tasks=_tasks(), num_samples=1)
    sampling = getattr(experiment.eval_configs, split_name)

    def should_not_restore(*_args, **_kwargs):
        raise AssertionError("role mismatch must fail before partial restore")

    def should_not_build(_request):
        raise AssertionError("role mismatch must fail before job construction")

    monkeypatch.setattr(
        "whetstone.evaluation.drivers.code_comp.direct.index_partial_records",
        should_not_restore,
    )
    with pytest.raises(ValueError, match="does not match split role"):
        run_direct_eval(
            experiment,
            candidate_body=DIRECT_WRAPPER_BODY_NAIVE,
            candidate_id="d1-role-mismatch",
            sampling=sampling,
            execution_policy=execution_policy(max_attempts=1),
            row_job_factory=should_not_build,
            evaluation_binding=evaluation_binding(
                sampling, official=official_binding
            ),
        )


def test_direct_evaluator_resume_skips_recorded_rows(tmp_path: Path) -> None:
    tasks = _tasks(2)
    experiment = build_direct_experiment(tasks=tasks, num_samples=1)
    log = PartialLog(path=tmp_path / "d1.partial.jsonl")
    first = run_direct_eval(
        experiment,
        candidate_body=DIRECT_WRAPPER_BODY_NAIVE,
        candidate_id="d1-naive",
        sampling=experiment.eval_configs.internal,
        execution_policy=execution_policy(max_attempts=1),
        row_job_factory=_passing_jobs(),
        evaluation_binding=evaluation_binding(
            experiment.eval_configs.internal
        ),
        partial_log=log,
    )

    def boom(_instance, _repeat: int, _drive_ordinal: int):
        raise AssertionError("recorded rows must not be called again")

    resumed = run_direct_eval(
        experiment,
        candidate_body=DIRECT_WRAPPER_BODY_NAIVE,
        candidate_id="d1-naive",
        sampling=experiment.eval_configs.internal,
        execution_policy=execution_policy(max_attempts=1),
        row_job_factory=row_job_factory(boom),
        evaluation_binding=evaluation_binding(
            experiment.eval_configs.internal
        ),
        partial_log=log,
    )
    assert (
        resumed.submission_score_aggregate == first.submission_score_aggregate
    )
    assert resumed.outputs == first.outputs


def test_d1_resume_requires_exact_evaluation_binding(tmp_path: Path) -> None:
    experiment = build_direct_experiment(tasks=_tasks(2), num_samples=1)
    sampling = experiment.eval_configs.internal
    binding_a = evaluation_binding(sampling)
    binding_b = binding_a.model_copy(update={"campaign": "other-campaign"})
    log = PartialLog(path=tmp_path / "d1-binding.partial")

    run_direct_eval(
        experiment,
        candidate_body=DIRECT_WRAPPER_BODY_NAIVE,
        candidate_id="d1-binding",
        sampling=sampling,
        execution_policy=execution_policy(max_attempts=1),
        row_job_factory=_passing_jobs(),
        evaluation_binding=binding_a,
        partial_log=log,
    )
    identities_a = {record.request_hash for record in log.load()}

    served_b: list[str] = []
    run_direct_eval(
        experiment,
        candidate_body=DIRECT_WRAPPER_BODY_NAIVE,
        candidate_id="d1-binding",
        sampling=sampling,
        execution_policy=execution_policy(max_attempts=1),
        row_job_factory=_passing_jobs(served=served_b),
        evaluation_binding=binding_b,
        partial_log=log,
    )

    assert len(served_b) == len(sampling.tasks)
    identities_b = {
        record.request_hash
        for record in log.load()
        if record.request_hash not in identities_a
    }
    assert len(identities_b) == len(identities_a)


def test_d1_pending_ordinal_zero_resumes_at_ordinal_one(
    tmp_path: Path, monkeypatch
) -> None:
    experiment = build_direct_experiment(
        tasks=_tasks(1), num_samples=1, internal_n=1, official_n=1
    )
    sampling = experiment.eval_configs.internal
    log = PartialLog(path=tmp_path / "d1-redrive.partial")
    pending = DirectRowOutcome(
        submission_score=None,
        output_text=None,
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
        "whetstone.evaluation.drivers.code_comp.direct.run_call_pool",
        crash_after_ordinal_zero,
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        run_direct_eval(
            experiment,
            candidate_body=DIRECT_WRAPPER_BODY_NAIVE,
            candidate_id="d1-redrive",
            sampling=sampling,
            execution_policy=execution_policy(max_attempts=1),
            row_job_factory=row_job_factory(lambda *_args: pending),
            evaluation_binding=evaluation_binding(sampling),
            partial_log=log,
        )
    assert {record.redrive_pending for record in log.load()} == {True}

    monkeypatch.undo()
    resumed_ordinals: list[int] = []

    def success(_instance, _repeat: int, drive_ordinal: int):
        resumed_ordinals.append(drive_ordinal)
        return _successful_outcome()

    run_direct_eval(
        experiment,
        candidate_body=DIRECT_WRAPPER_BODY_NAIVE,
        candidate_id="d1-redrive",
        sampling=sampling,
        execution_policy=execution_policy(max_attempts=1),
        row_job_factory=row_job_factory(success),
        evaluation_binding=evaluation_binding(sampling),
        partial_log=log,
    )
    assert resumed_ordinals == [1]
    assert {record.redrive_pending for record in log.load()} == {False, True}


def test_d1_terminal_timeout_is_persisted_and_not_repaid(
    tmp_path: Path, monkeypatch
) -> None:
    experiment = build_direct_experiment(
        tasks=_tasks(1),
        num_samples=1,
        internal_n=1,
        official_n=1,
    )
    log = PartialLog(path=tmp_path / "d1-timeout.partial.jsonl")
    walls: list[float | None] = []

    def timed_out_pool(
        specs, *, concurrency, is_rate_limited, max_wall_seconds
    ):
        del is_rate_limited
        walls.append(max_wall_seconds)
        return PoolOutcome(
            results=tuple(
                FanoutResult(
                    key=spec.key,
                    status=FanoutStatus.UNIT_TIMEOUT,
                )
                for spec in specs
            ),
            effective_concurrency=concurrency,
            concurrency_halved=False,
            deadline_reached=False,
            guard_timeouts=len(specs),
        )

    monkeypatch.setattr(
        "whetstone.evaluation.drivers.code_comp.direct.run_call_pool",
        timed_out_pool,
    )
    run_direct_eval(
        experiment,
        candidate_body=DIRECT_WRAPPER_BODY_NAIVE,
        candidate_id="d1-timeout",
        sampling=experiment.eval_configs.official,
        execution_policy=execution_policy(max_attempts=1),
        row_job_factory=_passing_jobs(),
        max_wall_seconds=10.0,
        evaluation_binding=evaluation_binding(
            experiment.eval_configs.official, official=True
        ),
        partial_log=log,
    )

    records = log.load()
    assert len(records) == 2
    assert {record.failure_code for record in records} == {"runner_timeout"}
    assert {record.redrive_pending for record in records} == {False, True}
    assert len({record.request_hash for record in records}) == 2
    assert {record.split_role for record in records} == {
        experiment.eval_configs.official.split_role
    }
    assert len(walls) == 2
    assert walls[1] is not None and walls[0] is not None
    assert walls[1] <= walls[0]

    def boom(_request):
        raise AssertionError("terminal timeout must restore without repayment")

    monkeypatch.undo()
    resumed = run_direct_eval(
        experiment,
        candidate_body=DIRECT_WRAPPER_BODY_NAIVE,
        candidate_id="d1-timeout",
        sampling=experiment.eval_configs.official,
        execution_policy=execution_policy(max_attempts=1),
        row_job_factory=boom,
        evaluation_binding=evaluation_binding(
            experiment.eval_configs.official, official=True
        ),
        partial_log=log,
    )
    assert resumed.submission_score_aggregate.rows_failed == 1


def test_d1_phase_deadline_is_missing_and_not_redriven(monkeypatch) -> None:
    experiment = build_direct_experiment(
        tasks=_tasks(1),
        num_samples=1,
        internal_n=1,
        official_n=1,
    )
    calls = 0

    def deadline_pool(
        specs, *, concurrency, is_rate_limited, max_wall_seconds
    ):
        nonlocal calls
        del is_rate_limited, max_wall_seconds
        calls += 1
        return PoolOutcome(
            results=tuple(
                FanoutResult(
                    key=spec.key,
                    status=FanoutStatus.OPERATION_DEADLINE,
                )
                for spec in specs
            ),
            effective_concurrency=concurrency,
            concurrency_halved=False,
            deadline_reached=True,
            guard_timeouts=0,
        )

    monkeypatch.setattr(
        "whetstone.evaluation.drivers.code_comp.direct.run_call_pool",
        deadline_pool,
    )
    result = run_direct_eval(
        experiment,
        candidate_body=DIRECT_WRAPPER_BODY_NAIVE,
        candidate_id="d1-deadline",
        sampling=experiment.eval_configs.official,
        execution_policy=execution_policy(max_attempts=1),
        row_job_factory=_passing_jobs(),
        evaluation_binding=evaluation_binding(
            experiment.eval_configs.official, official=True
        ),
    )

    assert calls == 1
    assert result.submission_score_aggregate.rows_missing == 1
    assert result.submission_score_aggregate.rows_failed == 0


def test_identifier_rename_is_whole_token_only() -> None:
    text = "foo(foo_bar, obj.foo, food); foo(1)"
    assert rename_identifier(text, "foo", "target") == (
        "target(foo_bar, obj.target, food); target(1)"
    )
