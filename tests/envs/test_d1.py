"""Focused D1 direct-generation environment-contract tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.envs.support import (
    execution_policy,
    process_row_job_factory,
    row_job_factory,
    synthetic_ed1_tasks,
)
from whetstone.envs.d1 import (
    D1_INPUT_ARMS,
    D1_SUBMISSION_SCORE_NAME,
    D1_WRAPPER_BODY_CEILING,
    D1_WRAPPER_BODY_NAIVE,
    build_d1_experiment,
    d1_ceiling_candidate,
    d1_initial_candidate,
    render_d1_frame,
)
from whetstone.envs.d1_eval import D1RowOutcome, _input_arm_text, run_d1_eval
from whetstone.envs.ed1 import (
    ED1_DATASET_REVISION,
    ED1_INVALID_BODY,
    Ed1BodyError,
    ed1_body_rejection,
)
from whetstone.envs.input_transform import (
    direct_prompt,
    rename_identifier,
    split_prompt,
)
from whetstone.execution.fanout import (
    FanoutResult,
    FanoutStatus,
    PoolOutcome,
)
from whetstone.execution.partials import PartialLog
from whetstone.optimization.mutation import MUTATION_FIELD


def _tasks(limit: int = 3):
    return synthetic_ed1_tasks(limit)


def _passing_jobs(*, served: list[str] | None = None):
    def outcome(instance, _repeat: int, _drive_ordinal: int):
        if served is not None:
            served.append(str(instance.id))
        return D1RowOutcome(
            submission_score=1.0,
            output_text="def rebuilt():\n    return 1\n",
            failed=False,
        )

    return row_job_factory(outcome)


def test_each_input_arm_has_distinct_graph_and_eval_identity() -> None:
    tasks = _tasks()
    graphs: set[str] = set()
    evals: set[str] = set()
    for arm in D1_INPUT_ARMS:
        experiment = build_d1_experiment(input_arm=arm, tasks=tasks)
        graphs.add(experiment.rollout_definition.graph_hash)
        evals.add(
            experiment.eval_configs.official.eval_config.config_identity_hash
        )
    assert len(graphs) == len(D1_INPUT_ARMS)
    assert len(evals) == len(D1_INPUT_ARMS)


def test_naive_prompt_matches_canonical_direct_prompt() -> None:
    tasks = _tasks(1)
    for arm in D1_INPUT_ARMS:
        experiment = build_d1_experiment(input_arm=arm, tasks=tasks)
        instance = experiment.eval_configs.internal.instances[0]
        body, _ = _input_arm_text(experiment, instance)
        actual = render_d1_frame(D1_WRAPPER_BODY_NAIVE, input_arm=body)
        task = experiment.humaneval_for(instance)
        expected = direct_prompt(
            f"direct_{arm}",
            split_prompt(task.prompt, task.entry_point),
            rename_token=experiment.rename_token,
        )
        assert actual == expected


def test_renamed_arm_scrubs_and_scores_renamed_entry_point() -> None:
    experiment = build_d1_experiment(input_arm="renamed", tasks=_tasks(1))
    instance = experiment.eval_configs.internal.instances[0]
    body, score_task = _input_arm_text(experiment, instance)
    original = experiment.humaneval_for(instance)
    assert original.entry_point not in body
    assert experiment.rename_token in body
    assert score_task.entry_point == experiment.rename_token
    assert original.entry_point not in score_task.test


def test_candidates_and_pass_only_reward_are_explicit() -> None:
    naive = d1_initial_candidate()
    ceiling = d1_ceiling_candidate()
    assert naive.payload[MUTATION_FIELD] == D1_WRAPPER_BODY_NAIVE
    assert ceiling.payload[MUTATION_FIELD] == D1_WRAPPER_BODY_CEILING
    assert naive.payload != ceiling.payload
    experiment = build_d1_experiment(tasks=_tasks())
    assert [term.name for term in experiment.reward_policy.terms] == [
        D1_SUBMISSION_SCORE_NAME
    ]
    assert experiment.dataset_revision == ED1_DATASET_REVISION


def test_body_restrictions_are_preflight_safe() -> None:
    rejection = ed1_body_rejection("Solve {input_arm} now.")
    assert rejection == ("{input_arm}",)
    assert ED1_INVALID_BODY
    assert ed1_body_rejection("Solve it carefully.") == ()
    tasks = _tasks(1)
    experiment = build_d1_experiment(tasks=tasks)
    served: list[str] = []

    with pytest.raises(Ed1BodyError) as error:
        run_d1_eval(
            experiment,
            candidate_body="Solve {input_arm} now.",
            candidate_id="invalid-body",
            sampling=experiment.eval_configs.internal,
            execution_policy=execution_policy(max_attempts=1),
            row_job_factory=_passing_jobs(served=served),
            apply_reward=False,
        )

    assert error.value.code == ED1_INVALID_BODY
    assert served == []


@pytest.mark.parametrize("arm", ["original", "docstring", "renamed"])
def test_direct_evaluator_records_exact_pass_rate(arm: str) -> None:
    tasks = _tasks(2)
    experiment = build_d1_experiment(
        input_arm=arm,
        tasks=tasks,
        repeats=2,
        internal_n=2,
        official_n=2,
    )
    result = run_d1_eval(
        experiment,
        candidate_body=D1_WRAPPER_BODY_NAIVE,
        candidate_id="d1-naive",
        sampling=experiment.eval_configs.internal,
        execution_policy=execution_policy(max_attempts=1),
        row_job_factory=_passing_jobs(),
        apply_reward=True,
    )
    assert result.submission_score_aggregate.name == D1_SUBMISSION_SCORE_NAME
    assert (
        result.submission_score_aggregate.aggregation_output.value
        == pytest.approx(1)
    )
    assert result.submission_score_aggregate.repeat_count == 2
    assert result.submission_score_aggregate.eval_config_hash == (
        experiment.eval_configs.internal.eval_config.config_identity_hash
    )
    assert result.per_task_counts == (2, 2)
    assert len(result.outputs) == 4
    assert result.reward is not None
    assert result.reward.input_citations[0].name == D1_SUBMISSION_SCORE_NAME


def test_d1_process_job_runs_real_row_driver() -> None:
    tasks = _tasks(1)
    experiment = build_d1_experiment(
        tasks=tasks,
        repeats=1,
        internal_n=1,
        official_n=1,
    )
    result = run_d1_eval(
        experiment,
        candidate_body=D1_WRAPPER_BODY_NAIVE,
        candidate_id="d1-process-job",
        sampling=experiment.eval_configs.internal,
        execution_policy=execution_policy(max_attempts=1),
        row_job_factory=process_row_job_factory(
            "tests.envs.process_workers:drive_d1_success"
        ),
        apply_reward=False,
    )

    assert result.submission_score_aggregate.rows_failed == 0
    assert result.submission_score_aggregate.aggregation_output.value == 1.0


def test_direct_evaluator_resume_skips_recorded_rows(tmp_path: Path) -> None:
    tasks = _tasks(2)
    experiment = build_d1_experiment(tasks=tasks, repeats=1)
    log = PartialLog(path=tmp_path / "d1.partial.jsonl")
    first = run_d1_eval(
        experiment,
        candidate_body=D1_WRAPPER_BODY_NAIVE,
        candidate_id="d1-naive",
        sampling=experiment.eval_configs.internal,
        execution_policy=execution_policy(max_attempts=1),
        row_job_factory=_passing_jobs(),
        apply_reward=False,
        partial_log=log,
    )

    def boom(_instance, _repeat: int, _drive_ordinal: int):
        raise AssertionError("recorded rows must not be called again")

    resumed = run_d1_eval(
        experiment,
        candidate_body=D1_WRAPPER_BODY_NAIVE,
        candidate_id="d1-naive",
        sampling=experiment.eval_configs.internal,
        execution_policy=execution_policy(max_attempts=1),
        row_job_factory=row_job_factory(boom),
        apply_reward=False,
        partial_log=log,
    )
    assert (
        resumed.submission_score_aggregate == first.submission_score_aggregate
    )


def test_d1_terminal_timeout_is_persisted_and_not_repaid(
    tmp_path: Path, monkeypatch
) -> None:
    experiment = build_d1_experiment(
        tasks=_tasks(1),
        repeats=1,
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

    monkeypatch.setattr("whetstone.envs.d1_eval.run_call_pool", timed_out_pool)
    run_d1_eval(
        experiment,
        candidate_body=D1_WRAPPER_BODY_NAIVE,
        candidate_id="d1-timeout",
        sampling=experiment.eval_configs.internal,
        execution_policy=execution_policy(max_attempts=1),
        row_job_factory=_passing_jobs(),
        max_wall_seconds=10.0,
        apply_reward=False,
        partial_log=log,
    )

    records = log.load()
    assert len(records) == 1
    assert records[0].failure_code == "runner_timeout"
    assert records[0].split_role == experiment.eval_configs.internal.split_role
    assert len(walls) == 2
    assert walls[1] is not None and walls[0] is not None
    assert walls[1] <= walls[0]

    def boom(_request):
        raise AssertionError("terminal timeout must restore without repayment")

    monkeypatch.undo()
    resumed = run_d1_eval(
        experiment,
        candidate_body=D1_WRAPPER_BODY_NAIVE,
        candidate_id="d1-timeout",
        sampling=experiment.eval_configs.internal,
        execution_policy=execution_policy(max_attempts=1),
        row_job_factory=boom,
        apply_reward=False,
        partial_log=log,
    )
    assert resumed.submission_score_aggregate.rows_failed == 1


def test_d1_phase_deadline_is_missing_and_not_redriven(monkeypatch) -> None:
    experiment = build_d1_experiment(
        tasks=_tasks(1),
        repeats=1,
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

    monkeypatch.setattr("whetstone.envs.d1_eval.run_call_pool", deadline_pool)
    result = run_d1_eval(
        experiment,
        candidate_body=D1_WRAPPER_BODY_NAIVE,
        candidate_id="d1-deadline",
        sampling=experiment.eval_configs.internal,
        execution_policy=execution_policy(max_attempts=1),
        row_job_factory=_passing_jobs(),
        apply_reward=False,
    )

    assert calls == 1
    assert result.submission_score_aggregate.rows_missing == 1
    assert result.submission_score_aggregate.rows_failed == 0


def test_identifier_rename_is_whole_token_only() -> None:
    text = "foo(foo_bar, obj.foo, food); foo(1)"
    assert rename_identifier(text, "foo", "target") == (
        "target(foo_bar, obj.target, food); target(1)"
    )
