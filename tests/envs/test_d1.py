"""Focused D1 direct-generation environment-contract tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.envs.support import (
    FakeTransport,
    execution_policy,
    synthetic_ed1_tasks,
)
from whetstone.envs.d1 import (
    D1_INPUT_ARMS,
    D1_PASS_RATE_NAME,
    D1_WRAPPER_BODY_CEILING,
    D1_WRAPPER_BODY_NAIVE,
    build_d1_experiment,
    d1_ceiling_candidate,
    d1_initial_candidate,
    render_d1_frame,
)
from whetstone.envs.d1_eval import _input_arm_text, run_d1_eval
from whetstone.envs.ed1 import (
    ED1_DATASET_REVISION,
    ED1_INVALID_BODY,
    Ed1BodyError,
    ed1_body_rejection,
)
from whetstone.envs.ed1_scoring import CodeScore
from whetstone.envs.input_transform import (
    direct_prompt,
    rename_identifier,
    split_prompt,
)
from whetstone.execution.partials import PartialLog
from whetstone.optimization.mutation import MUTATION_FIELD


def _tasks(limit: int = 3):
    return synthetic_ed1_tasks(limit)


def _passing_scorer(**_kwargs: object) -> CodeScore:
    return CodeScore(
        passed=True,
        infrastructure_unknown=False,
        outcome="passed",
    )


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
        D1_PASS_RATE_NAME
    ]
    assert experiment.dataset_revision == ED1_DATASET_REVISION


def test_body_restrictions_are_preflight_safe() -> None:
    rejection = ed1_body_rejection("Solve {input_arm} now.")
    assert rejection == ("{input_arm}",)
    assert ED1_INVALID_BODY
    assert ed1_body_rejection("Solve it carefully.") == ()
    tasks = _tasks(1)
    experiment = build_d1_experiment(tasks=tasks)
    transport = FakeTransport(reply=lambda _prompt: "unused")

    with pytest.raises(Ed1BodyError) as error:
        run_d1_eval(
            experiment,
            candidate_body="Solve {input_arm} now.",
            candidate_id="invalid-body",
            sampling=experiment.eval_configs.internal,
            execution_policy=execution_policy(max_attempts=1),
            transport=transport,
            scorer=_passing_scorer,
            apply_reward=False,
        )

    assert error.value.code == ED1_INVALID_BODY
    assert transport.served == []


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
        transport=FakeTransport(
            reply=lambda _prompt: "def rebuilt():\n    return 1\n"
        ),
        scorer=_passing_scorer,
        apply_reward=False,
    )
    assert result.pass_aggregate.aggregation_output.value == pytest.approx(1)
    assert result.pass_aggregate.repeat_count == 2
    assert result.pass_aggregate.eval_config_hash == (
        experiment.eval_configs.internal.eval_config.config_identity_hash
    )
    assert result.per_task_counts == (2, 2)
    assert len(result.outputs) == 4


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
        transport=FakeTransport(reply=lambda _prompt: "code"),
        scorer=_passing_scorer,
        apply_reward=False,
        partial_log=log,
    )

    def boom(_prompt: str) -> str:
        raise AssertionError("recorded rows must not be called again")

    resumed = run_d1_eval(
        experiment,
        candidate_body=D1_WRAPPER_BODY_NAIVE,
        candidate_id="d1-naive",
        sampling=experiment.eval_configs.internal,
        execution_policy=execution_policy(max_attempts=1),
        transport=FakeTransport(reply=boom),
        scorer=_passing_scorer,
        apply_reward=False,
        partial_log=log,
    )
    assert resumed.pass_aggregate == first.pass_aggregate


def test_identifier_rename_is_whole_token_only() -> None:
    text = "foo(foo_bar, obj.foo, food); foo(1)"
    assert rename_identifier(text, "foo", "target") == (
        "target(foo_bar, obj.target, food); target(1)"
    )
