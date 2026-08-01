"""Focused ED1 environment-contract tests with no orchestration dependency."""

from __future__ import annotations

from pathlib import Path

import pytest
from dr_code.execution import SubprocessStartError
from dr_code.humaneval import STRICT_FIELD_MARKER_PARSER_PROFILE

from tests.envs.support import (
    evaluation_binding,
    execution_policy,
    process_row_job_factory,
    row_job_factory,
    synthetic_ed1_tasks,
)
from whetstone.envs.ed1 import (
    ED1_CANONICAL_MODEL,
    ED1_DATASET_REVISION,
    ED1_ENV_NAME,
    ED1_INVALID_BODY,
    ED1_SUBMISSION_SCORE_NAME,
    ENCODER_BODY_A,
    Ed1BodyError,
    build_ed1_experiment,
    ed1_body_rejection,
    ed1_initial_candidate,
    render_encoder_frame,
)
from whetstone.envs.ed1_eval import Ed1RowOutcome, run_ed1_eval
from whetstone.envs.ed1_scoring import score_ed1_submission
from whetstone.envs.encdec_rollout import (
    DECODER_NODE_ID,
    ENCODER_NODE_ID,
    EVAL_NODE_ID,
    build_encdec_rollout_definition,
    encdec_graph_definition,
)
from whetstone.envs.sampling import Completeness
from whetstone.execution.fanout import (
    FanoutResult,
    FanoutStatus,
    PoolOutcome,
)
from whetstone.execution.partials import PartialLog
from whetstone.execution.prompt_cache import PromptResultCache
from whetstone.optimization.mutation import MUTATION_FIELD


def _tasks(limit: int = 3):
    return synthetic_ed1_tasks(limit)


def _successful_outcome(instance, *, encoder_text: str = "REBUILD:ok"):
    max_budget = round(0.5 * len(instance.prompt_inputs["input_code"]))
    return Ed1RowOutcome(
        primary_value=1.0,
        compression_value=0.5,
        encoder_text=encoder_text,
        decoder_text="def rebuilt():\n    return 1\n",
        failed=False,
        max_budget=max_budget,
        encoder_len=len(encoder_text),
    )


def _evaluate(
    *,
    tasks=None,
    repeats: int = 1,
    completeness: Completeness = Completeness.PROPAGATE,
    max_skip_fraction: float = 0.0,
    outcome_for=None,
    partial_log: PartialLog | None = None,
    apply_reward: bool = False,
):
    selected = tasks or _tasks()
    experiment = build_ed1_experiment(
        tasks=selected,
        internal_n=len(selected),
        official_n=len(selected),
        repeats=repeats,
        completeness=completeness,
        max_skip_fraction=max_skip_fraction,
    )
    candidate = ed1_initial_candidate()
    active_outcome = outcome_for or (
        lambda instance, _repeat, _drive: _successful_outcome(instance)
    )
    sampling = (
        experiment.eval_configs.internal
        if apply_reward
        else experiment.eval_configs.official
    )
    result = run_ed1_eval(
        experiment,
        candidate_template=str(candidate.payload[MUTATION_FIELD]),
        candidate_id=candidate.candidate_id,
        sampling=sampling,
        execution_policy=execution_policy(max_attempts=1),
        row_job_factory=row_job_factory(active_outcome),
        evaluation_binding=evaluation_binding(
            sampling, official=not apply_reward
        ),
        partial_log=partial_log,
    )
    return experiment, result


def test_encdec_graph_and_output_affecting_identity() -> None:
    definition = encdec_graph_definition()
    assert [node.node_id for node in definition.nodes] == [
        ENCODER_NODE_ID,
        DECODER_NODE_ID,
        EVAL_NODE_ID,
    ]
    assert definition.terminal_node_id == EVAL_NODE_ID
    base = build_encdec_rollout_definition(
        ED1_ENV_NAME,
        model=ED1_CANONICAL_MODEL,
        procedure_config_hash="a" * 64,
        budget_ratio=0.5,
    )
    ratio = build_encdec_rollout_definition(
        ED1_ENV_NAME,
        model=ED1_CANONICAL_MODEL,
        procedure_config_hash="a" * 64,
        budget_ratio=0.75,
    )
    model = build_encdec_rollout_definition(
        ED1_ENV_NAME,
        model="openai/gpt-5-nano",
        procedure_config_hash="a" * 64,
        budget_ratio=0.5,
    )
    assert base.graph_hash != ratio.graph_hash != model.graph_hash
    assert base.provider_call_config.definition.route.model == (
        ED1_CANONICAL_MODEL
    )


def test_humaneval_scoring_canonical_passes_wrong_fails() -> None:
    task = _tasks(1)[0].humaneval_task
    good = score_ed1_submission(
        raw_submission=task.ground_truth_code,
        task=task,
        timeout_seconds=30.0,
    )
    bad = score_ed1_submission(
        raw_submission=(
            f"def {task.entry_point}(*args, **kwargs):\n    return None\n"
        ),
        task=task,
        timeout_seconds=30.0,
    )
    assert good.passed and not good.infrastructure_unknown
    assert not bad.passed and not bad.infrastructure_unknown


@pytest.mark.parametrize(
    ("raw_submission", "expected_outcome"),
    [
        ("", "empty_submission"),
        ("x = 1", "no_top_level_functions"),
        ("```python\nthis is ???\n```", "extraction_failed"),
    ],
)
def test_humaneval_scoring_completed_rejections_are_definitive(
    raw_submission: str,
    expected_outcome: str,
) -> None:
    task = _tasks(1)[0].humaneval_task
    score = score_ed1_submission(raw_submission=raw_submission, task=task)

    assert score.outcome == expected_outcome
    assert score.infrastructure_unknown is False


def test_humaneval_scoring_projects_harness_failure() -> None:
    task = _tasks(1)[0].humaneval_task

    def unavailable(*, source: str, input_text: str, timeout_seconds: float):
        raise SubprocessStartError("subprocess unavailable")

    harness_failure = score_ed1_submission(
        raw_submission=task.ground_truth_code,
        task=task,
        run_in_subprocess=unavailable,
    )

    assert harness_failure.outcome == "harness_failure"
    assert harness_failure.infrastructure_unknown is True


def test_humaneval_scoring_honors_explicit_parser_profile() -> None:
    task = _tasks(1)[0].humaneval_task
    score = score_ed1_submission(
        raw_submission=task.ground_truth_code,
        task=task,
        parser_profile=STRICT_FIELD_MARKER_PARSER_PROFILE,
    )

    assert score.outcome == "extraction_failed"
    assert score.infrastructure_unknown is False


def test_body_validation_rejects_before_transport() -> None:
    assert ed1_body_rejection("Solve {input_code}") == ("{input_code}",)
    assert ed1_body_rejection("```python\npass\n```") == ("```",)
    assert ed1_body_rejection("Solve carefully.") == ()
    tasks = _tasks(1)
    experiment = build_ed1_experiment(tasks=tasks)
    served: list[str] = []

    def outcome(instance, _repeat: int, _drive_ordinal: int):
        served.append(str(instance.id))
        return _successful_outcome(instance)

    with pytest.raises(Ed1BodyError) as error:
        run_ed1_eval(
            experiment,
            candidate_template="Solve {input_code}",
            candidate_id="invalid-body",
            sampling=experiment.eval_configs.internal,
            execution_policy=execution_policy(max_attempts=1),
            row_job_factory=row_job_factory(outcome),
            evaluation_binding=evaluation_binding(
                experiment.eval_configs.internal
            ),
        )

    assert error.value.code == ED1_INVALID_BODY
    assert error.value.offending == ("{input_code}",)
    assert served == []


def test_no_budget_frame_omits_budget_instruction() -> None:
    rendered = render_encoder_frame(
        ENCODER_BODY_A,
        input_code="def f(): pass",
        max_budget=None,
    )
    assert "Use at most" not in rendered
    assert "```python\ndef f(): pass\n```" in rendered


def test_end_to_end_records_exact_dual_scores_and_outputs() -> None:
    experiment, result = _evaluate(repeats=2, apply_reward=True)
    assert result.primary_aggregate.name == ED1_SUBMISSION_SCORE_NAME
    assert result.primary_aggregate.aggregation_output.value == pytest.approx(
        1
    )
    compression = result.compression_aggregate.aggregation_output.value
    assert compression is not None and compression > 0
    assert result.primary_aggregate.eval_config_hash == (
        experiment.eval_configs.internal.eval_config.config_identity_hash
    )
    assert result.primary_aggregate.repeat_count == 2
    assert {row.metric_name for row in result.row_diags} == {
        ED1_SUBMISSION_SCORE_NAME
    }
    assert result.reward is not None
    assert result.reward.input_citations[0].name == ED1_SUBMISSION_SCORE_NAME
    assert len(result.outputs) == len(_tasks()) * 2
    assert all("ENCODER:" in (row.output_text or "") for row in result.outputs)
    assert experiment.dataset_revision == ED1_DATASET_REVISION


def test_ed1_process_job_runs_real_row_driver() -> None:
    tasks = _tasks(1)
    experiment = build_ed1_experiment(
        tasks=tasks,
        repeats=1,
        internal_n=1,
        official_n=1,
    )
    candidate = ed1_initial_candidate()
    result = run_ed1_eval(
        experiment,
        candidate_template=str(candidate.payload[MUTATION_FIELD]),
        candidate_id="ed1-process-job",
        sampling=experiment.eval_configs.internal,
        execution_policy=execution_policy(max_attempts=1),
        row_job_factory=process_row_job_factory(
            "tests.envs.process_workers:drive_ed1_success"
        ),
        evaluation_binding=evaluation_binding(
            experiment.eval_configs.internal
        ),
    )

    assert result.primary_aggregate.rows_failed == 0
    assert result.primary_aggregate.aggregation_output.value == 1.0
    assert result.outputs[0].output_text is not None


@pytest.mark.parametrize(
    ("split_name", "official_binding"),
    [("official", False), ("internal", True)],
)
def test_ed1_rejects_binding_role_mismatch_before_restore(
    monkeypatch, split_name: str, official_binding: bool
) -> None:
    experiment = build_ed1_experiment(tasks=_tasks(), repeats=1)
    sampling = getattr(experiment.eval_configs, split_name)
    candidate = ed1_initial_candidate()

    def should_not_restore(*_args, **_kwargs):
        raise AssertionError("role mismatch must fail before partial restore")

    def should_not_build(_request):
        raise AssertionError("role mismatch must fail before job construction")

    monkeypatch.setattr(
        "whetstone.envs.ed1_eval._restore_ed1_recorded",
        should_not_restore,
    )
    with pytest.raises(ValueError, match="does not match split role"):
        run_ed1_eval(
            experiment,
            candidate_template=str(candidate.payload[MUTATION_FIELD]),
            candidate_id="ed1-role-mismatch",
            sampling=sampling,
            execution_policy=execution_policy(max_attempts=1),
            row_job_factory=should_not_build,
            evaluation_binding=evaluation_binding(
                sampling, official=official_binding
            ),
        )


def test_budget_and_healthy_diagnostics_are_explicit() -> None:
    tasks = _tasks(1)
    long_description = "x" * (len(tasks[0].input_code) * 4)

    _, result = _evaluate(
        tasks=tasks,
        outcome_for=lambda instance, _repeat, _drive: _successful_outcome(
            instance, encoder_text=long_description
        ),
    )
    row = result.row_diags[0]
    assert row.max_budget == round(0.5 * len(tasks[0].input_code))
    assert row.encoder_len == len(long_description)
    assert row.over_budget is True
    assert row.failed is False
    assert result.diagnostics.present_rows == 1
    assert result.diagnostics.failed_rows == 0
    assert result.diagnostics.none_reason is None


def test_all_failed_diagnostics_name_dominant_failure() -> None:
    def failed(instance, _repeat: int, _drive_ordinal: int):
        return Ed1RowOutcome(
            primary_value=None,
            compression_value=None,
            encoder_text="REBUILD:ok",
            decoder_text="def rebuilt():\n    return 1\n",
            failed=True,
            failure_code="code_eval_infrastructure_unknown",
            max_budget=round(0.5 * len(instance.prompt_inputs["input_code"])),
            encoder_len=len("REBUILD:ok"),
        )

    _, result = _evaluate(outcome_for=failed)
    assert result.primary_aggregate.aggregation_output.value is None
    assert result.diagnostics.present_rows == 0
    assert result.diagnostics.failed_rows == 3
    assert result.diagnostics.none_reason is not None
    assert "code_eval_infrastructure_unknown" in (
        result.diagnostics.none_reason
    )


def test_bounded_skip_certifies_retained_scores_and_accounting() -> None:
    tasks = _tasks(4)
    calls = 0

    def outcome(instance, _repeat: int, _drive_ordinal: int):
        nonlocal calls
        calls += 1
        if calls == 1:
            return Ed1RowOutcome(
                primary_value=None,
                compression_value=None,
                encoder_text=None,
                decoder_text=None,
                failed=True,
                failure_code="code_eval_infrastructure_unknown",
            )
        return _successful_outcome(
            instance,
        )

    _, result = _evaluate(
        tasks=tasks,
        completeness=Completeness.SKIP,
        max_skip_fraction=0.30,
        outcome_for=outcome,
    )
    assert result.primary_aggregate.rows_failed == 1
    assert result.primary_aggregate.rows_present == 3
    assert result.primary_aggregate.aggregation_output.value == pytest.approx(
        1
    )
    assert result.per_task_counts == (1, 1, 1, 1)


def test_streaming_resume_restores_rows_without_transport(
    tmp_path: Path,
) -> None:
    tasks = _tasks(2)
    log = PartialLog(path=tmp_path / "ed1.partial.jsonl")
    experiment, first = _evaluate(tasks=tasks, partial_log=log)
    assert len(log.load()) == 2

    def boom(_instance, _repeat: int, _drive_ordinal: int):
        raise AssertionError("recorded rows must not be called again")

    candidate = ed1_initial_candidate()
    resumed = run_ed1_eval(
        experiment,
        candidate_template=str(candidate.payload[MUTATION_FIELD]),
        candidate_id=candidate.candidate_id,
        sampling=experiment.eval_configs.official,
        execution_policy=execution_policy(max_attempts=1),
        row_job_factory=row_job_factory(boom),
        evaluation_binding=evaluation_binding(
            experiment.eval_configs.official, official=True
        ),
        partial_log=log,
    )
    assert resumed.primary_aggregate == first.primary_aggregate
    assert resumed.compression_aggregate == first.compression_aggregate


def test_ed1_terminal_timeout_is_persisted_and_phase_deadline_is_missing(
    tmp_path: Path, monkeypatch
) -> None:
    tasks = _tasks(1)
    experiment = build_ed1_experiment(
        tasks=tasks,
        repeats=1,
        internal_n=1,
        official_n=1,
    )
    candidate = ed1_initial_candidate()
    log = PartialLog(path=tmp_path / "ed1-timeout.partial.jsonl")

    def timeout_pool(specs, *, concurrency, is_rate_limited, max_wall_seconds):
        del is_rate_limited, max_wall_seconds
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

    monkeypatch.setattr("whetstone.envs.ed1_eval.run_call_pool", timeout_pool)
    run_ed1_eval(
        experiment,
        candidate_template=str(candidate.payload[MUTATION_FIELD]),
        candidate_id="ed1-timeout",
        sampling=experiment.eval_configs.official,
        execution_policy=execution_policy(max_attempts=1),
        row_job_factory=row_job_factory(
            lambda instance, _repeat, _drive: _successful_outcome(instance)
        ),
        evaluation_binding=evaluation_binding(
            experiment.eval_configs.official, official=True
        ),
        partial_log=log,
    )

    records = log.load()
    assert len(records) == 1
    assert records[0].failure_code == "runner_timeout"
    assert records[0].split_role == experiment.eval_configs.official.split_role

    def boom(_request):
        raise AssertionError("terminal timeout must restore without repayment")

    resumed = run_ed1_eval(
        experiment,
        candidate_template=str(candidate.payload[MUTATION_FIELD]),
        candidate_id="ed1-timeout",
        sampling=experiment.eval_configs.official,
        execution_policy=execution_policy(max_attempts=1),
        row_job_factory=boom,
        evaluation_binding=evaluation_binding(
            experiment.eval_configs.official, official=True
        ),
        partial_log=log,
    )
    assert resumed.primary_aggregate.rows_failed == 1

    def deadline_pool(
        specs, *, concurrency, is_rate_limited, max_wall_seconds
    ):
        del is_rate_limited, max_wall_seconds
        return PoolOutcome(
            results=tuple(
                FanoutResult(
                    key=spec.key,
                    status=FanoutStatus.NOT_DISPATCHED,
                )
                for spec in specs
            ),
            effective_concurrency=concurrency,
            concurrency_halved=False,
            deadline_reached=True,
            guard_timeouts=0,
        )

    fresh_log = PartialLog(path=tmp_path / "ed1-deadline.partial.jsonl")
    monkeypatch.setattr("whetstone.envs.ed1_eval.run_call_pool", deadline_pool)
    missing = run_ed1_eval(
        experiment,
        candidate_template=str(candidate.payload[MUTATION_FIELD]),
        candidate_id="ed1-deadline",
        sampling=experiment.eval_configs.official,
        execution_policy=execution_policy(max_attempts=1),
        row_job_factory=row_job_factory(
            lambda instance, _repeat, _drive: _successful_outcome(instance)
        ),
        evaluation_binding=evaluation_binding(
            experiment.eval_configs.official, official=True
        ),
        partial_log=fresh_log,
    )
    assert missing.primary_aggregate.rows_missing == 1
    assert missing.primary_aggregate.rows_failed == 0
    assert fresh_log.load() == []


def test_process_job_cache_hit_and_provenance_are_persisted(
    tmp_path: Path,
) -> None:
    tasks = _tasks(1)
    experiment = build_ed1_experiment(
        tasks=tasks,
        repeats=1,
        internal_n=1,
        official_n=1,
    )
    candidate = ed1_initial_candidate()
    cache = PromptResultCache(tmp_path / "prompt-cache")
    job_factory = process_row_job_factory(
        "tests.envs.process_workers:drive_ed1_success"
    )
    first_log = PartialLog(path=tmp_path / "first.partial.jsonl")
    second_log = PartialLog(path=tmp_path / "second.partial.jsonl")

    for log in (first_log, second_log):
        run_ed1_eval(
            experiment,
            candidate_template=str(candidate.payload[MUTATION_FIELD]),
            candidate_id=candidate.candidate_id,
            sampling=experiment.eval_configs.internal,
            execution_policy=execution_policy(max_attempts=1),
            row_job_factory=job_factory,
            evaluation_binding=evaluation_binding(
                experiment.eval_configs.internal
            ),
            partial_log=log,
            cache=cache,
        )

    first = first_log.load()[0]
    second = second_log.load()[0]
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.cache_source_phase == first.phase
    assert second.cache_source_unit == first.unit
    assert second.cache_source_call_id == (
        f"{candidate.candidate_id}:{tasks[0].instance.id}#0:enc"
    )


def test_transient_encoder_failure_is_redriven_to_success() -> None:
    tasks = _tasks(1)
    experiment = build_ed1_experiment(
        tasks=tasks,
        repeats=1,
        internal_n=1,
        official_n=1,
    )
    candidate = ed1_initial_candidate()
    result = run_ed1_eval(
        experiment,
        candidate_template=str(candidate.payload[MUTATION_FIELD]),
        candidate_id=candidate.candidate_id,
        sampling=experiment.eval_configs.internal,
        execution_policy=execution_policy(max_attempts=1),
        row_job_factory=process_row_job_factory(
            "tests.envs.process_workers:drive_ed1_transient_then_success"
        ),
        evaluation_binding=evaluation_binding(
            experiment.eval_configs.internal
        ),
    )
    assert result.primary_aggregate.rows_failed == 0
    assert result.primary_aggregate.aggregation_output.value == pytest.approx(
        1
    )
