"""Focused ED1 environment-contract tests with no orchestration dependency."""

from __future__ import annotations

from pathlib import Path

import pytest
from dr_code.execution import SubprocessStartError
from dr_code.humaneval import STRICT_FIELD_MARKER_PARSER_PROFILE
from dr_providers import FailureClass

from tests.envs.support import (
    FakeTransport,
    constant_reply,
    evaluation_binding,
    execution_policy,
    process_row_job_factory,
    row_job_factory,
    synthetic_ed1_tasks,
)
from tests.provider.support import build_evidence, failure_outcome
from whetstone.envs.ed1 import (
    DECODER_TEMPLATE,
    ED1_BLENDED_REWARD_NAME,
    ED1_CANONICAL_MODEL,
    ED1_DATASET_REVISION,
    ED1_ENV_NAME,
    ED1_INVALID_BODY,
    ED1_SUBMISSION_SCORE_NAME,
    ENCODER_BODY_A,
    Ed1BodyError,
    build_ed1_blended_reward_policy,
    build_ed1_experiment,
    build_ed1_reward_policy,
    ed1_body_rejection,
    ed1_initial_candidate,
    render_encoder_frame,
    validate_ed1_body,
)
from whetstone.envs.ed1_blended import BoundedCompressionMetricConfig
from whetstone.envs.ed1_scoring import score_ed1_submission
from whetstone.envs.encdec_rollout import (
    DECODER_NODE_ID,
    ENCODER_NODE_ID,
    EVAL_NODE_ID,
    build_encdec_rollout_definition,
    encdec_graph_definition,
)
from whetstone.envs.reward import CandidateEvaluationFailure
from whetstone.envs.sampling import Completeness
from whetstone.evaluation.drivers.ed1 import (
    Ed1PartialPayload,
    Ed1RowOutcome,
    Ed1RowRequest,
    drive_ed1_row,
    run_ed1_eval,
)
from whetstone.evaluation.drivers.internal import _llm_component_step
from whetstone.evaluation.traces import ExecutedRowState
from whetstone.execution.fanout import (
    FanoutResult,
    FanoutStatus,
    PoolOutcome,
)
from whetstone.execution.partials import PartialLog
from whetstone.execution.prompt_cache import PromptResultCache
from whetstone.optimization.proposal.mutation import MUTATION_FIELD


def _tasks(limit: int = 3):
    return synthetic_ed1_tasks(limit)


def _successful_outcome(instance, *, encoder_text: str = "REBUILD:ok"):
    max_budget = round(0.5 * len(instance.prompt_inputs["input_code"]))
    decoder_text = "def rebuilt():\n    return 1\n"
    encoder_prompt = render_encoder_frame(
        ENCODER_BODY_A,
        input_code=instance.prompt_inputs["input_code"],
        max_budget=max_budget,
    )
    decoder_prompt = DECODER_TEMPLATE.format(encoder_output=encoder_text)
    return Ed1RowOutcome(
        primary_value=1.0,
        compression_value=0.5,
        encoder_text=encoder_text,
        decoder_text=decoder_text,
        row_state=ExecutedRowState.SUCCESS,
        executed_component_steps=(
            _llm_component_step(
                trace_index=0,
                component_id=ENCODER_NODE_ID,
                prompt=encoder_prompt,
                generation=encoder_text,
            ),
            _llm_component_step(
                trace_index=1,
                component_id=DECODER_NODE_ID,
                prompt=decoder_prompt,
                generation=decoder_text,
            ),
        ),
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
    blend_config: BoundedCompressionMetricConfig | None = None,
):
    selected = tasks or _tasks()
    experiment = build_ed1_experiment(
        tasks=selected,
        internal_n=len(selected),
        official_n=len(selected),
        repeats=repeats,
        completeness=completeness,
        max_skip_fraction=max_skip_fraction,
        blend_config=blend_config,
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
    output = result.outputs[0]
    assert output.output_text is not None
    assert [step.component_id for step in output.executed_component_steps] == [
        "encode",
        "decode",
    ]
    encoder_step = output.executed_component_steps[0]
    assert encoder_step.outputs == {
        "generation": "A compact executable reconstruction description."
    }
    assert encoder_step.outputs["generation"] != output.output_text
    assert encoder_step.inputs == {
        "prompt": render_encoder_frame(
            str(candidate.payload[MUTATION_FIELD]),
            input_code=tasks[0].input_code,
            max_budget=round(0.5 * len(tasks[0].input_code)),
        )
    }


def test_decoder_failure_preserves_only_the_real_encoder_step() -> None:
    tasks = _tasks(1)
    experiment = build_ed1_experiment(
        tasks=tasks, repeats=1, internal_n=1, official_n=1
    )
    instance = tasks[0].instance
    candidate = ed1_initial_candidate()
    policy = execution_policy(max_attempts=1)
    encoder_text = "encoder output that is not decoder code"
    encoder_transport = FakeTransport(constant_reply(encoder_text))
    calls = 0

    def transport(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return encoder_transport(request)
        return build_evidence(
            request=request,
            policy=policy.transport_policy,
            outcome=failure_outcome(
                failure_class=FailureClass.PERMANENT,
                message="decoder rejected",
            ),
        )

    def scorer_must_not_run(**_kwargs):
        raise AssertionError("decoder failure must not reach scoring")

    rollout = experiment.encdec_rollout
    assert rollout is not None
    outcome = drive_ed1_row(
        experiment=experiment,
        candidate_template=str(candidate.payload[MUTATION_FIELD]),
        instance=instance,
        provider_call_config=rollout.provider_call_config,
        execution_policy=policy,
        transport=transport,
        scorer=scorer_must_not_run,
        logical_call_id="decoder-failure",
        repeat_index=0,
        drive_ordinal=0,
        cache=None,
        cache_phase="internal_eval",
        cache_unit="decoder-failure",
    )

    assert outcome.row_state is ExecutedRowState.FAILED
    assert outcome.decoder_text is None
    assert outcome.encoder_text == encoder_text
    assert [
        step.component_id for step in outcome.executed_component_steps
    ] == ["encode"]
    assert outcome.executed_component_steps[0].outputs == {
        "generation": encoder_text
    }
    # A decoder failure does not erase the ENCODER leg's telemetry: that call
    # succeeded and its spend is real. The failed decoder still carries the
    # typed provider diagnostic.
    assert outcome.latency_s is not None
    assert outcome.provider_error is not None
    assert outcome.encoder_len == len(encoder_text)


def test_encoder_failure_still_reports_what_was_measured() -> None:
    # The encoder call failed, so there is no usage to report (tokens stay
    # None -- coverage-honest), but the budget the row was driven under and
    # the typed provider diagnostic ARE known and must not be dropped.
    tasks = _tasks(1)
    experiment = build_ed1_experiment(
        tasks=tasks, repeats=1, internal_n=1, official_n=1
    )
    instance = tasks[0].instance
    candidate = ed1_initial_candidate()
    policy = execution_policy(max_attempts=1)

    def transport(request):
        return build_evidence(
            request=request,
            policy=policy.transport_policy,
            outcome=failure_outcome(
                failure_class=FailureClass.PERMANENT,
                message="encoder rejected",
            ),
        )

    def scorer_must_not_run(**_kwargs):
        raise AssertionError("encoder failure must not reach scoring")

    rollout = experiment.encdec_rollout
    assert rollout is not None
    outcome = drive_ed1_row(
        experiment=experiment,
        candidate_template=str(candidate.payload[MUTATION_FIELD]),
        instance=instance,
        provider_call_config=rollout.provider_call_config,
        execution_policy=policy,
        transport=transport,
        scorer=scorer_must_not_run,
        logical_call_id="encoder-failure",
        repeat_index=0,
        drive_ordinal=0,
        cache=None,
        cache_phase="internal_eval",
        cache_unit="encoder-failure",
    )

    assert outcome.row_state is ExecutedRowState.FAILED
    assert outcome.encoder_text is None
    assert outcome.latency_s is not None
    assert outcome.provider_error is not None
    assert outcome.max_budget == round(0.5 * len(tasks[0].input_code))
    # No usage exists for a failed call: reporting 0 would fabricate coverage.
    assert outcome.prompt_tokens is None
    assert outcome.total_tokens is None


def test_ed1_v2_request_hash_is_pinned() -> None:
    tasks = _tasks(1)
    experiment = build_ed1_experiment(
        tasks=tasks, repeats=1, internal_n=1, official_n=1
    )
    candidate = ed1_initial_candidate()
    requests: list[Ed1RowRequest] = []
    base = row_job_factory(
        lambda instance, _repeat, _drive: _successful_outcome(instance)
    )

    def capture(request: Ed1RowRequest):
        requests.append(request)
        return base(request)

    run_ed1_eval(
        experiment,
        candidate_template=str(candidate.payload[MUTATION_FIELD]),
        candidate_id=candidate.candidate_id,
        sampling=experiment.eval_configs.internal,
        execution_policy=execution_policy(max_attempts=1),
        row_job_factory=capture,
        evaluation_binding=evaluation_binding(
            experiment.eval_configs.internal
        ),
    )

    assert requests[0].request_identity == (
        "18b96e7e0b651caf8c4477b14993acf4c112bfe161b6ed0aecb630307a702a4f"
    )


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
        "whetstone.evaluation.drivers.ed1.index_partial_records",
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
        completed = _successful_outcome(instance)
        return Ed1RowOutcome(
            primary_value=None,
            compression_value=None,
            encoder_text="REBUILD:ok",
            decoder_text="def rebuilt():\n    return 1\n",
            row_state=ExecutedRowState.FAILED,
            executed_component_steps=completed.executed_component_steps,
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


def _one_failed_row(instance, _repeat: int, _drive_ordinal: int):
    """One task fails outright; the rest succeed (an INCOMPLETE evaluation)."""
    if str(instance.id).endswith("/0"):
        return Ed1RowOutcome(
            primary_value=None,
            compression_value=None,
            encoder_text=None,
            decoder_text=None,
            row_state=ExecutedRowState.FAILED,
            executed_component_steps=(),
            failure_code="code_eval_infrastructure_unknown",
        )
    return _successful_outcome(instance)


def test_blended_reward_refuses_an_incomplete_evaluation() -> None:
    # A failed row folds into per-task scores as 0.0, so a raw mean of the
    # blended rewards would CERTIFY an evaluation the primary aggregate
    # refuses (value None under PROPAGATE). The blended path must gate on the
    # same completeness signal: no primary value -> typed failure, not a
    # silently-deflated Reward.
    with pytest.raises(CandidateEvaluationFailure):
        _evaluate(
            outcome_for=_one_failed_row,
            apply_reward=True,
            blend_config=BoundedCompressionMetricConfig(weight=0.1),
        )


def test_unblended_and_blended_agree_on_refusing_incompleteness() -> None:
    # The non-blended branch already refuses; the two branches must not
    # disagree about whether the SAME evaluation is certifiable.
    with pytest.raises(CandidateEvaluationFailure):
        _evaluate(outcome_for=_one_failed_row, apply_reward=True)


def test_complete_evaluation_produces_exact_blended_reward() -> None:
    blend = BoundedCompressionMetricConfig(weight=0.1)
    _, result = _evaluate(
        apply_reward=True,
        blend_config=blend,
    )
    primary = 1.0
    compression_ratio = 0.5
    compression_score = (blend.max_compression_ratio - compression_ratio) / (
        blend.max_compression_ratio - blend.min_compression_ratio
    )
    expected_value = primary * (
        (1.0 - blend.weight) + blend.weight * compression_score
    )

    reward = result.reward
    assert reward is not None
    assert result.per_task_primary == (1.0, 1.0, 1.0)
    assert result.per_task_compression == (0.5, 0.5, 0.5)
    assert reward.value == pytest.approx(expected_value)
    assert reward.reward_name == "reward"
    assert reward.reward_policy.policy_name == (
        "whetstone.env.ed1.blended_reward|"
        "primary_score_with_bounded_compression_penalty|"
        "w=0.1|min=0.01|max=4"
    )
    assert [citation.name for citation in reward.input_citations] == [
        ED1_BLENDED_REWARD_NAME
    ]
    assert reward.evidence_refs == (
        result.primary_aggregate.record_ref(),
        result.compression_aggregate.record_ref(),
    )


def test_advertised_reward_policy_matches_the_policy_applied() -> None:
    # The experiment ADVERTISES reward_policy; reward time builds the blended
    # policy from blend_config. A reader (and any consumer keying on the
    # policy identity) must not see the pass-only policy on a blended cell.
    blend = BoundedCompressionMetricConfig(weight=0.1)
    blended = build_ed1_experiment(tasks=_tasks(1), blend_config=blend)
    expected = build_ed1_blended_reward_policy(blend, env_name=ED1_ENV_NAME)
    assert blended.reward_policy.policy_name == expected.policy_name
    assert blended.reward_policy == expected

    plain = build_ed1_experiment(tasks=_tasks(1))
    assert plain.reward_policy == build_ed1_reward_policy()
    assert plain.reward_policy.policy_name != expected.policy_name


def test_blend_config_identity_reaches_the_advertised_policy_name() -> None:
    # A different weight is a different comparable config; the advertised
    # policy name must carry that distinction, not collapse it.
    names = {
        build_ed1_experiment(
            tasks=_tasks(1),
            blend_config=BoundedCompressionMetricConfig(weight=w),
        ).reward_policy.policy_name
        for w in (0.05, 0.10, 0.20)
    }
    assert len(names) == 3


def test_malformed_brace_is_a_typed_body_rejection() -> None:
    # An unmatched '{' makes the render contract's parser raise a BARE
    # ValueError; unguarded that surfaces as an untyped crash at eval start
    # instead of the promised typed ED1_INVALID_BODY rejection.
    for body in ("Explain {code", "Explain code}", "Explain {a{b} thing"):
        assert ed1_body_rejection(body)
        with pytest.raises(Ed1BodyError) as excinfo:
            validate_ed1_body(body)
        assert ED1_INVALID_BODY in str(excinfo.value)
        assert excinfo.value.offending


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
                row_state=ExecutedRowState.FAILED,
                executed_component_steps=(),
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
    records = log.load()
    assert len(records) == 2
    assert {record.raw_response for record in records} == {""}
    for record in records:
        Ed1PartialPayload.from_json_value(record.observation_payload)

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
    assert resumed.outputs == first.outputs


def test_ed1_partial_payload_rejects_shape_and_type_drift() -> None:
    completed = _successful_outcome(_tasks(1)[0].instance)
    valid = {
        "compression_value": 0.5,
        "encoder_text": completed.encoder_text,
        "decoder_text": completed.decoder_text,
        "attractor_pull": None,
        "max_budget": completed.max_budget,
        "encoder_len": completed.encoder_len,
        "row_state": ExecutedRowState.SUCCESS,
        "executed_component_steps": completed.executed_component_steps,
    }
    Ed1PartialPayload.model_validate(valid)

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        Ed1PartialPayload.model_validate({**valid, "unexpected": True})
    with pytest.raises(ValueError, match="valid number"):
        Ed1PartialPayload.model_validate({**valid, "compression_value": "0.5"})


def test_ed1_resume_requires_exact_evaluation_binding(tmp_path: Path) -> None:
    tasks = _tasks(2)
    experiment = build_ed1_experiment(tasks=tasks, repeats=1)
    sampling = experiment.eval_configs.internal
    candidate = ed1_initial_candidate()
    binding_a = evaluation_binding(sampling)
    binding_b = binding_a.model_copy(update={"campaign": "other-campaign"})
    log = PartialLog(path=tmp_path / "ed1-binding.partial")

    run_ed1_eval(
        experiment,
        candidate_template=str(candidate.payload[MUTATION_FIELD]),
        candidate_id=candidate.candidate_id,
        sampling=sampling,
        execution_policy=execution_policy(max_attempts=1),
        row_job_factory=row_job_factory(
            lambda instance, _repeat, _drive: _successful_outcome(instance)
        ),
        evaluation_binding=binding_a,
        partial_log=log,
    )
    identities_a = {record.request_identity for record in log.load()}

    served_b: list[str] = []

    def successful_b(instance, _repeat: int, _drive_ordinal: int):
        served_b.append(str(instance.id))
        return _successful_outcome(instance)

    run_ed1_eval(
        experiment,
        candidate_template=str(candidate.payload[MUTATION_FIELD]),
        candidate_id=candidate.candidate_id,
        sampling=sampling,
        execution_policy=execution_policy(max_attempts=1),
        row_job_factory=row_job_factory(successful_b),
        evaluation_binding=binding_b,
        partial_log=log,
    )

    assert len(served_b) == len(sampling.instances)
    identities_b = {
        record.request_identity
        for record in log.load()
        if record.request_identity not in identities_a
    }
    assert len(identities_b) == len(identities_a)


def test_ed1_pending_ordinal_zero_resumes_at_ordinal_one(
    tmp_path: Path, monkeypatch
) -> None:
    tasks = _tasks(1)
    experiment = build_ed1_experiment(
        tasks=tasks, repeats=1, internal_n=1, official_n=1
    )
    sampling = experiment.eval_configs.internal
    candidate = ed1_initial_candidate()
    log = PartialLog(path=tmp_path / "ed1-redrive.partial")
    pending = Ed1RowOutcome(
        primary_value=None,
        compression_value=None,
        encoder_text=None,
        decoder_text=None,
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
        "whetstone.evaluation.drivers.ed1.run_call_pool",
        crash_after_ordinal_zero,
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        run_ed1_eval(
            experiment,
            candidate_template=str(candidate.payload[MUTATION_FIELD]),
            candidate_id=candidate.candidate_id,
            sampling=sampling,
            execution_policy=execution_policy(max_attempts=1),
            row_job_factory=row_job_factory(lambda *_args: pending),
            evaluation_binding=evaluation_binding(sampling),
            partial_log=log,
        )
    assert {record.redrive_pending for record in log.load()} == {True}

    monkeypatch.undo()
    resumed_ordinals: list[int] = []

    def success(instance, _repeat: int, drive_ordinal: int):
        resumed_ordinals.append(drive_ordinal)
        return _successful_outcome(instance)

    run_ed1_eval(
        experiment,
        candidate_template=str(candidate.payload[MUTATION_FIELD]),
        candidate_id=candidate.candidate_id,
        sampling=sampling,
        execution_policy=execution_policy(max_attempts=1),
        row_job_factory=row_job_factory(success),
        evaluation_binding=evaluation_binding(sampling),
        partial_log=log,
    )
    assert resumed_ordinals == [1]
    assert {record.redrive_pending for record in log.load()} == {False, True}


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

    monkeypatch.setattr(
        "whetstone.evaluation.drivers.ed1.run_call_pool", timeout_pool
    )
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
    assert len(records) == 2
    assert {record.failure_code for record in records} == {"runner_timeout"}
    assert {record.redrive_pending for record in records} == {False, True}
    assert len({record.request_identity for record in records}) == 2
    assert {record.split_role for record in records} == {
        experiment.eval_configs.official.split_role
    }

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
    monkeypatch.setattr(
        "whetstone.evaluation.drivers.ed1.run_call_pool", deadline_pool
    )
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
    assert second.observation_payload == first.observation_payload


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
