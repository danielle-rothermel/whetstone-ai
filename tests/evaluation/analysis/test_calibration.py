from __future__ import annotations

import pytest
from dr_store import ObjectStore, SqliteBackend

from tests.envs.support import execution_policy, synthetic_ed1_tasks
from tests.evaluation.support import _binding, _engine
from whetstone.core.roles import EvaluationRole
from whetstone.envs.ed1 import (
    DECODER_TEMPLATE,
    ENCODER_BODY_B,
    build_ed1_experiment,
    render_encoder_frame,
)
from whetstone.evaluation import engine as engine_module
from whetstone.evaluation.analysis.calibration import run_anchor_calibration
from whetstone.evaluation.analysis.power import PowerConfig
from whetstone.evaluation.drivers.ed1 import (
    Ed1RowOutcome,
    Ed1RowRequest,
    Ed1RowResult,
)
from whetstone.evaluation.drivers.internal import _llm_component_step
from whetstone.evaluation.engine import EvaluationEngine
from whetstone.evaluation.traces import ExecutedRowState
from whetstone.execution.fanout import ProcessJob
from whetstone.experiment.binding import (
    EVALUATION_BINDING_SCHEMA_VERSION,
    EvaluationBinding,
)

_CALIBRATION_BASELINE_PURPOSE = "test-calibration-baseline"
_CALIBRATION_CEILING_PURPOSE = "test-calibration-ceiling"


def _calibration_row(request: Ed1RowRequest) -> ProcessJob:
    instance = request.instance.to_instance()
    encoder_text = "A compact reconstruction description."
    decoder_text = "def reconstructed():\n    return 1\n"
    encoder_prompt = render_encoder_frame(
        request.candidate_template,
        input_code=instance.prompt_inputs["input_code"],
        max_budget=None,
    )
    task_passed = request.candidate_template == ENCODER_BODY_B or not str(
        instance.id
    ).endswith("/0")
    outcome = Ed1RowOutcome(
        primary_value=float(task_passed),
        compression_value=0.5,
        encoder_text=encoder_text,
        decoder_text=decoder_text,
        row_state=ExecutedRowState.SUCCESS,
        executed_component_steps=(
            _llm_component_step(
                trace_index=0,
                component_id="encode",
                prompt=encoder_prompt,
                generation=encoder_text,
            ),
            _llm_component_step(
                trace_index=1,
                component_id="decode",
                prompt=DECODER_TEMPLATE.format(encoder_output=encoder_text),
                generation=decoder_text,
            ),
        ),
        max_budget=None,
        encoder_len=len(encoder_text),
    )
    result = Ed1RowResult(
        request_identity=request.request_identity,
        outcome=outcome,
    )
    return ProcessJob(
        entrypoint="tests.envs.process_workers:return_payload",
        payload=result.model_dump(mode="json"),
    )


def _internal_engine_and_binding(tmp_path):
    store = ObjectStore(
        SqliteBackend(tmp_path / "internal-calibration.sqlite")
    )
    engine = _engine(tmp_path, store=store, repeats=2)
    binding = _binding(engine, campaign="internal-calibration-test")
    return engine, binding, store, engine.experiment


def _ed1_engine_and_binding(tmp_path, *, concurrency: int = 2):
    tasks = synthetic_ed1_tasks(3)
    experiment = build_ed1_experiment(
        tasks=tasks,
        internal_n=3,
        official_n=3,
        repeats=2,
        budget_ratio=None,
    )
    store = ObjectStore(SqliteBackend(tmp_path / "calibration.sqlite"))
    policy = execution_policy()
    engine = EvaluationEngine(
        store=store,
        experiment=experiment,
        sampling=experiment.eval_configs.internal,
        execution_policy=policy,
        row_job_factory=_calibration_row,
        concurrency=concurrency,
    )
    binding = EvaluationBinding(
        schema_version=EVALUATION_BINDING_SCHEMA_VERSION,
        eval_config=engine.eval_config_ref,
        role=EvaluationRole.INTERNAL,
        campaign="calibration-test",
        provider_execution_policy_ref=engine.provider_execution_policy_ref,
    )
    return engine, binding, store, experiment


@pytest.mark.process_integration
def test_internal_calibration_evaluates_aligned_anchors(tmp_path) -> None:
    engine, binding, _store, experiment = _internal_engine_and_binding(
        tmp_path
    )
    task_ids = experiment.eval_configs.internal.task_set.task_identities[:2]

    result = run_anchor_calibration(
        engine=engine,
        evaluation_binding=binding,
        baseline_candidate=experiment.initial_candidate,
        ceiling_candidate=experiment.ceiling_candidate,
        baseline_purpose="internal-calibration-baseline",
        ceiling_purpose="internal-calibration-ceiling",
        task_ids=task_ids,
        pool_ceiling=len(
            experiment.eval_configs.internal.task_set.task_identities
        ),
        power_config=PowerConfig(trials=10, repeat_cap=2, seed=3),
        bootstrap_resamples=50,
        bootstrap_seed=5,
    )

    baseline = result.baseline.evidence
    ceiling = result.ceiling.evidence
    assert baseline.task_identities == ceiling.task_identities == task_ids
    assert baseline.purpose == "internal-calibration-baseline"
    assert ceiling.purpose == "internal-calibration-ceiling"
    assert len(baseline.per_task_values) == len(task_ids)
    assert baseline.per_task_values == ceiling.per_task_values


@pytest.mark.process_integration
def test_calibration_evaluates_aligned_anchors_and_plans_power(
    tmp_path, monkeypatch
) -> None:
    engine, binding, store, experiment = _ed1_engine_and_binding(
        tmp_path, concurrency=7
    )
    observed_concurrency: list[int] = []
    canonical_run = engine_module.run_ed1_eval

    def recording_run(*args, **kwargs):
        observed_concurrency.append(kwargs["concurrency"])
        return canonical_run(*args, **kwargs)

    monkeypatch.setattr(engine_module, "run_ed1_eval", recording_run)
    task_ids = ("Synthetic/2", "Synthetic/0")
    result = run_anchor_calibration(
        engine=engine,
        evaluation_binding=binding,
        baseline_candidate=experiment.initial_candidate,
        ceiling_candidate=experiment.ceiling_candidate,
        baseline_purpose=_CALIBRATION_BASELINE_PURPOSE,
        ceiling_purpose=_CALIBRATION_CEILING_PURPOSE,
        baseline_log_label="hand-engineered baseline",
        ceiling_log_label="hand-engineered comparison anchor",
        task_ids=task_ids,
        pool_ceiling=3,
        power_config=PowerConfig(repeat_cap=3, trials=100, seed=17),
        bootstrap_resamples=200,
        bootstrap_seed=19,
    )

    baseline = result.baseline.evidence
    ceiling = result.ceiling.evidence
    assert observed_concurrency == [7, 7]
    assert (
        result.evaluation_binding.eval_config
        == baseline.evaluation_binding.eval_config
    )
    assert result.evaluation_binding.eval_config != binding.eval_config
    assert baseline.evaluation_binding == ceiling.evaluation_binding
    assert baseline.task_identities == ceiling.task_identities == task_ids
    assert baseline.repeat_count == ceiling.repeat_count == 2
    assert baseline.per_task_counts == ceiling.per_task_counts == (2, 2)
    assert (
        baseline.row_accounting.planned == ceiling.row_accounting.planned == 4
    )
    assert baseline.purpose == _CALIBRATION_BASELINE_PURPOSE
    assert ceiling.purpose == _CALIBRATION_CEILING_PURPOSE
    assert baseline.reward_ref is not None
    assert ceiling.reward_ref is not None

    compression_score = (4.0 - 0.5) / (4.0 - 0.01)
    passing_reward = 0.9 + 0.1 * compression_score
    assert baseline.per_task_values == pytest.approx((passing_reward, 0.0))
    assert ceiling.per_task_values == pytest.approx(
        (passing_reward, passing_reward)
    )
    assert result.paired_delta_ci.point == pytest.approx(passing_reward / 2)
    assert result.power.certified_headroom == pytest.approx(passing_reward / 2)
    assert result.power.pool_ceiling == 3
    assert result.power.decomposition.anchor_repeats == 2
    assert store.get(result.baseline.evidence_ref.reference)
    assert store.get(result.ceiling.evidence_ref.reference)
    for cited in baseline.reward_ref.record.evidence_refs:
        assert store.get(cited.reference)


def test_calibration_rejects_an_impossible_pool_before_evaluation(
    tmp_path,
) -> None:
    engine, binding, _store, experiment = _ed1_engine_and_binding(tmp_path)

    with pytest.raises(ValueError, match="pool_ceiling cannot be smaller"):
        run_anchor_calibration(
            engine=engine,
            evaluation_binding=binding,
            baseline_candidate=experiment.initial_candidate,
            ceiling_candidate=experiment.ceiling_candidate,
            baseline_purpose=_CALIBRATION_BASELINE_PURPOSE,
            ceiling_purpose=_CALIBRATION_CEILING_PURPOSE,
            task_ids=("Synthetic/0", "Synthetic/1"),
            pool_ceiling=1,
            power_config=PowerConfig(trials=1),
            bootstrap_resamples=1,
        )


def test_calibration_reports_each_paid_evaluation_boundary(tmp_path) -> None:
    engine, binding, _store, experiment = _ed1_engine_and_binding(tmp_path)
    messages: list[str] = []

    run_anchor_calibration(
        engine=engine,
        evaluation_binding=binding,
        baseline_candidate=experiment.initial_candidate,
        ceiling_candidate=experiment.ceiling_candidate,
        baseline_purpose=_CALIBRATION_BASELINE_PURPOSE,
        ceiling_purpose=_CALIBRATION_CEILING_PURPOSE,
        baseline_log_label="hand-engineered baseline",
        ceiling_log_label="hand-engineered comparison anchor",
        task_ids=("Synthetic/0",),
        pool_ceiling=1,
        power_config=PowerConfig(trials=1),
        bootstrap_resamples=1,
        log=messages.append,
    )

    assert messages == [
        "Starting hand-engineered baseline evaluation (2 rows)",
        "Completed hand-engineered baseline evaluation "
        "(present=2/2, missing=0, failed=0, invalid=0)",
        "Starting hand-engineered comparison anchor evaluation (2 rows)",
        "Completed hand-engineered comparison anchor evaluation "
        "(present=2/2, missing=0, failed=0, invalid=0)",
    ]
