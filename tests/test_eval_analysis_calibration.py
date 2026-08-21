"""End-to-end tests for anchor calibration over constructed anchor evidence.

These run the real toy evaluation engine with a scripted eval-node runner, so
every per-task score is chosen by the test. That makes the calibrated delta,
its confidence interval, and the certified headroom exactly predictable from
the fixture rather than pinned from a prior run.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest
from dr_store.testing import temp_sqlite_store

from whetstone.eval.analysis.calibration import (
    AnchorCalibrationResult,
    run_anchor_calibration,
)
from whetstone.eval.analysis.power import PowerConfig
from whetstone.eval.metadata import metadata_with_purpose
from whetstone.eval.protocol import (
    EvalEvidenceWithRef,
    EvalRejected,
    EvalRequest,
    EvalTaskView,
)
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.eval.schema import EvalFailureEvidence
from whetstone.optim.contracts import ResolutionClass, ResolutionDetail
from whetstone.testing.toy.experiment import ToyTask, build_toy_experiment

#: The ceiling arm's template carries this marker, so the scripted runner can
#: tell the two anchors apart from the rendered prompt alone.
CEILING_MARKER = "CEILING"
BASELINE_TEMPLATE = "Reply to: {prompt}"
CEILING_TEMPLATE = f"{CEILING_MARKER} answer: {{prompt}}"

#: Per-task scores for each arm: the ceiling beats the baseline by exactly
#: 0.3 on every task, so the paired delta is a constant shift.
BASELINE_SCORES = (0.2, 0.4, 0.6)
CEILING_LIFT = 0.3


class ScriptedEvalProcedureRunner:
    """Return a caller-chosen score per (task, anchor arm)."""

    def __init__(self, scores: Mapping[tuple[str, str], float]) -> None:
        self._scores = dict(scores)

    def run_eval_node(
        self,
        *,
        node_id: str,
        node_inputs: Mapping[str, object],
        evaluation_procedure_config_hash: str,
        task: EvalTaskView,
    ) -> tuple[float | None, object | None, dict[str, object]]:
        _ = (node_id, evaluation_procedure_config_hash)
        generation = node_inputs.get("provider_generation")
        text = generation if isinstance(generation, str) else str(generation or "")
        arm = "ceiling" if CEILING_MARKER in text else "baseline"
        return self._scores[(task.task_id, arm)], {"text": text}, {}


def _toy_tasks(count: int) -> tuple[ToyTask, ...]:
    return tuple(
        ToyTask(task_id=f"task-{index}", prompt_inputs={"prompt": f"h{index}"})
        for index in range(count)
    )


def _scripted_scores(
    tasks: tuple[ToyTask, ...],
) -> dict[tuple[str, str], float]:
    scores: dict[tuple[str, str], float] = {}
    for task, baseline in zip(tasks, BASELINE_SCORES, strict=True):
        scores[(task.task_id, "baseline")] = baseline
        scores[(task.task_id, "ceiling")] = baseline + CEILING_LIFT
    return scores


def _calibrate(
    *,
    num_seeds: int = 2,
    task_count: int = 3,
    pool_ceiling: int = 3,
    task_ids: tuple[str, ...] | None = None,
    power_config: PowerConfig | None = None,
) -> AnchorCalibrationResult:
    tasks = _toy_tasks(task_count)
    experiment = build_toy_experiment(
        internal_tasks=tasks,
        num_seeds=num_seeds,
        initial_template=BASELINE_TEMPLATE,
        ceiling_template=CEILING_TEMPLATE,
    )
    with temp_sqlite_store() as store:
        engine = ReferenceEvalRuntimeConfig().build_engine(
            store,
            experiment=experiment,
            eval_runner=ScriptedEvalProcedureRunner(_scripted_scores(tasks)),
        )
        return run_anchor_calibration(
            engine=engine,
            baseline_candidate=experiment.initial_candidate,
            ceiling_candidate=experiment.ceiling_candidate,
            baseline_purpose="calibration-baseline",
            ceiling_purpose="calibration-ceiling",
            task_ids=task_ids or tuple(task.task_id for task in tasks),
            pool_ceiling=pool_ceiling,
            power_config=power_config,
            bootstrap_resamples=200,
            bootstrap_seed=1,
        )


def test_calibration_reports_the_scripted_per_task_scores() -> None:
    result = _calibrate()

    assert result.baseline.evidence.per_task_values == pytest.approx(
        BASELINE_SCORES
    )
    assert result.ceiling.evidence.per_task_values == pytest.approx(
        tuple(score + CEILING_LIFT for score in BASELINE_SCORES)
    )


def test_calibration_delta_ci_is_the_constant_scripted_lift() -> None:
    # The ceiling beats the baseline by exactly CEILING_LIFT on every task,
    # so every paired resample has that same delta and the interval is
    # degenerate. Exact up to floating-point accumulation.
    result = _calibrate()
    ci = result.paired_delta_ci

    assert ci.point == pytest.approx(CEILING_LIFT)
    assert ci.low == pytest.approx(CEILING_LIFT)
    assert ci.high == pytest.approx(CEILING_LIFT)
    assert ci.excludes_zero()
    assert ci.resamples == 200
    assert ci.level == 0.95


def test_calibration_power_uses_the_calibrated_anchors_and_repeats() -> None:
    result = _calibrate(num_seeds=2, pool_ceiling=3)
    power = result.power

    assert power.naive_mean == pytest.approx(sum(BASELINE_SCORES) / 3)
    assert power.ceiling_mean == pytest.approx(
        sum(BASELINE_SCORES) / 3 + CEILING_LIFT
    )
    assert power.certified_headroom == pytest.approx(CEILING_LIFT)
    assert power.pool_ceiling == 3
    # anchor_samples is the subset engine's repeat count, not the task count.
    assert power.decomposition.anchor_samples == 2
    assert power.decomposition.n_tasks_observed == 3


def test_calibration_honours_a_caller_supplied_power_config() -> None:
    result = _calibrate(power_config=PowerConfig(alpha=0.5, sample_cap=2))

    assert result.power.config.alpha == 0.5
    assert result.power.target_gap == pytest.approx(0.5 * CEILING_LIFT)
    assert max(point.num_seeds for point in result.power.surface) == 2


def test_calibration_evidence_agrees_with_the_subset_sampling_identity() -> None:
    result = _calibrate(num_seeds=2)
    baseline = result.baseline.evidence
    ceiling = result.ceiling.evidence

    assert baseline.task_hashes == ceiling.task_hashes
    assert len(baseline.task_hashes) == 3
    assert baseline.num_seeds == 2
    assert baseline.per_task_counts == (2, 2, 2)
    assert baseline.row_accounting.planned == 6
    assert baseline.row_accounting.present == 6
    assert baseline.eval_config_ref == result.eval_config_ref
    assert ceiling.eval_config_ref == result.eval_config_ref
    assert baseline.graph_hash == ceiling.graph_hash


def test_calibration_restricts_evaluation_to_the_requested_task_subset() -> None:
    # Calibration takes task IDs and subsets the engine itself; only the two
    # named tasks may appear in the resulting evidence.
    result = _calibrate(
        task_count=3, task_ids=("task-0", "task-2"), pool_ceiling=3
    )

    assert len(result.baseline.evidence.task_hashes) == 2
    assert result.baseline.evidence.per_task_values == pytest.approx(
        (BASELINE_SCORES[0], BASELINE_SCORES[2])
    )
    assert result.baseline.evidence.row_accounting.planned == 4


def test_calibration_rejects_task_ids_the_sampling_does_not_own() -> None:
    with pytest.raises(ValueError, match="unknown task IDs"):
        _calibrate(task_ids=("task-0", "task-missing"))


class _StubbedEvalEngine:
    """Delegate to a real engine but return a fixed evaluation result."""

    def __init__(self, inner, result) -> None:
        self._inner = inner
        self._result = result

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    def for_task_ids(self, task_ids: tuple[str, ...]) -> _StubbedEvalEngine:
        return _StubbedEvalEngine(self._inner.for_task_ids(task_ids), self._result)

    def evaluate(self, request: EvalRequest):
        return self._result


def _calibrate_with_result(result) -> AnchorCalibrationResult:
    tasks = _toy_tasks(2)
    experiment = build_toy_experiment(
        internal_tasks=tasks,
        num_seeds=1,
        initial_template=BASELINE_TEMPLATE,
        ceiling_template=CEILING_TEMPLATE,
    )
    with temp_sqlite_store() as store:
        engine = ReferenceEvalRuntimeConfig().build_engine(
            store, experiment=experiment
        )
        return run_anchor_calibration(
            engine=_StubbedEvalEngine(engine, result),
            baseline_candidate=experiment.initial_candidate,
            ceiling_candidate=experiment.ceiling_candidate,
            baseline_purpose="calibration-baseline",
            ceiling_purpose="calibration-ceiling",
            task_ids=("task-0", "task-1"),
            pool_ceiling=2,
            bootstrap_resamples=50,
        )


def _successful_evidence() -> EvalEvidenceWithRef:
    tasks = _toy_tasks(2)
    experiment = build_toy_experiment(
        internal_tasks=tasks,
        num_seeds=1,
        initial_template=BASELINE_TEMPLATE,
        ceiling_template=CEILING_TEMPLATE,
    )
    with temp_sqlite_store() as store:
        engine = ReferenceEvalRuntimeConfig().build_engine(
            store, experiment=experiment
        ).for_task_ids(("task-0", "task-1"))
        evaluated = engine.evaluate(
            EvalRequest(
                request_id="fixture",
                candidate=experiment.initial_candidate,
                metadata=metadata_with_purpose("fixture"),
            )
        )
    assert isinstance(evaluated, EvalEvidenceWithRef)
    return evaluated


def test_calibration_raises_when_an_anchor_evaluation_is_rejected() -> None:
    rejected = EvalRejected(
        detail=ResolutionDetail(
            classification=ResolutionClass.PROVIDER,
            message="provider refused the anchor",
        )
    )

    with pytest.raises(ValueError, match="rejected: provider refused"):
        _calibrate_with_result(rejected)


def test_calibration_raises_when_an_anchor_evaluation_fails() -> None:
    evaluated = _successful_evidence()
    failure = EvalFailureEvidence(
        candidate=evaluated.evidence.candidate,
        eval_config_ref=evaluated.evidence.eval_config_ref,
        eval_role=evaluated.evidence.eval_role,
        exception_type="RuntimeError",
        message="provider exploded",
    )

    with pytest.raises(ValueError, match="failed: provider exploded"):
        _calibrate_with_result(
            EvalEvidenceWithRef(
                evidence=failure, evidence_ref=evaluated.evidence_ref
            )
        )


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"task_hashes": ("drifted-a", "drifted-b")}, "changed task identity order"),
        ({"num_seeds": 9}, "changed sample count"),
        ({"per_task_counts": (3, 3)}, "changed per-task sample counts"),
        ({"reward_ref": None}, "requires internal reward evidence"),
    ],
)
def test_calibration_rejects_evidence_that_drifted_from_the_plan(
    update: dict[str, object], message: str
) -> None:
    evaluated = _successful_evidence()
    drifted = evaluated.evidence.model_copy(update=update)

    with pytest.raises(ValueError, match=message):
        _calibrate_with_result(
            EvalEvidenceWithRef(
                evidence=drifted, evidence_ref=evaluated.evidence_ref
            )
        )


def test_calibration_requires_the_internal_evaluation_split() -> None:
    tasks = _toy_tasks(2)
    experiment = build_toy_experiment(
        internal_tasks=tasks,
        num_seeds=1,
        initial_template=BASELINE_TEMPLATE,
        ceiling_template=CEILING_TEMPLATE,
    )
    with temp_sqlite_store() as store:
        official = ReferenceEvalRuntimeConfig(
            split_role="official"
        ).build_engine(store, experiment=experiment)

        with pytest.raises(ValueError, match="requires internal evaluation"):
            run_anchor_calibration(
                engine=official,
                baseline_candidate=experiment.initial_candidate,
                ceiling_candidate=experiment.ceiling_candidate,
                baseline_purpose="calibration-baseline",
                ceiling_purpose="calibration-ceiling",
                task_ids=("task-c",),
                pool_ceiling=1,
            )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"pool_ceiling": 1}, "pool_ceiling cannot be smaller"),
        ({"bootstrap_level": 0.0}, "bootstrap_level must be in"),
        ({"bootstrap_level": 1.0}, "bootstrap_level must be in"),
        ({"bootstrap_resamples": 0}, "bootstrap_resamples must be at least 1"),
    ],
)
def test_calibration_rejects_out_of_contract_arguments(
    kwargs: dict[str, object], message: str
) -> None:
    tasks = _toy_tasks(2)
    experiment = build_toy_experiment(
        internal_tasks=tasks,
        num_seeds=1,
        initial_template=BASELINE_TEMPLATE,
        ceiling_template=CEILING_TEMPLATE,
    )
    call: dict[str, object] = {
        "baseline_purpose": "calibration-baseline",
        "ceiling_purpose": "calibration-ceiling",
        "task_ids": ("task-0", "task-1"),
        "pool_ceiling": 2,
    }
    call.update(kwargs)
    with temp_sqlite_store() as store:
        engine = ReferenceEvalRuntimeConfig().build_engine(
            store, experiment=experiment
        )

        with pytest.raises(ValueError, match=message):
            run_anchor_calibration(
                engine=engine,
                baseline_candidate=experiment.initial_candidate,
                ceiling_candidate=experiment.ceiling_candidate,
                **call,
            )


def test_calibration_logs_progress_for_both_anchors() -> None:
    # The log callback is the operator-facing progress contract; it must
    # report the planned row count and per-anchor completion.
    messages: list[str] = []
    tasks = _toy_tasks(3)
    experiment = build_toy_experiment(
        internal_tasks=tasks,
        num_seeds=2,
        initial_template=BASELINE_TEMPLATE,
        ceiling_template=CEILING_TEMPLATE,
    )
    with temp_sqlite_store() as store:
        engine = ReferenceEvalRuntimeConfig().build_engine(
            store,
            experiment=experiment,
            eval_runner=ScriptedEvalProcedureRunner(_scripted_scores(tasks)),
        )
        run_anchor_calibration(
            engine=engine,
            baseline_candidate=experiment.initial_candidate,
            ceiling_candidate=experiment.ceiling_candidate,
            baseline_purpose="calibration-baseline",
            ceiling_purpose="calibration-ceiling",
            task_ids=tuple(task.task_id for task in tasks),
            pool_ceiling=3,
            bootstrap_resamples=50,
            log=messages.append,
        )

    assert any("Starting baseline anchor" in line for line in messages)
    assert any("Completed baseline anchor" in line for line in messages)
    assert any("Starting comparison anchor" in line for line in messages)
    assert any("Completed comparison anchor" in line for line in messages)
    # 3 tasks x 2 repeats
    assert any("(6 rows)" in line for line in messages)
    assert any("present=6/6" in line for line in messages)
