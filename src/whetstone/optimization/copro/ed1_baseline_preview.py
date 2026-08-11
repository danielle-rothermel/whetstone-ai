from __future__ import annotations

import json
from collections.abc import Callable

from dr_store import ObjectStore
from pydantic import BaseModel, ConfigDict, StrictStr

from whetstone.envs.ed1 import (
    ED1_DEFAULT_BLEND_CONFIG,
    Ed1Instance,
    build_ed1_experiment,
)
from whetstone.envs.ed1_blended import BoundedCompressionMetricConfig
from whetstone.envs.ed1_calibration import run_ed1_calibration
from whetstone.envs.ed1_scoring import CodeBatchScorer
from whetstone.evaluation.analysis.power import PowerConfig, PowerResult
from whetstone.evaluation.analysis.statistics import (
    DEFAULT_RESAMPLES,
    BootstrapCI,
)
from whetstone.evaluation.engine import EngineEvaluation, EvaluationEngine
from whetstone.evaluation.schema import (
    EvaluationComponentTraces,
    EvaluationEvidence,
    EvaluationOutputsRecord,
)
from whetstone.execution.partials import PartialLog
from whetstone.execution.prompt_cache import PromptResultCache
from whetstone.experiment.binding import EvaluationBinding
from whetstone.experiment.task_selection import TaskRoleSelection
from whetstone.optimization.copro.ed1_scoring_preview import (
    Ed1ScoringPreflight,
    Ed1ScoringRuntimeSummary,
    ed1_preview_evaluation_binding,
    run_ed1_scoring_preflight,
)
from whetstone.optimization.copro.ed1_task_model import (
    Ed1TaskModelConfig,
    ed1_task_model_row_job,
)
from whetstone.optimization.proposal.mutation import MUTATION_FIELD


class Ed1BaselineArmPreview(BaseModel):
    """One baseline arm with its exact evaluation and rendered row evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: StrictStr
    instruction: StrictStr
    evidence: EvaluationEvidence
    outputs: EvaluationOutputsRecord
    component_traces: EvaluationComponentTraces


class Ed1BaselinePreviewTranscript(BaseModel):
    """Serializable two-anchor calibration, bootstrap, and power preview."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_ids: tuple[StrictStr, ...]
    task_selection: TaskRoleSelection | None
    pool_ceiling: int
    budget_ratio: float | None
    concurrency: int
    task_model: Ed1TaskModelConfig
    runtime: Ed1ScoringRuntimeSummary
    blend_config: BoundedCompressionMetricConfig
    preflight: Ed1ScoringPreflight
    evaluation_binding: EvaluationBinding
    baseline: Ed1BaselineArmPreview
    ceiling: Ed1BaselineArmPreview
    paired_delta_ci: BootstrapCI
    power: PowerResult


class Ed1BaselineSweepTranscript(BaseModel):
    """All budget modes in one launch-ready baseline calibration sweep."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_ids: tuple[StrictStr, ...]
    task_selection: TaskRoleSelection | None
    budget_ratios: tuple[float | None, ...]
    previews: tuple[Ed1BaselinePreviewTranscript, ...]


def _load_outputs(
    store: ObjectStore,
    evidence: EvaluationEvidence,
) -> EvaluationOutputsRecord:
    raw = store.get(evidence.outputs_ref.reference)
    if raw is None:
        raise RuntimeError("persisted baseline outputs are missing")
    return EvaluationOutputsRecord.model_validate(raw)


def _load_component_traces(
    store: ObjectStore,
    evidence: EvaluationEvidence,
) -> EvaluationComponentTraces:
    raw = store.get(evidence.component_traces_ref.reference)
    if raw is None:
        raise RuntimeError("persisted baseline component traces are missing")
    return EvaluationComponentTraces.model_validate_json(json.dumps(raw))


def _selected_tasks(
    tasks: tuple[Ed1Instance, ...],
    task_ids: tuple[str, ...],
) -> tuple[Ed1Instance, ...]:
    if not task_ids:
        raise ValueError("baseline preview requires at least one task ID")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("baseline preview task IDs must be unique")
    by_id = {task.humaneval_task.task_id: task for task in tasks}
    missing = tuple(task_id for task_id in task_ids if task_id not in by_id)
    if missing:
        raise ValueError(f"baseline preview task IDs are unknown: {missing}")
    return tuple(by_id[task_id] for task_id in task_ids)


def _arm(
    *,
    label: str,
    store: ObjectStore,
    evaluated: EngineEvaluation,
) -> Ed1BaselineArmPreview:
    instruction = evaluated.evidence.candidate.record.payload[MUTATION_FIELD]
    if not isinstance(instruction, str):
        raise ValueError("ED1 baseline instruction must be text")
    return Ed1BaselineArmPreview(
        label=label,
        instruction=instruction,
        evidence=evaluated.evidence,
        outputs=_load_outputs(store, evaluated.evidence),
        component_traces=_load_component_traces(store, evaluated.evidence),
    )


def run_ed1_baseline_preview(
    *,
    store: ObjectStore,
    tasks: tuple[Ed1Instance, ...],
    task_ids: tuple[str, ...],
    pool_ceiling: int,
    task_model: Ed1TaskModelConfig,
    batch_scorer: CodeBatchScorer,
    runtime: Ed1ScoringRuntimeSummary,
    task_selection: TaskRoleSelection | None = None,
    preflight_task: Ed1Instance | None = None,
    budget_ratio: float | None = None,
    concurrency: int = 1,
    partial_log: PartialLog | None = None,
    prompt_cache: PromptResultCache | None = None,
    repeats: int = 1,
    blend_config: BoundedCompressionMetricConfig = ED1_DEFAULT_BLEND_CONFIG,
    power_config: PowerConfig | None = None,
    bootstrap_level: float = 0.95,
    bootstrap_resamples: int = DEFAULT_RESAMPLES,
    bootstrap_seed: int = 0,
    log: Callable[[str], None] | None = None,
) -> Ed1BaselinePreviewTranscript:
    """Evaluate both hand-engineered anchors on one exact shared binding."""

    if task_selection is not None and task_selection.task_ids != task_ids:
        raise ValueError(
            "baseline preview task IDs do not match the selected manifest role"
        )
    if concurrency < 1:
        raise ValueError("baseline preview concurrency must be positive")
    selected = _selected_tasks(tasks, task_ids)
    budget_label = (
        "unbudgeted"
        if budget_ratio is None
        else f"budget ratio {budget_ratio:g}"
    )
    if log is not None:
        log(f"{budget_label}: starting scoring-runtime preflight")
    preflight = run_ed1_scoring_preflight(
        (preflight_task or selected[0],), batch_scorer
    )
    if log is not None:
        log(
            f"{budget_label}: scoring-runtime preflight completed "
            f"({preflight.task_id}: {preflight.outcome})"
        )
    experiment = build_ed1_experiment(
        provider_call_config=task_model.provider_call_config,
        budget_ratio=budget_ratio,
        tasks=tasks,
        internal_n=len(tasks),
        official_n=len(tasks),
        repeats=repeats,
        blend_config=blend_config,
    )
    engine = EvaluationEngine(
        store=store,
        experiment=experiment,
        sampling=experiment.eval_configs.internal,
        execution_policy=task_model.execution_policy,
        row_job_factory=ed1_task_model_row_job(task_model),
        concurrency=concurrency,
        partial_log=partial_log,
        prompt_cache=prompt_cache,
        batch_scorer=batch_scorer,
    )
    binding = ed1_preview_evaluation_binding(
        engine,
        runtime,
        campaign="ed1-baseline-preview",
        task_model_kind=task_model.kind.value,
    )
    calibration = run_ed1_calibration(
        engine=engine,
        evaluation_binding=binding,
        task_ids=task_ids,
        pool_ceiling=pool_ceiling,
        power_config=power_config,
        bootstrap_level=bootstrap_level,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        log=(
            None
            if log is None
            else lambda message: log(f"{budget_label}: {message}")
        ),
    )
    return Ed1BaselinePreviewTranscript(
        task_ids=task_ids,
        task_selection=task_selection,
        pool_ceiling=pool_ceiling,
        budget_ratio=budget_ratio,
        concurrency=concurrency,
        task_model=task_model,
        runtime=runtime,
        blend_config=blend_config,
        preflight=preflight,
        evaluation_binding=calibration.evaluation_binding,
        baseline=_arm(
            label="hand-engineered baseline",
            store=store,
            evaluated=calibration.baseline,
        ),
        ceiling=_arm(
            label="hand-engineered comparison anchor",
            store=store,
            evaluated=calibration.ceiling,
        ),
        paired_delta_ci=calibration.paired_delta_ci,
        power=calibration.power,
    )


def run_ed1_baseline_sweep(
    *,
    store: ObjectStore,
    tasks: tuple[Ed1Instance, ...],
    task_ids: tuple[str, ...],
    pool_ceiling: int,
    task_model: Ed1TaskModelConfig,
    batch_scorer: CodeBatchScorer,
    runtime: Ed1ScoringRuntimeSummary,
    budget_ratios: tuple[float | None, ...],
    task_selection: TaskRoleSelection | None = None,
    preflight_task: Ed1Instance | None = None,
    concurrency: int = 1,
    partial_log: PartialLog | None = None,
    prompt_cache: PromptResultCache | None = None,
    repeats: int = 1,
    blend_config: BoundedCompressionMetricConfig = ED1_DEFAULT_BLEND_CONFIG,
    power_config: PowerConfig | None = None,
    bootstrap_level: float = 0.95,
    bootstrap_resamples: int = DEFAULT_RESAMPLES,
    bootstrap_seed: int = 0,
    log: Callable[[str], None] | None = None,
) -> Ed1BaselineSweepTranscript:
    """Evaluate both anchors under each requested budget framing."""
    if not budget_ratios:
        raise ValueError("baseline sweep requires at least one budget mode")
    if len(set(budget_ratios)) != len(budget_ratios):
        raise ValueError("baseline sweep budget modes must be unique")
    previews = tuple(
        run_ed1_baseline_preview(
            store=store,
            tasks=tasks,
            task_ids=task_ids,
            task_selection=task_selection,
            preflight_task=preflight_task,
            pool_ceiling=pool_ceiling,
            task_model=task_model,
            batch_scorer=batch_scorer,
            runtime=runtime,
            budget_ratio=budget_ratio,
            concurrency=concurrency,
            partial_log=partial_log,
            prompt_cache=prompt_cache,
            repeats=repeats,
            blend_config=blend_config,
            power_config=power_config,
            bootstrap_level=bootstrap_level,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
            log=log,
        )
        for budget_ratio in budget_ratios
    )
    return Ed1BaselineSweepTranscript(
        task_ids=task_ids,
        task_selection=task_selection,
        budget_ratios=budget_ratios,
        previews=previews,
    )


__all__ = [
    "Ed1BaselineArmPreview",
    "Ed1BaselinePreviewTranscript",
    "Ed1BaselineSweepTranscript",
    "run_ed1_baseline_preview",
    "run_ed1_baseline_sweep",
]
