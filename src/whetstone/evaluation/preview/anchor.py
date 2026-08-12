from __future__ import annotations

from collections.abc import Callable

from dr_store import ObjectStore
from pydantic import BaseModel, ConfigDict, StrictStr

from whetstone.evaluation.analysis.calibration import run_anchor_calibration
from whetstone.evaluation.analysis.power import PowerConfig, PowerResult
from whetstone.evaluation.analysis.statistics import (
    DEFAULT_RESAMPLES,
    BootstrapCI,
)
from whetstone.evaluation.engine import EngineEvaluation, EvaluationEngine
from whetstone.evaluation.preview.persisted import (
    load_component_traces,
    load_evaluation_outputs,
)
from whetstone.evaluation.preview.preflight import (
    PreviewMetadata,
    ScoringPreflight,
)
from whetstone.evaluation.schema import (
    EvaluationComponentTraces,
    EvaluationEvidence,
    EvaluationOutputsRecord,
)
from whetstone.experiment.binding import EvaluationBinding
from whetstone.experiment.candidate import Candidate
from whetstone.experiment.task_selection import TaskRoleSelection
from whetstone.optimization.proposal.mutation import MUTATION_FIELD

__all__ = [
    "AnchorConfigPreview",
    "BaselinePreviewTranscript",
    "BaselineSweepTranscript",
    "PreviewMetadata",
    "ScoringPreflight",
    "run_baseline_preview",
    "run_baseline_sweep",
]


def _calibration_task_hashes(
    engine: EvaluationEngine,
    task_ids: tuple[str, ...],
) -> tuple[str, ...]:
    """Map caller task IDs to the engine sampling's canonical task hashes."""
    hashes = engine.sampling.task_set.task_hashes
    if set(task_ids).issubset(hashes):
        return task_ids
    by_task_id = {
        str(task.id): task_hash
        for task, task_hash in zip(
            engine.sampling.tasks,
            hashes,
            strict=True,
        )
    }
    unknown = tuple(
        task_id for task_id in task_ids if task_id not in by_task_id
    )
    if unknown:
        raise ValueError(
            f"baseline preview task IDs are unknown to sampling: {unknown!r}"
        )
    return tuple(by_task_id[task_id] for task_id in task_ids)


class AnchorConfigPreview(BaseModel):
    """One anchor config with its exact evaluation and rendered rows."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: StrictStr
    instruction: StrictStr
    evidence: EvaluationEvidence
    outputs: EvaluationOutputsRecord
    component_traces: EvaluationComponentTraces


class BaselinePreviewTranscript(BaseModel):
    """Serializable two-anchor calibration, bootstrap, and power preview."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_ids: tuple[StrictStr, ...]
    task_selection: TaskRoleSelection | None
    pool_ceiling: int
    budget_ratio: float | None
    concurrency: int
    preflight: ScoringPreflight
    metadata: PreviewMetadata
    evaluation_binding: EvaluationBinding
    baseline: AnchorConfigPreview
    ceiling: AnchorConfigPreview
    paired_delta_ci: BootstrapCI
    power: PowerResult


class BaselineSweepTranscript(BaseModel):
    """All budget modes in one launch-ready baseline calibration sweep."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_ids: tuple[StrictStr, ...]
    task_selection: TaskRoleSelection | None
    budget_ratios: tuple[float | None, ...]
    previews: tuple[BaselinePreviewTranscript, ...]


def _anchor_config_preview(
    *,
    label: str,
    store: ObjectStore,
    evaluated: EngineEvaluation,
) -> AnchorConfigPreview:
    instruction = evaluated.evidence.candidate.record.payload[MUTATION_FIELD]
    if not isinstance(instruction, str):
        raise ValueError("baseline anchor instruction must be text")
    return AnchorConfigPreview(
        label=label,
        instruction=instruction,
        evidence=evaluated.evidence,
        outputs=load_evaluation_outputs(store, evaluated.evidence),
        component_traces=load_component_traces(store, evaluated.evidence),
    )


def run_baseline_preview(
    *,
    store: ObjectStore,
    engine: EvaluationEngine,
    evaluation_binding: EvaluationBinding,
    baseline_candidate: Candidate,
    ceiling_candidate: Candidate,
    task_ids: tuple[str, ...],
    pool_ceiling: int,
    preflight: ScoringPreflight,
    metadata: PreviewMetadata,
    task_selection: TaskRoleSelection | None = None,
    budget_ratio: float | None = None,
    concurrency: int = 1,
    baseline_purpose: str = "calibration-baseline",
    ceiling_purpose: str = "calibration-ceiling",
    baseline_log_label: str = "hand-engineered baseline",
    ceiling_log_label: str = "hand-engineered comparison anchor",
    power_config: PowerConfig | None = None,
    bootstrap_level: float = 0.95,
    bootstrap_resamples: int = DEFAULT_RESAMPLES,
    bootstrap_seed: int = 0,
    log: Callable[[str], None] | None = None,
) -> BaselinePreviewTranscript:
    """Evaluate both anchor candidates on one exact shared binding."""

    if not task_ids:
        raise ValueError("baseline preview requires at least one task ID")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("baseline preview task IDs must be unique")
    if task_selection is not None and task_selection.task_ids != task_ids:
        raise ValueError(
            "baseline preview task IDs do not match the selected manifest role"
        )
    if concurrency < 1:
        raise ValueError("baseline preview concurrency must be positive")
    budget_label = (
        "unbudgeted"
        if budget_ratio is None
        else f"budget ratio {budget_ratio:g}"
    )
    calibration_task_ids = _calibration_task_hashes(engine, task_ids)
    calibration = run_anchor_calibration(
        engine=engine,
        evaluation_binding=evaluation_binding,
        baseline_candidate=baseline_candidate,
        ceiling_candidate=ceiling_candidate,
        baseline_purpose=baseline_purpose,
        ceiling_purpose=ceiling_purpose,
        baseline_log_label=baseline_log_label,
        ceiling_log_label=ceiling_log_label,
        task_ids=calibration_task_ids,
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
    return BaselinePreviewTranscript(
        task_ids=task_ids,
        task_selection=task_selection,
        pool_ceiling=pool_ceiling,
        budget_ratio=budget_ratio,
        concurrency=concurrency,
        preflight=preflight,
        metadata=metadata,
        evaluation_binding=calibration.evaluation_binding,
        baseline=_anchor_config_preview(
            label=baseline_log_label,
            store=store,
            evaluated=calibration.baseline,
        ),
        ceiling=_anchor_config_preview(
            label=ceiling_log_label,
            store=store,
            evaluated=calibration.ceiling,
        ),
        paired_delta_ci=calibration.paired_delta_ci,
        power=calibration.power,
    )


def run_baseline_sweep(
    *,
    preview_factory: Callable[[float | None], BaselinePreviewTranscript],
    task_ids: tuple[str, ...],
    budget_ratios: tuple[float | None, ...],
    task_selection: TaskRoleSelection | None = None,
) -> BaselineSweepTranscript:
    """Evaluate both anchors under each requested budget framing."""
    if not budget_ratios:
        raise ValueError("baseline sweep requires at least one budget mode")
    if len(set(budget_ratios)) != len(budget_ratios):
        raise ValueError("baseline sweep budget modes must be unique")
    previews = tuple(
        preview_factory(budget_ratio) for budget_ratio in budget_ratios
    )
    return BaselineSweepTranscript(
        task_ids=task_ids,
        task_selection=task_selection,
        budget_ratios=budget_ratios,
        previews=previews,
    )
