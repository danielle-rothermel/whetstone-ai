from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from dr_store import ObjectStore

from whetstone.envs.code_comp.dataset import CodeCompTaskInstance
from whetstone.envs.code_comp.modes.encdec import (
    EncDecExperiment,
    EncDecTaskModelConfig,
    ed1_preview_metadata,
)
from whetstone.envs.code_comp.registry import (
    CodeCompMode,
    build_code_comp_experiment,
)
from whetstone.envs.code_comp.reward.blended import (
    ED1_DEFAULT_BLEND_CONFIG,
    BoundedCompressionMetricConfig,
    ed1_blended_aggregate_values,
)
from whetstone.envs.code_comp.runtime import (
    Ed1ScoringRuntimeSummary,
    ed1_environment_fingerprint,
)
from whetstone.envs.code_comp.scoring import (
    CodeBatchScorer,
    run_encdec_scoring_preflight,
)
from whetstone.execution.partials import PartialLog
from whetstone.execution.prompt_cache import PromptResultCache
from whetstone.experiment.candidate import Candidate
from whetstone.experiment.task_selection import TaskRoleSelection

if TYPE_CHECKING:
    from whetstone.evaluation.analysis.power import PowerConfig
    from whetstone.evaluation.engine import EvaluationEngine
    from whetstone.evaluation.preview.anchor import (
        BaselinePreviewTranscript,
        BaselineSweepTranscript,
    )
    from whetstone.experiment.binding import EvaluationBinding
    from whetstone.optimization.copro.ed1_dry_run import (
        Ed1CoproRoundAttempt,
        Ed1CoproSweepPoint,
        Ed1CoproSweepRanges,
    )
    from whetstone.optimization.copro.scoring_preview import (
        CandidateProgress,
        ScoringTranscript,
    )
    from whetstone.optimization.proposal.proposer import (
        ProposerRouteConfig,
        ProposerTransport,
    )


def build_ed1_preview_engine(
    *,
    store: ObjectStore,
    experiment: EncDecExperiment,
    task_model: EncDecTaskModelConfig,
    batch_scorer: CodeBatchScorer,
    concurrency: int = 1,
    partial_log: PartialLog | None = None,
    prompt_cache: PromptResultCache | None = None,
) -> EvaluationEngine:
    """Build one ED1 preview evaluation engine for matrix and debug scripts."""

    from whetstone.evaluation.drivers.code_comp.row_jobs import (
        ed1_task_model_row_job,
    )
    from whetstone.evaluation.engine import EvaluationEngine

    return EvaluationEngine(
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


def _selected_ed1_tasks(
    tasks: tuple[CodeCompTaskInstance, ...],
    task_ids: tuple[str, ...],
) -> tuple[CodeCompTaskInstance, ...]:
    if not task_ids:
        raise ValueError("baseline preview requires at least one task ID")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("baseline preview task IDs must be unique")
    by_id = {task.humaneval_task.task_id: task for task in tasks}
    missing = tuple(task_id for task_id in task_ids if task_id not in by_id)
    if missing:
        raise ValueError(f"baseline preview task IDs are unknown: {missing}")
    return tuple(by_id[task_id] for task_id in task_ids)


def run_ed1_anchor_baseline_preview(
    *,
    store: ObjectStore,
    tasks: tuple[CodeCompTaskInstance, ...],
    task_ids: tuple[str, ...],
    pool_ceiling: int,
    task_model: EncDecTaskModelConfig,
    batch_scorer: CodeBatchScorer,
    runtime: Ed1ScoringRuntimeSummary,
    task_selection: TaskRoleSelection | None = None,
    preflight_task: CodeCompTaskInstance | None = None,
    budget_ratio: float | None = None,
    concurrency: int = 1,
    partial_log: PartialLog | None = None,
    prompt_cache: PromptResultCache | None = None,
    repeats: int = 1,
    blend_config: BoundedCompressionMetricConfig = ED1_DEFAULT_BLEND_CONFIG,
    power_config: PowerConfig | None = None,
    bootstrap_level: float = 0.95,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 0,
    log: Callable[[str], None] | None = None,
) -> BaselinePreviewTranscript:
    """Wire ED1 execution and run the generic anchor baseline preview."""

    from whetstone.evaluation.preview.anchor import run_baseline_preview
    from whetstone.evaluation.preview.binding import preview_evaluation_binding

    selected = _selected_ed1_tasks(tasks, task_ids)
    budget_label = (
        "unbudgeted"
        if budget_ratio is None
        else f"budget ratio {budget_ratio:g}"
    )
    if log is not None:
        log(f"{budget_label}: starting scoring-runtime preflight")
    preflight = run_encdec_scoring_preflight(
        (preflight_task or selected[0],), batch_scorer
    )
    if log is not None:
        log(
            f"{budget_label}: scoring-runtime preflight completed "
            f"({preflight.task_id}: {preflight.outcome})"
        )
    experiment = cast(
        EncDecExperiment,
        build_code_comp_experiment(
            CodeCompMode.ENCDEC,
            provider_call_config=task_model.provider_call_config,
            budget_ratio=budget_ratio,
            tasks=tasks,
            internal_n=len(tasks),
            official_n=len(tasks),
            repeats=repeats,
            blend_config=blend_config,
        ),
    )
    engine = build_ed1_preview_engine(
        store=store,
        experiment=experiment,
        task_model=task_model,
        batch_scorer=batch_scorer,
        concurrency=concurrency,
        partial_log=partial_log,
        prompt_cache=prompt_cache,
    )
    binding = preview_evaluation_binding(
        engine,
        campaign="ed1-baseline-preview",
        provenance_note=(
            f"{task_model.kind.value}-generation-real-humaneval-scoring"
        ),
        environment_fingerprint=ed1_environment_fingerprint(runtime),
    )
    return run_baseline_preview(
        store=store,
        engine=engine,
        evaluation_binding=binding,
        baseline_candidate=experiment.initial_candidate,
        ceiling_candidate=experiment.ceiling_candidate,
        task_ids=task_ids,
        pool_ceiling=pool_ceiling,
        preflight=preflight,
        metadata=ed1_preview_metadata(
            task_model=task_model,
            runtime=runtime,
            blend_config=blend_config,
        ),
        task_selection=task_selection,
        budget_ratio=budget_ratio,
        concurrency=concurrency,
        baseline_purpose="ed1-calibration-baseline",
        ceiling_purpose="ed1-calibration-ceiling",
        power_config=power_config,
        bootstrap_level=bootstrap_level,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        log=log,
    )


def run_ed1_anchor_baseline_sweep(
    *,
    store: ObjectStore,
    tasks: tuple[CodeCompTaskInstance, ...],
    task_ids: tuple[str, ...],
    pool_ceiling: int,
    task_model: EncDecTaskModelConfig,
    batch_scorer: CodeBatchScorer,
    runtime: Ed1ScoringRuntimeSummary,
    budget_ratios: tuple[float | None, ...],
    task_selection: TaskRoleSelection | None = None,
    preflight_task: CodeCompTaskInstance | None = None,
    concurrency: int = 1,
    partial_log: PartialLog | None = None,
    prompt_cache: PromptResultCache | None = None,
    repeats: int = 1,
    blend_config: BoundedCompressionMetricConfig = ED1_DEFAULT_BLEND_CONFIG,
    power_config: PowerConfig | None = None,
    bootstrap_level: float = 0.95,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 0,
    log: Callable[[str], None] | None = None,
) -> BaselineSweepTranscript:
    """Evaluate both ED1 anchors under each requested budget framing."""

    from whetstone.evaluation.preview.anchor import run_baseline_sweep

    return run_baseline_sweep(
        preview_factory=lambda budget_ratio: run_ed1_anchor_baseline_preview(
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
        ),
        task_ids=task_ids,
        budget_ratios=budget_ratios,
        task_selection=task_selection,
    )


def run_ed1_copro_scoring_preview(
    *,
    store: ObjectStore,
    tasks: tuple[CodeCompTaskInstance, ...],
    sweep: Ed1CoproSweepRanges,
    proposer_kind: str,
    proposer_config: ProposerRouteConfig,
    proposer_transport: ProposerTransport,
    task_model: EncDecTaskModelConfig,
    batch_scorer: CodeBatchScorer,
    runtime: Ed1ScoringRuntimeSummary,
    task_selection: TaskRoleSelection | None = None,
    preflight_task: CodeCompTaskInstance | None = None,
    concurrency: int = 1,
    repeats: int = 1,
    blend_config: BoundedCompressionMetricConfig = ED1_DEFAULT_BLEND_CONFIG,
    proposal_observer: Callable[[Ed1CoproRoundAttempt], None] | None = None,
    candidate_observer: Callable[[CandidateProgress], None] | None = None,
) -> ScoringTranscript:
    """Wire ED1 execution and run the generic COPRO scoring preview."""

    from whetstone.evaluation.preview.binding import preview_evaluation_binding
    from whetstone.optimization.copro.ed1_dry_run import Ed1CoproPreviewTask
    from whetstone.optimization.copro.scoring_preview import (
        run_copro_scoring_preview,
    )

    if not tasks:
        raise ValueError("COPRO scoring preview requires at least one task")
    if repeats < 1:
        raise ValueError("COPRO scoring preview repeats must be positive")
    task_ids = tuple(task.humaneval_task.task_id for task in tasks)
    preflight = run_encdec_scoring_preflight(
        (preflight_task or tasks[0],), batch_scorer
    )
    first = tasks[0]
    preview_task = Ed1CoproPreviewTask(
        task_id=first.humaneval_task.task_id,
        input_code=first.input_code,
    )

    def _experiment_for(settings: Ed1CoproSweepPoint) -> EncDecExperiment:
        return cast(
            EncDecExperiment,
            build_code_comp_experiment(
                CodeCompMode.ENCDEC,
                provider_call_config=task_model.provider_call_config,
                budget_ratio=settings.budget_ratio,
                tasks=tasks,
                internal_n=len(tasks),
                official_n=len(tasks),
                repeats=repeats,
                blend_config=blend_config,
            ),
        )

    def engine_factory(settings: Ed1CoproSweepPoint) -> EvaluationEngine:
        return build_ed1_preview_engine(
            store=store,
            experiment=_experiment_for(settings),
            task_model=task_model,
            batch_scorer=batch_scorer,
            concurrency=concurrency,
        )

    def binding_factory(
        engine: EvaluationEngine,
        settings: Ed1CoproSweepPoint,
    ) -> EvaluationBinding:
        del settings
        return preview_evaluation_binding(
            engine,
            campaign="copro-scoring-preview",
            provenance_note=(
                f"{task_model.kind.value}-generation-real-humaneval-scoring"
            ),
            environment_fingerprint=ed1_environment_fingerprint(runtime),
        )

    def initial_candidate_factory(settings: Ed1CoproSweepPoint) -> Candidate:
        return _experiment_for(settings).initial_candidate

    return run_copro_scoring_preview(
        store=store,
        sweep=sweep,
        preview_task=preview_task,
        proposer_kind=proposer_kind,
        proposer_config=proposer_config,
        proposer_transport=proposer_transport,
        engine_factory=engine_factory,
        binding_factory=binding_factory,
        initial_candidate_factory=initial_candidate_factory,
        aggregate_values_fn=ed1_blended_aggregate_values,
        preflight=preflight,
        metadata=ed1_preview_metadata(
            task_model=task_model,
            runtime=runtime,
            blend_config=blend_config,
        ),
        task_ids=task_ids,
        task_selection=task_selection,
        concurrency=concurrency,
        proposal_observer=proposal_observer,
        candidate_observer=candidate_observer,
    )


__all__ = [
    "build_ed1_preview_engine",
    "run_ed1_anchor_baseline_preview",
    "run_ed1_anchor_baseline_sweep",
    "run_ed1_copro_scoring_preview",
]
