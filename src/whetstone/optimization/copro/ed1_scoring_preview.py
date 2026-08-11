from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from dr_store import ObjectStore
from pydantic import BaseModel, ConfigDict, StrictStr

from whetstone.envs.ed1 import (
    ED1_DEFAULT_BLEND_CONFIG,
    Ed1Instance,
    build_ed1_experiment,
)
from whetstone.envs.ed1_blended import BoundedCompressionMetricConfig
from whetstone.envs.ed1_preview import (
    Ed1ScoringPreflight,
    Ed1ScoringRuntimeSummary,
    ed1_environment_fingerprint,
    run_ed1_scoring_preflight,
)
from whetstone.envs.ed1_scoring import CodeBatchScorer
from whetstone.evaluation.engine import EvaluationEngine
from whetstone.evaluation.preview.binding import preview_evaluation_binding
from whetstone.evaluation.preview.persisted import (
    load_aggregate_value,
    load_component_traces,
    load_evaluation_outputs,
)
from whetstone.evaluation.preview.resolution import evaluate_and_resolve
from whetstone.evaluation.schema import (
    EvaluationComponentTraces,
    EvaluationEvidence,
    EvaluationOutputsRecord,
)
from whetstone.experiment.binding import EvaluationBinding
from whetstone.experiment.candidate import (
    Candidate,
    CandidateRef,
    candidate_reference,
)
from whetstone.experiment.task_selection import TaskRoleSelection
from whetstone.optimization.contracts import IntentResolution
from whetstone.optimization.copro.adapter import (
    CoproAttempt,
    CoproDriver,
    CoproFinalization,
    CoproState,
)
from whetstone.optimization.copro.ed1_dry_run import (
    Ed1CoproPreviewTask,
    Ed1CoproRoundAttempt,
    Ed1CoproRoundPreview,
    Ed1CoproSweepPoint,
    Ed1CoproSweepRanges,
    attempt_ed1_copro_round,
)
from whetstone.optimization.copro.ed1_task_model import (
    Ed1TaskModelConfig,
    ed1_task_model_row_job,
)
from whetstone.optimization.proposal.proposer import (
    ProposerRouteConfig,
    ProposerTransport,
)


class Ed1ScoredCandidate(BaseModel):
    """One exact candidate evaluation and its COPRO projection."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    occurrence_ordinal: int
    candidate: CandidateRef
    evidence: EvaluationEvidence
    outputs: EvaluationOutputsRecord
    component_traces: EvaluationComponentTraces
    primary_value: float | None
    compression_value: float | None
    resolution: IntentResolution
    attempt: CoproAttempt


class Ed1CoproScoredRound(BaseModel):
    """One proposed and measured breadth-sized COPRO round."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    preview: Ed1CoproRoundPreview
    evaluations: tuple[Ed1ScoredCandidate, ...]
    state_after: CoproState


class Ed1CoproScoringPoint(BaseModel):
    """All rounds and final ranking for one sweep point."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    settings: Ed1CoproSweepPoint
    evaluation_binding: EvaluationBinding
    rounds: tuple[Ed1CoproScoredRound, ...]
    finalization: CoproFinalization


class Ed1CoproScoringTranscript(BaseModel):
    """Serializable dummy-generation, real-scoring COPRO transcript."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sweep: Ed1CoproSweepRanges
    task_ids: tuple[StrictStr, ...]
    task_selection: TaskRoleSelection | None
    task_model: Ed1TaskModelConfig
    concurrency: int
    runtime: Ed1ScoringRuntimeSummary
    blend_config: BoundedCompressionMetricConfig
    preflight: Ed1ScoringPreflight
    points: tuple[Ed1CoproScoringPoint, ...]


@dataclass(frozen=True, slots=True)
class Ed1CoproCandidateProgress:
    """One candidate evaluation starting or completing within a round."""

    settings: Ed1CoproSweepPoint
    round_index: int
    candidate_index: int
    candidate_count: int
    candidate: CandidateRef
    result: Ed1ScoredCandidate | None


class Ed1CoproRoundFailure(RuntimeError):
    """A proposal round failed after preserving its complete attempt."""

    def __init__(self, attempt: Ed1CoproRoundAttempt) -> None:
        self.attempt = attempt
        super().__init__(attempt.terminal_failure or "COPRO proposal failed")


def _candidate_sequence(
    preview: Ed1CoproRoundPreview,
) -> tuple[Candidate, ...]:
    candidates = tuple(
        mutation.candidate.record for mutation in preview.candidate_mutations
    )
    if preview.round_plan.include_initial_candidate:
        return (*candidates, preview.starting_state.initial_candidate)
    return candidates


def _build_ed1_scored_candidate(
    *,
    store: ObjectStore,
    engine: EvaluationEngine,
    binding: EvaluationBinding,
    candidate: Candidate,
    purpose: str,
    run_id: str,
    round_index: int,
    occurrence_ordinal: int,
) -> Ed1ScoredCandidate:
    evaluated, resolution = evaluate_and_resolve(
        engine,
        binding,
        candidate,
        purpose=purpose,
        run_id=run_id,
        step_index=round_index,
        occurrence_ordinal=occurrence_ordinal,
        message="measured by the ED1 scoring preview",
    )
    reward_ref = evaluated.evidence.reward_ref
    if reward_ref is None:
        raise RuntimeError("internal ED1 evaluation returned no Reward")
    if len(reward_ref.record.evidence_refs) != 2:
        raise RuntimeError(
            "ED1 blended Reward must cite primary and compression aggregates"
        )
    attempt = CoproAttempt.from_resolution(
        occurrence_ordinal=occurrence_ordinal,
        round_index=round_index,
        resolution=resolution,
        expected_run_id=run_id,
        expected_evaluation_binding=binding,
        expected_reward_policy_hash=reward_ref.record.reward_policy_hash,
    )
    return Ed1ScoredCandidate(
        occurrence_ordinal=occurrence_ordinal,
        candidate=evaluated.evidence.candidate,
        evidence=evaluated.evidence,
        outputs=load_evaluation_outputs(store, evaluated.evidence),
        component_traces=load_component_traces(store, evaluated.evidence),
        primary_value=load_aggregate_value(
            store, reward_ref.record.evidence_refs[0]
        ),
        compression_value=load_aggregate_value(
            store, reward_ref.record.evidence_refs[1]
        ),
        resolution=resolution,
        attempt=attempt,
    )


def run_ed1_copro_scoring_preview(
    *,
    store: ObjectStore,
    tasks: tuple[Ed1Instance, ...],
    sweep: Ed1CoproSweepRanges,
    proposer_kind: str,
    proposer_config: ProposerRouteConfig,
    proposer_transport: ProposerTransport,
    task_model: Ed1TaskModelConfig,
    batch_scorer: CodeBatchScorer,
    runtime: Ed1ScoringRuntimeSummary,
    task_selection: TaskRoleSelection | None = None,
    preflight_task: Ed1Instance | None = None,
    concurrency: int = 1,
    repeats: int = 1,
    blend_config: BoundedCompressionMetricConfig = ED1_DEFAULT_BLEND_CONFIG,
    proposal_observer: Callable[[Ed1CoproRoundAttempt], None] | None = None,
    candidate_observer: Callable[[Ed1CoproCandidateProgress], None]
    | None = None,
) -> Ed1CoproScoringTranscript:
    """Run proposal drafts through deterministic ED1 generation and scoring."""

    if not tasks:
        raise ValueError("COPRO scoring preview requires at least one task")
    if repeats < 1:
        raise ValueError("COPRO scoring preview repeats must be positive")
    if concurrency < 1:
        raise ValueError("COPRO scoring preview concurrency must be positive")
    task_ids = tuple(task.humaneval_task.task_id for task in tasks)
    if task_selection is not None and task_selection.task_ids != task_ids:
        raise ValueError(
            "COPRO scoring task IDs do not match the selected manifest role"
        )
    preflight = run_ed1_scoring_preflight(
        (preflight_task or tasks[0],), batch_scorer
    )
    first = tasks[0]
    preview_task = Ed1CoproPreviewTask(
        task_id=first.humaneval_task.task_id,
        input_code=first.input_code,
    )
    points: list[Ed1CoproScoringPoint] = []
    for settings in sweep.expand():
        experiment = build_ed1_experiment(
            provider_call_config=task_model.provider_call_config,
            budget_ratio=settings.budget_ratio,
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
            batch_scorer=batch_scorer,
        )
        binding = preview_evaluation_binding(
            engine,
            campaign="copro-scoring-preview",
            provenance_note=(
                f"{task_model.kind.value}-generation-real-humaneval-scoring"
            ),
            environment_fingerprint=ed1_environment_fingerprint(runtime),
        )
        lifecycle = CoproDriver(settings.copro)
        state = lifecycle.initial_state(experiment.initial_candidate)
        run_id = f"copro-scoring-preview:{settings.sweep_ordinal}"
        rounds: list[Ed1CoproScoredRound] = []
        for round_index in range(settings.copro.depth):
            proposal_attempt = attempt_ed1_copro_round(
                settings=settings,
                state=state,
                preview_task=preview_task,
                proposer_kind=proposer_kind,
                proposer_config=proposer_config,
                transport=proposer_transport,
                request_ordinal=(
                    settings.sweep_ordinal * settings.copro.depth + round_index
                ),
            )
            if proposal_observer is not None:
                proposal_observer(proposal_attempt)
            if not proposal_attempt.succeeded:
                raise Ed1CoproRoundFailure(proposal_attempt)
            preview = proposal_attempt.require_preview()
            start = round_index * settings.copro.breadth
            candidates = _candidate_sequence(preview)
            evaluations_list: list[Ed1ScoredCandidate] = []
            for index, candidate in enumerate(candidates):
                if candidate_observer is not None:
                    candidate_observer(
                        Ed1CoproCandidateProgress(
                            settings=settings,
                            round_index=round_index,
                            candidate_index=index,
                            candidate_count=len(candidates),
                            candidate=candidate_reference(candidate),
                            result=None,
                        )
                    )
                evaluated = _build_ed1_scored_candidate(
                    store=store,
                    engine=engine,
                    binding=binding,
                    candidate=candidate,
                    purpose=preview.round_plan.proposal_mode,
                    run_id=run_id,
                    round_index=round_index,
                    occurrence_ordinal=start + index,
                )
                evaluations_list.append(evaluated)
                if candidate_observer is not None:
                    candidate_observer(
                        Ed1CoproCandidateProgress(
                            settings=settings,
                            round_index=round_index,
                            candidate_index=index,
                            candidate_count=len(candidates),
                            candidate=evaluated.candidate,
                            result=evaluated,
                        )
                    )
            evaluations = tuple(evaluations_list)
            state = lifecycle.fold_round(
                state,
                tuple(item.attempt for item in evaluations),
            )
            rounds.append(
                Ed1CoproScoredRound(
                    preview=preview,
                    evaluations=evaluations,
                    state_after=state,
                )
            )
        points.append(
            Ed1CoproScoringPoint(
                settings=settings,
                evaluation_binding=binding,
                rounds=tuple(rounds),
                finalization=lifecycle.finalize(state),
            )
        )
    return Ed1CoproScoringTranscript(
        sweep=sweep,
        task_ids=task_ids,
        task_selection=task_selection,
        task_model=task_model,
        concurrency=concurrency,
        runtime=runtime,
        blend_config=blend_config,
        preflight=preflight,
        points=tuple(points),
    )


__all__ = [
    "Ed1CoproCandidateProgress",
    "Ed1CoproRoundFailure",
    "Ed1CoproScoredRound",
    "Ed1CoproScoringPoint",
    "Ed1CoproScoringTranscript",
    "Ed1ScoredCandidate",
    "run_ed1_copro_scoring_preview",
]
