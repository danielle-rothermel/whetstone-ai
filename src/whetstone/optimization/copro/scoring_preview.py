from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from dr_store import ObjectStore
from pydantic import BaseModel, ConfigDict, StrictStr

from whetstone.evaluation.engine import EvaluationEngine
from whetstone.evaluation.preview.preflight import (
    PreviewMetadata,
    ScoringPreflight,
)
from whetstone.evaluation.preview.scored import (
    ScoredCandidate,
    build_scored_candidate,
)
from whetstone.experiment.binding import EvaluationBinding
from whetstone.experiment.candidate import (
    Candidate,
    CandidateRef,
    candidate_reference,
)
from whetstone.experiment.reward import RewardRef
from whetstone.experiment.task_selection import TaskRoleSelection
from whetstone.optimization.copro.adapter import (
    CoproDriver,
    CoproFinalization,
    CoproState,
)
from whetstone.optimization.copro.code_comp.dry_run import (
    CodeCompCoproPreviewTask,
    CodeCompCoproRoundAttempt,
    CodeCompCoproRoundPreview,
    CodeCompCoproSweepPoint,
    CodeCompCoproSweepRanges,
    attempt_ed1_copro_round,
)
from whetstone.optimization.proposal.proposer import (
    ProposerRouteConfig,
    ProposerTransport,
)

__all__ = [
    "CandidateProgress",
    "RoundFailure",
    "ScoredRound",
    "ScoringPoint",
    "ScoringTranscript",
    "run_copro_scoring_preview",
]


class ScoredRound(BaseModel):
    """One proposed and measured breadth-sized COPRO round."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    preview: CodeCompCoproRoundPreview
    evaluations: tuple[ScoredCandidate, ...]
    state_after: CoproState


class ScoringPoint(BaseModel):
    """All rounds and final ranking for one sweep point."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    settings: CodeCompCoproSweepPoint
    evaluation_binding: EvaluationBinding
    rounds: tuple[ScoredRound, ...]
    finalization: CoproFinalization


class ScoringTranscript(BaseModel):
    """Serializable proposal-generation and real-scoring COPRO transcript."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sweep: CodeCompCoproSweepRanges
    task_ids: tuple[StrictStr, ...]
    task_selection: TaskRoleSelection | None
    concurrency: int
    preflight: ScoringPreflight
    metadata: PreviewMetadata
    points: tuple[ScoringPoint, ...]


@dataclass(frozen=True, slots=True)
class CandidateProgress:
    """One candidate evaluation starting or completing within a round."""

    settings: CodeCompCoproSweepPoint
    round_index: int
    candidate_index: int
    candidate_count: int
    candidate: CandidateRef
    result: ScoredCandidate | None


class RoundFailure(RuntimeError):
    """A proposal round failed after preserving its complete attempt."""

    def __init__(self, attempt: CodeCompCoproRoundAttempt) -> None:
        self.attempt = attempt
        super().__init__(attempt.terminal_failure or "COPRO proposal failed")


def _candidate_sequence(
    preview: CodeCompCoproRoundPreview,
) -> tuple[Candidate, ...]:
    candidates = tuple(
        mutation.candidate.record for mutation in preview.candidate_mutations
    )
    if preview.round_plan.include_initial_candidate:
        return (*candidates, preview.starting_state.initial_candidate)
    return candidates


def run_copro_scoring_preview(
    *,
    store: ObjectStore,
    sweep: CodeCompCoproSweepRanges,
    preview_task: CodeCompCoproPreviewTask,
    proposer_kind: str,
    proposer_config: ProposerRouteConfig,
    proposer_transport: ProposerTransport,
    engine_factory: Callable[[CodeCompCoproSweepPoint], EvaluationEngine],
    binding_factory: Callable[
        [EvaluationEngine, CodeCompCoproSweepPoint], EvaluationBinding
    ],
    initial_candidate_factory: Callable[[CodeCompCoproSweepPoint], Candidate],
    aggregate_values_fn: Callable[
        [ObjectStore, RewardRef], tuple[float | None, ...]
    ],
    preflight: ScoringPreflight,
    metadata: PreviewMetadata,
    task_ids: tuple[str, ...],
    task_selection: TaskRoleSelection | None = None,
    concurrency: int = 1,
    resolution_message: str = "measured by the COPRO scoring preview",
    proposal_observer: Callable[[CodeCompCoproRoundAttempt], None]
    | None = None,
    candidate_observer: Callable[[CandidateProgress], None] | None = None,
) -> ScoringTranscript:
    """Run proposal drafts through real generation and scoring."""

    if not task_ids:
        raise ValueError("COPRO scoring preview requires at least one task")
    if concurrency < 1:
        raise ValueError("COPRO scoring preview concurrency must be positive")
    if task_selection is not None and task_selection.task_ids != task_ids:
        raise ValueError(
            "COPRO scoring task IDs do not match the selected manifest role"
        )
    points: list[ScoringPoint] = []
    for settings in sweep.expand():
        engine = engine_factory(settings)
        binding = binding_factory(engine, settings)
        lifecycle = CoproDriver(settings.copro)
        state = lifecycle.initial_state(initial_candidate_factory(settings))
        run_id = f"copro-scoring-preview:{settings.sweep_ordinal}"
        rounds: list[ScoredRound] = []
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
                raise RoundFailure(proposal_attempt)
            preview = proposal_attempt.require_preview()
            start = round_index * settings.copro.breadth
            candidates = _candidate_sequence(preview)
            evaluations_list: list[ScoredCandidate] = []
            for index, candidate in enumerate(candidates):
                if candidate_observer is not None:
                    candidate_observer(
                        CandidateProgress(
                            settings=settings,
                            round_index=round_index,
                            candidate_index=index,
                            candidate_count=len(candidates),
                            candidate=candidate_reference(candidate),
                            result=None,
                        )
                    )
                scored = build_scored_candidate(
                    store=store,
                    engine=engine,
                    binding=binding,
                    candidate=candidate,
                    purpose=preview.round_plan.proposal_mode,
                    run_id=run_id,
                    round_index=round_index,
                    occurrence_ordinal=start + index,
                    message=resolution_message,
                    aggregate_values_fn=aggregate_values_fn,
                )
                evaluations_list.append(scored)
                if candidate_observer is not None:
                    candidate_observer(
                        CandidateProgress(
                            settings=settings,
                            round_index=round_index,
                            candidate_index=index,
                            candidate_count=len(candidates),
                            candidate=scored.candidate,
                            result=scored,
                        )
                    )
            evaluations = tuple(evaluations_list)
            state = lifecycle.fold_round(
                state,
                tuple(item.attempt for item in evaluations),
            )
            rounds.append(
                ScoredRound(
                    preview=preview,
                    evaluations=evaluations,
                    state_after=state,
                )
            )
        points.append(
            ScoringPoint(
                settings=settings,
                evaluation_binding=binding,
                rounds=tuple(rounds),
                finalization=lifecycle.finalize(state),
            )
        )
    return ScoringTranscript(
        sweep=sweep,
        task_ids=task_ids,
        task_selection=task_selection,
        concurrency=concurrency,
        preflight=preflight,
        metadata=metadata,
        points=tuple(points),
    )
