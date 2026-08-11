from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from dr_store import ObjectStore
from pydantic import BaseModel, ConfigDict, StrictStr

from whetstone.core.identity import TypedRef
from whetstone.core.roles import EvaluationRole
from whetstone.envs.ed1 import (
    ED1_DEFAULT_BLEND_CONFIG,
    Ed1Instance,
    build_ed1_experiment,
)
from whetstone.envs.ed1_blended import (
    BoundedCompressionMetricConfig,
)
from whetstone.envs.ed1_runtime import Ed1RuntimeProbe
from whetstone.envs.ed1_scoring import (
    CodeBatchScorer,
    CodeScore,
    CodeScoringInput,
)
from whetstone.envs.task_selection import TaskRoleSelection
from whetstone.evaluation import AggregationOutput
from whetstone.evaluation.engine import EvaluationEngine, EvaluationRequest
from whetstone.evaluation.schema import (
    EvaluationComponentTraces,
    EvaluationEvidence,
    EvaluationOutputsRecord,
)
from whetstone.experiment.binding import (
    EVALUATION_BINDING_SCHEMA_VERSION,
    EvaluationBinding,
    ExecutionEnvironmentFingerprint,
)
from whetstone.experiment.candidate import (
    Candidate,
    CandidateRef,
    candidate_reference,
)
from whetstone.optimization.contracts import (
    INTENT_RESOLUTION_SCHEMA_VERSION,
    EvaluationIntent,
    IntentOutcome,
    IntentResolution,
    ResolutionClass,
    ResolutionDetail,
)
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

ED1_SCORING_PREFLIGHT_TASK_ID = "HumanEval/0"


class Ed1ScoringRuntimeSummary(BaseModel):
    """Runtime identity displayed and persisted with a scoring preview."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evaluation_python: StrictStr
    dr_code_version: StrictStr
    runtime_identity_hash: StrictStr
    probe: Ed1RuntimeProbe


class Ed1ScoringPreflight(BaseModel):
    """Ground-truth check completed before candidate evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: StrictStr
    passed: bool
    infrastructure_unknown: bool
    outcome: StrictStr


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


def _one_score(scores: Sequence[CodeScore], *, context: str) -> CodeScore:
    if len(scores) != 1:
        raise ValueError(
            f"{context} returned {len(scores)} scores, expected 1"
        )
    return scores[0]


def run_ed1_scoring_preflight(
    tasks: tuple[Ed1Instance, ...],
    batch_scorer: CodeBatchScorer,
) -> Ed1ScoringPreflight:
    task = tasks[0].humaneval_task
    score = _one_score(
        batch_scorer(
            (
                CodeScoringInput(
                    raw_submission=task.ground_truth_code,
                    task=task,
                ),
            )
        ),
        context="runtime preflight",
    )
    result = Ed1ScoringPreflight(
        task_id=task.task_id,
        passed=score.passed,
        infrastructure_unknown=score.infrastructure_unknown,
        outcome=score.outcome,
    )
    if result.infrastructure_unknown or not result.passed:
        raise RuntimeError(
            "HumanEval ground-truth runtime preflight did not pass: "
            f"{result.outcome}"
        )
    return result


def _load_outputs(
    store: ObjectStore, evidence: EvaluationEvidence
) -> EvaluationOutputsRecord:
    raw = store.get(evidence.outputs_ref.reference)
    if raw is None:
        raise RuntimeError("persisted evaluation outputs are missing")
    return EvaluationOutputsRecord.model_validate(raw)


def _load_component_traces(
    store: ObjectStore, evidence: EvaluationEvidence
) -> EvaluationComponentTraces:
    raw = store.get(evidence.component_traces_ref.reference)
    if raw is None:
        raise RuntimeError("persisted component traces are missing")
    return EvaluationComponentTraces.model_validate_json(json.dumps(raw))


def _load_aggregate_value(
    store: ObjectStore, reference: TypedRef
) -> float | None:
    raw = store.get(reference.reference)
    if not isinstance(raw, dict):
        raise RuntimeError("persisted rollout aggregate is missing")
    output = AggregationOutput.model_validate(raw.get("aggregation_output"))
    return output.value


def ed1_preview_evaluation_binding(
    engine: EvaluationEngine,
    runtime: Ed1ScoringRuntimeSummary,
    *,
    campaign: str,
    task_model_kind: str,
) -> EvaluationBinding:
    return EvaluationBinding(
        schema_version=EVALUATION_BINDING_SCHEMA_VERSION,
        eval_config=engine.eval_config_ref,
        role=EvaluationRole.INTERNAL,
        campaign=campaign,
        provider_execution_policy_ref=(engine.provider_execution_policy_ref),
        environment_fingerprint=ExecutionEnvironmentFingerprint(
            dependency_versions=(
                ("dr-code", runtime.dr_code_version),
                ("numpy", runtime.probe.numpy_version),
            ),
            runtime_identity=runtime.runtime_identity_hash,
        ),
        provenance_note=(
            f"{task_model_kind}-generation-real-humaneval-scoring"
        ),
    )


def _candidate_sequence(
    preview: Ed1CoproRoundPreview,
) -> tuple[Candidate, ...]:
    candidates = tuple(
        mutation.candidate.record for mutation in preview.candidate_mutations
    )
    if preview.round_plan.include_initial_candidate:
        return (*candidates, preview.starting_state.initial_candidate)
    return candidates


def _score_candidate(
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
    request = EvaluationRequest(
        candidate=candidate,
        evaluation_binding=binding,
        purpose=purpose,
    )
    evaluated = engine.evaluate(request)
    reward_ref = evaluated.evidence.reward_ref
    if reward_ref is None:
        raise RuntimeError("internal ED1 evaluation returned no Reward")
    if len(reward_ref.record.evidence_refs) != 2:
        raise RuntimeError(
            "ED1 blended Reward must cite primary and compression aggregates"
        )
    intent = EvaluationIntent(
        intent_id=(
            f"{run_id}:{round_index}:{occurrence_ordinal}:"
            f"{evaluated.evidence.candidate.identity_hash}"
        ),
        candidate=evaluated.evidence.candidate,
        target_eval_config=binding.eval_config,
        evaluation_binding=binding,
        purpose=purpose,
        run_id=run_id,
        step_index=round_index,
        expected_reward_policy_hash=reward_ref.record.reward_policy_hash,
    )
    resolution = IntentResolution(
        schema_version=INTENT_RESOLUTION_SCHEMA_VERSION,
        intent=intent,
        outcome=IntentOutcome.COMPLETED,
        detail=ResolutionDetail(
            classification=ResolutionClass.MEASURED,
            message="measured by the ED1 scoring preview",
        ),
        evaluation_result_ref=evaluated.evidence_ref,
        reward_evidence_refs=reward_ref.record.evidence_refs,
        resolved_eval_config=binding.eval_config,
        reward_ref=reward_ref,
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
        outputs=_load_outputs(store, evaluated.evidence),
        component_traces=_load_component_traces(store, evaluated.evidence),
        primary_value=_load_aggregate_value(
            store, reward_ref.record.evidence_refs[0]
        ),
        compression_value=_load_aggregate_value(
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
        binding = ed1_preview_evaluation_binding(
            engine,
            runtime,
            campaign="copro-scoring-preview",
            task_model_kind=task_model.kind.value,
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
                evaluated = _score_candidate(
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
    "ED1_SCORING_PREFLIGHT_TASK_ID",
    "Ed1CoproCandidateProgress",
    "Ed1CoproRoundFailure",
    "Ed1CoproScoredRound",
    "Ed1CoproScoringPoint",
    "Ed1CoproScoringTranscript",
    "Ed1ScoredCandidate",
    "Ed1ScoringPreflight",
    "Ed1ScoringRuntimeSummary",
    "ed1_preview_evaluation_binding",
    "run_ed1_copro_scoring_preview",
    "run_ed1_scoring_preflight",
]
