from __future__ import annotations

from collections.abc import Callable

from dr_store import ObjectStore
from pydantic import BaseModel, ConfigDict

from whetstone.evaluation.protocol import EvaluationEngine, eval_is_success
from whetstone.evaluation.preview.persisted import (
    load_component_traces,
    load_evaluation_outputs,
)
from whetstone.evaluation.preview.resolution import evaluate_and_resolve
from whetstone.evaluation.schema import (
    EvaluationComponentTraces,
    EvaluationEvidence,
    EvaluationOutputsRecord,
)
from whetstone.experiment.candidate import Candidate, CandidateRef
from whetstone.experiment.reward import RewardRef
from whetstone.optimization.contracts import IntentOutcome, IntentResolution

__all__ = [
    "ScoredEvaluation",
    "build_scored_evaluation",
]


class ScoredEvaluation(BaseModel):
    """One exact candidate evaluation and its optimizer-facing projection."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    occurrence_ordinal: int
    candidate: CandidateRef
    evidence: EvaluationEvidence
    outputs: EvaluationOutputsRecord
    component_traces: EvaluationComponentTraces
    aggregate_values: tuple[float | None, ...]
    resolution: IntentResolution


def build_scored_evaluation(
    *,
    store: ObjectStore,
    engine: EvaluationEngine,
    candidate: Candidate,
    purpose: str,
    run_id: str,
    round_index: int,
    occurrence_ordinal: int,
    message: str,
    aggregate_values_fn: Callable[
        [ObjectStore, RewardRef], tuple[float | None, ...]
    ],
) -> ScoredEvaluation:
    evaluated, resolution = evaluate_and_resolve(
        engine,
        candidate,
        purpose=purpose,
        run_id=run_id,
        step_index=round_index,
        occurrence_ordinal=occurrence_ordinal,
        message=message,
    )
    if resolution.outcome is not IntentOutcome.COMPLETED:
        raise RuntimeError(
            f"scored evaluation requires COMPLETED resolution, got "
            f"{resolution.outcome.value}: {resolution.detail.message}"
        )
    if not eval_is_success(evaluated):
        raise RuntimeError("scored evaluation requires successful evidence")
    evidence = evaluated.evidence
    assert isinstance(evidence, EvaluationEvidence)
    reward_ref = evidence.reward_ref
    if reward_ref is None:
        raise RuntimeError("internal evaluation returned no Reward")
    return ScoredEvaluation(
        occurrence_ordinal=occurrence_ordinal,
        candidate=evidence.candidate,
        evidence=evidence,
        outputs=load_evaluation_outputs(store, evidence),
        component_traces=load_component_traces(store, evidence),
        aggregate_values=aggregate_values_fn(store, reward_ref),
        resolution=resolution,
    )
