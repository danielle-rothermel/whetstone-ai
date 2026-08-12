from __future__ import annotations

from collections.abc import Callable

from dr_store import ObjectStore
from pydantic import BaseModel, ConfigDict

from whetstone.evaluation.engine import EvaluationEngine
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
from whetstone.experiment.binding import EvaluationBinding
from whetstone.experiment.candidate import Candidate, CandidateRef
from whetstone.experiment.reward import RewardRef
from whetstone.optimization.contracts import IntentResolution
from whetstone.optimization.copro.adapter import CoproAttempt

__all__ = [
    "ScoredCandidate",
    "build_scored_candidate",
]


class ScoredCandidate(BaseModel):
    """One exact candidate evaluation and its optimizer projection."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    occurrence_ordinal: int
    candidate: CandidateRef
    evidence: EvaluationEvidence
    outputs: EvaluationOutputsRecord
    component_traces: EvaluationComponentTraces
    aggregate_values: tuple[float | None, ...]
    resolution: IntentResolution
    attempt: CoproAttempt


def build_scored_candidate(
    *,
    store: ObjectStore,
    engine: EvaluationEngine,
    binding: EvaluationBinding,
    candidate: Candidate,
    purpose: str,
    run_id: str,
    round_index: int,
    occurrence_ordinal: int,
    message: str,
    aggregate_values_fn: Callable[
        [ObjectStore, RewardRef], tuple[float | None, ...]
    ],
) -> ScoredCandidate:
    evaluated, resolution = evaluate_and_resolve(
        engine,
        binding,
        candidate,
        purpose=purpose,
        run_id=run_id,
        step_index=round_index,
        occurrence_ordinal=occurrence_ordinal,
        message=message,
    )
    reward_ref = evaluated.evidence.reward_ref
    if reward_ref is None:
        raise RuntimeError("internal evaluation returned no Reward")
    attempt = CoproAttempt.from_resolution(
        occurrence_ordinal=occurrence_ordinal,
        round_index=round_index,
        resolution=resolution,
        expected_run_id=run_id,
        expected_evaluation_binding=binding,
        expected_reward_policy_hash=reward_ref.record.reward_policy_hash,
    )
    return ScoredCandidate(
        occurrence_ordinal=occurrence_ordinal,
        candidate=evaluated.evidence.candidate,
        evidence=evaluated.evidence,
        outputs=load_evaluation_outputs(store, evaluated.evidence),
        component_traces=load_component_traces(store, evaluated.evidence),
        aggregate_values=aggregate_values_fn(store, reward_ref),
        resolution=resolution,
        attempt=attempt,
    )
