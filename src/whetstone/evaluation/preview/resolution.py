from __future__ import annotations

from whetstone.evaluation.engine import (
    EngineEvaluation,
    EvaluationEngine,
    EvaluationRequest,
)
from whetstone.experiment.binding import EvaluationBinding
from whetstone.experiment.candidate import Candidate
from whetstone.optimization.contracts import (
    INTENT_RESOLUTION_SCHEMA_VERSION,
    EvaluationIntent,
    IntentOutcome,
    IntentResolution,
    ResolutionClass,
    ResolutionDetail,
)

__all__ = [
    "build_evaluation_intent",
    "build_measured_resolution",
    "evaluate_and_resolve",
]


def build_evaluation_intent(
    evaluated: EngineEvaluation,
    binding: EvaluationBinding,
    *,
    purpose: str,
    run_id: str,
    step_index: int,
    occurrence_ordinal: int,
) -> EvaluationIntent:
    reward_ref = evaluated.evidence.reward_ref
    if reward_ref is None:
        raise RuntimeError("internal evaluation returned no Reward")
    return EvaluationIntent(
        intent_id=(
            f"{run_id}:{step_index}:{occurrence_ordinal}:"
            f"{evaluated.evidence.candidate.identity_hash}"
        ),
        candidate=evaluated.evidence.candidate,
        target_eval_config=binding.eval_config,
        evaluation_binding=binding,
        purpose=purpose,
        run_id=run_id,
        step_index=step_index,
        expected_reward_policy_hash=reward_ref.record.reward_policy_hash,
    )


def build_measured_resolution(
    evaluated: EngineEvaluation,
    intent: EvaluationIntent,
    *,
    message: str,
) -> IntentResolution:
    reward_ref = evaluated.evidence.reward_ref
    if reward_ref is None:
        raise RuntimeError("internal evaluation returned no Reward")
    return IntentResolution(
        schema_version=INTENT_RESOLUTION_SCHEMA_VERSION,
        intent=intent,
        outcome=IntentOutcome.COMPLETED,
        detail=ResolutionDetail(
            classification=ResolutionClass.MEASURED,
            message=message,
        ),
        evaluation_result_ref=evaluated.evidence_ref,
        reward_evidence_refs=reward_ref.record.evidence_refs,
        resolved_eval_config=intent.evaluation_binding.eval_config,
        reward_ref=reward_ref,
    )


def evaluate_and_resolve(
    engine: EvaluationEngine,
    binding: EvaluationBinding,
    candidate: Candidate,
    *,
    purpose: str,
    run_id: str,
    step_index: int,
    occurrence_ordinal: int,
    message: str,
) -> tuple[EngineEvaluation, IntentResolution]:
    evaluated = engine.evaluate(
        EvaluationRequest(
            candidate=candidate,
            evaluation_binding=binding,
            purpose=purpose,
        )
    )
    intent = build_evaluation_intent(
        evaluated,
        binding,
        purpose=purpose,
        run_id=run_id,
        step_index=step_index,
        occurrence_ordinal=occurrence_ordinal,
    )
    resolution = build_measured_resolution(
        evaluated,
        intent,
        message=message,
    )
    return evaluated, resolution
