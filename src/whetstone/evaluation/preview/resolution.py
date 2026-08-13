from __future__ import annotations

from whetstone.evaluation.metadata import metadata_with_purpose
from whetstone.evaluation.protocol import (
    EngineEvaluation,
    EvalRequest,
    EvaluationEngine,
)
from whetstone.experiment.candidate import Candidate
from whetstone.optimization.contracts import (
    INTENT_RESOLUTION_SCHEMA_VERSION,
    IntentOutcome,
    IntentResolution,
    OptimEvalRequest,
    ResolutionClass,
    ResolutionDetail,
)

__all__ = [
    "build_measured_resolution",
    "build_optim_eval_request",
    "build_rejected_resolution",
    "build_scored_resolution",
    "evaluate_and_resolve",
]


def build_optim_eval_request(
    evaluated: EngineEvaluation,
    engine: EvaluationEngine,
    *,
    purpose: str,
    run_id: str,
    step_index: int,
    occurrence_ordinal: int,
) -> OptimEvalRequest:
    reward_ref = evaluated.evidence.reward_ref
    if reward_ref is None:
        raise RuntimeError("internal evaluation returned no Reward")
    return OptimEvalRequest(
        optim_run_id=run_id,
        optim_step_index=step_index,
        eval_request=EvalRequest(
            request_id=(
                f"{run_id}:{step_index}:{occurrence_ordinal}:"
                f"{evaluated.evidence.candidate.identity_hash}"
            ),
            candidate=evaluated.evidence.candidate.record,
            metadata=metadata_with_purpose(purpose),
        ),
        expected_reward_policy_hash=reward_ref.record.reward_policy_hash,
    )


def build_measured_resolution(
    evaluated: EngineEvaluation,
    optim_eval_request: OptimEvalRequest,
    engine: EvaluationEngine,
    *,
    message: str,
) -> IntentResolution:
    reward_ref = evaluated.evidence.reward_ref
    if reward_ref is None:
        raise RuntimeError("internal evaluation returned no Reward")
    return IntentResolution(
        schema_version=INTENT_RESOLUTION_SCHEMA_VERSION,
        optim_eval_request=optim_eval_request,
        outcome=IntentOutcome.COMPLETED,
        detail=ResolutionDetail(
            classification=ResolutionClass.MEASURED,
            message=message,
        ),
        evaluation_result_ref=evaluated.evidence_ref,
        reward_evidence_refs=reward_ref.record.evidence_refs,
        resolved_eval_config=engine.eval_config_ref,
        reward_ref=reward_ref,
    )


def build_rejected_resolution(
    optim_eval_request: OptimEvalRequest,
    engine: EvaluationEngine,
    *,
    message: str,
) -> IntentResolution:
    return IntentResolution(
        schema_version=INTENT_RESOLUTION_SCHEMA_VERSION,
        optim_eval_request=optim_eval_request,
        outcome=IntentOutcome.REJECTED,
        detail=ResolutionDetail(
            classification=ResolutionClass.VALIDATION,
            message=message,
        ),
        resolved_eval_config=engine.eval_config_ref,
    )


def build_scored_resolution(
    evaluated: EngineEvaluation,
    optim_eval_request: OptimEvalRequest,
    engine: EvaluationEngine,
    *,
    message: str,
) -> IntentResolution:
    return build_measured_resolution(
        evaluated,
        optim_eval_request,
        engine,
        message=message,
    )


def evaluate_and_resolve(
    engine: EvaluationEngine,
    candidate: Candidate,
    *,
    purpose: str,
    run_id: str,
    step_index: int,
    occurrence_ordinal: int,
    message: str,
) -> tuple[EngineEvaluation, IntentResolution]:
    evaluated = engine.evaluate(
        EvalRequest(
            request_id=(
                f"{run_id}:{step_index}:{occurrence_ordinal}:{purpose}"
            ),
            candidate=candidate,
            metadata=metadata_with_purpose(purpose),
        )
    )
    optim_eval_request = build_optim_eval_request(
        evaluated,
        engine,
        purpose=purpose,
        run_id=run_id,
        step_index=step_index,
        occurrence_ordinal=occurrence_ordinal,
    )
    resolution = build_measured_resolution(
        evaluated, optim_eval_request, engine, message=message
    )
    return evaluated, resolution
