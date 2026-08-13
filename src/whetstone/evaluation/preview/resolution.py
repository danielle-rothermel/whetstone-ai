from __future__ import annotations

from whetstone.core.identity import TerminalFailure
from whetstone.evaluation.metadata import metadata_with_purpose
from whetstone.evaluation.protocol import (
    EvalEvidenceWithRef,
    EvalRequest,
    EvalResult,
    EvaluationEngine,
    eval_is_rejected,
    eval_is_success,
)
from whetstone.evaluation.schema import EvaluationEvidence, EvaluationFailureEvidence
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
    "build_failed_resolution",
    "build_measured_resolution",
    "build_optim_eval_request",
    "build_rejected_resolution",
    "build_scored_resolution",
    "evaluate_and_resolve",
]


def build_optim_eval_request(
    evaluated: EvalEvidenceWithRef,
    engine: EvaluationEngine,
    *,
    purpose: str,
    run_id: str,
    step_index: int,
    occurrence_ordinal: int,
) -> OptimEvalRequest:
    if not isinstance(evaluated.evidence, EvaluationEvidence):
        raise RuntimeError("optim eval request requires successful evidence")
    evidence = evaluated.evidence
    reward_ref = evidence.reward_ref
    if reward_ref is None:
        raise RuntimeError("internal evaluation returned no Reward")
    return OptimEvalRequest(
        optim_run_id=run_id,
        optim_step_index=step_index,
        eval_request=EvalRequest(
            request_id=(
                f"{run_id}:{step_index}:{occurrence_ordinal}:"
                f"{evidence.candidate.identity_hash}"
            ),
            candidate=evidence.candidate.record,
            metadata=metadata_with_purpose(purpose),
        ),
        expected_reward_policy_hash=reward_ref.record.reward_policy_hash,
    )


def build_optim_eval_request_for_eval_request(
    request: EvalRequest,
    engine: EvaluationEngine,
    *,
    run_id: str,
    step_index: int,
) -> OptimEvalRequest:
    return OptimEvalRequest(
        optim_run_id=run_id,
        optim_step_index=step_index,
        eval_request=request,
        expected_reward_policy_hash=engine.reward_policy_identity_hash(),
    )


def build_measured_resolution(
    evaluated: EvalEvidenceWithRef,
    optim_eval_request: OptimEvalRequest,
    engine: EvaluationEngine,
    *,
    message: str,
) -> IntentResolution:
    if not isinstance(evaluated.evidence, EvaluationEvidence):
        raise RuntimeError("measured resolution requires successful evidence")
    evidence = evaluated.evidence
    reward_ref = evidence.reward_ref
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
    detail: ResolutionDetail,
) -> IntentResolution:
    return IntentResolution(
        schema_version=INTENT_RESOLUTION_SCHEMA_VERSION,
        optim_eval_request=optim_eval_request,
        outcome=IntentOutcome.REJECTED,
        detail=detail,
        resolved_eval_config=engine.eval_config_ref,
    )


def build_failed_resolution(
    evaluated: EvalEvidenceWithRef,
    optim_eval_request: OptimEvalRequest,
    engine: EvaluationEngine,
) -> IntentResolution:
    if not isinstance(evaluated.evidence, EvaluationFailureEvidence):
        raise RuntimeError("failed resolution requires failure evidence")
    failure = evaluated.evidence
    terminal_failure = TerminalFailure(
        code=f"evaluation_{failure.exception_type}",
        message=failure.message,
        details={
            "evidence_schema": evaluated.evidence_ref.schema_name,
            "evidence_content_hash": evaluated.evidence_ref.content_hash,
        },
    )
    return IntentResolution(
        schema_version=INTENT_RESOLUTION_SCHEMA_VERSION,
        optim_eval_request=optim_eval_request,
        outcome=IntentOutcome.FAILED,
        detail=ResolutionDetail(
            classification=ResolutionClass.INFRASTRUCTURE,
            message=failure.message,
        ),
        evaluation_result_ref=evaluated.evidence_ref,
        reward_evidence_refs=(),
        resolved_eval_config=engine.eval_config_ref,
        terminal_failure=terminal_failure,
    )


def build_scored_resolution(
    evaluated: EvalEvidenceWithRef,
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
) -> tuple[EvalResult, IntentResolution]:
    request = EvalRequest(
        request_id=(
            f"{run_id}:{step_index}:{occurrence_ordinal}:{purpose}"
        ),
        candidate=candidate,
        metadata=metadata_with_purpose(purpose),
    )
    result = engine.evaluate(request)
    if eval_is_rejected(result):
        optim_eval_request = build_optim_eval_request_for_eval_request(
            request,
            engine,
            run_id=run_id,
            step_index=step_index,
        )
        return result, build_rejected_resolution(
            optim_eval_request,
            engine,
            detail=result.detail,
        )
    if not isinstance(result, EvalEvidenceWithRef):
        raise TypeError(f"unexpected evaluation result: {result!r}")
    if isinstance(result.evidence, EvaluationFailureEvidence):
        optim_eval_request = build_optim_eval_request_for_eval_request(
            request,
            engine,
            run_id=run_id,
            step_index=step_index,
        )
        return result, build_failed_resolution(
            result, optim_eval_request, engine
        )
    if not eval_is_success(result):
        raise TypeError(f"unexpected evaluation result: {result!r}")
    optim_eval_request = build_optim_eval_request(
        result,
        engine,
        purpose=purpose,
        run_id=run_id,
        step_index=step_index,
        occurrence_ordinal=occurrence_ordinal,
    )
    resolution = build_measured_resolution(
        result, optim_eval_request, engine, message=message
    )
    return result, resolution
