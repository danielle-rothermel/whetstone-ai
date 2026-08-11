"""Shared helpers for evaluation restart and forgery pathway tests."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from dr_store import ObjectNotFoundError, ObjectStore

from tests.evaluation.support import (
    _bind_without_validation,
    _binding,
    _completed_resolution,
    _publish_attestation,
    _put_typed,
)
from whetstone.coordination.evaluation_service import EngineEvaluationService
from whetstone.core.identity import TypedRef
from whetstone.core.roles import EvaluationRole
from whetstone.evaluation.engine import (
    EngineEvaluation,
    EvaluationEngine,
    EvaluationRequest,
)
from whetstone.evaluation.schema import (
    EVALUATION_OUTPUTS_SCHEMA,
    EvaluationOutputsRecord,
)
from whetstone.evaluation.schema_names import EVALUATION_EVIDENCE_SCHEMA
from whetstone.optimization.contracts import EvaluationIntent


@dataclass(frozen=True, slots=True)
class EvaluatedIntentBundle:
    evaluated: EngineEvaluation
    intent: EvaluationIntent
    alternate_outputs_ref: TypedRef


def evaluate_intent(
    engine: EvaluationEngine,
    intent: EvaluationIntent,
) -> EngineEvaluation:
    return engine.evaluate(
        EvaluationRequest(
            candidate=intent.candidate.record,
            evaluation_binding=intent.evaluation_binding,
            purpose=intent.purpose,
        )
    )


def evaluate_intent_bundle(
    engine: EvaluationEngine,
    intent: EvaluationIntent,
) -> EvaluatedIntentBundle:
    evaluated = evaluate_intent(engine, intent)
    other_candidate = intent.candidate.record.model_copy(
        update={"candidate_id": "candidate-b"}
    )
    alternate = engine.evaluate(
        EvaluationRequest(
            candidate=other_candidate,
            evaluation_binding=intent.evaluation_binding,
            purpose=intent.purpose,
        )
    )
    return EvaluatedIntentBundle(
        evaluated=evaluated,
        intent=intent,
        alternate_outputs_ref=alternate.evidence.outputs_ref,
    )


def build_forged_evidence(
    forgery: str,
    *,
    evaluated: EngineEvaluation,
    intent: EvaluationIntent,
    engine: EvaluationEngine,
    store: ObjectStore,
    alternate_outputs_ref: TypedRef | None = None,
) -> TypedRef:
    evidence = evaluated.evidence
    evidence_update: dict[str, object] = {}

    if forgery == "candidate":
        if alternate_outputs_ref is None:
            raise ValueError(
                "alternate_outputs_ref is required for candidate forgery"
            )
        evidence_update["outputs_ref"] = alternate_outputs_ref
    elif forgery == "evidence_binding":
        evidence_update["evaluation_binding"] = _binding(
            engine,
            campaign="forged-binding",
        )
    elif forgery == "evidence_purpose":
        evidence_update["purpose"] = "forged-purpose"
    elif forgery == "evidence_dataset":
        evidence_update["dataset_hash"] = "forged-dataset"
    elif forgery == "aggregate_value":
        assert evidence.aggregate_value is not None
        evidence_update["aggregate_value"] = evidence.aggregate_value + 1.0
    elif forgery == "missing_output":
        evidence_update["outputs_ref"] = TypedRef(
            schema_name=EVALUATION_OUTPUTS_SCHEMA,
            content_hash="f" * 64,
        )
    else:
        outputs_content = EvaluationOutputsRecord.model_validate(
            store.get(evidence.outputs_ref.reference)
        ).record_content()
        if forgery == "output_binding":
            outputs_content["evaluation_binding"] = _binding(
                engine,
                campaign="forged-output-binding",
            ).model_dump(mode="json")
        elif forgery == "output_purpose":
            outputs_content["purpose"] = "forged-purpose"
        elif forgery == "output_role":
            official_binding = _binding(
                engine,
                role=EvaluationRole.OFFICIAL,
                campaign="forged-output-role",
            )
            outputs_content["evaluation_binding"] = (
                official_binding.model_dump(mode="json")
            )
            outputs_content["evaluation_role"] = "official"
        elif forgery == "output_split":
            outputs_content["split_role"] = "official"
        elif forgery == "output_task":
            outputs_content["task_hashes"] = ["forged-task"]
            outputs_content["outputs"][0]["task_hash"] = "forged-task"
        elif forgery == "output_repeat":
            outputs_content["num_samples"] = 2
        elif forgery == "output_trace":
            outputs_content["outputs"][0]["rendered_prompt"] = "forged prompt"
        elif forgery == "output_metadata":
            outputs_content["outputs"][0].update(
                {
                    "output_text": "forged output",
                    "finish_reason": "length",
                    "provider_error": {"type": "forged"},
                    "failure_code": "forged_failure",
                }
            )
        elif forgery == "output_score":
            outputs_content["outputs"][0]["score"] = 0.0
        elif forgery == "output_empty":
            outputs_content["outputs"] = []
        else:
            raise AssertionError(f"unhandled forgery {forgery}")
        evidence_update["outputs_ref"] = _put_typed(
            store,
            EVALUATION_OUTPUTS_SCHEMA,
            outputs_content,
        )

    forged_evidence = evidence.model_copy(update=evidence_update)
    return _put_typed(
        store,
        EVALUATION_EVIDENCE_SCHEMA,
        forged_evidence.record_content(),
    )


def assert_restart_rejects_forged_resolution(
    *,
    store: ObjectStore,
    engine: EvaluationEngine,
    intent: EvaluationIntent,
    evaluated: EngineEvaluation,
    forged_evidence_ref: TypedRef,
) -> None:
    forged_resolution = _completed_resolution(intent, evaluated).model_copy(
        update={"evaluation_result_ref": forged_evidence_ref}
    )
    service = EngineEvaluationService(store=store, engine=engine)
    _publish_attestation(
        service=service,
        intent=intent,
        resolution=_completed_resolution(intent, evaluated),
    )
    with pytest.raises((ObjectNotFoundError, ValueError)):
        service._bind(intent, forged_resolution)
    _bind_without_validation(
        store=store,
        service=service,
        intent=intent,
        resolution=forged_resolution,
    )

    with pytest.raises((ObjectNotFoundError, ValueError)):
        service.resolve_evaluation_intent(intent)


def assert_restart_rejects_forgery(
    *,
    store: ObjectStore,
    engine: EvaluationEngine,
    bundle: EvaluatedIntentBundle,
    forgery: str,
) -> None:
    forged_evidence_ref = build_forged_evidence(
        forgery,
        evaluated=bundle.evaluated,
        intent=bundle.intent,
        engine=engine,
        store=store,
        alternate_outputs_ref=bundle.alternate_outputs_ref,
    )
    assert_restart_rejects_forged_resolution(
        store=store,
        engine=engine,
        intent=bundle.intent,
        evaluated=bundle.evaluated,
        forged_evidence_ref=forged_evidence_ref,
    )
