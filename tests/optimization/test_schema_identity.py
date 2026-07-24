"""Typed optimization serialization-boundary contracts."""

import pytest
from pydantic import ValidationError

from whetstone.optimization import (
    CandidateRef,
    EvalConfigRef,
    EvaluationIntent,
    IntentOutcome,
    IntentResolution,
    ResolutionClass,
    ResolutionDetail,
    TypedRef,
    candidate_reference,
    eval_config_reference,
)

from .support import (
    FULL_A,
    candidate,
    eval_config,
    make_intent,
    proposal_request,
)


def test_candidate_ref_binds_exact_record_content_and_identity() -> None:
    record = candidate()
    ref = candidate_reference(record)
    assert ref.record == record
    assert ref.identity_hash == record.identity_hash()
    with pytest.raises(ValidationError, match="exact candidate"):
        CandidateRef(
            record=record,
            record_ref=TypedRef(schema_name="wrong", content_hash=FULL_A),
            identity_hash=record.identity_hash(),
        )


def test_eval_config_ref_binds_exact_typed_record_and_identity() -> None:
    record = eval_config()
    ref = eval_config_reference(record)
    assert ref.record == record
    assert ref.identity_hash == record.config_identity_hash
    with pytest.raises(ValidationError, match="identity_hash"):
        EvalConfigRef(
            record=record,
            record_ref=ref.record_ref,
            identity_hash=FULL_A,
        )


def test_intent_has_exact_refs_and_no_loose_identity_fields() -> None:
    proposed = candidate("P1")
    intent = make_intent(proposed)
    dumped = intent.model_dump()
    assert dumped["candidate"]["record"]["candidate_id"] == "P1"
    assert dumped["target_eval_config"]["record"]["config_identity_hash"]
    assert "candidate_id" not in dumped
    assert "target_eval_config_ref" not in dumped
    assert "target_eval_config_hash" not in dumped
    with pytest.raises(ValidationError):
        EvaluationIntent.model_validate({**dumped, "candidate_id": "P1"})


def test_only_pre_execution_rejection_may_have_empty_evidence() -> None:
    intent = make_intent(candidate("P1"))
    rejected = IntentResolution(
        intent=intent,
        outcome=IntentOutcome.REJECTED,
        detail=ResolutionDetail(
            classification=ResolutionClass.VALIDATION,
            message="bad candidate",
        ),
        resolved_eval_config=intent.target_eval_config,
    )
    assert rejected.evaluation_evidence_refs == ()
    with pytest.raises(ValidationError, match="requires execution evidence"):
        IntentResolution(
            intent=intent,
            outcome=IntentOutcome.FAILED,
            detail=ResolutionDetail(
                classification=ResolutionClass.UNSCORABLE,
                message="could not score",
            ),
            resolved_eval_config=intent.target_eval_config,
        )


def test_resolution_rejects_a_different_eval_config() -> None:
    intent = make_intent(candidate("P1"))
    with pytest.raises(ValidationError, match="exact target"):
        IntentResolution(
            intent=intent,
            outcome=IntentOutcome.REJECTED,
            detail=ResolutionDetail(
                classification=ResolutionClass.VALIDATION,
                message="rejected",
            ),
            resolved_eval_config=eval_config_reference(eval_config("e" * 64)),
        )


def test_noninitial_request_requires_prior_ref() -> None:
    with pytest.raises(ValidationError, match="prior result"):
        proposal_request(step_index=1)
