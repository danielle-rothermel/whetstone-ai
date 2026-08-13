from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from typing import Any

from dr_store import ObjectStore

from whetstone.coordination.evaluation_claims import (
    EvaluationClaims,
    EvaluationIntentClaim,
    _ignore_renewal_publication,
    _OwnedClaim,
    _wait_for_renewal,
)
from whetstone.core.effects.authority import ReplayPolicy
from whetstone.core.identity import (
    TerminalFailure,
    TypedRef,
    typed_ref_for_record,
)
from whetstone.evaluation.protocol import EvalRequest, EvaluationEngine
from whetstone.evaluation.evidence_validation import (
    EvaluationEvidenceValidation,
)
from whetstone.evaluation.schema import (
    EvaluationFailureEvidence,
    EvaluationFailureEvidenceRef,
)
from whetstone.evaluation.schema_names import EVALUATION_FAILURE_SCHEMA
from whetstone.experiment.candidate import candidate_reference
from whetstone.experiment.sampling import evaluation_role_for_split
from whetstone.optimization.contracts import (
    INTENT_RESOLUTION_SCHEMA,
    INTENT_RESOLUTION_SCHEMA_VERSION,
    IntentOutcome,
    IntentResolution,
    OptimEvalRequest,
    ResolutionClass,
    ResolutionDetail,
)

_EVALUATION_SERVICE_NAMESPACE = "whetstone.evaluation_service.v3"


class EngineEvaluationService(EvaluationClaims, EvaluationEvidenceValidation):
    def __init__(
        self,
        *,
        store: ObjectStore,
        engine: EvaluationEngine,
        claim_lease_seconds: float = 300.0,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        _renewal_wait: Callable[[float, threading.Event], bool] = (
            _wait_for_renewal
        ),
        _renewal_published: Callable[[EvaluationIntentClaim], None] = (
            _ignore_renewal_publication
        ),
    ) -> None:
        if claim_lease_seconds <= 0:
            raise ValueError("claim_lease_seconds must be positive")
        self._store = store
        self._engine = engine
        self._claim_lease_seconds = claim_lease_seconds
        self._clock = clock
        self._sleep = sleep
        self._renewal_wait = _renewal_wait
        self._renewal_published = _renewal_published
        self._owner_id = uuid.uuid4().hex
        self._resolve_lock = threading.Lock()

    @property
    def replay_policy(self) -> ReplayPolicy:
        return ReplayPolicy.DURABLE_WORKFLOW

    def validate_resolution_graph(self, resolution: IntentResolution) -> None:
        self._validate_result_graph(
            resolution,
            expected_optim_eval_request=resolution.optim_eval_request,
        )

    def _load(
        self,
        reference: Any,
        *,
        expected_optim_eval_request: OptimEvalRequest,
    ) -> IntentResolution:
        record_ref = self._typed_ref(reference)
        if record_ref.schema_name != INTENT_RESOLUTION_SCHEMA:
            raise ValueError("Intent Resolution ref has the wrong schema")
        resolution = IntentResolution.model_validate(
            self._store.get(record_ref.reference)
        )
        if (
            typed_ref_for_record(
                INTENT_RESOLUTION_SCHEMA,
                resolution.model_dump(mode="json"),
            )
            != record_ref
        ):
            raise ValueError("persisted Intent Resolution ref is not exact")
        self._validate_result_graph(
            resolution,
            expected_optim_eval_request=expected_optim_eval_request,
        )
        return resolution

    @staticmethod
    def _intent_ref(optim_eval_request: OptimEvalRequest) -> TypedRef:
        return typed_ref_for_record(
            "whetstone.optim_eval_request",
            optim_eval_request.model_dump(mode="json"),
        )

    @classmethod
    def _key(cls, optim_eval_request: OptimEvalRequest) -> str:
        return (
            f"{_EVALUATION_SERVICE_NAMESPACE}.intent_resolution:"
            f"{cls._intent_ref(optim_eval_request).content_hash}"
        )

    @classmethod
    def _claim_key(
        cls,
        optim_eval_request: OptimEvalRequest,
        event_ordinal: int,
    ) -> str:
        return (
            f"{_EVALUATION_SERVICE_NAMESPACE}.intent_claim:"
            f"{cls._intent_ref(optim_eval_request).content_hash}"
            f"#{event_ordinal}"
        )

    @staticmethod
    def _typed_ref(reference: Any) -> TypedRef:
        if isinstance(reference, TypedRef):
            return reference
        return TypedRef(
            schema_name=reference.schema,
            content_hash=reference.content_hash,
        )

    def _evaluate_and_bind(
        self,
        optim_eval_request: OptimEvalRequest,
        owned: _OwnedClaim,
    ) -> IntentResolution:
        self._persist_intent_targets(optim_eval_request)
        resolved_eval_config = self._engine.eval_config_ref
        request = EvalRequest(
            request_id=optim_eval_request.eval_request.request_id,
            candidate=optim_eval_request.eval_request.candidate,
            metadata=optim_eval_request.eval_request.metadata,
        )
        try:
            self._engine.validate_request(request)
        except (KeyError, TypeError, ValueError) as exc:
            return self._bind_if_owned(
                optim_eval_request,
                IntentResolution(
                    schema_version=INTENT_RESOLUTION_SCHEMA_VERSION,
                    optim_eval_request=optim_eval_request,
                    outcome=IntentOutcome.REJECTED,
                    detail=ResolutionDetail(
                        classification=ResolutionClass.VALIDATION,
                        message=str(exc) or type(exc).__name__,
                    ),
                    resolved_eval_config=resolved_eval_config,
                ),
                owned,
            )
        try:
            self._assert_generation_current(optim_eval_request, owned)
            evaluated = self._engine.evaluate(request)
        except Exception as exc:
            candidate_ref = candidate_reference(
                optim_eval_request.eval_request.candidate
            )
            failure = EvaluationFailureEvidence(
                candidate=candidate_ref,
                eval_config_ref=self._engine.eval_config_ref,
                eval_role=evaluation_role_for_split(
                    self._engine.sampling.split_role
                ),
                provider_execution_policy_ref=(
                    self._engine.provider_execution_policy_ref
                ),
                metadata=optim_eval_request.eval_request.metadata,
                exception_type=type(exc).__name__,
                message=str(exc) or type(exc).__name__,
            )
            persisted_ref, _ = self._store.put(
                EVALUATION_FAILURE_SCHEMA, failure.record_content()
            )
            failure_ref = EvaluationFailureEvidenceRef(
                record=failure,
                record_ref=self._typed_ref(persisted_ref),
            )
            terminal_failure = TerminalFailure(
                code=f"evaluation_{failure.exception_type}",
                message=failure.message,
                details={
                    "evidence_schema": failure_ref.record_ref.schema_name,
                    "evidence_content_hash": (
                        failure_ref.record_ref.content_hash
                    ),
                },
            )
            return self._bind_if_owned(
                optim_eval_request,
                IntentResolution(
                    schema_version=INTENT_RESOLUTION_SCHEMA_VERSION,
                    optim_eval_request=optim_eval_request,
                    outcome=IntentOutcome.FAILED,
                    detail=ResolutionDetail(
                        classification=ResolutionClass.INFRASTRUCTURE,
                        message=failure.message,
                    ),
                    evaluation_result_ref=failure_ref.record_ref,
                    reward_evidence_refs=(),
                    resolved_eval_config=resolved_eval_config,
                    terminal_failure=terminal_failure,
                ),
                owned,
            )
        reward_ref = evaluated.evidence.reward_ref
        reward_evidence_refs = (
            () if reward_ref is None else reward_ref.record.evidence_refs
        )
        return self._bind_if_owned(
            optim_eval_request,
            IntentResolution(
                schema_version=INTENT_RESOLUTION_SCHEMA_VERSION,
                optim_eval_request=optim_eval_request,
                outcome=IntentOutcome.COMPLETED,
                detail=ResolutionDetail(
                    classification=ResolutionClass.MEASURED,
                    message="candidate evaluated under exact sampling binding",
                ),
                evaluation_result_ref=evaluated.evidence_ref,
                reward_evidence_refs=reward_evidence_refs,
                resolved_eval_config=resolved_eval_config,
                reward_ref=reward_ref,
            ),
            owned,
        )


__all__ = ["EngineEvaluationService"]