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
from whetstone.evaluation.engine import EvaluationEngine, EvaluationRequest
from whetstone.evaluation.evidence_validation import (
    EvaluationEvidenceValidation,
)
from whetstone.evaluation.schema import (
    EvaluationFailureEvidence,
    EvaluationFailureEvidenceRef,
)
from whetstone.evaluation.schema_names import EVALUATION_FAILURE_SCHEMA
from whetstone.optimization.contracts import (
    INTENT_RESOLUTION_SCHEMA_VERSION,
    EvaluationIntent,
    IntentOutcome,
    IntentResolution,
    ResolutionClass,
    ResolutionDetail,
)

_EVALUATION_SERVICE_NAMESPACE = "whetstone.evaluation_service.v3"


class EngineEvaluationService(EvaluationClaims, EvaluationEvidenceValidation):
    """Persist one authoritative resolution per immutable intent across
    processes and restarts; an expired uncommitted attempt may be retried.
    """

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
        """Return the recovery policy of the durable evaluator workflow."""
        return ReplayPolicy.DURABLE_WORKFLOW

    def validate_resolution_graph(self, resolution: IntentResolution) -> None:
        """Validate one exact result graph without mutating durable state."""
        self._validate_result_graph(
            resolution,
            expected_intent=resolution.intent,
        )

    @staticmethod
    def _intent_ref(intent: EvaluationIntent) -> TypedRef:
        return typed_ref_for_record(
            "whetstone.evaluation_intent", intent.model_dump(mode="json")
        )

    @classmethod
    def _key(cls, intent: EvaluationIntent) -> str:
        return (
            f"{_EVALUATION_SERVICE_NAMESPACE}.intent_resolution:"
            f"{cls._intent_ref(intent).content_hash}"
        )

    @classmethod
    def _claim_key(
        cls,
        intent: EvaluationIntent,
        event_ordinal: int,
    ) -> str:
        return (
            f"{_EVALUATION_SERVICE_NAMESPACE}.intent_claim:"
            f"{cls._intent_ref(intent).content_hash}"
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
        intent: EvaluationIntent,
        owned: _OwnedClaim,
    ) -> IntentResolution:
        self._persist_intent_targets(intent)
        if intent.target_eval_config != self._engine.eval_config_ref:
            return self._bind_if_owned(
                intent,
                IntentResolution(
                    schema_version=INTENT_RESOLUTION_SCHEMA_VERSION,
                    intent=intent,
                    outcome=IntentOutcome.REJECTED,
                    detail=ResolutionDetail(
                        classification=ResolutionClass.VALIDATION,
                        message=(
                            "intent target Eval Config is not the engine's "
                            "exact sampling binding"
                        ),
                    ),
                    resolved_eval_config=intent.target_eval_config,
                ),
                owned,
            )
        request = EvaluationRequest(
            candidate=intent.candidate.record,
            evaluation_binding=intent.evaluation_binding,
            purpose=intent.purpose,
        )
        try:
            self._engine.validate_request(request)
        except (KeyError, TypeError, ValueError) as exc:
            return self._bind_if_owned(
                intent,
                IntentResolution(
                    schema_version=INTENT_RESOLUTION_SCHEMA_VERSION,
                    intent=intent,
                    outcome=IntentOutcome.REJECTED,
                    detail=ResolutionDetail(
                        classification=ResolutionClass.VALIDATION,
                        message=str(exc) or type(exc).__name__,
                    ),
                    resolved_eval_config=intent.target_eval_config,
                ),
                owned,
            )
        try:
            self._assert_generation_current(intent, owned)
            evaluated = self._engine.evaluate(request)
        except Exception as exc:
            failure = EvaluationFailureEvidence(
                candidate=intent.candidate,
                evaluation_binding=intent.evaluation_binding,
                purpose=intent.purpose,
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
                intent,
                IntentResolution(
                    schema_version=INTENT_RESOLUTION_SCHEMA_VERSION,
                    intent=intent,
                    outcome=IntentOutcome.FAILED,
                    detail=ResolutionDetail(
                        classification=ResolutionClass.INFRASTRUCTURE,
                        message=failure.message,
                    ),
                    evaluation_result_ref=failure_ref.record_ref,
                    reward_evidence_refs=(),
                    resolved_eval_config=intent.target_eval_config,
                    terminal_failure=terminal_failure,
                ),
                owned,
            )
        reward_ref = evaluated.evidence.reward_ref
        reward_evidence_refs = (
            () if reward_ref is None else reward_ref.record.evidence_refs
        )
        return self._bind_if_owned(
            intent,
            IntentResolution(
                schema_version=INTENT_RESOLUTION_SCHEMA_VERSION,
                intent=intent,
                outcome=IntentOutcome.COMPLETED,
                detail=ResolutionDetail(
                    classification=ResolutionClass.MEASURED,
                    message="candidate evaluated under exact sampling binding",
                ),
                evaluation_result_ref=evaluated.evidence_ref,
                reward_evidence_refs=reward_evidence_refs,
                resolved_eval_config=intent.target_eval_config,
                reward_ref=reward_ref,
            ),
            owned,
        )


__all__ = ["EngineEvaluationService"]
