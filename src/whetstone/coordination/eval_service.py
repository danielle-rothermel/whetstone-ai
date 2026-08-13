from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum, verify, UNIQUE
from typing import Any

from dr_store import ObjectStore

from whetstone.coordination.eval_claims import (
    EvalClaims,
    EvalIntentClaim,
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
from whetstone.eval.protocol import (
    EvalEvidenceWithRef,
    EvalRejected,
    EvalRequest,
    EvalEngine,
)
from whetstone.eval.evidence_validation import (
    EvalEvidenceValidation,
)
from whetstone.eval.schema import (
    EvalEvidence,
    EvalFailureEvidence,
)
from whetstone.optim.contracts import (
    INTENT_RESOLUTION_SCHEMA,
    INTENT_RESOLUTION_SCHEMA_VERSION,
    IntentOutcome,
    IntentResolution,
    OptimEvalRequest,
    ResolutionClass,
    ResolutionDetail,
)

_EVAL_SERVICE_NAMESPACE = "whetstone.eval_service.v3"
_PLATFORM_INTENT_NAMESPACE = "whetstone.platform_eval_intent.v1"


@verify(UNIQUE)
class EvalDispatchMode(StrEnum):
    INLINE = "inline"
    PLATFORM = "platform"


class EvalPlatformDeferred(RuntimeError):
    """Evaluation intent persisted for platform row fan-out."""


@dataclass(frozen=True, slots=True)
class EvalExecutionContext:
    dispatch_mode: EvalDispatchMode = EvalDispatchMode.INLINE


class EvalEngineService(EvalClaims, EvalEvidenceValidation):
    def __init__(
        self,
        *,
        store: ObjectStore,
        engine: EvalEngine,
        dispatch_mode: EvalDispatchMode = EvalDispatchMode.INLINE,
        claim_lease_seconds: float = 300.0,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        _renewal_wait: Callable[[float, threading.Event], bool] = (
            _wait_for_renewal
        ),
        _renewal_published: Callable[[EvalIntentClaim], None] = (
            _ignore_renewal_publication
        ),
    ) -> None:
        if claim_lease_seconds <= 0:
            raise ValueError("claim_lease_seconds must be positive")
        self._store = store
        self._engine = engine
        self._dispatch_mode = dispatch_mode
        self._claim_lease_seconds = claim_lease_seconds
        self._clock = clock
        self._sleep = sleep
        self._renewal_wait = _renewal_wait
        self._renewal_published = _renewal_published
        self._owner_id = uuid.uuid4().hex
        self._resolve_lock = threading.Lock()
        self._active_context: EvalExecutionContext | None = None

    @property
    def replay_policy(self) -> ReplayPolicy:
        return ReplayPolicy.DURABLE_WORKFLOW

    @property
    def dispatch_mode(self) -> EvalDispatchMode:
        return self._dispatch_mode

    def set_dispatch_mode(self, mode: EvalDispatchMode) -> EvalDispatchMode:
        previous = self._dispatch_mode
        self._dispatch_mode = mode
        return previous

    def _effective_context(
        self,
        context: EvalExecutionContext | None,
    ) -> EvalExecutionContext:
        if context is not None:
            return context
        return EvalExecutionContext(dispatch_mode=self._dispatch_mode)

    @classmethod
    def _platform_intent_key(cls, optim_eval_request: OptimEvalRequest) -> str:
        return (
            f"{_PLATFORM_INTENT_NAMESPACE}.pending:"
            f"{cls._intent_ref(optim_eval_request).content_hash}"
        )

    def persist_platform_intent(
        self,
        optim_eval_request: OptimEvalRequest,
        *,
        context: EvalExecutionContext | None = None,
    ) -> TypedRef:
        effective = self._effective_context(context)
        if effective.dispatch_mode is not EvalDispatchMode.PLATFORM:
            raise ValueError("platform intent persistence requires PLATFORM mode")
        self._persist_intent_targets(optim_eval_request)
        reference, _ = self._store.put(
            "whetstone.optim_eval_request",
            optim_eval_request.model_dump(mode="json"),
        )
        key = self._platform_intent_key(optim_eval_request)
        self._store.bind(key, reference)
        return TypedRef(
            schema_name=reference.schema,
            content_hash=reference.content_hash,
        )

    def resolve_optim_eval_request(
        self,
        optim_eval_request: OptimEvalRequest,
        *,
        context: EvalExecutionContext | None = None,
    ) -> IntentResolution:
        with self._resolve_lock:
            previous = self._active_context
            self._active_context = self._effective_context(context)
            try:
                return self._resolve_claimed(optim_eval_request)
            finally:
                self._active_context = previous

    def load_platform_intent(
        self,
        optim_eval_request: OptimEvalRequest,
    ) -> OptimEvalRequest | None:
        bound = self._store.resolve(self._platform_intent_key(optim_eval_request))
        if bound is None:
            return None
        return OptimEvalRequest.model_validate(self._store.get(bound))

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
            f"{_EVAL_SERVICE_NAMESPACE}.intent_resolution:"
            f"{cls._intent_ref(optim_eval_request).content_hash}"
        )

    @classmethod
    def _claim_key(
        cls,
        optim_eval_request: OptimEvalRequest,
        event_ordinal: int,
    ) -> str:
        return (
            f"{_EVAL_SERVICE_NAMESPACE}.intent_claim:"
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
        effective = self._effective_context(self._active_context)
        if effective.dispatch_mode is EvalDispatchMode.PLATFORM:
            self.persist_platform_intent(
                optim_eval_request,
                context=effective,
            )
            raise EvalPlatformDeferred(
                "evaluation intent deferred to platform eval stages"
            )
        resolved_eval_config = self._engine.eval_config_ref
        request = EvalRequest(
            request_id=optim_eval_request.eval_request.request_id,
            candidate=optim_eval_request.eval_request.candidate,
            metadata=optim_eval_request.eval_request.metadata,
        )
        self._assert_generation_current(optim_eval_request, owned)
        result = self._engine.evaluate(request)
        match result:
            case EvalRejected(detail=detail):
                return self._bind_if_owned(
                    optim_eval_request,
                    IntentResolution(
                        schema_version=INTENT_RESOLUTION_SCHEMA_VERSION,
                        optim_eval_request=optim_eval_request,
                        outcome=IntentOutcome.REJECTED,
                        detail=detail,
                        resolved_eval_config=resolved_eval_config,
                    ),
                    owned,
                )
            case EvalEvidenceWithRef(
                evidence=EvalFailureEvidence() as failure,
                evidence_ref=evidence_ref,
            ):
                terminal_failure = TerminalFailure(
                    code=f"evaluation_{failure.exception_type}",
                    message=failure.message,
                    details={
                        "evidence_schema": evidence_ref.schema_name,
                        "evidence_content_hash": evidence_ref.content_hash,
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
                        eval_result_ref=evidence_ref,
                        reward_evidence_refs=(),
                        resolved_eval_config=resolved_eval_config,
                        terminal_failure=terminal_failure,
                    ),
                    owned,
                )
            case EvalEvidenceWithRef(
                evidence=EvalEvidence() as evidence,
                evidence_ref=evidence_ref,
            ):
                reward_ref = evidence.reward_ref
                reward_evidence_refs = (
                    ()
                    if reward_ref is None
                    else reward_ref.record.evidence_refs
                )
                return self._bind_if_owned(
                    optim_eval_request,
                    IntentResolution(
                        schema_version=INTENT_RESOLUTION_SCHEMA_VERSION,
                        optim_eval_request=optim_eval_request,
                        outcome=IntentOutcome.COMPLETED,
                        detail=ResolutionDetail(
                            classification=ResolutionClass.MEASURED,
                            message=(
                                "candidate evaluated under exact sampling "
                                "binding"
                            ),
                        ),
                        eval_result_ref=evidence_ref,
                        reward_evidence_refs=reward_evidence_refs,
                        resolved_eval_config=resolved_eval_config,
                        reward_ref=reward_ref,
                    ),
                    owned,
                )
            case _:
                raise TypeError(f"unexpected evaluation result: {result!r}")


__all__ = [
    "EvalDispatchMode",
    "EvalEngineService",
    "EvalExecutionContext",
    "EvalPlatformDeferred",
]
