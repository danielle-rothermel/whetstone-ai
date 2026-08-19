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
from whetstone.eval.row_slice import RowEvalCompletion, RowEvalOutcome, RowEvalSlice
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

    def _effective_context(
        self,
        context: EvalExecutionContext | None,
    ) -> EvalExecutionContext:
        if context is not None:
            return context
        return EvalExecutionContext()

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

    def _clear_platform_intent(self, optim_eval_request: OptimEvalRequest) -> None:
        self._store.evict_bindings([self._platform_intent_key(optim_eval_request)])

    def resolve_platform_intent_from_row_outcomes(
        self,
        optim_eval_request: OptimEvalRequest,
        *,
        row_outcomes: tuple[RowEvalOutcome, ...],
    ) -> IntentResolution:
        with self._resolve_lock:
            previous = self._active_context
            self._active_context = EvalExecutionContext(
                dispatch_mode=EvalDispatchMode.INLINE
            )
            try:
                return self._resolve_claimed_with_row_outcomes(
                    optim_eval_request,
                    row_outcomes,
                )
            finally:
                self._active_context = previous

    def resolve_platform_intent_from_row_slices(
        self,
        optim_eval_request: OptimEvalRequest,
        *,
        row_slices: tuple[RowEvalSlice, ...],
    ) -> IntentResolution:
        row_outcomes = tuple(
            RowEvalOutcome(
                task_id=row_slice.task_id,
                seed_index=row_slice.seed_index,
                evidence=row_slice.evidence,
                supplemental_aggregate_refs=row_slice.supplemental_aggregate_refs,
            )
            for row_slice in row_slices
        )
        return self.resolve_platform_intent_from_row_outcomes(
            optim_eval_request,
            row_outcomes=row_outcomes,
        )

    def _resolve_claimed_with_row_outcomes(
        self,
        intent: OptimEvalRequest,
        row_outcomes: tuple[RowEvalOutcome, ...],
    ) -> IntentResolution:
        existing = self._store.resolve(self._key(intent))
        if existing is not None:
            return self._load(existing, expected_optim_eval_request=intent)
        attested = self._attested_resolution(intent)
        if attested is not None:
            return self._bind(intent, attested)
        owned = self._claim(intent)
        existing = self._store.resolve(self._key(intent))
        if existing is not None:
            return self._load(existing, expected_optim_eval_request=intent)
        attested = self._attested_resolution(intent)
        if attested is not None:
            return self._bind(intent, attested)
        if owned is None:
            raise RuntimeError("evaluation claim resolved without a result")
        return self._assemble_with_heartbeat_outcomes(intent, row_outcomes, owned)

    def _resolve_claimed_with_row_slices(
        self,
        intent: OptimEvalRequest,
        row_slices: tuple[RowEvalSlice, ...],
    ) -> IntentResolution:
        existing = self._store.resolve(self._key(intent))
        if existing is not None:
            return self._load(existing, expected_optim_eval_request=intent)
        attested = self._attested_resolution(intent)
        if attested is not None:
            return self._bind(intent, attested)
        owned = self._claim(intent)
        existing = self._store.resolve(self._key(intent))
        if existing is not None:
            return self._load(existing, expected_optim_eval_request=intent)
        attested = self._attested_resolution(intent)
        if attested is not None:
            return self._bind(intent, attested)
        if owned is None:
            raise RuntimeError("evaluation claim resolved without a result")
        return self._assemble_with_heartbeat(intent, row_slices, owned)

    def _assemble_with_heartbeat_outcomes(
        self,
        intent: OptimEvalRequest,
        row_outcomes: tuple[RowEvalOutcome, ...],
        owned: _OwnedClaim,
    ) -> IntentResolution:
        stop = threading.Event()
        heartbeat_errors: list[Exception] = []

        def heartbeat() -> None:
            interval = self._claim_lease_seconds / 3
            while True:
                try:
                    if self._renewal_wait(interval, stop):
                        return
                    self._renew_claim(intent, owned)
                except Exception as exc:
                    heartbeat_errors.append(exc)
                    return

        self._renew_claim(intent, owned)
        thread = threading.Thread(
            target=heartbeat,
            name=f"evaluation-heartbeat-{owned.generation}",
            daemon=True,
        )
        thread.start()
        try:
            self._assert_generation_current(intent, owned)
            resolution = self._assemble_and_bind_outcomes(
                intent,
                row_outcomes,
                owned,
            )
        finally:
            stop.set()
            thread.join()
        if heartbeat_errors and self._store.resolve(self._key(intent)) is None:
            raise RuntimeError("evaluation lease heartbeat failed") from (
                heartbeat_errors[0]
            )
        return resolution

    def _assemble_and_bind_outcomes(
        self,
        optim_eval_request: OptimEvalRequest,
        row_outcomes: tuple[RowEvalOutcome, ...],
        owned: _OwnedClaim,
    ) -> IntentResolution:
        self._persist_intent_targets(optim_eval_request)
        self._assert_generation_current(optim_eval_request, owned)
        resolved_eval_config = self._engine.eval_config_ref
        self._clear_platform_intent(optim_eval_request)
        for outcome in row_outcomes:
            if outcome.rejected_detail is not None:
                return self._bind_if_owned(
                    optim_eval_request,
                    IntentResolution(
                        schema_version=INTENT_RESOLUTION_SCHEMA_VERSION,
                        optim_eval_request=optim_eval_request,
                        outcome=IntentOutcome.REJECTED,
                        detail=outcome.rejected_detail,
                        resolved_eval_config=resolved_eval_config,
                    ),
                    owned,
                )
        for outcome in row_outcomes:
            if outcome.failure is not None:
                if outcome.evidence_ref is None:
                    raise ValueError("failed row outcome is missing evidence_ref")
                terminal_failure = TerminalFailure(
                    code=f"evaluation_{outcome.failure.exception_type}",
                    message=outcome.failure.message,
                    details={
                        "evidence_schema": outcome.evidence_ref.schema_name,
                        "evidence_content_hash": outcome.evidence_ref.content_hash,
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
                            message=outcome.failure.message,
                        ),
                        eval_result_ref=outcome.evidence_ref,
                        reward_evidence_refs=(),
                        resolved_eval_config=resolved_eval_config,
                        terminal_failure=terminal_failure,
                    ),
                    owned,
                )
        row_slices = tuple(
            RowEvalSlice(
                task_id=outcome.task_id,
                seed_index=outcome.seed_index,
                evidence=outcome.evidence,
                supplemental_aggregate_refs=outcome.supplemental_aggregate_refs,
            )
            for outcome in row_outcomes
            if outcome.evidence is not None
        )
        if not row_slices:
            raise ValueError("row outcomes contain no success evidence")
        return self._assemble_and_bind(
            optim_eval_request,
            row_slices,
            owned,
            clear_platform_intent=False,
        )

    def _assemble_with_heartbeat(
        self,
        intent: OptimEvalRequest,
        row_slices: tuple[RowEvalSlice, ...],
        owned: _OwnedClaim,
    ) -> IntentResolution:
        stop = threading.Event()
        heartbeat_errors: list[Exception] = []

        def heartbeat() -> None:
            interval = self._claim_lease_seconds / 3
            while True:
                try:
                    if self._renewal_wait(interval, stop):
                        return
                    self._renew_claim(intent, owned)
                except Exception as exc:
                    heartbeat_errors.append(exc)
                    return

        self._renew_claim(intent, owned)
        thread = threading.Thread(
            target=heartbeat,
            name=f"evaluation-heartbeat-{owned.generation}",
            daemon=True,
        )
        thread.start()
        try:
            self._assert_generation_current(intent, owned)
            resolution = self._assemble_and_bind(intent, row_slices, owned)
        finally:
            stop.set()
            thread.join()
        if heartbeat_errors and self._store.resolve(self._key(intent)) is None:
            raise RuntimeError("evaluation lease heartbeat failed") from (
                heartbeat_errors[0]
            )
        return resolution

    def _assemble_and_bind(
        self,
        optim_eval_request: OptimEvalRequest,
        row_slices: tuple[RowEvalSlice, ...],
        owned: _OwnedClaim,
        *,
        clear_platform_intent: bool = True,
    ) -> IntentResolution:
        self._persist_intent_targets(optim_eval_request)
        self._assert_generation_current(optim_eval_request, owned)
        resolved_eval_config = self._engine.eval_config_ref
        request = EvalRequest(
            request_id=optim_eval_request.eval_request.request_id,
            candidate=optim_eval_request.eval_request.candidate,
            metadata=optim_eval_request.eval_request.metadata,
        )
        result = self._engine.assemble_from_row_slices(
            request,
            row_slices=row_slices,
        )
        if clear_platform_intent:
            self._clear_platform_intent(optim_eval_request)
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
