"""Restart-safe PR4 EvaluationService backed by the canonical engine."""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from dr_store import BindingConflictError, BindStatus, ObjectStore

from whetstone.evaluation.engine import EvaluationEngine, EvaluationRequest
from whetstone.evaluation.schema import (
    EVALUATION_FAILURE_SCHEMA,
    EVALUATION_INTENT_CLAIM_SCHEMA,
    INTENT_RESOLUTION_SCHEMA,
    EvaluationFailureEvidence,
    EvaluationIntentClaim,
)
from whetstone.optimization.identity import TypedRef, typed_ref_for_record
from whetstone.optimization.schema import (
    EvaluationIntent,
    IntentOutcome,
    IntentResolution,
    ResolutionClass,
    ResolutionDetail,
)


@dataclass(frozen=True, slots=True)
class _OwnedClaim:
    intent_ref: TypedRef
    generation: int


class _LeaseLostError(RuntimeError):
    pass


class EngineEvaluationService:
    """Resolve each immutable intent exactly once across process restarts."""

    def __init__(
        self,
        *,
        store: ObjectStore,
        engine: EvaluationEngine,
        claim_lease_seconds: float = 300.0,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if claim_lease_seconds <= 0:
            raise ValueError("claim_lease_seconds must be positive")
        self._store = store
        self._engine = engine
        self._claim_lease_seconds = claim_lease_seconds
        self._clock = clock
        self._sleep = sleep
        self._owner_id = uuid.uuid4().hex
        self._resolve_lock = threading.Lock()

    @staticmethod
    def _intent_ref(intent: EvaluationIntent) -> TypedRef:
        return typed_ref_for_record(
            "whetstone.evaluation_intent", intent.model_dump(mode="json")
        )

    @classmethod
    def _key(cls, intent: EvaluationIntent) -> str:
        return (
            "whetstone.intent_resolution:"
            f"{cls._intent_ref(intent).content_hash}"
        )

    @classmethod
    def _claim_key(
        cls,
        intent: EvaluationIntent,
        event_ordinal: int,
    ) -> str:
        return (
            "whetstone.intent_evaluation_claim:"
            f"{cls._intent_ref(intent).content_hash}"
            f"#{event_ordinal}"
        )

    def _load(self, reference: Any) -> IntentResolution:
        return IntentResolution.model_validate(self._store.get(reference))

    def _bind(
        self, intent: EvaluationIntent, resolution: IntentResolution
    ) -> IntentResolution:
        content = resolution.model_dump(mode="json")
        reference, _ = self._store.put(INTENT_RESOLUTION_SCHEMA, content)
        try:
            self._store.bind(self._key(intent), reference)
        except BindingConflictError:
            winner = self._store.resolve(self._key(intent))
            assert winner is not None
            loaded = self._load(winner)
            if loaded.intent != intent:
                raise ValueError(
                    "durable Intent Resolution belongs to another intent"
                ) from None
            return loaded
        return resolution

    def _load_claim(self, reference: Any) -> EvaluationIntentClaim:
        return EvaluationIntentClaim.model_validate(self._store.get(reference))

    def _latest_claim(
        self,
        intent: EvaluationIntent,
    ) -> EvaluationIntentClaim | None:
        latest: EvaluationIntentClaim | None = None
        event_ordinal = 0
        intent_ref = self._intent_ref(intent)
        while True:
            bound = self._store.resolve(self._claim_key(intent, event_ordinal))
            if bound is None:
                return latest
            claim = self._load_claim(bound)
            if (
                claim.intent_ref != intent_ref
                or claim.event_ordinal != event_ordinal
            ):
                raise ValueError(
                    "durable evaluation claim has invalid lease identity"
                )
            if latest is None:
                if claim.generation != 0 or claim.heartbeat_ordinal != 0:
                    raise ValueError(
                        "durable evaluation claim stream has invalid origin"
                    )
            elif claim.owner_id == latest.owner_id:
                if (
                    claim.generation != latest.generation
                    or claim.heartbeat_ordinal != latest.heartbeat_ordinal + 1
                ):
                    raise ValueError(
                        "durable evaluation claim has invalid renewal order"
                    )
            elif (
                claim.generation != latest.generation + 1
                or claim.heartbeat_ordinal != 0
            ):
                raise ValueError(
                    "durable evaluation claim has invalid takeover order"
                )
            latest = claim
            event_ordinal += 1

    def _append_claim_event(
        self,
        *,
        intent: EvaluationIntent,
        intent_ref: TypedRef,
        prior: EvaluationIntentClaim | None,
        generation: int,
        heartbeat_ordinal: int,
    ) -> EvaluationIntentClaim:
        if prior is None:
            event_ordinal = 0
            if generation != 0 or heartbeat_ordinal != 0:
                raise ValueError("initial evaluation claim must start at zero")
        elif generation == prior.generation:
            event_ordinal = prior.event_ordinal + 1
            if (
                prior.owner_id != self._owner_id
                or heartbeat_ordinal != prior.heartbeat_ordinal + 1
            ):
                raise _LeaseLostError(
                    "evaluation lease cannot be renewed by another owner"
                )
        else:
            event_ordinal = prior.event_ordinal + 1
            if generation != prior.generation + 1 or heartbeat_ordinal != 0:
                raise ValueError("evaluation takeover must start a generation")
            if prior.expires_at > self._clock():
                raise _LeaseLostError(
                    "evaluation lease cannot be taken over before expiry"
                )
        claim = EvaluationIntentClaim(
            intent_ref=intent_ref,
            owner_id=self._owner_id,
            event_ordinal=event_ordinal,
            generation=generation,
            heartbeat_ordinal=heartbeat_ordinal,
            expires_at=float(self._clock() + self._claim_lease_seconds),
        )
        reference, _ = self._store.put(
            EVALUATION_INTENT_CLAIM_SCHEMA,
            claim.model_dump(mode="json"),
        )
        try:
            status = self._store.bind(
                self._claim_key(intent, event_ordinal),
                reference,
            )
        except BindingConflictError:
            status = None
        if status not in (None, BindStatus.BOUND, BindStatus.IDEMPOTENT):
            raise _LeaseLostError(
                "evaluation claim event was not durably bound"
            )
        bound = self._store.resolve(self._claim_key(intent, event_ordinal))
        assert bound is not None
        persisted = self._load_claim(bound)
        if (
            persisted.intent_ref != intent_ref
            or persisted.event_ordinal != event_ordinal
        ):
            raise ValueError(
                "durable evaluation claim has invalid event identity"
            )
        return persisted

    def _renew_claim(
        self,
        intent: EvaluationIntent,
        owned: _OwnedClaim,
    ) -> None:
        latest = self._latest_claim(intent)
        if (
            latest is None
            or latest.owner_id != self._owner_id
            or latest.generation != owned.generation
        ):
            raise _LeaseLostError(
                "evaluation lease is not owned by this resolver"
            )
        winner = self._append_claim_event(
            intent=intent,
            intent_ref=owned.intent_ref,
            prior=latest,
            generation=owned.generation,
            heartbeat_ordinal=latest.heartbeat_ordinal + 1,
        )
        if (
            winner.owner_id != self._owner_id
            or winner.generation != owned.generation
        ):
            raise _LeaseLostError(
                "evaluation lease renewal lost claim arbitration"
            )

    def _assert_generation_current(
        self,
        intent: EvaluationIntent,
        owned: _OwnedClaim,
    ) -> None:
        latest = self._latest_claim(intent)
        if (
            latest is None
            or latest.owner_id != self._owner_id
            or latest.generation != owned.generation
        ):
            raise _LeaseLostError(
                "evaluation lease is not owned by this resolver"
            )

    def _claim(self, intent: EvaluationIntent) -> _OwnedClaim | None:
        """Acquire the current durable lease or await its resolution.

        After a crashed owner's persisted lease expires, a fresh resolver
        claims the next append-only generation and safely retries. Concurrent
        live resolvers observe the winning unexpired claim and wait for its
        terminal resolution.
        """
        intent_ref = self._intent_ref(intent)
        while True:
            if self._store.resolve(self._key(intent)) is not None:
                return
            winner = self._latest_claim(intent)
            if winner is None:
                winner = self._append_claim_event(
                    intent=intent,
                    intent_ref=intent_ref,
                    prior=None,
                    generation=0,
                    heartbeat_ordinal=0,
                )
            if winner.owner_id == self._owner_id:
                return _OwnedClaim(
                    intent_ref=intent_ref,
                    generation=winner.generation,
                )
            remaining = winner.expires_at - self._clock()
            if remaining <= 0:
                takeover = self._append_claim_event(
                    intent=intent,
                    intent_ref=intent_ref,
                    prior=winner,
                    generation=winner.generation + 1,
                    heartbeat_ordinal=0,
                )
                if takeover.owner_id == self._owner_id:
                    return _OwnedClaim(
                        intent_ref=intent_ref,
                        generation=takeover.generation,
                    )
                continue
            self._sleep(min(0.05, remaining))

    def _evaluate_with_heartbeat(
        self,
        intent: EvaluationIntent,
        owned: _OwnedClaim,
    ) -> IntentResolution:
        stop = threading.Event()
        heartbeat_errors: list[Exception] = []

        def heartbeat() -> None:
            interval = self._claim_lease_seconds / 3
            while not stop.wait(interval):
                try:
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
            resolution = self._evaluate_and_bind(intent, owned)
        finally:
            stop.set()
            thread.join()
        if heartbeat_errors:
            raise RuntimeError("evaluation lease heartbeat failed") from (
                heartbeat_errors[0]
            )
        return resolution

    def _resolve_claimed(self, intent: EvaluationIntent) -> IntentResolution:
        existing = self._store.resolve(self._key(intent))
        if existing is not None:
            return self._load(existing)
        owned = self._claim(intent)
        existing = self._store.resolve(self._key(intent))
        if existing is not None:
            return self._load(existing)
        if owned is None:
            raise RuntimeError("evaluation claim resolved without a result")
        return self._evaluate_with_heartbeat(intent, owned)

    def resolve_evaluation_intent(
        self, intent: EvaluationIntent
    ) -> IntentResolution:
        with self._resolve_lock:
            return self._resolve_claimed(intent)

    def _bind_if_owned(
        self,
        intent: EvaluationIntent,
        resolution: IntentResolution,
        owned: _OwnedClaim,
    ) -> IntentResolution:
        self._assert_generation_current(intent, owned)
        return self._bind(intent, resolution)

    def _evaluate_and_bind(
        self,
        intent: EvaluationIntent,
        owned: _OwnedClaim,
    ) -> IntentResolution:
        if intent.target_eval_config != self._engine.eval_config_ref:
            return self._bind_if_owned(
                intent,
                IntentResolution(
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
        try:
            self._engine.preflight(intent.candidate.record)
        except (KeyError, TypeError, ValueError) as exc:
            return self._bind_if_owned(
                intent,
                IntentResolution(
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
            evaluated = self._engine.evaluate(
                EvaluationRequest(
                    candidate=intent.candidate.record,
                    evaluation_role=intent.context_role,
                    evaluation_context_id=intent.intent_id,
                    purpose=intent.purpose,
                )
            )
        except Exception as exc:
            failure = EvaluationFailureEvidence(
                candidate=intent.candidate,
                eval_config=intent.target_eval_config,
                purpose=intent.purpose,
                exception_type=type(exc).__name__,
                message=str(exc) or type(exc).__name__,
            )
            ref, _ = self._store.put(
                EVALUATION_FAILURE_SCHEMA, failure.record_content()
            )
            return self._bind_if_owned(
                intent,
                IntentResolution(
                    intent=intent,
                    outcome=IntentOutcome.FAILED,
                    detail=ResolutionDetail(
                        classification=ResolutionClass.INFRASTRUCTURE,
                        message=failure.message,
                    ),
                    evaluation_evidence_refs=(
                        TypedRef(
                            schema_name=ref.schema,
                            content_hash=ref.content_hash,
                        ),
                    ),
                    resolved_eval_config=intent.target_eval_config,
                ),
                owned,
            )
        return self._bind_if_owned(
            intent,
            IntentResolution(
                intent=intent,
                outcome=IntentOutcome.COMPLETED,
                detail=ResolutionDetail(
                    classification=ResolutionClass.MEASURED,
                    message="candidate evaluated under exact sampling binding",
                ),
                evaluation_evidence_refs=(evaluated.evidence_ref,),
                resolved_eval_config=intent.target_eval_config,
                reward_ref=evaluated.evidence.reward_ref,
            ),
            owned,
        )


__all__ = ["EngineEvaluationService"]
