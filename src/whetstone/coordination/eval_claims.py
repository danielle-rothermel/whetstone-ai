from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from dr_store import BindingConflictError, BindStatus
from pydantic import (
    BaseModel,
    ConfigDict,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)

from whetstone.core.identity import (
    IdentityHash,
    TypedRef,
)
from whetstone.eval.protocol import EvalEngine
from whetstone.optim.contracts import (
    INTENT_RESOLUTION_SCHEMA,
    OptimEvalRequest,
    IntentOutcome,
    IntentResolution,
)

EVAL_RESULT_ATTESTATION_SCHEMA = (
    "whetstone.eval_result_attestation"
)
EVAL_INTENT_CLAIM_SCHEMA = "whetstone.eval_intent_claim"


class EvalIntentClaim(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    intent_ref: TypedRef
    owner_id: StrictStr
    event_ordinal: StrictInt
    generation: StrictInt
    heartbeat_ordinal: StrictInt
    expires_at: StrictFloat
    result_attestation_ref: TypedRef | None = None


class EvalResultAttestation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    graph_hash: IdentityHash
    resolution: IntentResolution

    @model_validator(mode="after")
    def _validate(self) -> EvalResultAttestation:
        if self.resolution.outcome not in {
            IntentOutcome.COMPLETED,
            IntentOutcome.FAILED,
        }:
            raise ValueError(
                "an Evaluation Result Attestation requires a terminal "
                "executed outcome"
            )
        return self

    def record_content(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


@dataclass(frozen=True, slots=True)
class _OwnedClaim:
    intent_ref: TypedRef
    generation: int


class _LeaseLostError(RuntimeError):
    pass


def _wait_for_renewal(interval: float, stop: threading.Event) -> bool:
    return stop.wait(interval)


def _ignore_renewal_publication(
    _claim: EvalIntentClaim,
) -> None:
    pass


class EvalClaims:
    _store: Any
    _engine: EvalEngine
    _owner_id: str
    _clock: Any
    _sleep: Any
    _claim_lease_seconds: float
    _renewal_wait: Any
    _renewal_published: Any
    _resolve_lock: Any
    _validate_result_graph: Any
    _load: Any
    _load_exact: Any
    _evaluate_and_bind: Any

    if TYPE_CHECKING:

        @staticmethod
        def _intent_ref(intent: OptimEvalRequest) -> TypedRef: ...

        @classmethod
        def _key(cls, intent: OptimEvalRequest) -> str: ...

        @classmethod
        def _claim_key(
            cls,
            intent: OptimEvalRequest,
            event_ordinal: int,
        ) -> str: ...

        @staticmethod
        def _typed_ref(reference: Any) -> TypedRef: ...

    def _bind(
        self, intent: OptimEvalRequest, resolution: IntentResolution
    ) -> IntentResolution:
        self._validate_result_graph(
            resolution, expected_optim_eval_request=intent
        )
        content = resolution.model_dump(mode="json")
        reference, _ = self._store.put(INTENT_RESOLUTION_SCHEMA, content)
        try:
            self._store.bind(self._key(intent), reference)
        except BindingConflictError:
            winner = self._store.resolve(self._key(intent))
            assert winner is not None
            loaded = self._load(winner, expected_optim_eval_request=intent)
            return loaded
        return resolution

    def _load_claim(self, reference: Any) -> EvalIntentClaim:
        return EvalIntentClaim.model_validate(self._store.get(reference))

    def _load_result_attestation(
        self,
        reference: Any,
        *,
        expected_optim_eval_request: OptimEvalRequest,
    ) -> EvalResultAttestation:
        _attestation_ref, content = self._load_exact(
            reference,
            expected_schema=EVAL_RESULT_ATTESTATION_SCHEMA,
        )
        attestation = EvalResultAttestation.model_validate(content)
        if (
            attestation.resolution.optim_eval_request
            != expected_optim_eval_request
        ):
            raise ValueError(
                "Evaluation Result Attestation belongs to another "
                "Optim Eval Request"
            )
        if attestation.graph_hash != self._engine.plan_snapshot.graph_hash:
            raise ValueError(
                "Evaluation Result Attestation uses another generation graph"
            )
        return attestation

    def _latest_claim(
        self,
        intent: OptimEvalRequest,
    ) -> EvalIntentClaim | None:
        latest: EvalIntentClaim | None = None
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
            elif latest.result_attestation_ref is not None:
                raise ValueError(
                    "durable evaluation claim stream continues after its "
                    "terminal attestation"
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
        intent: OptimEvalRequest,
        intent_ref: TypedRef,
        prior: EvalIntentClaim | None,
        generation: int,
        heartbeat_ordinal: int,
        result_attestation_ref: TypedRef | None = None,
    ) -> EvalIntentClaim:
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
            if prior.result_attestation_ref is not None:
                raise _LeaseLostError(
                    "terminal evaluation claim cannot be extended"
                )
        else:
            event_ordinal = prior.event_ordinal + 1
            if generation != prior.generation + 1 or heartbeat_ordinal != 0:
                raise ValueError("evaluation takeover must start a generation")
            if prior.expires_at > self._clock():
                raise _LeaseLostError(
                    "evaluation lease cannot be taken over before expiry"
                )
        claim = EvalIntentClaim(
            intent_ref=intent_ref,
            owner_id=self._owner_id,
            event_ordinal=event_ordinal,
            generation=generation,
            heartbeat_ordinal=heartbeat_ordinal,
            expires_at=float(self._clock() + self._claim_lease_seconds),
            result_attestation_ref=result_attestation_ref,
        )
        reference, _ = self._store.put(
            EVAL_INTENT_CLAIM_SCHEMA,
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

    def _publish_result_attestation(
        self,
        *,
        intent: OptimEvalRequest,
        resolution: IntentResolution,
        owned: _OwnedClaim,
    ) -> EvalResultAttestation:
        self._validate_result_graph(
            resolution,
            expected_optim_eval_request=intent,
            require_attestation=False,
        )
        attestation = EvalResultAttestation(
            graph_hash=self._engine.plan_snapshot.graph_hash,
            resolution=resolution,
        )
        persisted, _ = self._store.put(
            EVAL_RESULT_ATTESTATION_SCHEMA,
            attestation.record_content(),
        )
        attestation_ref = self._typed_ref(persisted)
        while True:
            latest = self._latest_claim(intent)
            if (
                latest is None
                or latest.owner_id != self._owner_id
                or latest.generation != owned.generation
            ):
                raise _LeaseLostError(
                    "evaluation lease is not owned by this resolver"
                )
            if latest.result_attestation_ref is not None:
                existing = self._load_result_attestation(
                    latest.result_attestation_ref,
                    expected_optim_eval_request=intent,
                )
                if existing != attestation:
                    raise _LeaseLostError(
                        "terminal evaluation claim names another result"
                    )
                return existing
            winner = self._append_claim_event(
                intent=intent,
                intent_ref=owned.intent_ref,
                prior=latest,
                generation=owned.generation,
                heartbeat_ordinal=latest.heartbeat_ordinal + 1,
                result_attestation_ref=attestation_ref,
            )
            if (
                winner.owner_id != self._owner_id
                or winner.generation != owned.generation
            ):
                raise _LeaseLostError(
                    "evaluation result lost claim arbitration"
                )
            if winner.result_attestation_ref == attestation_ref:
                return attestation
            if winner.result_attestation_ref is not None:
                raise _LeaseLostError(
                    "terminal evaluation claim names another result"
                )

    def _attested_resolution(
        self,
        intent: OptimEvalRequest,
    ) -> IntentResolution | None:
        latest = self._latest_claim(intent)
        if latest is None or latest.result_attestation_ref is None:
            return None
        return self._load_result_attestation(
            latest.result_attestation_ref,
            expected_optim_eval_request=intent,
        ).resolution

    def _renew_claim(
        self,
        intent: OptimEvalRequest,
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
        if latest.result_attestation_ref is not None:
            return
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
        self._renewal_published(winner)

    def _assert_generation_current(
        self,
        intent: OptimEvalRequest,
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

    def _claim(self, intent: OptimEvalRequest) -> _OwnedClaim | None:
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
            if winner.result_attestation_ref is not None:
                return None
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
        intent: OptimEvalRequest,
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
            resolution = self._evaluate_and_bind(intent, owned)
        finally:
            stop.set()
            thread.join()
        if heartbeat_errors and self._store.resolve(self._key(intent)) is None:
            raise RuntimeError("evaluation lease heartbeat failed") from (
                heartbeat_errors[0]
            )
        return resolution

    def _resolve_claimed(self, intent: OptimEvalRequest) -> IntentResolution:
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
        return self._evaluate_with_heartbeat(intent, owned)

    def resolve_optim_eval_request(
        self, optim_eval_request: OptimEvalRequest
    ) -> IntentResolution:
        with self._resolve_lock:
            return self._resolve_claimed(optim_eval_request)

    def _bind_if_owned(
        self,
        intent: OptimEvalRequest,
        resolution: IntentResolution,
        owned: _OwnedClaim,
    ) -> IntentResolution:
        if resolution.outcome in {
            IntentOutcome.COMPLETED,
            IntentOutcome.FAILED,
        }:
            self._publish_result_attestation(
                intent=intent,
                resolution=resolution,
                owned=owned,
            )
        else:
            self._assert_generation_current(intent, owned)
        return self._bind(intent, resolution)


__all__ = [
    "EVAL_INTENT_CLAIM_SCHEMA",
    "EVAL_RESULT_ATTESTATION_SCHEMA",
    "EvalClaims",
    "EvalIntentClaim",
    "EvalResultAttestation",
]
