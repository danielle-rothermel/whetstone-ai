"""Provider Call Attempt wrapper and terminal Provider Call Result.

A :class:`ProviderCallAttempt` is the serializable Whetstone logical-attempt
wrapper. It contains:

* the logical call identity (stable across attempts of one logical call),
* the 1-based attempt number,
* the Provider Execution Policy identity,
* timing (start/end monotonic-ish timestamps and derived latency),
* exactly ONE stable :class:`ProviderInvocationEvidence` artifact, and
* its Whetstone semantic classification (a Generation or a Provider Semantic
  Failure).

It does not guarantee exactly one physical wire call; it is one *logical*
attempt observation. Provider bodies live inside the invocation evidence.

A :class:`ProviderCallResult` is the terminal semantic Result for one logical
provider call: the Provider Call Request identity, the ordered completed
attempts, and the final Generation or Provider Semantic Failure. An exhausted
Provider Semantic Failure is expected terminal domain output — a valid Result,
never an exception.
"""

from __future__ import annotations

import math
from typing import Any

from dr_providers import ProviderInvocationEvidence
from pydantic import (
    BaseModel,
    ConfigDict,
    StrictInt,
    StrictStr,
    model_validator,
)

from whetstone.identity import require_full_hash
from whetstone.provider.classification import (
    Generation,
    ProviderSemanticFailure,
    SemanticFailureClass,
    classify_outcome,
)

__all__ = [
    "PROVIDER_CALL_ATTEMPT_SCHEMA",
    "PROVIDER_CALL_ATTEMPT_SCHEMA_VERSION",
    "PROVIDER_CALL_RESULT_SCHEMA",
    "PROVIDER_CALL_RESULT_SCHEMA_VERSION",
    "ProviderCallAttempt",
    "ProviderCallResult",
]

PROVIDER_CALL_ATTEMPT_SCHEMA = "whetstone.provider_call_attempt"
PROVIDER_CALL_ATTEMPT_SCHEMA_VERSION = 1
PROVIDER_CALL_RESULT_SCHEMA = "whetstone.provider_call_result"
PROVIDER_CALL_RESULT_SCHEMA_VERSION = 1


class ProviderCallAttempt(BaseModel):
    """Serializable Whetstone logical-attempt wrapper.

    One completed logical attempt: identity, attempt number, execution-policy
    identity, timing, one Provider Invocation Evidence artifact, and its
    semantic classification. This type is infrastructure-free and fully
    serializable on its own.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = PROVIDER_CALL_ATTEMPT_SCHEMA_VERSION
    #: Stable identity of the logical call, constant across its attempts.
    logical_call_id: StrictStr
    #: 1-based attempt number within the logical call.
    attempt_number: StrictInt
    #: Identity Hash of the composing Provider Execution Policy.
    execution_policy_hash: StrictStr
    #: Wall-clock timing of the attempt (seconds, injectable clock).
    started_at: float
    ended_at: float
    #: Exactly one stable transport evidence artifact (provider bodies here).
    evidence: ProviderInvocationEvidence
    #: The Whetstone semantic classification of this attempt.
    generation: Generation | None = None
    semantic_failure: ProviderSemanticFailure | None = None

    @model_validator(mode="after")
    def _validate(self) -> ProviderCallAttempt:
        if not self.logical_call_id:
            raise ValueError("logical_call_id must be non-empty")
        if self.attempt_number < 1:
            raise ValueError("attempt_number must be a positive integer")
        require_full_hash(
            self.execution_policy_hash, field="execution_policy_hash"
        )
        if not math.isfinite(self.started_at) or self.started_at < 0:
            raise ValueError("started_at must be finite and non-negative")
        if not math.isfinite(self.ended_at) or self.ended_at < 0:
            raise ValueError("ended_at must be finite and non-negative")
        if self.ended_at < self.started_at:
            raise ValueError("ended_at cannot precede started_at")
        # Exactly one classification side is present.
        has_generation = self.generation is not None
        has_failure = self.semantic_failure is not None
        if has_generation == has_failure:
            raise ValueError(
                "a ProviderCallAttempt holds exactly one of generation or "
                "semantic_failure"
            )
        expected = classify_outcome(self.evidence.outcome)
        if has_generation and self.generation != expected:
            raise ValueError(
                "generation must exactly reclassify the evidence outcome"
            )
        if has_failure and self.semantic_failure != expected:
            raise ValueError(
                "semantic_failure must exactly reclassify the evidence outcome"
            )
        return self

    @property
    def latency_seconds(self) -> float:
        return self.ended_at - self.started_at

    @property
    def latency_ms(self) -> int:
        return round(self.latency_seconds * 1000)

    @property
    def succeeded(self) -> bool:
        return self.generation is not None

    @property
    def failure_class(self) -> SemanticFailureClass | None:
        if self.semantic_failure is None:
            return None
        return self.semantic_failure.failure_class

    def to_stable_dict(self) -> dict[str, Any]:
        """Stable serialized form for checkpointing/persistence."""
        return self.model_dump(mode="json")


class ProviderCallResult(BaseModel):
    """Terminal semantic Result for one logical provider call.

    Request identity, ordered completed attempts, and the final Generation or
    Provider Semantic Failure. An exhausted Provider Semantic Failure is a
    valid terminal Result, not an exception.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = PROVIDER_CALL_RESULT_SCHEMA_VERSION
    logical_call_id: StrictStr
    #: Provider Call Request identity payload (config-ref + transcript).
    request_identity: dict[str, Any]
    execution_policy_hash: StrictStr
    #: Ordered completed attempts (attempt 1 .. N).
    attempts: tuple[ProviderCallAttempt, ...]
    generation: Generation | None = None
    semantic_failure: ProviderSemanticFailure | None = None

    @model_validator(mode="after")
    def _validate(self) -> ProviderCallResult:
        if not self.logical_call_id:
            raise ValueError("logical_call_id must be non-empty")
        if not self.attempts:
            raise ValueError("a ProviderCallResult has at least one attempt")
        require_full_hash(
            self.execution_policy_hash, field="execution_policy_hash"
        )
        # Attempts are ordered, contiguous, 1-based, and share identity.
        for index, attempt in enumerate(self.attempts, start=1):
            if attempt.attempt_number != index:
                raise ValueError(
                    "attempts must be ordered 1..N with contiguous numbers"
                )
            if attempt.logical_call_id != self.logical_call_id:
                raise ValueError(
                    "every attempt shares the Result's logical_call_id"
                )
            if attempt.execution_policy_hash != self.execution_policy_hash:
                raise ValueError(
                    "every attempt shares the Result's execution_policy_hash"
                )
        # Exactly one terminal side is present.
        has_generation = self.generation is not None
        has_failure = self.semantic_failure is not None
        if has_generation == has_failure:
            raise ValueError(
                "a ProviderCallResult holds exactly one of generation or "
                "semantic_failure"
            )
        # The terminal outcome must equal the last attempt's classification.
        last = self.attempts[-1]
        if has_generation and self.generation != last.generation:
            raise ValueError(
                "terminal generation must equal the final attempt's generation"
            )
        if has_failure and self.semantic_failure != last.semantic_failure:
            raise ValueError(
                "terminal failure must equal the final attempt's failure"
            )
        first_evidence_identities = self.attempts[0].evidence.model_dump(
            mode="json",
            include={"request_identity", "policy_identity"},
        )
        expected_policy_identity = first_evidence_identities["policy_identity"]
        for attempt in self.attempts:
            evidence_identities = attempt.evidence.model_dump(
                mode="json",
                include={"request_identity", "policy_identity"},
            )
            if (
                evidence_identities["request_identity"]
                != self.request_identity
            ):
                raise ValueError(
                    "every attempt evidence request identity must equal the "
                    "Result's request_identity"
                )
            if (
                evidence_identities["policy_identity"]
                != expected_policy_identity
            ):
                raise ValueError(
                    "every attempt evidence policy identity must agree"
                )
        return self

    @property
    def succeeded(self) -> bool:
        return self.generation is not None

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    def to_stable_dict(self) -> dict[str, Any]:
        """Stable serialized form for checkpointing/persistence."""
        return self.model_dump(mode="json")
