"""Provider Execution Policy: Whetstone semantic orchestration policy.

The Provider Execution Policy composes exactly ONE Provider Transport Policy
*reference* with only Whetstone semantic concerns:

* a bounded logical-attempt count,
* per-semantic-class retry eligibility (a closed map over
  :class:`SemanticFailureClass`),
* a deterministic backoff schedule.

It deliberately duplicates NONE of the transport policy's operational fields:
no credentials, no timeout, and no native retry count. Native retries are
pinned to zero by construction — the policy references the transport policy by
its identity payload and asserts ``native_retry_count == 0``; it never carries
its own copy of that field. Whetstone owns all semantic retry, so any nonzero
native retry would be a second, uncoordinated retry loop.

The policy is identity-bearing: its Identity Payload (transport-policy identity
+ classification/backoff/attempt config) hashes through dr-serialize to a full
64-character Provider Execution Policy Identity Hash that each Provider Call
Attempt references.
"""

from __future__ import annotations

from typing import Any

from dr_providers import ProviderTransportPolicy
from dr_serialize import build_identity_document, identity_document_hash
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_validator,
    model_validator,
)

from whetstone.provider.classification import SemanticFailureClass

__all__ = [
    "PROVIDER_EXECUTION_POLICY_SCHEMA",
    "PROVIDER_EXECUTION_POLICY_SCHEMA_VERSION",
    "BackoffSchedule",
    "ProviderExecutionPolicy",
    "default_retry_eligibility",
]

PROVIDER_EXECUTION_POLICY_SCHEMA = "whetstone.provider_execution_policy"
PROVIDER_EXECUTION_POLICY_SCHEMA_VERSION = 1


def default_retry_eligibility() -> dict[SemanticFailureClass, bool]:
    """Conservative default per-class retry eligibility.

    Transient/rate/timeout classes are retryable; a clean provider rejection,
    a malformed response, and a blank generation are not (retrying the same
    request is not expected to change a deterministic provider "no" or a
    structural response defect). Callers may override any entry.
    """
    return {
        SemanticFailureClass.TRANSPORT_ERROR: True,
        SemanticFailureClass.RATE_LIMIT: True,
        SemanticFailureClass.TIMEOUT: True,
        SemanticFailureClass.PROVIDER_REJECTION: False,
        SemanticFailureClass.BLANK_GENERATION: False,
        SemanticFailureClass.MALFORMED_RESPONSE: False,
    }


class BackoffSchedule(BaseModel):
    """Deterministic backoff schedule keyed by prior-attempt count.

    ``delay_for(attempt_number)`` returns the sleep, in seconds, taken BEFORE
    logical attempt ``attempt_number`` (1-based). The first attempt has zero
    delay. The schedule is a pure function of the attempt number — no jitter,
    no wall-clock, no randomness — so replay is byte-identical.

    ``base_seconds`` scaled geometrically by ``multiplier`` per subsequent
    attempt, capped at ``max_seconds``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    base_seconds: float = 1.0
    multiplier: float = 2.0
    max_seconds: float = 60.0

    @model_validator(mode="after")
    def _validate(self) -> BackoffSchedule:
        if self.base_seconds < 0:
            raise ValueError("base_seconds cannot be negative")
        if self.multiplier < 1:
            raise ValueError("multiplier must be >= 1")
        if self.max_seconds < self.base_seconds:
            raise ValueError("max_seconds cannot be below base_seconds")
        return self

    def delay_for(self, attempt_number: int) -> float:
        """Deterministic pre-attempt delay for a 1-based attempt number."""
        if attempt_number < 1:
            raise ValueError("attempt_number must be a positive integer")
        if attempt_number == 1:
            return 0.0
        delay = self.base_seconds * (self.multiplier ** (attempt_number - 2))
        return min(delay, self.max_seconds)

    def identity_payload(self) -> dict[str, Any]:
        return {
            "base_seconds": self.base_seconds,
            "multiplier": self.multiplier,
            "max_seconds": self.max_seconds,
        }


class ProviderExecutionPolicy(BaseModel):
    """Whetstone semantic orchestration policy over one transport policy.

    Composes one :class:`ProviderTransportPolicy` reference with a bounded
    logical-attempt count, per-class retry eligibility, and a deterministic
    backoff schedule. Carries no credentials, timeout, or native retry field.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The single referenced transport policy. Native retries MUST be zero.
    transport_policy: ProviderTransportPolicy
    #: Bounded maximum number of logical attempts (>= 1).
    max_attempts: StrictInt = 3
    #: Closed per-class retry eligibility map (complete over the taxonomy).
    retry_eligibility: dict[SemanticFailureClass, bool] = Field(
        default_factory=default_retry_eligibility
    )
    backoff: BackoffSchedule = Field(default_factory=BackoffSchedule)

    @field_validator("max_attempts")
    @classmethod
    def _bounded(cls, value: int) -> int:
        if value < 1:
            raise ValueError("max_attempts must be a positive integer")
        return value

    @model_validator(mode="after")
    def _validate(self) -> ProviderExecutionPolicy:
        # Native transport retries are pinned to zero: Whetstone owns retry.
        if self.transport_policy.native_retry_count != 0:
            raise ValueError(
                "Provider Execution Policy requires the referenced transport "
                "policy's native_retry_count to be zero; Whetstone owns all "
                "semantic retry"
            )
        # The eligibility map must be complete and closed over the taxonomy.
        missing = set(SemanticFailureClass) - set(self.retry_eligibility)
        if missing:
            raise ValueError(
                "retry_eligibility must cover every SemanticFailureClass; "
                f"missing: {sorted(c.value for c in missing)}"
            )
        return self

    def is_retryable(self, failure_class: SemanticFailureClass) -> bool:
        """Per-class retry eligibility lookup (closed over the taxonomy)."""
        return self.retry_eligibility[failure_class]

    def delay_before(self, attempt_number: int) -> float:
        """Deterministic backoff delay before a 1-based logical attempt."""
        return self.backoff.delay_for(attempt_number)

    def identity_payload(self) -> dict[str, Any]:
        """Semantic identity: transport-policy identity + semantic config.

        References the transport policy by its identity payload (env-var name
        only, never a secret) and adds only the Whetstone semantic fields.
        """
        return {
            "transport_policy": self.transport_policy.identity_payload(),
            "max_attempts": self.max_attempts,
            "retry_eligibility": {
                cls.value: self.retry_eligibility[cls]
                for cls in sorted(
                    self.retry_eligibility, key=lambda c: c.value
                )
            },
            "backoff": self.backoff.identity_payload(),
        }

    @property
    def identity_hash(self) -> str:
        """Full 64-char Provider Execution Policy Identity Hash."""
        document = build_identity_document(
            schema=PROVIDER_EXECUTION_POLICY_SCHEMA,
            schema_version=PROVIDER_EXECUTION_POLICY_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )
        return identity_document_hash(document)
