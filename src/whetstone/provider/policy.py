from __future__ import annotations

import math
from typing import Any

from dr_providers import ProviderKind, ProviderTransportPolicy, policy_for
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
    "default_transport_policy",
]

PROVIDER_EXECUTION_POLICY_SCHEMA = "whetstone.provider_execution_policy"
PROVIDER_EXECUTION_POLICY_SCHEMA_VERSION = 1


def default_retry_eligibility() -> dict[SemanticFailureClass, bool]:
    return {
        SemanticFailureClass.TRANSPORT_ERROR: True,
        SemanticFailureClass.RATE_LIMIT: True,
        SemanticFailureClass.TIMEOUT: True,
        SemanticFailureClass.PROVIDER_REJECTION: False,
        SemanticFailureClass.BLANK_PROVIDER_GENERATION: False,
        SemanticFailureClass.MALFORMED_RESPONSE: False,
    }


def default_transport_policy(
    *,
    api_key_env: str,
    provider_kind: ProviderKind = ProviderKind.OPENAI,
    timeout_seconds: float = 30.0,
) -> ProviderTransportPolicy:
    return policy_for(
        provider_kind,
        api_key_env=api_key_env,
        timeout_seconds=timeout_seconds,
        connect_timeout_seconds=min(10.0, timeout_seconds),
        idle_timeout_seconds=timeout_seconds,
        max_connections=10,
        max_keepalive_connections=5,
        max_request_bytes=1_048_576,
        max_response_bytes=10_485_760,
    )


class BackoffSchedule(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    base_seconds: float = 1.0
    multiplier: float = 2.0
    max_seconds: float = 60.0

    @model_validator(mode="after")
    def _validate(self) -> BackoffSchedule:
        if not math.isfinite(self.base_seconds):
            raise ValueError("base_seconds must be finite")
        if not math.isfinite(self.multiplier):
            raise ValueError("multiplier must be finite")
        if not math.isfinite(self.max_seconds):
            raise ValueError("max_seconds must be finite")
        if self.base_seconds < 0:
            raise ValueError("base_seconds cannot be negative")
        if self.multiplier < 1:
            raise ValueError("multiplier must be >= 1")
        if self.max_seconds < self.base_seconds:
            raise ValueError("max_seconds cannot be below base_seconds")
        return self

    def delay_for(self, attempt_number: int) -> float:
        if attempt_number < 1:
            raise ValueError("attempt_number must be a positive integer")
        if attempt_number == 1:
            return 0.0
        if (
            self.base_seconds == 0
            or self.multiplier == 1
            or self.base_seconds == self.max_seconds
        ):
            return self.base_seconds

        try:
            delay = self.base_seconds * self.multiplier ** (attempt_number - 2)
        except OverflowError:
            return self.max_seconds
        if not math.isfinite(delay):
            return self.max_seconds
        return min(delay, self.max_seconds)

    def identity_payload(self) -> dict[str, Any]:
        return {
            "base_seconds": self.base_seconds,
            "multiplier": self.multiplier,
            "max_seconds": self.max_seconds,
        }


class ProviderExecutionPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    transport_policy: ProviderTransportPolicy

    max_attempts: StrictInt = 3

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
        missing = set(SemanticFailureClass) - set(self.retry_eligibility)
        if missing:
            raise ValueError(
                "retry_eligibility must cover every SemanticFailureClass; "
                f"missing: {sorted(c.value for c in missing)}"
            )
        return self

    def is_retryable(self, failure_class: SemanticFailureClass) -> bool:
        return self.retry_eligibility[failure_class]

    def delay_before(self, attempt_number: int) -> float:
        return self.backoff.delay_for(attempt_number)

    def identity_payload(self) -> dict[str, Any]:
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
        document = build_identity_document(
            schema=PROVIDER_EXECUTION_POLICY_SCHEMA,
            schema_version=PROVIDER_EXECUTION_POLICY_SCHEMA_VERSION,
            payload=self.identity_payload(),
        )
        return identity_document_hash(document)
