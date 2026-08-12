from __future__ import annotations

import math
from typing import Any, Literal

from dr_providers import ProviderInvocationEvidence
from pydantic import (
    BaseModel,
    ConfigDict,
    StrictInt,
    StrictStr,
    model_validator,
)

from whetstone.core.identity import require_full_hash
from whetstone.provider.classification import (
    ProviderGeneration,
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
PROVIDER_CALL_ATTEMPT_SCHEMA_VERSION = 2
PROVIDER_CALL_RESULT_SCHEMA = "whetstone.provider_call_result"
PROVIDER_CALL_RESULT_SCHEMA_VERSION = 2


class ProviderCallAttempt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[2] = PROVIDER_CALL_ATTEMPT_SCHEMA_VERSION

    logical_call_id: StrictStr

    attempt_number: StrictInt

    execution_policy_hash: StrictStr

    started_at: float
    ended_at: float

    evidence: ProviderInvocationEvidence

    provider_generation: ProviderGeneration | None = None
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

        has_provider_generation = self.provider_generation is not None
        has_failure = self.semantic_failure is not None
        if has_provider_generation == has_failure:
            raise ValueError(
                "a ProviderCallAttempt holds exactly one of "
                "provider_generation or semantic_failure"
            )
        expected = classify_outcome(self.evidence.outcome)
        if has_provider_generation and self.provider_generation != expected:
            raise ValueError(
                "provider_generation must exactly reclassify the evidence "
                "outcome"
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
        return self.provider_generation is not None

    @property
    def failure_class(self) -> SemanticFailureClass | None:
        if self.semantic_failure is None:
            return None
        return self.semantic_failure.failure_class

    def to_stable_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class ProviderCallResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[2] = PROVIDER_CALL_RESULT_SCHEMA_VERSION
    logical_call_id: StrictStr

    request_hash: dict[str, Any]
    execution_policy_hash: StrictStr

    attempts: tuple[ProviderCallAttempt, ...]
    provider_generation: ProviderGeneration | None = None
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

        has_provider_generation = self.provider_generation is not None
        has_failure = self.semantic_failure is not None
        if has_provider_generation == has_failure:
            raise ValueError(
                "a ProviderCallResult holds exactly one of "
                "provider_generation or semantic_failure"
            )

        last = self.attempts[-1]
        if (
            has_provider_generation
            and self.provider_generation != last.provider_generation
        ):
            raise ValueError(
                "terminal provider_generation must equal the final attempt's "
                "provider_generation"
            )
        if has_failure and self.semantic_failure != last.semantic_failure:
            raise ValueError(
                "terminal failure must equal the final attempt's failure"
            )
        first_request_hash = self.attempts[0].evidence.request_identity_hash
        first_policy = self.attempts[0].evidence.policy_identity
        for attempt in self.attempts:
            if attempt.evidence.request_identity_hash != first_request_hash:
                raise ValueError(
                    "every attempt evidence request identity must agree"
                )
            if attempt.evidence.policy_identity != first_policy:
                raise ValueError(
                    "every attempt evidence policy identity must agree"
                )
        return self

    @property
    def succeeded(self) -> bool:
        return self.provider_generation is not None

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    def to_stable_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
