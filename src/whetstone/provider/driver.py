from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from dr_providers import ProviderCallRequest, ProviderInvocationEvidence

from whetstone.provider.attempt import ProviderCallAttempt, ProviderCallResult
from whetstone.provider.classification import (
    ProviderGeneration,
    classify_outcome,
)
from whetstone.provider.policy import ProviderExecutionPolicy

__all__ = [
    "TransportCall",
    "run_provider_call",
]


TransportCall = Callable[[ProviderCallRequest], ProviderInvocationEvidence]


Clock = Callable[[], float]


Sleep = Callable[[float], None]


def _default_clock() -> float:
    import time

    return time.monotonic()


def _no_sleep(_seconds: float) -> None:
    return None


@dataclass(frozen=True)
class _Driver:
    request: ProviderCallRequest
    policy: ProviderExecutionPolicy
    transport: TransportCall
    logical_call_id: str
    clock: Clock = field(default=_default_clock)
    sleep: Sleep = field(default=_no_sleep)

    def run(self) -> ProviderCallResult:
        policy_hash = self.policy.identity_hash
        attempts: list[ProviderCallAttempt] = []
        for attempt_number in range(1, self.policy.max_attempts + 1):
            delay = self.policy.delay_before(attempt_number)
            if delay > 0:
                self.sleep(delay)

            started_at = self.clock()
            evidence = self.transport(self.request)
            if evidence.request_identity_hash != self.request.identity_hash:
                raise ValueError(
                    "transport evidence request identity does not match the "
                    "invoked request"
                )
            expected_policy = self.policy.transport_policy.identity_payload()
            if evidence.policy_identity != expected_policy:
                raise ValueError(
                    "transport evidence policy identity does not match the "
                    "invoked transport policy"
                )
            ended_at = self.clock()

            classification = classify_outcome(evidence.outcome)
            if isinstance(classification, ProviderGeneration):
                attempt = ProviderCallAttempt(
                    logical_call_id=self.logical_call_id,
                    attempt_number=attempt_number,
                    execution_policy_hash=policy_hash,
                    started_at=started_at,
                    ended_at=ended_at,
                    evidence=evidence,
                    provider_generation=classification,
                )
                attempts.append(attempt)
                return ProviderCallResult(
                    logical_call_id=self.logical_call_id,
                    request_hash=self.request.identity_payload(),
                    execution_policy_hash=policy_hash,
                    attempts=tuple(attempts),
                    provider_generation=classification,
                )

            attempt = ProviderCallAttempt(
                logical_call_id=self.logical_call_id,
                attempt_number=attempt_number,
                execution_policy_hash=policy_hash,
                started_at=started_at,
                ended_at=ended_at,
                evidence=evidence,
                semantic_failure=classification,
            )
            attempts.append(attempt)

            retry_eligible = self.policy.is_retryable(
                classification.failure_class
            )
            if not retry_eligible:
                break

        last = attempts[-1]
        return ProviderCallResult(
            logical_call_id=self.logical_call_id,
            request_hash=self.request.identity_payload(),
            execution_policy_hash=policy_hash,
            attempts=tuple(attempts),
            semantic_failure=last.semantic_failure,
        )


def run_provider_call(
    *,
    request: ProviderCallRequest,
    policy: ProviderExecutionPolicy,
    transport: TransportCall,
    logical_call_id: str,
    clock: Clock | None = None,
    sleep: Sleep | None = None,
) -> ProviderCallResult:
    if not logical_call_id:
        raise ValueError("logical_call_id must be non-empty")
    driver = _Driver(
        request=request,
        policy=policy,
        transport=transport,
        logical_call_id=logical_call_id,
        clock=clock or _default_clock,
        sleep=sleep or _no_sleep,
    )
    return driver.run()
