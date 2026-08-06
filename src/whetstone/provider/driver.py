from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from dr_providers import ProviderCallRequest, ProviderInvocationEvidence

from whetstone.provider.attempt import ProviderCallAttempt, ProviderCallResult
from whetstone.provider.classification import Generation, classify_outcome
from whetstone.provider.policy import ProviderExecutionPolicy

__all__ = [
    "TransportCall",
    "run_provider_call",
]

#: The injectable transport callable. dr-providers' ``HttpProvider.invoke``
#: satisfies it; tests inject a scripted/fake peer. It returns exactly one
#: stable Provider Invocation Evidence per physical invocation.
TransportCall = Callable[[ProviderCallRequest], ProviderInvocationEvidence]

#: Injectable monotonic clock hook returning seconds. Defaults to a real
#: monotonic clock; tests inject a deterministic sequence.
Clock = Callable[[], float]

#: Injectable sleep hook. Defaults to no-op (the pure driver never blocks);
#: callers may inject their own scheduling mechanism.
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
            # Deterministic pre-attempt backoff (zero before the first).
            delay = self.policy.delay_before(attempt_number)
            if delay > 0:
                self.sleep(delay)

            started_at = self.clock()
            evidence = self.transport(self.request)
            evidence_identities = evidence.model_dump(
                mode="json",
                include={"request_identity", "policy_identity"},
            )
            if (
                evidence_identities["request_identity"]
                != self.request.identity_payload()
            ):
                raise ValueError(
                    "transport evidence request identity does not match the "
                    "invoked request"
                )
            if (
                evidence_identities["policy_identity"]
                != self.policy.transport_policy.identity_payload()
            ):
                raise ValueError(
                    "transport evidence policy identity does not match the "
                    "invoked transport policy"
                )
            ended_at = self.clock()

            classification = classify_outcome(evidence.outcome)
            if isinstance(classification, Generation):
                attempt = ProviderCallAttempt(
                    logical_call_id=self.logical_call_id,
                    attempt_number=attempt_number,
                    execution_policy_hash=policy_hash,
                    started_at=started_at,
                    ended_at=ended_at,
                    evidence=evidence,
                    generation=classification,
                )
                attempts.append(attempt)
                return ProviderCallResult(
                    logical_call_id=self.logical_call_id,
                    request_identity=self.request.identity_payload(),
                    execution_policy_hash=policy_hash,
                    attempts=tuple(attempts),
                    generation=classification,
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

            # Stop early on a non-retry-eligible failure. If this was the last
            # bounded attempt, fall through to terminal exhaustion regardless.
            retry_eligible = self.policy.is_retryable(
                classification.failure_class
            )
            if not retry_eligible:
                break

        # Exhausted (bound reached) or stopped on a non-eligible failure.
        # Both are expected terminal domain output: a valid Result carrying the
        # final Provider Semantic Failure, never an exception.
        last = attempts[-1]
        return ProviderCallResult(
            logical_call_id=self.logical_call_id,
            request_identity=self.request.identity_payload(),
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
    """Run the bounded attempt loop for one logical provider call.

    Pure and DBOS-free. Deterministic given the same recorded transport
    outcomes: same attempt sequence and byte-identical terminal Result.

    Args:
        request: the immutable Provider Call Request.
        policy: the composing Provider Execution Policy (bounds + backoff +
            per-class retry eligibility).
        transport: the injectable transport callable returning one Provider
            Invocation Evidence per physical invocation.
        logical_call_id: stable identity of the logical call.
        clock: injectable monotonic clock hook (seconds). Defaults to real.
        sleep: injectable sleep hook for backoff. Defaults to no-op.
    """
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
