"""Pure inspection helpers over terminal provider-call results."""

from __future__ import annotations

from dataclasses import dataclass

from whetstone.execution.fanout import GUARD_MARGIN_SECONDS
from whetstone.provider.attempt import ProviderCallResult
from whetstone.provider.classification import SemanticFailureClass
from whetstone.provider.policy import ProviderExecutionPolicy

__all__ = [
    "CallTelemetry",
    "call_telemetry",
    "failure_code_of",
    "guard_deadline_seconds",
    "is_rate_limit_failure",
    "is_transient_transport_failure",
]


@dataclass(frozen=True, slots=True)
class CallTelemetry:
    """Usage, latency, and provider provenance for one call."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    latency_s: float | None = None
    finish_reason: str | None = None
    provider_error: dict[str, object] | None = None


def call_telemetry(result: ProviderCallResult | None) -> CallTelemetry:
    """Extract coverage-honest telemetry from a provider-call result."""
    if result is None:
        return CallTelemetry()
    if not result.succeeded or result.generation is None:
        return CallTelemetry(
            latency_s=_accepted_latency(result),
            provider_error=_provider_error_of(result),
        )
    usage = result.generation.response.usage
    finish_reason = result.generation.response.finish_reason
    if usage is None:
        return CallTelemetry(
            latency_s=_accepted_latency(result),
            finish_reason=finish_reason,
        )
    return CallTelemetry(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        reasoning_tokens=getattr(usage, "reasoning_tokens", None),
        latency_s=_accepted_latency(result),
        finish_reason=finish_reason,
    )


def _provider_error_of(
    result: ProviderCallResult | None,
) -> dict[str, object] | None:
    if result is None or result.succeeded or result.semantic_failure is None:
        return None
    failure = result.semantic_failure
    body: dict[str, object] = {
        "failure_class": failure.failure_class.value,
        "message": failure.message,
    }
    if failure.transport_failure is not None:
        body["transport_failure"] = failure.transport_failure.model_dump(
            mode="json"
        )
    if failure.rejected_response is not None:
        body["rejected_response"] = failure.rejected_response.model_dump(
            mode="json"
        )
    return body


def _accepted_latency(result: ProviderCallResult | None) -> float | None:
    if result is None or not result.attempts:
        return None
    attempt = next(
        (
            candidate
            for candidate in result.attempts
            if candidate.generation is not None
        ),
        result.attempts[-1],
    )
    return max(0.0, attempt.ended_at - attempt.started_at)


_TRANSIENT_CLASSES = frozenset(
    {
        SemanticFailureClass.TRANSPORT_ERROR,
        SemanticFailureClass.RATE_LIMIT,
        SemanticFailureClass.TIMEOUT,
    }
)


def is_transient_transport_failure(result: ProviderCallResult) -> bool:
    """Whether one additional bounded observation drive is eligible."""
    return (
        not result.succeeded
        and result.semantic_failure is not None
        and result.semantic_failure.failure_class in _TRANSIENT_CLASSES
    )


def failure_code_of(result: ProviderCallResult) -> str:
    """Return the most specific stable code for a failed call."""
    if result.succeeded or result.semantic_failure is None:
        return ""
    failure = result.semantic_failure
    code = getattr(failure.transport_failure, "code", None)
    if code:
        return str(code)
    return failure.failure_class.value


def is_rate_limit_failure(result: ProviderCallResult) -> bool:
    """Whether any attempt observed a rate-limit failure."""
    return any(
        attempt.failure_class is SemanticFailureClass.RATE_LIMIT
        for attempt in result.attempts
    )


def guard_deadline_seconds(
    policy: ProviderExecutionPolicy,
    *,
    wire_calls_per_unit: int = 1,
) -> float:
    """Return the transport-cap-based deadline for one fanout unit."""
    cap = policy.transport_policy.timeout_seconds
    return cap * max(1, wire_calls_per_unit) + GUARD_MARGIN_SECONDS
