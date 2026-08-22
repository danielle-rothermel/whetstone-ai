from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

from whetstone.provider.attempt import ProviderCallResult
from whetstone.provider.classification import SemanticFailureClass
from whetstone.provider.policy import ProviderExecutionPolicy

if TYPE_CHECKING:
    from dr_providers import ProviderTransportResponse

__all__ = [
    "GUARD_MARGIN_SECONDS",
    "CallTelemetry",
    "call_telemetry",
    "failure_code_of",
    "guard_deadline_seconds",
    "is_rate_limit_failure",
    "is_transient_transport_failure",
]

GUARD_MARGIN_SECONDS = 15.0


@dataclass(frozen=True, slots=True)
class CallTelemetry:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    latency_s: float | None = None
    finish_reason: str | None = None
    provider_error: dict[str, object] | None = None
    #: Provider-reported price for this call, when the provider returned one.
    #: Never estimated: absent means the price is unknown, not zero.
    provider_cost: float | None = None


def call_telemetry(result: ProviderCallResult | None) -> CallTelemetry:
    """Accounting for one logical call, summed over every billed attempt.

    A logical call can cost more than one provider response. When the
    execution policy retries a response-level failure such as
    ``BLANK_PROVIDER_GENERATION``, the rejected response was still generated
    and still billed, and it stays on the attempt's ``rejected_response``.
    Reporting only the terminal generation would drop that spend, so tokens
    and ``provider_cost`` are summed across every attempt that carried a
    response -- accepted or rejected.

    The same rule makes a *failed* call report its usage: a call that
    exhausted its retries having been billed for each one contributes its
    tokens and price like any other, while ``provider_error`` still marks it
    failed for scoring. Only an attempt that never got a response back (a
    transport failure) contributes nothing, because nothing was billed.

    ``finish_reason`` describes the outcome the caller acts on, so it comes
    from the accepted generation alone and stays absent for a failed call.
    """
    if result is None:
        return CallTelemetry()
    responses = tuple(_billed_responses(result))
    prompt_tokens, completion_tokens = None, None
    total_tokens, reasoning_tokens = None, None
    provider_cost: float | None = None
    for response in responses:
        if response.cost is not None:
            provider_cost = (provider_cost or 0.0) + response.cost.total_cost
        usage = response.usage
        if usage is None:
            continue
        prompt_tokens = _add(prompt_tokens, usage.prompt_tokens)
        completion_tokens = _add(completion_tokens, usage.completion_tokens)
        total_tokens = _add(total_tokens, usage.total_tokens)
        reasoning_tokens = _add(
            reasoning_tokens, getattr(usage, "reasoning_tokens", None)
        )
    accepted = result.provider_generation
    return CallTelemetry(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        reasoning_tokens=reasoning_tokens,
        latency_s=_accepted_latency(result),
        finish_reason=(
            accepted.response.stop_reason if accepted is not None else None
        ),
        provider_error=_provider_error_of(result),
        provider_cost=provider_cost,
    )


def _add(running: int | None, value: int | None) -> int | None:
    """Sum token counts, keeping ``None`` for "no attempt reported this"."""
    if value is None:
        return running
    return value if running is None else running + value


def _billed_responses(
    result: ProviderCallResult,
) -> Iterator[ProviderTransportResponse]:
    """Every provider response this logical call was billed for, in order.

    An attempt is billed when the provider produced a response: the accepted
    generation, or a response the classifier rejected (blank, malformed) but
    the provider had already generated and charged for. An attempt that
    failed at the transport carries no response and is not billed.
    """
    for attempt in result.attempts:
        if attempt.provider_generation is not None:
            yield attempt.provider_generation.response
        elif (
            attempt.semantic_failure is not None
            and attempt.semantic_failure.rejected_response is not None
        ):
            yield attempt.semantic_failure.rejected_response


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
            if candidate.provider_generation is not None
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
    return (
        not result.succeeded
        and result.semantic_failure is not None
        and result.semantic_failure.failure_class in _TRANSIENT_CLASSES
    )


def failure_code_of(result: ProviderCallResult) -> str:
    if result.succeeded or result.semantic_failure is None:
        return ""
    failure = result.semantic_failure
    code = getattr(failure.transport_failure, "code", None)
    if code:
        return str(code)
    return failure.failure_class.value


def is_rate_limit_failure(result: ProviderCallResult) -> bool:
    return any(
        attempt.failure_class is SemanticFailureClass.RATE_LIMIT
        for attempt in result.attempts
    )


def guard_deadline_seconds(
    policy: ProviderExecutionPolicy,
    *,
    wire_calls_per_unit: int = 1,
) -> float:
    if wire_calls_per_unit < 1:
        raise ValueError("wire_calls_per_unit must be a positive integer")
    attempt_timeouts = (
        policy.max_attempts * policy.transport_policy.timeout_seconds
    )
    backoff_seconds = sum(
        policy.delay_before(attempt_number)
        for attempt_number in range(1, policy.max_attempts + 1)
    )
    return (
        wire_calls_per_unit * (attempt_timeouts + backoff_seconds)
        + GUARD_MARGIN_SECONDS
    )
