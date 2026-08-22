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
    "PROVIDER_ERROR_KEY",
    "REJECTED_RESPONSE_KEY",
    "CallTelemetry",
    "call_telemetry",
    "evidences_provider_response",
    "failure_code_of",
    "guard_deadline_seconds",
    "is_rate_limit_failure",
    "is_transient_transport_failure",
]

GUARD_MARGIN_SECONDS = 15.0

#: Persisted key the failure body from :func:`call_telemetry` is stored
#: under, on an ``EvalOutputRow`` and on a failed ``ProposalDraft``'s
#: response evidence alike.
PROVIDER_ERROR_KEY = "provider_error"

#: Key inside that body carrying the generation the classifier rejected. It
#: is present exactly when the provider produced -- and therefore billed for
#: -- a response, which is what separates a response-level semantic failure
#: from a transport failure that got nothing back.
REJECTED_RESPONSE_KEY = "rejected_response"


def evidences_provider_response(provider_error: object) -> bool:
    """Whether a persisted failure body shows the provider answered.

    Both cost roles ask this one question of the same persisted body, so a
    rejected response that carried no usage telemetry is counted as the
    billed provider call it was rather than dropped as a transport failure.
    """
    if not isinstance(provider_error, dict):
        return False
    return provider_error.get(REJECTED_RESPONSE_KEY) is not None


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

    Aggregation is all-or-nothing per field, because a partial sum presented
    as a complete total is worse than an absent one. ``provider_cost`` is
    reported only when *every* billed attempt carried a price; if any billed
    attempt was unpriced the field is ``None`` and the row counts as unpriced,
    which is the same honesty rule ``RoleCost.usd`` obeys one level up.
    Tokens follow the same rule: a directional count is reported only when
    every billed attempt reported it, so a row whose retry lacked a token
    breakdown reports ``None`` -- and is counted in
    ``rows_missing_token_breakdown`` -- rather than publishing one attempt's
    tokens as the call's total.

    ``finish_reason`` describes the outcome the caller acts on, so it comes
    from the accepted generation alone and stays absent for a failed call.
    """
    if result is None:
        return CallTelemetry()
    responses = tuple(_billed_responses(result))
    prompt_tokens = _sum_across(responses, "prompt_tokens")
    completion_tokens = _sum_across(responses, "completion_tokens")
    total_tokens = _sum_across(responses, "total_tokens")
    reasoning_tokens = _sum_across(responses, "reasoning_tokens")
    provider_cost: float | None = None
    if responses and all(
        response.cost is not None for response in responses
    ):
        provider_cost = sum(
            response.cost.total_cost
            for response in responses
            if response.cost is not None
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


def _sum_across(
    responses: tuple[ProviderTransportResponse, ...], field: str
) -> int | None:
    """Sum one token field, or ``None`` when any billed attempt omitted it.

    Completeness is per field and per attempt: one attempt without the count
    makes the call's total unknowable, so the field is withheld rather than
    reported as the partial sum of the attempts that did have it.
    """
    if not responses:
        return None
    values: list[int] = []
    for response in responses:
        usage = response.usage
        value = None if usage is None else getattr(usage, field, None)
        if value is None:
            return None
        values.append(int(value))
    return sum(values)


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
        body[REJECTED_RESPONSE_KEY] = failure.rejected_response.model_dump(
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
