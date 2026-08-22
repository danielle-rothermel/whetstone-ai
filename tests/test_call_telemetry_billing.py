"""``call_telemetry`` accounts for every provider response a call was billed for.

One logical call can cost more than one provider response: a retried
response-level failure was generated and billed before the retry succeeded,
and a call that exhausted its retries was billed for each attempt. These
tests pin that the telemetry a row persists -- which is what run cost is
later derived from -- carries that whole spend rather than only the terminal
generation's.
"""

from __future__ import annotations

import pytest
from dr_providers import (
    CostInfo,
    ProviderInvocationEvidence,
    ProviderTransportResponse,
    TokenUsage,
)
from dr_providers.outcomes.models import (
    ProviderTransportFailure,
    RecoverabilityClass,
)

from whetstone.execution.call_support import call_telemetry
from whetstone.provider.attempt import ProviderCallAttempt, ProviderCallResult
from whetstone.provider.classification import (
    ProviderGeneration,
    classify_outcome,
)

_HASH = "a" * 64


def _response(
    *,
    text: str,
    prompt: int | None = None,
    completion: int | None = None,
    usd: float | None = None,
) -> ProviderTransportResponse:
    usage = None
    if prompt is not None or completion is not None:
        usage = TokenUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=(prompt or 0) + (completion or 0),
        )
    return ProviderTransportResponse(
        text=text,
        usage=usage,
        cost=None if usd is None else CostInfo(total_cost=usd),
        stop_reason="stop",
    )


def _attempt(
    number: int,
    *,
    response: ProviderTransportResponse | None = None,
    transport_failure: bool = False,
) -> ProviderCallAttempt:
    if transport_failure:
        evidence = ProviderInvocationEvidence(
            request_identity_hash=_HASH,
            failure=ProviderTransportFailure(
                recoverability=RecoverabilityClass.TRANSIENT,
                message="connection reset",
                status_code=503,
            ),
        )
    else:
        evidence = ProviderInvocationEvidence(
            request_identity_hash=_HASH, response=response
        )
    classification = classify_outcome(evidence.outcome)
    outcome_field = (
        {"provider_generation": classification}
        if isinstance(classification, ProviderGeneration)
        else {"semantic_failure": classification}
    )
    return ProviderCallAttempt(
        logical_call_id="call-1",
        attempt_number=number,
        execution_policy_hash=_HASH,
        started_at=float(number),
        ended_at=float(number) + 0.5,
        evidence=evidence,
        **outcome_field,
    )


def _result(*attempts: ProviderCallAttempt) -> ProviderCallResult:
    last = attempts[-1]
    return ProviderCallResult(
        logical_call_id="call-1",
        request_hash={"identity": _HASH},
        execution_policy_hash=_HASH,
        attempts=attempts,
        provider_generation=last.provider_generation,
        semantic_failure=last.semantic_failure,
    )


def test_a_single_successful_attempt_reports_its_own_usage() -> None:
    result = _result(
        _attempt(1, response=_response(text="answer", prompt=9, completion=4, usd=0.5))
    )

    telemetry = call_telemetry(result)

    assert telemetry.prompt_tokens == 9
    assert telemetry.completion_tokens == 4
    assert telemetry.provider_cost == pytest.approx(0.5)
    assert telemetry.provider_error is None
    assert telemetry.finish_reason is not None


def test_a_retried_call_sums_every_billed_attempt() -> None:
    # The blank generation was produced and billed before the retry landed.
    # Reporting only the terminal generation loses that spend.
    rejected = _attempt(
        1, response=_response(text="   ", prompt=7, completion=0, usd=0.25)
    )
    accepted = _attempt(
        2, response=_response(text="answer", prompt=9, completion=4, usd=0.5)
    )

    telemetry = call_telemetry(_result(rejected, accepted))

    assert telemetry.prompt_tokens == 16
    assert telemetry.completion_tokens == 4
    assert telemetry.total_tokens == 20
    assert telemetry.provider_cost == pytest.approx(0.75)
    assert telemetry.provider_error is None


def test_a_billed_failure_reports_its_usage_and_stays_failed() -> None:
    # Retries exhausted: every attempt was generated and billed, and the call
    # still failed. Both facts have to survive into the persisted row.
    first = _attempt(
        1, response=_response(text="  ", prompt=7, completion=0, usd=0.25)
    )
    second = _attempt(
        2, response=_response(text="   ", prompt=6, completion=0, usd=0.2)
    )

    telemetry = call_telemetry(_result(first, second))

    assert telemetry.prompt_tokens == 13
    assert telemetry.provider_cost == pytest.approx(0.45)
    assert telemetry.provider_error is not None
    assert telemetry.finish_reason is None


def test_a_transport_failure_is_not_billed() -> None:
    # Nothing came back, so nothing was charged.
    telemetry = call_telemetry(_result(_attempt(1, transport_failure=True)))

    assert telemetry.prompt_tokens is None
    assert telemetry.completion_tokens is None
    assert telemetry.provider_cost is None
    assert telemetry.provider_error is not None


def test_a_transport_failure_before_a_success_adds_nothing() -> None:
    dropped = _attempt(1, transport_failure=True)
    accepted = _attempt(
        2, response=_response(text="answer", prompt=9, completion=4, usd=0.5)
    )

    telemetry = call_telemetry(_result(dropped, accepted))

    assert telemetry.prompt_tokens == 9
    assert telemetry.provider_cost == pytest.approx(0.5)


def test_a_priced_attempt_without_a_token_split_keeps_its_price() -> None:
    # A provider may price a call without breaking tokens out by direction.
    result = _result(_attempt(1, response=_response(text="answer", usd=0.4)))

    telemetry = call_telemetry(result)

    assert telemetry.prompt_tokens is None
    assert telemetry.provider_cost == pytest.approx(0.4)


def test_an_unpriced_attempt_contributes_no_price(tmp_path) -> None:
    # A price is never inferred: absent means unknown, not zero.
    rejected = _attempt(1, response=_response(text=" ", prompt=7, completion=0))
    accepted = _attempt(
        2, response=_response(text="answer", prompt=9, completion=4)
    )

    telemetry = call_telemetry(_result(rejected, accepted))

    assert telemetry.prompt_tokens == 16
    assert telemetry.provider_cost is None


def test_no_call_at_all_reports_empty_telemetry() -> None:
    assert call_telemetry(None).prompt_tokens is None
    assert call_telemetry(None).provider_cost is None
