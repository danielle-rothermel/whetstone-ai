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

from whetstone.execution.call_support import (
    PROVIDER_ERROR_KEY,
    call_telemetry,
)
from whetstone.optim.proposal.proposer import ProposalDraft
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


def test_a_partly_priced_retry_reports_no_price() -> None:
    # One billed attempt was priced and the other was not, so the call's real
    # price is unknowable. Publishing the priced attempt's 0.50 would classify
    # the row as fully priced and let an understated total present itself as
    # complete, which is exactly what RoleCost.usd refuses one level up.
    unpriced = _attempt(1, response=_response(text=" ", prompt=7, completion=0))
    priced = _attempt(
        2, response=_response(text="answer", prompt=9, completion=4, usd=0.5)
    )

    telemetry = call_telemetry(_result(unpriced, priced))

    assert telemetry.provider_cost is None
    assert telemetry.prompt_tokens == 16


def test_a_retry_missing_a_token_split_reports_no_tokens() -> None:
    # The first attempt was billed but reported no usage at all. Summing only
    # the second attempt's tokens would present one attempt's usage as the
    # whole call's; the row instead reports no token breakdown.
    untracked = _attempt(1, response=_response(text=" ", usd=0.25))
    accepted = _attempt(
        2, response=_response(text="answer", prompt=9, completion=4, usd=0.5)
    )

    telemetry = call_telemetry(_result(untracked, accepted))

    assert telemetry.prompt_tokens is None
    assert telemetry.completion_tokens is None
    assert telemetry.total_tokens is None
    # Both attempts were priced, so the price total stays complete.
    assert telemetry.provider_cost == pytest.approx(0.75)


def test_proposer_accounting_sums_every_billed_attempt() -> None:
    # The proposer side bills retries exactly like the task-model side: it
    # projects the same call_telemetry aggregate onto the draft's usage map,
    # so a rejected-then-retried proposal reports both attempts' spend.
    from whetstone.optim.proposal.proposer import _billed_accounting

    rejected = _attempt(
        1, response=_response(text="   ", prompt=7, completion=1, usd=0.25)
    )
    accepted = _attempt(
        2, response=_response(text="answer", prompt=9, completion=4, usd=0.5)
    )

    usage, cost, _ = _billed_accounting(_result(rejected, accepted))

    assert usage["prompt_tokens"] == 16
    assert usage["completion_tokens"] == 5
    assert cost == pytest.approx(0.75)


def test_proposer_accounting_withholds_a_partial_retry_price() -> None:
    # Same honesty rule one level down: an unpriced billed attempt makes the
    # whole proposer call unpriced rather than publishing the priced half.
    unpriced = _attempt(1, response=_response(text=" ", prompt=7, completion=1))
    priced = _attempt(
        2, response=_response(text="answer", prompt=9, completion=4, usd=0.5)
    )

    from whetstone.optim.proposal.proposer import _billed_accounting

    usage, cost, _ = _billed_accounting(_result(unpriced, priced))

    assert usage["prompt_tokens"] == 16
    assert cost is None


def _failure_draft(result: ProviderCallResult) -> ProposalDraft:
    """The failure draft ``ProviderProposerTransport`` builds for a result.

    Built through ``_billed_accounting`` and the same evidence shape the
    transport writes, so what is under test is the accounting rule rather
    than a hand-assembled draft.
    """
    from whetstone.optim.proposal.proposer import _billed_accounting

    usage, cost, provider_error = _billed_accounting(result)
    return ProposalDraft.failure(
        detail="provider proposer failed",
        request_evidence={"logical_call_id": "call-1"},
        response_evidence={
            "logical_call_id": "call-1",
            PROVIDER_ERROR_KEY: provider_error,
        },
        usage=usage,
        cost=cost,
    )


def test_a_rejected_proposer_response_without_telemetry_is_a_call() -> None:
    # A provider that returns a blank generation and reports neither usage
    # nor a price still ran the model and still charged for it; only the
    # response classifier turned the answer down. The rejected response on
    # the persisted failure body is what separates that from a transport
    # failure, so the call is counted -- as an unpriced one with an unknown
    # token breakdown, which withholds the role's usd rather than completing
    # it from spend that was never measured.
    rejected = _attempt(1, response=_response(text="   "))
    draft = _failure_draft(_result(rejected))

    usage = draft.call_usage()

    assert usage is not None
    assert usage.call_id == "call-1"
    assert usage.prompt_tokens is None
    assert usage.completion_tokens is None
    assert usage.usd is None


def test_a_proposer_transport_failure_is_still_not_a_call() -> None:
    # Nothing came back, so nothing was billed: counting it would add an
    # unpriced call for spend that never happened.
    draft = _failure_draft(_result(_attempt(1, transport_failure=True)))

    assert draft.call_usage() is None
