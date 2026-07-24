"""Semantic failure taxonomy and Generation acceptance.

dr-providers returns a typed no-throw Provider Transport Outcome — either a
Provider Transport Response (a successful wire body carrying least-processed
``text``) or a Provider Transport Failure (a typed transport-level failure).
Neither asserts Whetstone semantic acceptance. This module is the sole owner
of that acceptance decision:

* **Generation acceptance** — a :class:`Generation` is the accepted *nonblank*
  semantic text projected by Whetstone from a Provider Transport Response. If
  the projected text is blank or whitespace-only it is NOT a Generation; it is
  classified as a :class:`ProviderSemanticFailure` of class
  ``BLANK_GENERATION`` that retains the rejected transport response.

* **Semantic failure classification** — every Provider Transport Outcome maps
  deterministically to exactly one closed :class:`SemanticFailureClass`. A
  Provider Transport Failure is classified from its transport ``failure_class``
  (and, where informative, its ``status_code``); a blank/whitespace response
  is classified ``BLANK_GENERATION``; a response that projects no ``text``
  field at all is ``MALFORMED_RESPONSE``.

The taxonomy is a *closed* enum: classification is total over the Provider
Transport Outcome union, so an exhausted retry loop can always name its cause.
The rejected causal evidence (the transport failure or the rejected response)
is retained on every :class:`ProviderSemanticFailure` so retry policy and the
terminal Provider Call Result carry their cause, never just a label.
"""

from __future__ import annotations

from enum import StrEnum

from dr_providers import (
    FailureClass,
    ProviderTransportFailure,
    ProviderTransportOutcome,
    ProviderTransportResponse,
)
from pydantic import BaseModel, ConfigDict, StrictStr

__all__ = [
    "Generation",
    "ProviderSemanticFailure",
    "SemanticFailureClass",
    "accept_generation",
    "classify_outcome",
    "is_blank",
]

# HTTP status the transport layer classifies as RATE_LIMITED / timeout-ish.
_TIMEOUT_STATUS_CODES = frozenset({408})


class SemanticFailureClass(StrEnum):
    """Closed Whetstone semantic failure taxonomy.

    Every Provider Transport Outcome that is not an accepted Generation maps to
    exactly one of these values. The set is closed: retry policy keys on these
    values and an exhausted loop always has one to report.
    """

    #: A Provider Transport Failure whose cause is a wire/connection-level
    #: transport error (a transient transport fault that is not a rate limit or
    #: a clean provider rejection): e.g. connection reset, 5xx, 409/425.
    TRANSPORT_ERROR = "transport-error"
    #: A Provider Transport Failure the provider signalled as a rate/quota
    #: limit (HTTP 429 / RATE_LIMITED / RESOURCE_EXHAUSTION).
    RATE_LIMIT = "rate-limit"
    #: A Provider Transport Failure whose cause is a request timeout.
    TIMEOUT = "timeout"
    #: A Provider Transport Failure the provider rejected permanently
    #: (bad request, auth, model rejection): a clean provider "no".
    PROVIDER_REJECTION = "provider-rejection"
    #: A successful Provider Transport Response whose projected semantic text
    #: is blank or whitespace-only — rejected as a Generation.
    BLANK_GENERATION = "blank-generation"
    #: A Provider Transport Outcome Whetstone cannot project semantic text
    #: from: a response missing its ``text`` projection, or a transport failure
    #: with an unknown/unclassifiable transport class.
    MALFORMED_RESPONSE = "malformed-response"


class Generation(BaseModel):
    """Accepted nonblank semantic text projected from a Transport Response.

    A Generation is the LLM Call Node's primary output. It carries the exact
    accepted ``text`` and retains the causal :class:`ProviderTransportResponse`
    it was projected from as provenance (the full provider result). It is not a
    transport response and not a Provider Call Result.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: StrictStr
    response: ProviderTransportResponse

    def model_post_init(self, _context: object) -> None:
        # A Generation is nonblank by construction; acceptance is the only
        # constructor callers should use, but enforce the invariant here too.
        if is_blank(self.text):
            raise ValueError("Generation text must be nonblank")


class ProviderSemanticFailure(BaseModel):
    """Whetstone-classified semantic failure retaining its causal evidence.

    Retains either the causal Provider Transport Failure OR the rejected
    Provider Transport Response (exactly one), plus the closed taxonomy value
    used by retry policy. It is expected domain output, never an exception.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    failure_class: SemanticFailureClass
    message: StrictStr
    #: The causal transport failure, when the transport itself failed.
    transport_failure: ProviderTransportFailure | None = None
    #: The rejected transport response, when a response was returned but not
    #: accepted as a Generation (blank/whitespace or missing text).
    rejected_response: ProviderTransportResponse | None = None

    def model_post_init(self, _context: object) -> None:
        has_failure = self.transport_failure is not None
        has_response = self.rejected_response is not None
        if has_failure == has_response:
            raise ValueError(
                "a ProviderSemanticFailure retains exactly one of "
                "transport_failure or rejected_response"
            )


def is_blank(text: str | None) -> bool:
    """A projection is blank when it is ``None`` or whitespace-only."""
    return text is None or not text.strip()


def _classify_transport_failure(
    failure: ProviderTransportFailure,
) -> SemanticFailureClass:
    """Deterministically map a transport failure to a semantic class."""
    status = failure.status_code
    failure_class = failure.failure_class
    if status in _TIMEOUT_STATUS_CODES:
        return SemanticFailureClass.TIMEOUT
    if failure_class in (
        FailureClass.RATE_LIMITED,
        FailureClass.RESOURCE_EXHAUSTION,
    ):
        return SemanticFailureClass.RATE_LIMIT
    if failure_class is FailureClass.TRANSIENT:
        return SemanticFailureClass.TRANSPORT_ERROR
    if failure_class is FailureClass.PERMANENT:
        return SemanticFailureClass.PROVIDER_REJECTION
    # FailureClass.UNKNOWN and any future/unrecognized transport class.
    return SemanticFailureClass.MALFORMED_RESPONSE


def accept_generation(
    response: ProviderTransportResponse,
) -> Generation | ProviderSemanticFailure:
    """Project a Generation from a Transport Response, or classify a failure.

    Acceptance is nonblank semantic ``text``. A blank/whitespace-only text is
    a ``BLANK_GENERATION`` failure retaining the rejected response.
    """
    if is_blank(response.text):
        return ProviderSemanticFailure(
            failure_class=SemanticFailureClass.BLANK_GENERATION,
            message="provider returned a blank or whitespace-only generation",
            rejected_response=response,
        )
    return Generation(text=response.text, response=response)


def classify_outcome(
    outcome: ProviderTransportOutcome,
) -> Generation | ProviderSemanticFailure:
    """Deterministically classify any Provider Transport Outcome.

    Total over the closed transport-outcome union: a Provider Transport
    Response projects a Generation (or a blank-generation failure); a Provider
    Transport Failure classifies to exactly one semantic failure class.
    """
    if isinstance(outcome, ProviderTransportResponse):
        return accept_generation(outcome)
    return ProviderSemanticFailure(
        failure_class=_classify_transport_failure(outcome),
        message=outcome.message,
        transport_failure=outcome,
    )
