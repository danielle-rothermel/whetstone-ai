from __future__ import annotations

from enum import StrEnum

from dr_providers import (
    RecoverabilityClass,
    ProviderTransportFailure,
    ProviderTransportOutcome,
    ProviderTransportResponse,
)
from pydantic import BaseModel, ConfigDict, StrictStr

__all__ = [
    "ProviderGeneration",
    "ProviderSemanticFailure",
    "SemanticFailureClass",
    "accept_provider_generation",
    "classify_outcome",
    "is_blank",
]


_TIMEOUT_STATUS_CODES = frozenset({408})


class SemanticFailureClass(StrEnum):
    TRANSPORT_ERROR = "transport-error"

    RATE_LIMIT = "rate-limit"

    TIMEOUT = "timeout"

    PROVIDER_REJECTION = "provider-rejection"

    BLANK_PROVIDER_GENERATION = "blank-provider-generation"

    MALFORMED_RESPONSE = "malformed-response"


class ProviderGeneration(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: StrictStr
    response: ProviderTransportResponse

    def model_post_init(self, _context: object) -> None:

        if is_blank(self.text):
            raise ValueError("ProviderGeneration text must be nonblank")
        if self.text != self.response.text:
            raise ValueError(
                "ProviderGeneration text must equal its causal response text"
            )


class ProviderSemanticFailure(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    failure_class: SemanticFailureClass
    message: StrictStr

    transport_failure: ProviderTransportFailure | None = None

    rejected_response: ProviderTransportResponse | None = None

    def model_post_init(self, _context: object) -> None:
        has_failure = self.transport_failure is not None
        has_response = self.rejected_response is not None
        if has_failure == has_response:
            raise ValueError(
                "a ProviderSemanticFailure retains exactly one of "
                "transport_failure or rejected_response"
            )


def is_blank(text: str) -> bool:
    return not text.strip()


def _classify_transport_failure(
    failure: ProviderTransportFailure,
) -> SemanticFailureClass:
    status = failure.status_code
    recoverability = failure.recoverability
    if status in _TIMEOUT_STATUS_CODES:
        return SemanticFailureClass.TIMEOUT
    if recoverability in (
        RecoverabilityClass.RATE_LIMITED,
        RecoverabilityClass.RESOURCE_EXHAUSTION,
    ):
        return SemanticFailureClass.RATE_LIMIT
    if recoverability is RecoverabilityClass.TRANSIENT:
        return SemanticFailureClass.TRANSPORT_ERROR
    if recoverability is RecoverabilityClass.PERMANENT:
        return SemanticFailureClass.PROVIDER_REJECTION

    return SemanticFailureClass.MALFORMED_RESPONSE


def accept_provider_generation(
    response: ProviderTransportResponse,
) -> ProviderGeneration | ProviderSemanticFailure:
    if is_blank(response.text):
        return ProviderSemanticFailure(
            failure_class=SemanticFailureClass.BLANK_PROVIDER_GENERATION,
            message="provider returned a blank or whitespace-only generation",
            rejected_response=response,
        )
    return ProviderGeneration(text=response.text, response=response)


def classify_outcome(
    outcome: ProviderTransportOutcome,
) -> ProviderGeneration | ProviderSemanticFailure:
    if isinstance(outcome, ProviderTransportResponse):
        return accept_provider_generation(outcome)
    return ProviderSemanticFailure(
        failure_class=_classify_transport_failure(outcome),
        message=outcome.message,
        transport_failure=outcome,
    )
