from __future__ import annotations

from collections.abc import Callable

from dr_providers import (
    ProviderCallRequest,
    ProviderInvocationEvidence,
    ProviderTransportPolicy,
)
from dr_providers.outcomes.evidence import ProviderHttpRequestEvidence
from dr_providers.outcomes.models import ProviderTransportResponse

from whetstone.provider.driver import TransportCall
from whetstone.provider.policy import ProviderExecutionPolicy

__all__ = [
    "FakeLlmTransport",
    "fake_llm_transport",
    "fake_llm_transport_factory",
]


def fake_llm_transport(
    *,
    transport_policy: ProviderTransportPolicy,
    text_factory: Callable[[ProviderCallRequest], str] | None = None,
) -> TransportCall:
    """Return a deterministic transport that echoes the last user message."""

    def _transport(request: ProviderCallRequest) -> ProviderInvocationEvidence:
        if text_factory is not None:
            text = text_factory(request)
        else:
            messages = request.transcript.messages
            prompt = messages[-1].content if messages else ""
            text = f"generated: {prompt}"
        response = ProviderTransportResponse(text=text, stop_reason="stop")
        return ProviderInvocationEvidence.build(
            request=request,
            policy=transport_policy,
            http_request=ProviderHttpRequestEvidence(
                method="POST",
                url="http://whetstone.fake/llm",
                headers={},
                body={},
                body_bytes=0,
            ),
            outcome=response,
        )

    return _transport


class FakeLlmTransport:
    """Callable transport wrapper for sandbox graph previews."""

    def __init__(
        self,
        *,
        transport_policy: ProviderTransportPolicy,
        text_factory: Callable[[ProviderCallRequest], str] | None = None,
    ) -> None:
        self._transport = fake_llm_transport(
            transport_policy=transport_policy,
            text_factory=text_factory,
        )

    def __call__(
        self, request: ProviderCallRequest
    ) -> ProviderInvocationEvidence:
        return self._transport(request)


def fake_llm_transport_factory(
    policy: ProviderExecutionPolicy,
) -> FakeLlmTransport:
    return FakeLlmTransport(transport_policy=policy.transport_policy)
