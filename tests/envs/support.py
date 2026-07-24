"""Shared fixtures for the whetstone env-adapter tests.

Builds a scripted fake transport (no network, no DBOS, no live paid call)
that returns a per-request generation text so the internal-eval loop can be
driven deterministically, plus small helpers to construct env pools and
execution policies.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from dr_providers import (
    ProviderCallRequest,
    ProviderInvocationEvidence,
    ProviderKind,
    ProviderTransportPolicy,
    ProviderTransportResponse,
    RawHttpRequest,
    policy_for,
)

from whetstone.provider.policy import ProviderExecutionPolicy

API_KEY_ENV = "OPENROUTER_API_KEY"

#: A text-returning callable keyed on the request's user-message content
#: (the rendered prompt), so a fake can answer differently per task.
ReplyFn = Callable[[str], str]


def transport_policy() -> ProviderTransportPolicy:
    return policy_for(
        ProviderKind.OPENROUTER,
        api_key_env=API_KEY_ENV,
        base_url="https://example.test/v1",
        native_retry_count=0,
    )


def execution_policy(*, max_attempts: int = 1) -> ProviderExecutionPolicy:
    return ProviderExecutionPolicy(
        transport_policy=transport_policy(),
        max_attempts=max_attempts,
    )


def _response(text: str) -> ProviderTransportResponse:
    return ProviderTransportResponse(
        text=text,
        raw_body={"choices": [{"message": {"content": text}}]},
        response_id="resp-1",
        model="test-model",
        finish_reason="stop",
    )


def _prompt_of(request: ProviderCallRequest) -> str:
    messages = request.transcript.messages
    return messages[-1].content if messages else ""


@dataclass
class FakeTransport:
    """A scripted transport: maps a rendered prompt to a reply text.

    ``reply`` is a pure function of the rendered prompt (the user message),
    so the fake can return the correct answer for one task and a wrong answer
    for another. Records every request it served for determinism assertions.
    No network is touched.
    """

    reply: ReplyFn
    policy: ProviderTransportPolicy = field(default_factory=transport_policy)
    served: list[ProviderCallRequest] = field(default_factory=list)

    def __call__(
        self, request: ProviderCallRequest
    ) -> ProviderInvocationEvidence:
        self.served.append(request)
        text = self.reply(_prompt_of(request))
        raw_request = RawHttpRequest.build(
            url="https://example.test/v1/chat/completions",
            headers={"Authorization": "Bearer k", "content-type": "json"},
            body={"model": "test-model"},
        )
        return ProviderInvocationEvidence.build(
            request=request,
            policy=self.policy,
            raw_request=raw_request,
            outcome=_response(text),
        )


def constant_reply(text: str) -> ReplyFn:
    """A reply function returning ``text`` for every prompt."""

    def _reply(_prompt: str) -> str:
        return text

    return _reply
