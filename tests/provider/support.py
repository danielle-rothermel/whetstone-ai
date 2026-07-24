"""Shared fixtures for the Whetstone semantic provider-layer tests.

Builds real dr-providers Provider Call Requests, real Provider Transport
Policies / Whetstone Provider Execution Policies, and a recording transport
peer that returns real Provider Invocation Evidence from a scripted list of
Provider Transport Outcomes. No network, no DBOS.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dr_providers import (
    FailureClass,
    MessageRole,
    PromptMessage,
    ProviderCallRequest,
    ProviderInvocationEvidence,
    ProviderKind,
    ProviderTransportFailure,
    ProviderTransportOutcome,
    ProviderTransportPolicy,
    ProviderTransportResponse,
    RawHttpRequest,
    Transcript,
    openrouter_chat_config,
    policy_for,
)

from whetstone.provider.policy import (
    BackoffSchedule,
    ProviderExecutionPolicy,
)

API_KEY_ENV = "OPENROUTER_API_KEY"


def build_request(*, content: str = "hello") -> ProviderCallRequest:
    return ProviderCallRequest(
        config=openrouter_chat_config(model="test-model"),
        transcript=Transcript(
            messages=(PromptMessage(role=MessageRole.USER, content=content),)
        ),
    )


def build_transport_policy(
    *, native_retry_count: int = 0
) -> ProviderTransportPolicy:
    return policy_for(
        ProviderKind.OPENROUTER,
        api_key_env=API_KEY_ENV,
        base_url="https://example.test/v1",
        native_retry_count=native_retry_count,
    )


def build_execution_policy(
    *,
    max_attempts: int = 3,
    transport_policy: ProviderTransportPolicy | None = None,
    backoff: BackoffSchedule | None = None,
    retry_eligibility: dict | None = None,
) -> ProviderExecutionPolicy:
    kwargs: dict = {
        "transport_policy": transport_policy or build_transport_policy(),
        "max_attempts": max_attempts,
    }
    if backoff is not None:
        kwargs["backoff"] = backoff
    if retry_eligibility is not None:
        kwargs["retry_eligibility"] = retry_eligibility
    return ProviderExecutionPolicy(**kwargs)


def response_outcome(*, text: str) -> ProviderTransportResponse:
    return ProviderTransportResponse(
        text=text,
        raw_body={"choices": [{"message": {"content": text}}]},
        response_id="resp-1",
        model="test-model",
        finish_reason="stop",
    )


def failure_outcome(
    *,
    failure_class: FailureClass,
    message: str = "transport failed",
    status_code: int | None = None,
) -> ProviderTransportFailure:
    return ProviderTransportFailure(
        failure_class=failure_class,
        message=message,
        retryable=failure_class
        in (FailureClass.TRANSIENT, FailureClass.RATE_LIMITED),
        raw_request={"model": "test-model"},
        raw_response_body={"error": message},
        status_code=status_code,
    )


def build_evidence(
    *,
    request: ProviderCallRequest,
    policy: ProviderTransportPolicy,
    outcome: ProviderTransportOutcome,
) -> ProviderInvocationEvidence:
    raw_request = RawHttpRequest.build(
        url="https://example.test/v1/chat/completions",
        headers={"Authorization": "Bearer test-key", "content-type": "json"},
        body={"model": "test-model"},
    )
    return ProviderInvocationEvidence.build(
        request=request,
        policy=policy,
        raw_request=raw_request,
        outcome=outcome,
    )


@dataclass
class RecordingTransport:
    """A transport callable that replays scripted outcomes as evidence.

    Consumes ``outcomes`` in order (the last repeats). Records every request
    it served and every evidence it produced so tests can assert determinism.
    """

    request: ProviderCallRequest
    transport_policy: ProviderTransportPolicy
    outcomes: list[ProviderTransportOutcome]
    served: list[ProviderCallRequest] = field(default_factory=list)
    produced: list[ProviderInvocationEvidence] = field(default_factory=list)

    def __call__(
        self, request: ProviderCallRequest
    ) -> ProviderInvocationEvidence:
        index = min(len(self.served), len(self.outcomes) - 1)
        outcome = self.outcomes[index]
        self.served.append(request)
        evidence = build_evidence(
            request=request,
            policy=self.transport_policy,
            outcome=outcome,
        )
        self.produced.append(evidence)
        return evidence


@dataclass
class FakeClock:
    """Deterministic monotonic clock: increments a fixed step per read."""

    step: float = 0.5
    _t: float = 0.0

    def __call__(self) -> float:
        value = self._t
        self._t += self.step
        return value


@dataclass
class SleepRecorder:
    """Records every backoff delay requested without actually sleeping."""

    delays: list[float] = field(default_factory=list)

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)
