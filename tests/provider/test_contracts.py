from __future__ import annotations

from typing import Any

from dr_providers import FailureClass

from tests.provider import support as s
from whetstone.provider.attempt import ProviderCallResult
from whetstone.provider.driver import run_provider_call
from whetstone.provider.policy import BackoffSchedule

LONG_BODY_TEXT = "Z" * 5000
SECRET_TOKEN = "sk-super-secret-credential-value-1234567890"


def _run_exhausting() -> tuple[ProviderCallResult, s.RecordingTransport]:
    request = s.build_request()
    transport_policy = s.build_transport_policy()
    policy = s.build_execution_policy(
        transport_policy=transport_policy,
        max_attempts=3,
        backoff=BackoffSchedule(base_seconds=0.0, max_seconds=0.0),
    )
    transport = s.RecordingTransport(
        request=request,
        transport_policy=transport_policy,
        outcomes=[
            s.failure_outcome(
                failure_class=FailureClass.TRANSIENT,
                message="upstream 503",
            )
        ],
    )
    result = run_provider_call(
        request=request,
        policy=policy,
        transport=transport,
        logical_call_id="lc-exhaust",
        clock=s.FakeClock(),
        sleep=s.SleepRecorder(),
    )
    return result, transport


def _iter_strings(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, str):
        found.append(value)
    elif isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str):
                found.append(key)
            found.extend(_iter_strings(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.extend(_iter_strings(item))
    return found


class TestExhaustionIsExpectedDomainOutput:
    def test_exhaustion_returns_a_valid_result_not_an_exception(self) -> None:
        result, _ = _run_exhausting()
        assert isinstance(result, ProviderCallResult)
        assert not result.succeeded
        assert result.semantic_failure is not None
        rebuilt = ProviderCallResult.model_validate(result.to_stable_dict())
        assert rebuilt.to_stable_dict() == result.to_stable_dict()


class TestNoCredentialMaterialInPersistedShapes:
    def test_no_secret_token_anywhere_in_persisted_result(self) -> None:
        request = s.build_request()
        transport_policy = s.build_transport_policy()
        policy = s.build_execution_policy(transport_policy=transport_policy)

        def leaky_transport(req):
            from dr_providers import (
                ProviderHttpRequestEvidence,
                ProviderInvocationEvidence,
            )

            http_request = ProviderHttpRequestEvidence.build(
                url="https://example.test/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {SECRET_TOKEN}",
                    "x-api-key": SECRET_TOKEN,
                },
                body={"model": "test-model"},
            )
            return ProviderInvocationEvidence.build(
                request=req,
                policy=transport_policy,
                http_request=http_request,
                outcome=s.response_outcome(text="ok"),
            )

        result = run_provider_call(
            request=request,
            policy=policy,
            transport=leaky_transport,
            logical_call_id="lc-cred",
            clock=s.FakeClock(),
        )
        strings = _iter_strings(result.to_stable_dict())
        assert all(SECRET_TOKEN not in text for text in strings)
        assert not any(text.startswith("Bearer ") for text in strings)
        assert "<redacted>" in strings


class TestEvidenceNeverTruncated:
    def test_full_response_body_survives_into_persisted_attempt(self) -> None:
        request = s.build_request()
        transport_policy = s.build_transport_policy()
        policy = s.build_execution_policy(transport_policy=transport_policy)
        transport = s.RecordingTransport(
            request=request,
            transport_policy=transport_policy,
            outcomes=[s.response_outcome(text=LONG_BODY_TEXT)],
        )
        result = run_provider_call(
            request=request,
            policy=policy,
            transport=transport,
            logical_call_id="lc-notrunc",
            clock=s.FakeClock(),
        )
        stable = result.to_stable_dict()
        blob = str(stable)
        assert LONG_BODY_TEXT in blob
        assert result.provider_generation is not None
        assert result.provider_generation.text == LONG_BODY_TEXT
        assert len(result.provider_generation.text) == 5000

    def test_full_failure_body_survives_into_persisted_attempt(self) -> None:
        request = s.build_request()
        transport_policy = s.build_transport_policy()
        policy = s.build_execution_policy(
            transport_policy=transport_policy,
            max_attempts=1,
        )
        long_message = "E" * 4096
        transport = s.RecordingTransport(
            request=request,
            transport_policy=transport_policy,
            outcomes=[
                s.failure_outcome(
                    failure_class=FailureClass.PERMANENT,
                    message=long_message,
                )
            ],
        )
        result = run_provider_call(
            request=request,
            policy=policy,
            transport=transport,
            logical_call_id="lc-failbody",
            clock=s.FakeClock(),
        )
        blob = str(result.to_stable_dict())
        assert long_message in blob
