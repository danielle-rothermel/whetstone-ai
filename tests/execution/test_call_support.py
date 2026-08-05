"""Provider-call inspection, provenance, and re-drive eligibility."""

from __future__ import annotations

import pytest
from dr_providers import FailureClass, ProviderKind, policy_for

from tests.provider import support as s
from whetstone.execution.call_support import (
    GUARD_MARGIN_SECONDS,
    call_telemetry,
    failure_code_of,
    guard_deadline_seconds,
    is_rate_limit_failure,
    is_transient_transport_failure,
)
from whetstone.provider.attempt import ProviderCallResult
from whetstone.provider.driver import run_provider_call
from whetstone.provider.policy import BackoffSchedule, ProviderExecutionPolicy


def _policy(
    *,
    timeout_seconds: float,
    max_attempts: int,
) -> ProviderExecutionPolicy:
    return ProviderExecutionPolicy(
        transport_policy=policy_for(
            ProviderKind.OPENROUTER,
            api_key_env="OPENROUTER_API_KEY",
            base_url="https://example.test/v1",
            timeout_seconds=timeout_seconds,
            native_retry_count=0,
        ),
        max_attempts=max_attempts,
    )


def _result(
    outcomes,
    *,
    logical_call_id: str,
) -> ProviderCallResult:
    request = s.build_request()
    transport_policy = s.build_transport_policy()
    policy = s.build_execution_policy(
        transport_policy=transport_policy,
        max_attempts=len(outcomes),
        backoff=BackoffSchedule(base_seconds=0.0, max_seconds=0.0),
    )
    transport = s.RecordingTransport(
        request=request,
        transport_policy=transport_policy,
        outcomes=outcomes,
    )
    return run_provider_call(
        request=request,
        policy=policy,
        transport=transport,
        logical_call_id=logical_call_id,
        clock=s.FakeClock(),
        sleep=s.SleepRecorder(),
    )


def test_guard_deadline_covers_attempts_backoff_and_wire_calls() -> None:
    policy = ProviderExecutionPolicy(
        transport_policy=policy_for(
            ProviderKind.OPENROUTER,
            api_key_env="OPENROUTER_API_KEY",
            base_url="https://example.test/v1",
            timeout_seconds=10.0,
            native_retry_count=0,
        ),
        max_attempts=4,
        backoff=BackoffSchedule(
            base_seconds=2.0,
            multiplier=2.0,
            max_seconds=5.0,
        ),
    )
    per_wire_call = 4 * 10.0 + (0.0 + 2.0 + 4.0 + 5.0)
    assert guard_deadline_seconds(policy) == (
        per_wire_call + GUARD_MARGIN_SECONDS
    )
    assert guard_deadline_seconds(policy, wire_calls_per_unit=2) == (
        2 * per_wire_call + GUARD_MARGIN_SECONDS
    )


def test_guard_deadline_rejects_nonpositive_wire_call_count() -> None:
    policy = _policy(timeout_seconds=10.0, max_attempts=1)
    with pytest.raises(ValueError, match="positive"):
        guard_deadline_seconds(policy, wire_calls_per_unit=0)


def test_only_transient_terminal_failures_are_redrive_eligible() -> None:
    transient = _result(
        [
            s.failure_outcome(
                failure_class=FailureClass.TRANSIENT,
                message="connection reset",
            )
        ],
        logical_call_id="transient",
    )
    permanent = _result(
        [
            s.failure_outcome(
                failure_class=FailureClass.PERMANENT,
                message="400 bad request",
            )
        ],
        logical_call_id="permanent",
    )
    success = _result(
        [s.response_outcome(text="ok")],
        logical_call_id="success",
    )
    assert is_transient_transport_failure(transient)
    assert not is_transient_transport_failure(permanent)
    assert not is_transient_transport_failure(success)


def test_rate_limit_detection_inspects_original_attempt_provenance() -> None:
    recovered = _result(
        [
            s.failure_outcome(
                failure_class=FailureClass.RATE_LIMITED,
                message="limited",
                status_code=429,
            ),
            s.response_outcome(text="ok"),
        ],
        logical_call_id="recovered",
    )
    assert recovered.succeeded
    assert is_rate_limit_failure(recovered)


def test_failure_code_and_full_provider_error_are_preserved() -> None:
    failed = _result(
        [
            s.failure_outcome(
                failure_class=FailureClass.PERMANENT,
                message="400 bad request",
            )
        ],
        logical_call_id="failed",
    )
    telemetry = call_telemetry(failed)
    assert failure_code_of(failed) == "provider-rejection"
    assert telemetry.provider_error is not None
    assert telemetry.provider_error["failure_class"] == "provider-rejection"
    assert "400 bad request" in str(telemetry.provider_error["message"])


def test_success_telemetry_and_absent_result_are_coverage_honest() -> None:
    successful = _result(
        [s.response_outcome(text="hello")],
        logical_call_id="successful",
    )
    telemetry = call_telemetry(successful)
    assert telemetry.finish_reason == "stop"
    assert telemetry.latency_s == 0.5
    assert telemetry.provider_error is None
    assert call_telemetry(None).latency_s is None
