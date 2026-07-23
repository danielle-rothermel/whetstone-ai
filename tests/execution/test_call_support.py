"""Pure-inspection helpers over a terminal Provider Call Result.

Covers the guard-deadline math (aligned to the new transport cap + 15s
semantics, NOT the old cap x max_attempts) and the transient-transport-failure
predicate that drives FIX 2's single bounded observation re-drive.
"""

from __future__ import annotations

from dr_providers import FailureClass, policy_for

from tests.provider import support as s
from whetstone.execution.call_support import (
    call_telemetry,
    guard_deadline_seconds,
    is_transient_transport_failure,
)
from whetstone.execution.fanout import GUARD_MARGIN_SECONDS
from whetstone.provider.attempt import ProviderCallResult
from whetstone.provider.driver import run_provider_call
from whetstone.provider.policy import BackoffSchedule, ProviderExecutionPolicy


def _policy(*, timeout_seconds: float, max_attempts: int) -> (
    ProviderExecutionPolicy
):
    return ProviderExecutionPolicy(
        transport_policy=policy_for(
            api_key_env="OPENROUTER_API_KEY",
            base_url="https://example.test/v1",
            timeout_seconds=timeout_seconds,
            native_retry_count=0,
        ),
        max_attempts=max_attempts,
    )


def test_guard_deadline_is_cap_plus_margin_not_retry_stacked() -> None:
    # The runner guard tracks the transport's SINGLE-call absolute cap plus the
    # fixed margin, independent of max_attempts. Under the old flat-deadline
    # model this was cap x max_attempts + 10 (600 x 3 + 10 = 1810), which let
    # stacked semantic retries trip the guard before the transport bound.
    policy = _policy(timeout_seconds=600.0, max_attempts=3)
    assert guard_deadline_seconds(policy) == 600.0 + GUARD_MARGIN_SECONDS
    assert GUARD_MARGIN_SECONDS == 15.0
    # max_attempts must NOT scale the guard anymore.
    policy_more_attempts = _policy(timeout_seconds=600.0, max_attempts=8)
    assert guard_deadline_seconds(policy_more_attempts) == (
        guard_deadline_seconds(policy)
    )


def _exhausted_transient() -> ProviderCallResult:
    request = s.build_request()
    transport_policy = s.build_transport_policy()
    policy = s.build_execution_policy(
        transport_policy=transport_policy,
        max_attempts=2,
        backoff=BackoffSchedule(base_seconds=0.0, max_seconds=0.0),
    )
    transport = s.RecordingTransport(
        request=request,
        transport_policy=transport_policy,
        outcomes=[
            s.failure_outcome(
                failure_class=FailureClass.TRANSIENT, message="503"
            ),
            s.failure_outcome(
                failure_class=FailureClass.TRANSIENT, message="503"
            ),
        ],
    )
    return run_provider_call(
        request=request, policy=policy, transport=transport,
        logical_call_id="lc-transient", clock=s.FakeClock(),
        sleep=s.SleepRecorder(),
    )


def _clean_rejection() -> ProviderCallResult:
    request = s.build_request()
    transport_policy = s.build_transport_policy()
    policy = s.build_execution_policy(
        transport_policy=transport_policy, max_attempts=1,
        backoff=BackoffSchedule(base_seconds=0.0, max_seconds=0.0),
    )
    transport = s.RecordingTransport(
        request=request,
        transport_policy=transport_policy,
        outcomes=[
            s.failure_outcome(
                failure_class=FailureClass.PERMANENT, message="400 bad request"
            )
        ],
    )
    return run_provider_call(
        request=request, policy=policy, transport=transport,
        logical_call_id="lc-permanent", clock=s.FakeClock(),
        sleep=s.SleepRecorder(),
    )


def test_transient_transport_failure_true_for_exhausted_transient() -> None:
    result = _exhausted_transient()
    assert is_transient_transport_failure(result)


def test_transient_transport_failure_false_for_clean_rejection() -> None:
    # A PERMANENT provider rejection is NOT transient: re-driving the same
    # request will not change a deterministic "no", so it must not re-drive.
    result = _clean_rejection()
    assert not is_transient_transport_failure(result)


def _success() -> ProviderCallResult:
    request = s.build_request()
    transport_policy = s.build_transport_policy()
    policy = s.build_execution_policy(
        transport_policy=transport_policy, max_attempts=1,
        backoff=BackoffSchedule(base_seconds=0.0, max_seconds=0.0),
    )
    transport = s.RecordingTransport(
        request=request,
        transport_policy=transport_policy,
        outcomes=[s.response_outcome(text="hi")],
    )
    return run_provider_call(
        request=request, policy=policy, transport=transport,
        logical_call_id="lc-ok", clock=s.FakeClock(), sleep=s.SleepRecorder(),
    )


def test_call_telemetry_carries_finish_reason_on_success() -> None:
    # Task 26 item 3: the accepted Generation's finish_reason is surfaced so a
    # truncated ``length`` is distinguishable from a clean ``stop``.
    tel = call_telemetry(_success())
    assert tel.finish_reason == "stop"
    assert tel.provider_error is None


def test_call_telemetry_persists_full_provider_error_on_failure() -> None:
    # Task 26 item 2: a failed call carries the FULL typed provider diagnostic,
    # not just a short code -- so a 400 malformed-request is reconstructable.
    tel = call_telemetry(_clean_rejection())
    assert tel.provider_error is not None
    assert tel.provider_error["failure_class"]
    assert "400 bad request" in str(tel.provider_error["message"])


def test_call_telemetry_all_none_for_absent_result() -> None:
    tel = call_telemetry(None)
    assert tel.finish_reason is None
    assert tel.provider_error is None
    assert tel.latency_s is None
