"""Pure attempt-loop driver: bounded, deterministic, replay-identical."""

from __future__ import annotations

import pytest
from dr_providers import FailureClass, ProviderTransportOutcome

from tests.provider import support as s
from whetstone.provider.attempt import ProviderCallResult
from whetstone.provider.classification import (
    ProviderSemanticFailure,
    SemanticFailureClass,
)
from whetstone.provider.driver import Clock, run_provider_call
from whetstone.provider.policy import BackoffSchedule


def _run(
    *,
    outcomes: list[ProviderTransportOutcome],
    max_attempts: int = 3,
    retry_eligibility: dict[SemanticFailureClass, bool] | None = None,
    clock: Clock | None = None,
) -> tuple[ProviderCallResult, s.RecordingTransport, s.SleepRecorder]:
    request = s.build_request()
    transport_policy = s.build_transport_policy()
    policy = s.build_execution_policy(
        max_attempts=max_attempts,
        transport_policy=transport_policy,
        backoff=BackoffSchedule(base_seconds=1.0, multiplier=2.0),
        retry_eligibility=retry_eligibility,
    )
    transport = s.RecordingTransport(
        request=request,
        transport_policy=transport_policy,
        outcomes=outcomes,
    )
    sleep_recorder = s.SleepRecorder()
    result = run_provider_call(
        request=request,
        policy=policy,
        transport=transport,
        logical_call_id="lc-1",
        clock=clock or s.FakeClock(),
        sleep=sleep_recorder,
    )
    return result, transport, sleep_recorder


class TestSuccessPath:
    def test_first_attempt_success_stops(self) -> None:
        result, transport, sleeps = _run(
            outcomes=[s.response_outcome(text="ok")]
        )
        assert result.succeeded
        assert result.attempt_count == 1
        generation = result.generation
        assert generation is not None
        assert generation.text == "ok"
        assert len(transport.served) == 1
        # No backoff before the first attempt.
        assert sleeps.delays == []

    def test_retries_until_success(self) -> None:
        result, _transport, sleeps = _run(
            outcomes=[
                s.failure_outcome(failure_class=FailureClass.TRANSIENT),
                s.failure_outcome(failure_class=FailureClass.RATE_LIMITED),
                s.response_outcome(text="finally"),
            ]
        )
        assert result.succeeded
        assert result.attempt_count == 3
        generation = result.generation
        assert generation is not None
        assert generation.text == "finally"
        # Backoff taken before attempts 2 and 3 only.
        assert sleeps.delays == [1.0, 2.0]

    def test_attempts_are_ordered_and_share_identity(self) -> None:
        result, _, _ = _run(
            outcomes=[
                s.failure_outcome(failure_class=FailureClass.TRANSIENT),
                s.response_outcome(text="ok"),
            ]
        )
        for index, attempt in enumerate(result.attempts, start=1):
            assert attempt.attempt_number == index
            assert attempt.logical_call_id == "lc-1"
            assert (
                attempt.execution_policy_hash == result.execution_policy_hash
            )


class TestExhaustionIsExpectedOutput:
    def test_exhausted_failure_is_a_valid_terminal_result(self) -> None:
        # Exhaustion is EXPECTED domain output: a valid Result, not a raise.
        result, transport, _ = _run(
            outcomes=[s.failure_outcome(failure_class=FailureClass.TRANSIENT)],
            max_attempts=3,
        )
        assert isinstance(result, ProviderCallResult)
        assert not result.succeeded
        assert result.generation is None
        failure = result.semantic_failure
        assert isinstance(failure, ProviderSemanticFailure)
        assert failure.failure_class is SemanticFailureClass.TRANSPORT_ERROR
        # Bound was reached: exactly max_attempts physical calls.
        assert result.attempt_count == 3
        assert len(transport.served) == 3

    def test_non_retryable_failure_stops_immediately(self) -> None:
        result, transport, sleeps = _run(
            outcomes=[s.failure_outcome(failure_class=FailureClass.PERMANENT)],
            max_attempts=5,
        )
        assert not result.succeeded
        # provider-rejection is not retry-eligible: stop after one attempt.
        assert result.attempt_count == 1
        assert len(transport.served) == 1
        assert sleeps.delays == []
        failure = result.semantic_failure
        assert isinstance(failure, ProviderSemanticFailure)
        assert failure.failure_class is SemanticFailureClass.PROVIDER_REJECTION

    def test_blank_generation_exhausts_by_default(self) -> None:
        result, _, _ = _run(
            outcomes=[s.response_outcome(text="   ")], max_attempts=3
        )
        # Blank generation is not retry-eligible by default: one attempt.
        assert not result.succeeded
        assert result.attempt_count == 1
        failure = result.semantic_failure
        assert isinstance(failure, ProviderSemanticFailure)
        assert failure.failure_class is SemanticFailureClass.BLANK_GENERATION
        # The rejected response is retained on the terminal failure.
        assert failure.rejected_response is not None

    def test_terminal_failure_equals_last_attempt(self) -> None:
        result, _, _ = _run(
            outcomes=[s.failure_outcome(failure_class=FailureClass.TRANSIENT)],
            max_attempts=2,
        )
        assert result.semantic_failure == result.attempts[-1].semantic_failure


class TestReplayDeterminism:
    def test_same_recorded_outcomes_produce_identical_sequence(self) -> None:
        outcomes = [
            s.failure_outcome(failure_class=FailureClass.TRANSIENT),
            s.failure_outcome(failure_class=FailureClass.RATE_LIMITED),
            s.response_outcome(text="done"),
        ]
        first, _, _ = _run(outcomes=list(outcomes))
        second, _, _ = _run(outcomes=list(outcomes))
        # Byte-identical stable serialization of the whole Result.
        assert first.to_stable_dict() == second.to_stable_dict()

    def test_attempt_sequence_is_byte_identical(self) -> None:
        outcomes = [
            s.failure_outcome(failure_class=FailureClass.TRANSIENT),
            s.response_outcome(text="x"),
        ]
        first, _, _ = _run(outcomes=list(outcomes))
        second, _, _ = _run(outcomes=list(outcomes))
        assert [a.to_stable_dict() for a in first.attempts] == [
            a.to_stable_dict() for a in second.attempts
        ]

    def test_backoff_decisions_are_deterministic(self) -> None:
        outcomes = [
            s.failure_outcome(failure_class=FailureClass.TRANSIENT),
            s.failure_outcome(failure_class=FailureClass.TRANSIENT),
            s.response_outcome(text="x"),
        ]
        _, _, first_sleeps = _run(outcomes=list(outcomes))
        _, _, second_sleeps = _run(outcomes=list(outcomes))
        assert first_sleeps.delays == second_sleeps.delays


class TestInjectableHooks:
    def test_default_sleep_is_noop_no_blocking(self) -> None:
        # Without a sleep hook the pure driver never blocks (no wall clock).
        request = s.build_request()
        transport_policy = s.build_transport_policy()
        policy = s.build_execution_policy(
            transport_policy=transport_policy, max_attempts=2
        )
        transport = s.RecordingTransport(
            request=request,
            transport_policy=transport_policy,
            outcomes=[
                s.failure_outcome(failure_class=FailureClass.TRANSIENT),
                s.response_outcome(text="ok"),
            ],
        )
        # No clock, no sleep injected -> uses real monotonic; must not hang.
        result = run_provider_call(
            request=request,
            policy=policy,
            transport=transport,
            logical_call_id="lc-x",
        )
        assert result.succeeded

    def test_timing_recorded_from_injected_clock(self) -> None:
        result, _, _ = _run(
            outcomes=[s.response_outcome(text="ok")],
            clock=s.FakeClock(step=0.5),
        )
        attempt = result.attempts[0]
        assert attempt.latency_seconds == pytest.approx(0.5)
        assert attempt.latency_ms == 500

    def test_empty_logical_call_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="logical_call_id"):
            run_provider_call(
                request=s.build_request(),
                policy=s.build_execution_policy(),
                transport=s.RecordingTransport(
                    request=s.build_request(),
                    transport_policy=s.build_transport_policy(),
                    outcomes=[s.response_outcome(text="ok")],
                ),
                logical_call_id="",
            )
