from __future__ import annotations

import pytest
from dr_providers import FailureClass, ProviderTransportOutcome

import whetstone.provider.driver as provider_driver
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
        generation = result.provider_generation
        assert generation is not None
        assert generation.text == "ok"
        assert len(transport.served) == 1
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
        generation = result.provider_generation
        assert generation is not None
        assert generation.text == "finally"
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
        result, transport, _ = _run(
            outcomes=[s.failure_outcome(failure_class=FailureClass.TRANSIENT)],
            max_attempts=3,
        )
        assert isinstance(result, ProviderCallResult)
        assert not result.succeeded
        assert result.provider_generation is None
        failure = result.semantic_failure
        assert isinstance(failure, ProviderSemanticFailure)
        assert failure.failure_class is SemanticFailureClass.TRANSPORT_ERROR
        assert result.attempt_count == 3
        assert len(transport.served) == 3

    def test_non_retryable_failure_stops_immediately(self) -> None:
        result, transport, sleeps = _run(
            outcomes=[s.failure_outcome(failure_class=FailureClass.PERMANENT)],
            max_attempts=5,
        )
        assert not result.succeeded
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
        assert not result.succeeded
        assert result.attempt_count == 1
        failure = result.semantic_failure
        assert isinstance(failure, ProviderSemanticFailure)
        assert (
            failure.failure_class
            is SemanticFailureClass.BLANK_PROVIDER_GENERATION
        )
        assert failure.rejected_response is not None

    def test_terminal_failure_equals_last_attempt(self) -> None:
        result, _, _ = _run(
            outcomes=[s.failure_outcome(failure_class=FailureClass.TRANSIENT)],
            max_attempts=2,
        )
        assert result.semantic_failure == result.attempts[-1].semantic_failure


class TestReplayDeterminism:
    def test_same_recorded_outcomes_produce_same_stable_payload(self) -> None:
        outcomes = [
            s.failure_outcome(failure_class=FailureClass.TRANSIENT),
            s.failure_outcome(failure_class=FailureClass.RATE_LIMITED),
            s.response_outcome(text="done"),
        ]
        first, _, _ = _run(outcomes=list(outcomes))
        second, _, _ = _run(outcomes=list(outcomes))
        assert first.to_stable_dict() == second.to_stable_dict()

    def test_attempt_sequence_has_same_stable_payload(self) -> None:
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
    def test_default_sleep_hook_receives_backoff(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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
        default_sleep_delays: list[float] = []
        monkeypatch.setattr(
            provider_driver, "_no_sleep", default_sleep_delays.append
        )

        result = provider_driver.run_provider_call(
            request=request,
            policy=policy,
            transport=transport,
            logical_call_id="lc-x",
            clock=s.FakeClock(),
        )
        assert result.succeeded
        assert default_sleep_delays == [1.0]

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


class TestTransportEvidenceIdentity:
    def test_rejects_evidence_for_a_foreign_request(self) -> None:
        request = s.build_request()
        transport_policy = s.build_transport_policy()
        policy = s.build_execution_policy(transport_policy=transport_policy)

        def foreign_request_transport(_request):
            return s.build_evidence(
                request=s.build_request(content="foreign"),
                policy=transport_policy,
                outcome=s.response_outcome(text="ok"),
            )

        with pytest.raises(ValueError, match="request identity"):
            run_provider_call(
                request=request,
                policy=policy,
                transport=foreign_request_transport,
                logical_call_id="lc-foreign-request",
                clock=s.FakeClock(),
            )

    def test_rejects_evidence_for_a_foreign_policy(self) -> None:
        request = s.build_request()
        transport_policy = s.build_transport_policy()
        policy = s.build_execution_policy(transport_policy=transport_policy)
        foreign_policy = s.build_transport_policy(
            base_url="https://foreign.example/v1"
        )

        def foreign_policy_transport(invoked_request):
            return s.build_evidence(
                request=invoked_request,
                policy=foreign_policy,
                outcome=s.response_outcome(text="ok"),
            )

        with pytest.raises(ValueError, match="policy identity"):
            run_provider_call(
                request=request,
                policy=policy,
                transport=foreign_policy_transport,
                logical_call_id="lc-foreign-policy",
                clock=s.FakeClock(),
            )
