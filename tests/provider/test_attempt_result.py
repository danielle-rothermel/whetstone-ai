from __future__ import annotations

import pytest
from dr_providers import (
    FailureClass,
    ProviderCallRequest,
    ProviderTransportPolicy,
)

from tests.provider import support as s
from whetstone.provider.attempt import (
    ProviderCallAttempt,
    ProviderCallResult,
)
from whetstone.provider.classification import (
    ProviderGeneration,
    ProviderSemanticFailure,
    classify_outcome,
)


def _accepted_generation() -> ProviderGeneration:
    classification = classify_outcome(s.response_outcome(text="ok"))
    assert isinstance(classification, ProviderGeneration)
    return classification


def _transient_failure() -> ProviderSemanticFailure:
    classification = classify_outcome(
        s.failure_outcome(failure_class=FailureClass.TRANSIENT)
    )
    assert isinstance(classification, ProviderSemanticFailure)
    return classification


def _attempt(
    *,
    number: int,
    outcome,
    policy_hash: str,
    logical_call_id: str = "lc-1",
    request: ProviderCallRequest | None = None,
    transport_policy: ProviderTransportPolicy | None = None,
) -> ProviderCallAttempt:
    request = request or s.build_request()
    evidence = s.build_evidence(
        request=request,
        policy=transport_policy or s.build_transport_policy(),
        outcome=outcome,
    )
    classification = classify_outcome(outcome)
    if isinstance(classification, ProviderGeneration):
        generation = classification
        failure = None
    else:
        generation = None
        failure = classification
    return ProviderCallAttempt(
        logical_call_id=logical_call_id,
        attempt_number=number,
        execution_policy_hash=policy_hash,
        started_at=0.0,
        ended_at=0.25,
        evidence=evidence,
        provider_generation=generation,
        semantic_failure=failure,
    )


class TestProviderCallAttempt:
    def test_wrapper_carries_required_identity_fields(self) -> None:
        policy_hash = s.build_execution_policy().identity_hash
        attempt = _attempt(
            number=1,
            outcome=s.response_outcome(text="ok"),
            policy_hash=policy_hash,
        )
        assert attempt.logical_call_id == "lc-1"
        assert attempt.attempt_number == 1
        assert attempt.execution_policy_hash == policy_hash
        assert attempt.succeeded
        assert attempt.latency_ms == 250
        assert attempt.evidence is not None
        stable = attempt.to_stable_dict()
        assert stable["schema_version"] == 2
        assert "schema_version" not in attempt.evidence.identity_payload()

    def test_rejects_v1_schema(self) -> None:
        policy_hash = s.build_execution_policy().identity_hash
        attempt = _attempt(
            number=1,
            outcome=s.response_outcome(text="ok"),
            policy_hash=policy_hash,
        )
        with pytest.raises(ValueError, match="schema_version"):
            ProviderCallAttempt.model_validate(
                {**attempt.to_stable_dict(), "schema_version": 1}
            )

    def test_exactly_one_classification_side(self) -> None:
        policy_hash = s.build_execution_policy().identity_hash
        evidence = s.build_evidence(
            request=s.build_request(),
            policy=s.build_transport_policy(),
            outcome=s.response_outcome(text="ok"),
        )
        gen = _accepted_generation()
        with pytest.raises(ValueError, match="exactly one"):
            ProviderCallAttempt(
                logical_call_id="lc-1",
                attempt_number=1,
                execution_policy_hash=policy_hash,
                started_at=0.0,
                ended_at=1.0,
                evidence=evidence,
                provider_generation=gen,
                semantic_failure=_transient_failure(),
            )

    def test_rejects_short_policy_hash(self) -> None:
        with pytest.raises(ValueError, match="64-char"):
            _attempt(
                number=1,
                outcome=s.response_outcome(text="ok"),
                policy_hash="abc",
            )

    @pytest.mark.parametrize(
        "policy_hash",
        ["g" * 64, "A" * 64],
    )
    def test_rejects_noncanonical_policy_hash(self, policy_hash: str) -> None:
        with pytest.raises(ValueError, match="lowercase SHA-256"):
            _attempt(
                number=1,
                outcome=s.response_outcome(text="ok"),
                policy_hash=policy_hash,
            )

    def test_rejects_backwards_timing(self) -> None:
        policy_hash = s.build_execution_policy().identity_hash
        evidence = s.build_evidence(
            request=s.build_request(),
            policy=s.build_transport_policy(),
            outcome=s.response_outcome(text="ok"),
        )
        with pytest.raises(ValueError, match="ended_at"):
            ProviderCallAttempt(
                logical_call_id="lc-1",
                attempt_number=1,
                execution_policy_hash=policy_hash,
                started_at=2.0,
                ended_at=1.0,
                evidence=evidence,
                provider_generation=_accepted_generation(),
            )

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("started_at", float("nan")),
            ("started_at", float("inf")),
            ("started_at", float("-inf")),
            ("ended_at", float("nan")),
            ("ended_at", float("inf")),
            ("ended_at", float("-inf")),
        ],
    )
    def test_rejects_nonfinite_timing(self, field: str, value: float) -> None:
        policy_hash = s.build_execution_policy().identity_hash
        response = s.response_outcome(text="ok")
        started_at = value if field == "started_at" else 0.0
        ended_at = value if field == "ended_at" else 1.0
        with pytest.raises(ValueError, match=rf"{field} must be finite"):
            ProviderCallAttempt(
                logical_call_id="lc-1",
                attempt_number=1,
                execution_policy_hash=policy_hash,
                evidence=s.build_evidence(
                    request=s.build_request(),
                    policy=s.build_transport_policy(),
                    outcome=response,
                ),
                started_at=started_at,
                ended_at=ended_at,
                provider_generation=_accepted_generation(),
            )

    def test_success_evidence_rejects_failure_classification(self) -> None:
        policy_hash = s.build_execution_policy().identity_hash
        response = s.response_outcome(text="ok")
        failure = _transient_failure()
        with pytest.raises(ValueError, match="exactly reclassify"):
            ProviderCallAttempt(
                logical_call_id="lc-1",
                attempt_number=1,
                execution_policy_hash=policy_hash,
                started_at=0.0,
                ended_at=1.0,
                evidence=s.build_evidence(
                    request=s.build_request(),
                    policy=s.build_transport_policy(),
                    outcome=response,
                ),
                semantic_failure=failure,
            )

    def test_failure_evidence_rejects_generation(self) -> None:
        policy_hash = s.build_execution_policy().identity_hash
        failure = s.failure_outcome(failure_class=FailureClass.TRANSIENT)
        generation = _accepted_generation()
        with pytest.raises(ValueError, match="exactly reclassify"):
            ProviderCallAttempt(
                logical_call_id="lc-1",
                attempt_number=1,
                execution_policy_hash=policy_hash,
                started_at=0.0,
                ended_at=1.0,
                evidence=s.build_evidence(
                    request=s.build_request(),
                    policy=s.build_transport_policy(),
                    outcome=failure,
                ),
                provider_generation=generation,
            )


class TestProviderCallResult:
    @pytest.mark.precheck
    def test_ordered_attempts_and_terminal_success(self) -> None:
        policy_hash = s.build_execution_policy().identity_hash
        a1 = _attempt(
            number=1,
            outcome=s.failure_outcome(failure_class=FailureClass.TRANSIENT),
            policy_hash=policy_hash,
        )
        a2 = _attempt(
            number=2,
            outcome=s.response_outcome(text="ok"),
            policy_hash=policy_hash,
        )
        result = ProviderCallResult(
            logical_call_id="lc-1",
            request_hash=s.build_request().identity_payload(),
            execution_policy_hash=policy_hash,
            attempts=(a1, a2),
            provider_generation=a2.provider_generation,
        )
        assert result.succeeded
        assert result.attempt_count == 2

        stable = result.to_stable_dict()
        assert stable["schema_version"] == 2
        assert all(
            attempt["schema_version"] == 2 for attempt in stable["attempts"]
        )
        rebuilt = ProviderCallResult.model_validate(stable)
        assert rebuilt == result

    def test_rejects_v1_schema(self) -> None:
        policy_hash = s.build_execution_policy().identity_hash
        attempt = _attempt(
            number=1,
            outcome=s.response_outcome(text="ok"),
            policy_hash=policy_hash,
        )
        result = ProviderCallResult(
            logical_call_id="lc-1",
            request_hash=s.build_request().identity_payload(),
            execution_policy_hash=policy_hash,
            attempts=(attempt,),
            provider_generation=attempt.provider_generation,
        )
        with pytest.raises(ValueError, match="schema_version"):
            ProviderCallResult.model_validate(
                {**result.to_stable_dict(), "schema_version": 1}
            )

    def test_requires_contiguous_attempt_numbers(self) -> None:
        policy_hash = s.build_execution_policy().identity_hash
        a1 = _attempt(
            number=1,
            outcome=s.response_outcome(text="ok"),
            policy_hash=policy_hash,
        )
        a3 = _attempt(
            number=3,
            outcome=s.response_outcome(text="ok"),
            policy_hash=policy_hash,
        )
        with pytest.raises(ValueError, match="ordered 1"):
            ProviderCallResult(
                logical_call_id="lc-1",
                request_hash={},
                execution_policy_hash=policy_hash,
                attempts=(a1, a3),
                provider_generation=a3.provider_generation,
            )

    def test_terminal_must_match_last_attempt(self) -> None:
        policy_hash = s.build_execution_policy().identity_hash
        a1 = _attempt(
            number=1,
            outcome=s.failure_outcome(failure_class=FailureClass.TRANSIENT),
            policy_hash=policy_hash,
        )
        with pytest.raises(ValueError, match="final attempt"):
            ProviderCallResult(
                logical_call_id="lc-1",
                request_hash={},
                execution_policy_hash=policy_hash,
                attempts=(a1,),
                provider_generation=_accepted_generation(),
            )

    def test_requires_at_least_one_attempt(self) -> None:
        policy_hash = s.build_execution_policy().identity_hash
        with pytest.raises(ValueError, match="at least one attempt"):
            ProviderCallResult(
                logical_call_id="lc-1",
                request_hash={},
                execution_policy_hash=policy_hash,
                attempts=(),
                semantic_failure=_transient_failure(),
            )

    def test_request_hash_must_match_every_attempt_evidence(self) -> None:
        policy_hash = s.build_execution_policy().identity_hash
        attempt = _attempt(
            number=1,
            outcome=s.response_outcome(text="ok"),
            policy_hash=policy_hash,
        )
        with pytest.raises(ValueError, match="request identity"):
            ProviderCallResult(
                logical_call_id="lc-1",
                request_hash=s.build_request(
                    content="foreign"
                ).identity_payload(),
                execution_policy_hash=policy_hash,
                attempts=(attempt,),
                provider_generation=attempt.provider_generation,
            )

    def test_attempt_evidence_policy_identities_must_agree(self) -> None:
        policy_hash = s.build_execution_policy().identity_hash
        a1 = _attempt(
            number=1,
            outcome=s.failure_outcome(failure_class=FailureClass.TRANSIENT),
            policy_hash=policy_hash,
        )
        a2 = _attempt(
            number=2,
            outcome=s.response_outcome(text="ok"),
            policy_hash=policy_hash,
            transport_policy=s.build_transport_policy(
                base_url="https://foreign.example/v1"
            ),
        )
        with pytest.raises(ValueError, match="policy identity"):
            ProviderCallResult(
                logical_call_id="lc-1",
                request_hash=s.build_request().identity_payload(),
                execution_policy_hash=policy_hash,
                attempts=(a1, a2),
                provider_generation=a2.provider_generation,
            )
