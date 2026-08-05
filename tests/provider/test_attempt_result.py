"""Provider Call Attempt + Provider Call Result schema invariants."""

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
from whetstone.provider.classification import Generation, classify_outcome


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
    is_gen = isinstance(classification, Generation)
    generation = classification if is_gen else None
    failure = None if is_gen else classification
    return ProviderCallAttempt(
        logical_call_id=logical_call_id,
        attempt_number=number,
        execution_policy_hash=policy_hash,
        started_at=0.0,
        ended_at=0.25,
        evidence=evidence,
        generation=generation,
        semantic_failure=failure,  # type: ignore[arg-type]
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
        # Exactly one Provider Invocation Evidence artifact.
        assert attempt.evidence is not None

    def test_exactly_one_classification_side(self) -> None:
        policy_hash = s.build_execution_policy().identity_hash
        evidence = s.build_evidence(
            request=s.build_request(),
            policy=s.build_transport_policy(),
            outcome=s.response_outcome(text="ok"),
        )
        gen = classify_outcome(s.response_outcome(text="ok"))
        with pytest.raises(ValueError, match="exactly one"):
            ProviderCallAttempt(
                logical_call_id="lc-1",
                attempt_number=1,
                execution_policy_hash=policy_hash,
                started_at=0.0,
                ended_at=1.0,
                evidence=evidence,
                generation=gen,
                semantic_failure=classify_outcome(
                    s.failure_outcome(failure_class=FailureClass.TRANSIENT)
                ),
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
                generation=classify_outcome(s.response_outcome(text="ok")),
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
        kwargs = {"started_at": 0.0, "ended_at": 1.0, field: value}
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
                generation=classify_outcome(response),
                **kwargs,
            )

    def test_success_evidence_rejects_failure_classification(self) -> None:
        policy_hash = s.build_execution_policy().identity_hash
        response = s.response_outcome(text="ok")
        failure = classify_outcome(
            s.failure_outcome(failure_class=FailureClass.TRANSIENT)
        )
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
                semantic_failure=failure,  # type: ignore[arg-type]
            )

    def test_failure_evidence_rejects_generation(self) -> None:
        policy_hash = s.build_execution_policy().identity_hash
        failure = s.failure_outcome(failure_class=FailureClass.TRANSIENT)
        generation = classify_outcome(s.response_outcome(text="ok"))
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
                generation=generation,  # type: ignore[arg-type]
            )


class TestProviderCallResult:
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
            request_identity=s.build_request().identity_payload(),
            execution_policy_hash=policy_hash,
            attempts=(a1, a2),
            generation=a2.generation,
        )
        assert result.succeeded
        assert result.attempt_count == 2

        rebuilt = ProviderCallResult.model_validate(
            result.model_dump(mode="json")
        )
        assert rebuilt == result

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
                request_identity={},
                execution_policy_hash=policy_hash,
                attempts=(a1, a3),
                generation=a3.generation,
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
                request_identity={},
                execution_policy_hash=policy_hash,
                attempts=(a1,),
                # Wrong: terminal claims success while last attempt failed.
                generation=classify_outcome(s.response_outcome(text="ok")),
            )

    def test_requires_at_least_one_attempt(self) -> None:
        policy_hash = s.build_execution_policy().identity_hash
        with pytest.raises(ValueError, match="at least one attempt"):
            ProviderCallResult(
                logical_call_id="lc-1",
                request_identity={},
                execution_policy_hash=policy_hash,
                attempts=(),
                semantic_failure=classify_outcome(
                    s.failure_outcome(failure_class=FailureClass.TRANSIENT)
                ),
            )

    def test_request_identity_must_match_every_attempt_evidence(self) -> None:
        policy_hash = s.build_execution_policy().identity_hash
        attempt = _attempt(
            number=1,
            outcome=s.response_outcome(text="ok"),
            policy_hash=policy_hash,
        )
        with pytest.raises(ValueError, match="request identity"):
            ProviderCallResult(
                logical_call_id="lc-1",
                request_identity=s.build_request(
                    content="foreign"
                ).identity_payload(),
                execution_policy_hash=policy_hash,
                attempts=(attempt,),
                generation=attempt.generation,
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
                request_identity=s.build_request().identity_payload(),
                execution_policy_hash=policy_hash,
                attempts=(a1, a2),
                generation=a2.generation,
            )
