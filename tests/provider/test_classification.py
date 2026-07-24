"""Semantic failure taxonomy + Generation acceptance contract tests."""

from __future__ import annotations

import pytest
from dr_providers import (
    FailureClass,
    ProviderTransportFailure,
    ProviderTransportResponse,
)

from tests.provider import support as s
from whetstone.provider.classification import (
    Generation,
    ProviderSemanticFailure,
    SemanticFailureClass,
    accept_generation,
    classify_outcome,
    is_blank,
)


class TestGenerationAcceptance:
    def test_nonblank_response_projects_a_generation(self) -> None:
        response = s.response_outcome(text="the answer")
        result = accept_generation(response)
        assert isinstance(result, Generation)
        assert result.text == "the answer"
        # Retains the full causal transport response as provenance.
        assert result.response is response

    @pytest.mark.parametrize("text", ["", "   ", "\n\t ", "\r\n"])
    def test_blank_response_is_a_semantic_failure(self, text: str) -> None:
        response = s.response_outcome(text=text)
        result = accept_generation(response)
        assert isinstance(result, ProviderSemanticFailure)
        assert result.failure_class is SemanticFailureClass.BLANK_GENERATION
        # The rejected response is retained, never discarded.
        assert result.rejected_response is response
        assert result.transport_failure is None

    def test_generation_cannot_be_constructed_blank(self) -> None:
        with pytest.raises(ValueError, match="nonblank"):
            Generation(text="  ", response=s.response_outcome(text="x"))

    def test_is_blank(self) -> None:
        assert is_blank(None)
        assert is_blank("")
        assert is_blank("  \n")
        assert not is_blank("x")


class TestFailureTaxonomyIsClosedAndTotal:
    def test_every_transport_failure_class_maps(self) -> None:
        # Classification is total over the FailureClass enum.
        for failure_class in FailureClass:
            outcome = s.failure_outcome(failure_class=failure_class)
            result = classify_outcome(outcome)
            assert isinstance(result, ProviderSemanticFailure)
            assert result.failure_class in set(SemanticFailureClass)

    @pytest.mark.parametrize(
        ("failure_class", "expected"),
        [
            (FailureClass.RATE_LIMITED, SemanticFailureClass.RATE_LIMIT),
            (
                FailureClass.RESOURCE_EXHAUSTION,
                SemanticFailureClass.RATE_LIMIT,
            ),
            (FailureClass.TRANSIENT, SemanticFailureClass.TRANSPORT_ERROR),
            (
                FailureClass.PERMANENT,
                SemanticFailureClass.PROVIDER_REJECTION,
            ),
            (FailureClass.UNKNOWN, SemanticFailureClass.MALFORMED_RESPONSE),
        ],
    )
    def test_deterministic_class_mapping(
        self, failure_class: FailureClass, expected: SemanticFailureClass
    ) -> None:
        outcome = s.failure_outcome(failure_class=failure_class)
        result = classify_outcome(outcome)
        assert isinstance(result, ProviderSemanticFailure)
        assert result.failure_class is expected

    def test_timeout_status_wins_over_class(self) -> None:
        outcome = s.failure_outcome(
            failure_class=FailureClass.TRANSIENT, status_code=408
        )
        result = classify_outcome(outcome)
        assert isinstance(result, ProviderSemanticFailure)
        assert result.failure_class is SemanticFailureClass.TIMEOUT

    def test_transport_failure_retains_causal_evidence(self) -> None:
        outcome = s.failure_outcome(
            failure_class=FailureClass.TRANSIENT, message="reset"
        )
        result = classify_outcome(outcome)
        assert isinstance(result, ProviderSemanticFailure)
        assert result.transport_failure is outcome
        assert result.rejected_response is None
        assert result.message == "reset"

    def test_classification_is_deterministic(self) -> None:
        outcome = s.failure_outcome(failure_class=FailureClass.RATE_LIMITED)
        first = classify_outcome(outcome)
        second = classify_outcome(outcome)
        assert first.model_dump() == second.model_dump()


class TestSemanticFailureInvariants:
    def test_requires_exactly_one_causal_side(self) -> None:
        response = ProviderTransportResponse(text="x")
        failure = ProviderTransportFailure(
            failure_class=FailureClass.TRANSIENT,
            message="m",
            retryable=True,
        )
        # Both present -> rejected.
        with pytest.raises(ValueError, match="exactly one"):
            ProviderSemanticFailure(
                failure_class=SemanticFailureClass.TRANSPORT_ERROR,
                message="m",
                transport_failure=failure,
                rejected_response=response,
            )
        # Neither present -> rejected.
        with pytest.raises(ValueError, match="exactly one"):
            ProviderSemanticFailure(
                failure_class=SemanticFailureClass.TRANSPORT_ERROR,
                message="m",
            )

    def test_taxonomy_is_the_expected_closed_set(self) -> None:
        assert {c.value for c in SemanticFailureClass} == {
            "transport-error",
            "rate-limit",
            "timeout",
            "provider-rejection",
            "blank-generation",
            "malformed-response",
        }
