"""Provider Execution Policy composition + identity contract tests."""

from __future__ import annotations

import pytest

from tests.provider import support as s
from whetstone.provider.classification import SemanticFailureClass
from whetstone.provider.policy import (
    BackoffSchedule,
    ProviderExecutionPolicy,
    default_retry_eligibility,
)


class TestComposition:
    def test_composes_one_transport_policy_reference(self) -> None:
        transport = s.build_transport_policy()
        policy = s.build_execution_policy(transport_policy=transport)
        assert policy.transport_policy is transport

    def test_no_duplicated_transport_fields(self) -> None:
        # The execution policy carries no credentials/timeout/native-retry
        # field of its own; those live only on the referenced transport policy.
        fields = set(ProviderExecutionPolicy.model_fields)
        for forbidden in (
            "api_key_env",
            "api_key",
            "timeout_seconds",
            "native_retry_count",
            "base_url",
        ):
            assert forbidden not in fields

    def test_native_retries_pinned_zero(self) -> None:
        # A referenced transport policy with nonzero native retries is refused.
        with pytest.raises(ValueError, match="native_retry_count"):
            s.build_execution_policy(
                transport_policy=s.build_transport_policy(native_retry_count=2)
            )

    def test_bounded_attempt_count(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            s.build_execution_policy(max_attempts=0)

    def test_retry_eligibility_must_be_closed(self) -> None:
        partial = {SemanticFailureClass.RATE_LIMIT: True}
        with pytest.raises(ValueError, match="every SemanticFailureClass"):
            s.build_execution_policy(retry_eligibility=partial)

    def test_default_eligibility_is_conservative(self) -> None:
        eligibility = default_retry_eligibility()
        assert eligibility[SemanticFailureClass.TRANSPORT_ERROR]
        assert eligibility[SemanticFailureClass.RATE_LIMIT]
        assert eligibility[SemanticFailureClass.TIMEOUT]
        assert not eligibility[SemanticFailureClass.PROVIDER_REJECTION]
        assert not eligibility[SemanticFailureClass.BLANK_GENERATION]
        assert not eligibility[SemanticFailureClass.MALFORMED_RESPONSE]

    def test_is_retryable_lookup(self) -> None:
        policy = s.build_execution_policy()
        assert policy.is_retryable(SemanticFailureClass.RATE_LIMIT)
        assert not policy.is_retryable(
            SemanticFailureClass.PROVIDER_REJECTION
        )


class TestBackoff:
    def test_first_attempt_has_zero_delay(self) -> None:
        assert BackoffSchedule().delay_for(1) == 0.0

    def test_geometric_and_capped(self) -> None:
        schedule = BackoffSchedule(
            base_seconds=1.0, multiplier=2.0, max_seconds=5.0
        )
        assert schedule.delay_for(2) == 1.0
        assert schedule.delay_for(3) == 2.0
        assert schedule.delay_for(4) == 4.0
        assert schedule.delay_for(5) == 5.0  # capped
        assert schedule.delay_for(6) == 5.0

    def test_deterministic_no_jitter(self) -> None:
        schedule = BackoffSchedule()
        assert [schedule.delay_for(n) for n in range(1, 5)] == [
            schedule.delay_for(n) for n in range(1, 5)
        ]

    def test_rejects_bad_schedule(self) -> None:
        with pytest.raises(ValueError, match="multiplier"):
            BackoffSchedule(multiplier=0.5)
        with pytest.raises(ValueError, match="max_seconds"):
            BackoffSchedule(base_seconds=10.0, max_seconds=1.0)


class TestIdentity:
    def test_identity_hash_is_full_length(self) -> None:
        assert len(s.build_execution_policy().identity_hash) == 64

    def test_identity_excludes_secret(self) -> None:
        # Only the env-var NAME appears, never a secret; and the payload has no
        # timeout duplication under an execution-policy-owned key.
        payload = s.build_execution_policy().identity_payload()
        assert (
            payload["transport_policy"]["api_key_env"] == s.API_KEY_ENV
        )
        assert "api_key" not in payload
        assert "timeout_seconds" not in payload

    def test_semantic_change_changes_hash(self) -> None:
        base = s.build_execution_policy(max_attempts=3)
        more = s.build_execution_policy(max_attempts=5)
        assert base.identity_hash != more.identity_hash

    def test_backoff_change_changes_hash(self) -> None:
        base = s.build_execution_policy(
            backoff=BackoffSchedule(base_seconds=1)
        )
        other = s.build_execution_policy(
            backoff=BackoffSchedule(base_seconds=2)
        )
        assert base.identity_hash != other.identity_hash

    def test_eligibility_change_changes_hash(self) -> None:
        eligibility = default_retry_eligibility()
        eligibility[SemanticFailureClass.PROVIDER_REJECTION] = True
        base = s.build_execution_policy()
        other = s.build_execution_policy(retry_eligibility=eligibility)
        assert base.identity_hash != other.identity_hash

    def test_identity_is_stable(self) -> None:
        assert (
            s.build_execution_policy().identity_hash
            == s.build_execution_policy().identity_hash
        )
