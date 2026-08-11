"""SQLite real-time lease renewal pathway tests."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from tests.core.effects.authority_support import (
    _HASH_A,
    _HASH_B,
    _LEASE_DURATION,
    _acquire,
    _Backend,
    _request,
    _result_ref,
)
from whetstone.core.effects.authority import (
    AcquireOutcome,
    EffectAuthority,
    ReplayPolicy,
    StaleLeaseError,
    TerminalConflictError,
    TerminalFailure,
)


def test_renew_and_takeover_share_backend_authority_time(
    timed_backend: _Backend,
) -> None:
    request = _request()
    first = _acquire(timed_backend.authority, request)
    assert first.lease is not None

    timed_backend.advance_past(first.lease.expires_at - _LEASE_DURATION)
    renewed = timed_backend.authority.renew(
        first.lease,
        lease_duration=_LEASE_DURATION,
    )
    assert renewed.fence == first.lease.fence
    assert renewed.expires_at > first.lease.expires_at
    with pytest.raises(StaleLeaseError):
        timed_backend.authority.renew(
            first.lease,
            lease_duration=_LEASE_DURATION,
        )

    timed_backend.advance_past(renewed.expires_at)
    takeover = _acquire(
        timed_backend.authority,
        request,
        owner="worker-2",
        attempt="attempt-2",
    )
    assert takeover.outcome is AcquireOutcome.ACQUIRED
    assert takeover.lease is not None
    assert takeover.lease.owner_id == "worker-2"
    assert takeover.lease.attempt_id == "attempt-2"
    assert takeover.lease.fence == 2


@pytest.mark.parametrize(
    "policy",
    (ReplayPolicy.IDEMPOTENT, ReplayPolicy.DURABLE_WORKFLOW),
)
def test_stale_attempt_cannot_terminalize_after_takeover(
    timed_backend: _Backend,
    policy: ReplayPolicy,
) -> None:
    request = _request(policy=policy)
    first = _acquire(timed_backend.authority, request)
    assert first.lease is not None
    timed_backend.advance_past(first.lease.expires_at)
    takeover = _acquire(
        timed_backend.authority,
        request,
        owner="worker-2",
        attempt="attempt-2",
    )
    assert takeover.lease is not None

    with pytest.raises(StaleLeaseError):
        timed_backend.authority.succeed(
            first.lease,
            result_ref=_result_ref("first"),
        )
    terminal = timed_backend.authority.succeed(
        takeover.lease,
        result_ref=_result_ref("second"),
    )
    assert terminal.attempt_id == "attempt-2"
    assert terminal.result_ref == _result_ref("second")


def test_no_redrive_expiry_becomes_immutable_recovery_required(
    timed_backend: _Backend,
) -> None:
    request = _request(policy=ReplayPolicy.NO_REDRIVE)
    first = _acquire(timed_backend.authority, request)
    assert first.lease is not None
    timed_backend.advance_past(first.lease.expires_at)

    recovered = _acquire(
        timed_backend.authority,
        request,
        owner="worker-2",
        attempt="attempt-2",
    )
    assert recovered.outcome is AcquireOutcome.RECOVERY_REQUIRED
    assert recovered.terminal is not None
    assert recovered.terminal.failure is not None
    assert recovered.terminal.failure.code.startswith("effect-recovery:")
    assert recovered.terminal.failure.details["owner_id"] == "worker-1"
    assert recovered.terminal.failure.details["attempt_id"] == "attempt-1"

    replay = _acquire(
        timed_backend.authority,
        request,
        owner="worker-3",
        attempt="attempt-3",
    )
    assert replay == recovered
    with pytest.raises(TerminalConflictError):
        timed_backend.authority.succeed(
            first.lease,
            result_ref=_result_ref("late"),
        )


def test_request_hash_and_policy_are_immutable(
    backend: _Backend,
) -> None:
    original = _request()
    _acquire(backend.authority, original)

    divergent_identity = _acquire(
        backend.authority,
        _request(identity_hash=_HASH_B),
        owner="worker-2",
        attempt="attempt-2",
    )
    assert divergent_identity.outcome is AcquireOutcome.REQUEST_CONFLICT
    assert divergent_identity.existing_request_hash == _HASH_A
    assert divergent_identity.existing_replay_policy is ReplayPolicy.IDEMPOTENT

    divergent_policy = _acquire(
        backend.authority,
        _request(policy=ReplayPolicy.DURABLE_WORKFLOW),
        owner="worker-2",
        attempt="attempt-2",
    )
    assert divergent_policy.outcome is AcquireOutcome.REQUEST_CONFLICT


def test_success_and_failure_are_exact_immutable_and_replayed(
    backend: _Backend,
) -> None:
    success_request = _request(key="effect:success")
    success = _acquire(backend.authority, success_request)
    assert success.lease is not None
    terminal = backend.authority.succeed(
        success.lease,
        result_ref=_result_ref("result-a"),
    )
    assert (
        backend.authority.succeed(
            success.lease,
            result_ref=_result_ref("result-a"),
        )
        == terminal
    )
    with pytest.raises(TerminalConflictError):
        backend.authority.succeed(
            success.lease,
            result_ref=_result_ref("result-b"),
        )
    replay = _acquire(
        backend.authority,
        success_request,
        owner="worker-2",
        attempt="attempt-2",
    )
    assert replay.outcome is AcquireOutcome.SUCCEEDED
    assert replay.terminal == terminal

    failure_request = _request(key="effect:failure")
    failed = _acquire(backend.authority, failure_request)
    assert failed.lease is not None
    failure = TerminalFailure(
        code="provider_timeout",
        message="provider timed out",
        details={"provider": "test"},
    )
    failed_terminal = backend.authority.fail(
        failed.lease,
        result_ref=_result_ref("failed-result"),
        failure=failure,
    )
    failed_replay = _acquire(
        backend.authority,
        failure_request,
        owner="worker-2",
        attempt="attempt-2",
    )
    assert failed_replay.outcome is AcquireOutcome.FAILED
    assert failed_replay.terminal == failed_terminal
    with pytest.raises(TerminalConflictError):
        backend.authority.fail(
            failed.lease,
            result_ref=_result_ref("different"),
            failure=failure,
        )


def test_sqlite_maintenance_terminalization_surfaces_renewal_loss(
    tmp_path: Path,
) -> None:
    database = tmp_path / "renewal-loss.sqlite"
    authority = EffectAuthority.sqlite(database)
    acquired = _acquire(
        authority,
        _request(),
        duration=timedelta(milliseconds=1),
    )
    assert acquired.lease is not None
    backend = _Backend(authority, None, database)
    backend.advance_past(acquired.lease.expires_at)

    with pytest.raises(StaleLeaseError, match="effect lease is stale"):
        with authority.maintain(
            acquired.lease,
            lease_duration=timedelta(milliseconds=1),
        ) as maintenance:
            assert maintenance._stop.wait(timeout=1)
            maintenance.succeed(result_ref=_result_ref("too-late"))
