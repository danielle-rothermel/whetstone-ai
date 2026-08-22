"""Whetstone's usage contract over ``dr_store.lease.LeaseAuthority``.

These pin how whetstone *uses* leasing -- request identity built from
whetstone identity types, replay-policy semantics at the boundary, terminal
verification, and maintenance -- plus the whetstone-typed translation
(``TypedRef`` result refs, whetstone ``TerminalFailure``). dr-store owns and
tests the leasing internals; these tests do not re-verify them.
"""

from __future__ import annotations

import json
from datetime import timedelta
from typing import TYPE_CHECKING

import pytest
from dr_store.testing import (
    FakeClock,
    temp_sqlite_lease_authority,
)

from whetstone.core.identity import (
    ImmutableJsonObject,
    TerminalFailure,
    TypedRef,
    compute_identity_hash,
    compute_prefixed_identity_key,
    typed_ref_for_record,
)
from whetstone.core.leasing import (
    AcquireOutcome,
    EffectLease,
    EffectLeaseAuthority,
    EffectTerminal,
    ReplayPolicy,
    StaleLeaseError,
    TerminalConflictError,
    TerminalOutcome,
    effect_request,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

_SCHEMA = "whetstone.leasing_test_effect"
_SCHEMA_VERSION = 1
_KEY_PREFIX = "whetstone.leasing_test:"
_RESULT_SCHEMA = "whetstone.leasing_test_result"
_LEASE = timedelta(seconds=30)


def _request(
    *,
    payload: dict[str, object] | None = None,
    replay_policy: ReplayPolicy = ReplayPolicy.IDEMPOTENT,
):
    """Build a lease request the way whetstone's harnesses build one."""
    exact = {"call": "alpha"} if payload is None else payload
    return effect_request(
        semantic_key=compute_prefixed_identity_key(
            schema=_SCHEMA,
            schema_version=_SCHEMA_VERSION,
            prefix=_KEY_PREFIX,
            payload=exact,
        ),
        request_hash=compute_identity_hash(
            schema=_SCHEMA,
            schema_version=_SCHEMA_VERSION,
            payload=exact,
        ),
        replay_policy=replay_policy,
    )


def _result_ref(value: str = "ok") -> TypedRef:
    return typed_ref_for_record(_RESULT_SCHEMA, {"value": value})


def _failure(code: str = "leasing_test_failure") -> TerminalFailure:
    return TerminalFailure(
        code=code,
        message="the effect failed",
        details=ImmutableJsonObject({"attempt": 1}),
    )


@pytest.fixture
def authority() -> Iterator[tuple[EffectLeaseAuthority, FakeClock]]:
    clock = FakeClock()
    lease_authority = EffectLeaseAuthority.memory(clock=clock.now)
    try:
        yield lease_authority, clock
    finally:
        lease_authority.close()


def _acquire(
    lease_authority: EffectLeaseAuthority,
    request,
    *,
    owner_id: str = "owner-a",
    attempt_id: str = "attempt-a",
) -> EffectLease:
    acquisition = lease_authority.acquire(
        request,
        owner_id=owner_id,
        attempt_id=attempt_id,
        lease_duration=_LEASE,
    )
    assert acquisition.outcome is AcquireOutcome.ACQUIRED
    assert acquisition.lease is not None
    return acquisition.lease


def test_request_identity_comes_from_whetstone_identity_types() -> None:
    """Whetstone's OpaqueKey and IdentityHash build a valid lease request."""
    request = _request()

    assert request.semantic_key.startswith(_KEY_PREFIX)
    assert len(request.request_hash) == 64
    assert request.replay_policy is ReplayPolicy.IDEMPOTENT
    # Same payload is the same effect; a different payload is a different one.
    assert request == _request()
    assert _request(payload={"call": "beta"}).semantic_key != (
        request.semantic_key
    )


def test_a_second_holder_is_busy_until_the_lease_expires(
    authority: tuple[EffectLeaseAuthority, FakeClock],
) -> None:
    """A live lease refuses a different owner and reports its expiry."""
    lease_authority, _clock = authority
    request = _request()
    lease = _acquire(lease_authority, request)

    contender = lease_authority.acquire(
        request,
        owner_id="owner-b",
        attempt_id="attempt-b",
        lease_duration=_LEASE,
    )

    assert contender.outcome is AcquireOutcome.BUSY
    assert contender.busy_expires_at == lease.expires_at


def test_the_same_key_with_another_request_hash_conflicts(
    authority: tuple[EffectLeaseAuthority, FakeClock],
) -> None:
    """Rebinding a semantic key to another exact request is refused."""
    lease_authority, _clock = authority
    request = _request()
    _acquire(lease_authority, request)
    rebound = effect_request(
        semantic_key=request.semantic_key,
        request_hash=compute_identity_hash(
            schema=_SCHEMA,
            schema_version=_SCHEMA_VERSION,
            payload={"call": "beta"},
        ),
        replay_policy=ReplayPolicy.IDEMPOTENT,
    )

    conflict = lease_authority.acquire(
        rebound,
        owner_id="owner-b",
        attempt_id="attempt-b",
        lease_duration=_LEASE,
    )

    assert conflict.outcome is AcquireOutcome.REQUEST_CONFLICT
    assert conflict.existing_request_hash == request.request_hash
    assert conflict.existing_replay_policy is ReplayPolicy.IDEMPOTENT


def test_an_idempotent_effect_is_redriven_after_its_lease_expires(
    authority: tuple[EffectLeaseAuthority, FakeClock],
) -> None:
    """An abandoned IDEMPOTENT lease is taken over with a higher fence."""
    lease_authority, clock = authority
    request = _request(replay_policy=ReplayPolicy.IDEMPOTENT)
    first = _acquire(lease_authority, request)
    clock.advance(_LEASE * 2)

    takeover = lease_authority.acquire(
        request,
        owner_id="owner-b",
        attempt_id="attempt-b",
        lease_duration=_LEASE,
    )

    assert takeover.outcome is AcquireOutcome.ACQUIRED
    assert takeover.lease is not None
    assert takeover.lease.fence > first.fence
    assert takeover.lease.owner_id == "owner-b"


def test_a_no_redrive_effect_becomes_recovery_required_when_abandoned(
    authority: tuple[EffectLeaseAuthority, FakeClock],
) -> None:
    """An abandoned NO_REDRIVE lease terminalizes as recovery-required."""
    lease_authority, clock = authority
    request = _request(replay_policy=ReplayPolicy.NO_REDRIVE)
    _acquire(lease_authority, request)
    clock.advance(_LEASE * 2)

    recovery = lease_authority.acquire(
        request,
        owner_id="owner-b",
        attempt_id="attempt-b",
        lease_duration=_LEASE,
    )

    assert recovery.outcome is AcquireOutcome.RECOVERY_REQUIRED
    assert recovery.terminal is not None
    assert recovery.terminal.outcome is TerminalOutcome.RECOVERY_REQUIRED
    # Whetstone consumers raise their domain error off this failure.
    assert recovery.terminal.failure is not None
    assert isinstance(recovery.terminal.failure, TerminalFailure)
    assert recovery.terminal.result_ref is None


def test_a_succeeded_effect_replays_its_terminal_result_ref(
    authority: tuple[EffectLeaseAuthority, FakeClock],
) -> None:
    """Re-acquiring a succeeded effect returns the schema-typed result ref."""
    lease_authority, _clock = authority
    request = _request()
    lease = _acquire(lease_authority, request)
    result_ref = _result_ref()

    terminal = lease_authority.succeed(lease, result_ref=result_ref)
    replay = lease_authority.acquire(
        request,
        owner_id="owner-b",
        attempt_id="attempt-b",
        lease_duration=_LEASE,
    )

    assert terminal.result_ref == result_ref
    assert isinstance(terminal.result_ref, TypedRef)
    assert terminal.result_ref.schema_name == _RESULT_SCHEMA
    assert replay.outcome is AcquireOutcome.SUCCEEDED
    assert replay.terminal == terminal


def test_a_failed_effect_replays_its_whetstone_terminal_failure(
    authority: tuple[EffectLeaseAuthority, FakeClock],
) -> None:
    """A failed terminal round-trips whetstone's TerminalFailure exactly."""
    lease_authority, _clock = authority
    request = _request()
    lease = _acquire(lease_authority, request)
    failure = _failure()

    terminal = lease_authority.fail(
        lease, result_ref=_result_ref(), failure=failure
    )
    replay = lease_authority.acquire(
        request,
        owner_id="owner-b",
        attempt_id="attempt-b",
        lease_duration=_LEASE,
    )

    assert terminal.outcome is TerminalOutcome.FAILED
    assert terminal.failure == failure
    assert replay.outcome is AcquireOutcome.FAILED
    assert replay.terminal is not None
    assert replay.terminal.failure == failure


@pytest.mark.parametrize(
    ("message", "json_safe"),
    [
        ("provider error: " + "x" * 1200, True),
        ("   ", True),
        ("hello\x00world", True),
        ("bad\ud800surrogate", False),
    ],
    ids=["oversized", "whitespace-only", "nul", "unpaired-surrogate"],
)
def test_a_dirty_failure_message_round_trips_the_whetstone_failure(
    authority: tuple[EffectLeaseAuthority, FakeClock],
    message: str,
    json_safe: bool,
) -> None:
    """Provider text that dr-store would reject still publishes as FAILED.

    The write path coerces code/message into dr-store's accepted shape so a
    long or dirty provider message cannot raise inside ``maintain(...)`` and
    lose the lease. ``EffectTerminal.failure`` restores the original so
    persist-and-compare call sites stay equal. Unpaired surrogates cannot
    ride in JSON details, so ``verify_terminal`` is skipped for that case.
    """
    lease_authority, _clock = authority
    request = _request()
    lease = _acquire(lease_authority, request)
    failure = TerminalFailure(
        code="evaluation_RuntimeError",
        message=message,
        details=ImmutableJsonObject({"attempt": 1}),
    )

    terminal = lease_authority.fail(
        lease, result_ref=_result_ref(), failure=failure
    )

    assert terminal.outcome is TerminalOutcome.FAILED
    assert terminal.failure == failure
    if json_safe:
        assert lease_authority.verify_terminal(terminal) == terminal
    lease_failure = terminal.to_lease().failure
    assert lease_failure is not None
    assert lease_failure.details["untruncated_message"] == message
    assert lease_failure.details["attempt"] == 1
    if len(message) > 1024:
        assert len(lease_failure.message) == 1024
        assert lease_failure.message.endswith("...")
    replay = lease_authority.acquire(
        request,
        owner_id="owner-b",
        attempt_id="attempt-b",
        lease_duration=_LEASE,
    )
    assert replay.outcome is AcquireOutcome.FAILED
    assert replay.terminal == terminal
    assert replay.terminal is not None
    assert replay.terminal.failure == failure


def test_a_stale_lease_cannot_publish_a_terminal(
    authority: tuple[EffectLeaseAuthority, FakeClock],
) -> None:
    """A superseded lease holder is refused at terminal publication."""
    lease_authority, clock = authority
    request = _request()
    abandoned = _acquire(lease_authority, request)
    clock.advance(_LEASE * 2)
    _acquire(
        lease_authority, request, owner_id="owner-b", attempt_id="attempt-b"
    )

    with pytest.raises(StaleLeaseError):
        lease_authority.succeed(abandoned, result_ref=_result_ref())


def test_verify_terminal_accepts_only_the_authoritative_terminal(
    authority: tuple[EffectLeaseAuthority, FakeClock],
) -> None:
    """Whetstone re-verifies stored terminals through the authority."""
    lease_authority, _clock = authority
    request = _request()
    lease = _acquire(lease_authority, request)
    terminal = lease_authority.succeed(lease, result_ref=_result_ref())

    assert lease_authority.verify_terminal(terminal) == terminal

    forged = EffectTerminal.model_validate_json(
        json.dumps(
            terminal.model_dump(mode="json")
            | {"result_ref": _result_ref("forged").model_dump(mode="json")}
        )
    )
    with pytest.raises(TerminalConflictError):
        lease_authority.verify_terminal(forged)


def test_maintenance_terminalizes_through_the_handle(
    authority: tuple[EffectLeaseAuthority, FakeClock],
) -> None:
    """The maintenance handle publishes the terminal whetstone stores."""
    lease_authority, _clock = authority
    request = _request()
    lease = _acquire(lease_authority, request)
    result_ref = _result_ref()

    with lease_authority.maintain(
        lease, lease_duration=_LEASE
    ) as maintenance:
        terminal = maintenance.succeed(result_ref=result_ref)

    assert terminal.result_ref == result_ref
    assert lease_authority.verify_terminal(terminal) == terminal


def test_a_clean_maintenance_exit_requires_terminal_publication(
    authority: tuple[EffectLeaseAuthority, FakeClock],
) -> None:
    """Leaving the context without terminalizing is a programming error."""
    lease_authority, _clock = authority
    lease = _acquire(lease_authority, _request())

    with pytest.raises(RuntimeError, match="terminal publication"):
        with lease_authority.maintain(lease, lease_duration=_LEASE):
            pass


def test_the_effect_terminal_round_trips_through_its_stored_json() -> None:
    """EffectTerminal's JSON form is what whetstone persists and reloads."""
    with temp_sqlite_lease_authority() as lease_authority:
        whetstone_authority = EffectLeaseAuthority(lease_authority)
        request = _request()
        lease = _acquire(whetstone_authority, request)
        failure = _failure()
        terminal = whetstone_authority.fail(
            lease, result_ref=_result_ref(), failure=failure
        )

        reloaded = EffectTerminal.model_validate_json(
            terminal.model_dump_json()
        )

        assert reloaded == terminal
        assert reloaded.result_ref == _result_ref()
        assert reloaded.failure == failure
        # The reloaded terminal is still authoritative against the store.
        assert whetstone_authority.verify_terminal(reloaded) == terminal


def test_a_transient_terminalization_failure_does_not_poison_the_handle(
    authority: tuple[EffectLeaseAuthority, FakeClock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transient succeed() failure is retryable on the same handle.

    Whetstone's deleted EffectAuthority marked terminalization started before
    calling the authority, so one transient failure permanently poisoned the
    handle and the caller could only abandon the lease. dr-store restarts the
    renewer instead, so the caller retries inside the still-open context --
    new behavior whetstone gains from the cutover.
    """
    lease_authority, _clock = authority
    lease = _acquire(lease_authority, _request())
    result_ref = _result_ref()
    inner = lease_authority._authority
    original_succeed = inner.succeed
    attempts = {"count": 0}

    def flaky_succeed(held, *, result_ref):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise OSError("authority unavailable")
        return original_succeed(held, result_ref=result_ref)

    monkeypatch.setattr(inner, "succeed", flaky_succeed)

    with lease_authority.maintain(
        lease, lease_duration=_LEASE
    ) as maintenance:
        with pytest.raises(OSError, match="authority unavailable"):
            maintenance.succeed(result_ref=result_ref)
        # The renewer restarted and the lease is still held, so the same
        # handle terminalizes on retry rather than raising "already
        # terminalized".
        maintenance.check()
        assert maintenance.lease == lease
        terminal = maintenance.succeed(result_ref=result_ref)

    assert attempts["count"] == 2
    assert terminal.result_ref == result_ref
    assert lease_authority.verify_terminal(terminal) == terminal
