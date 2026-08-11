from __future__ import annotations

import inspect
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Condition, Event, Lock, Thread

import pytest
from pydantic import ValidationError

from tests.core.effects.authority_support import (
    _HASH_A,
    _HASH_B,
    _LEASE_DURATION,
    _NOW,
    _acquire,
    _request,
    _result_ref,
)
from tests.optimization.sqlite_time import wait_for_sqlite_authority_after
from whetstone.core.effects.authority import (
    AcquireOutcome,
    EffectAuthority,
    EffectLease,
    EffectTerminal,
    ReplayPolicy,
    StaleLeaseError,
    TerminalConflictError,
    TerminalFailure,
    TerminalOutcome,
)
from whetstone.core.identity import NonEmptyId, TypedRef


class _FakeClock:
    def __init__(self, now: datetime = _NOW) -> None:
        self._now = now
        self._lock = Lock()

    def __call__(self) -> datetime:
        with self._lock:
            return self._now

    def advance(self, duration: timedelta) -> None:
        with self._lock:
            self._now += duration


class _ScriptedRenewalWait:
    def __init__(self) -> None:
        self._condition = Condition()
        self._requested_intervals: list[float] = []
        self._releases = 0

    def wait(self, interval_seconds: float, stop: Event) -> bool:
        with self._condition:
            self._requested_intervals.append(interval_seconds)
            self._condition.notify_all()
            ready = self._condition.wait_for(
                lambda: stop.is_set() or self._releases > 0,
                timeout=2,
            )
            if not ready:
                raise TimeoutError("test did not script the next renewal wait")
            if stop.is_set():
                return True
            self._releases -= 1
            return False

    def wake(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def await_request_count(self, count: int) -> tuple[float, ...]:
        with self._condition:
            observed = self._condition.wait_for(
                lambda: len(self._requested_intervals) >= count,
                timeout=2,
            )
            if not observed:
                raise AssertionError(
                    f"renewer did not request {count} scripted waits"
                )
            return tuple(self._requested_intervals)

    def release_one(self) -> None:
        with self._condition:
            self._releases += 1
            self._condition.notify_all()


@dataclass(frozen=True, slots=True)
class _Backend:
    authority: EffectAuthority
    clock: _FakeClock | None
    database: Path | None

    def advance_past(self, instant: datetime) -> None:
        if self.clock is not None:
            now = self.clock()
            if now <= instant:
                self.clock.advance(instant - now + timedelta(microseconds=1))
            return
        assert self.database is not None
        wait_for_sqlite_authority_after(self.database, instant)


class _CoordinatedAuthority(EffectAuthority):
    def __init__(self, authority: EffectAuthority) -> None:
        self._authority = authority
        self._renewal_wait_strategy = authority._renewal_wait_strategy
        self.release_renewal = Event()
        self.renewal_entered = Event()
        self.terminal_entered = Event()
        self.renew_calls = 0
        self.terminal_lease: EffectLease | None = None

    def renew(
        self,
        lease: EffectLease,
        *,
        lease_duration: timedelta,
    ) -> EffectLease:
        self.renew_calls += 1
        self.renewal_entered.set()
        if not self.release_renewal.wait(timeout=2):
            raise TimeoutError("test did not release coordinated renewal")
        return self._authority.renew(
            lease,
            lease_duration=lease_duration,
        )

    def _validate_lease_duration(self, value: timedelta) -> timedelta:
        return self._authority._validate_lease_duration(value)

    def succeed(
        self,
        lease: EffectLease,
        *,
        result_ref: TypedRef,
    ) -> EffectTerminal:
        self.terminal_lease = lease
        self.terminal_entered.set()
        return self._authority.succeed(lease, result_ref=result_ref)

    def fail(
        self,
        lease: EffectLease,
        *,
        result_ref: TypedRef,
        failure: TerminalFailure,
    ) -> EffectTerminal:
        self.terminal_lease = lease
        self.terminal_entered.set()
        return self._authority.fail(
            lease,
            result_ref=result_ref,
            failure=failure,
        )


@pytest.fixture(
    name="backend",
    params=(
        "memory",
        pytest.param(
            "sqlite",
            marks=pytest.mark.sqlite_time_integration,
        ),
    ),
)
def backend_fixture(
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> _Backend:
    if request.param == "memory":
        clock = _FakeClock()
        return _Backend(EffectAuthority.memory(clock=clock), clock, None)
    database = tmp_path / "authority.sqlite"
    return _Backend(EffectAuthority.sqlite(database), None, database)


def test_public_transitions_expose_duration_but_no_process_time() -> None:
    assert list(inspect.signature(EffectAuthority.acquire).parameters) == [
        "self",
        "request",
        "owner_id",
        "attempt_id",
        "lease_duration",
    ]
    assert list(inspect.signature(EffectAuthority.renew).parameters) == [
        "self",
        "lease",
        "lease_duration",
    ]
    assert list(inspect.signature(EffectAuthority.succeed).parameters) == [
        "self",
        "lease",
        "result_ref",
    ]
    assert list(inspect.signature(EffectAuthority.fail).parameters) == [
        "self",
        "lease",
        "result_ref",
        "failure",
    ]
    authority = EffectAuthority.memory(clock=_FakeClock())
    with pytest.raises(TypeError, match="unexpected keyword argument 'now'"):
        authority.acquire(
            _request(),
            owner_id="worker",
            attempt_id="attempt",
            lease_duration=_LEASE_DURATION,
            now=datetime.max.replace(  # ty: ignore[unknown-argument]
                tzinfo=UTC
            ),
        )


@pytest.mark.parametrize(
    "duration",
    (timedelta(0), timedelta(microseconds=-1)),
)
def test_lease_duration_must_be_positive(duration: timedelta) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        _acquire(
            EffectAuthority.memory(clock=_FakeClock()),
            _request(),
            duration=duration,
        )


def test_same_attempt_replays_but_same_owner_new_attempt_is_busy(
    backend: _Backend,
) -> None:
    request = _request()
    first = _acquire(backend.authority, request)
    replay = _acquire(backend.authority, request)
    competing = _acquire(
        backend.authority,
        request,
        owner="worker-1",
        attempt="attempt-2",
    )

    assert first.outcome is AcquireOutcome.ACQUIRED
    assert replay == first
    assert first.lease is not None
    assert first.lease.attempt_id == "attempt-1"
    assert competing.outcome is AcquireOutcome.BUSY
    assert competing.busy_expires_at == first.lease.expires_at


def test_renew_and_takeover_share_backend_authority_time(
    backend: _Backend,
) -> None:
    request = _request()
    first = _acquire(backend.authority, request)
    assert first.lease is not None

    backend.advance_past(first.lease.expires_at - _LEASE_DURATION)
    renewed = backend.authority.renew(
        first.lease,
        lease_duration=_LEASE_DURATION,
    )
    assert renewed.fence == first.lease.fence
    assert renewed.expires_at > first.lease.expires_at
    with pytest.raises(StaleLeaseError):
        backend.authority.renew(
            first.lease,
            lease_duration=_LEASE_DURATION,
        )

    backend.advance_past(renewed.expires_at)
    takeover = _acquire(
        backend.authority,
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
    backend: _Backend,
    policy: ReplayPolicy,
) -> None:
    request = _request(policy=policy)
    first = _acquire(backend.authority, request)
    assert first.lease is not None
    backend.advance_past(first.lease.expires_at)
    takeover = _acquire(
        backend.authority,
        request,
        owner="worker-2",
        attempt="attempt-2",
    )
    assert takeover.lease is not None

    with pytest.raises(StaleLeaseError):
        backend.authority.succeed(
            first.lease,
            result_ref=_result_ref("first"),
        )
    terminal = backend.authority.succeed(
        takeover.lease,
        result_ref=_result_ref("second"),
    )
    assert terminal.attempt_id == "attempt-2"
    assert terminal.result_ref == _result_ref("second")


def test_no_redrive_expiry_becomes_immutable_recovery_required(
    backend: _Backend,
) -> None:
    request = _request(policy=ReplayPolicy.NO_REDRIVE)
    first = _acquire(backend.authority, request)
    assert first.lease is not None
    backend.advance_past(first.lease.expires_at)

    recovered = _acquire(
        backend.authority,
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
        backend.authority,
        request,
        owner="worker-3",
        attempt="attempt-3",
    )
    assert replay == recovered
    with pytest.raises(TerminalConflictError):
        backend.authority.succeed(
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


def test_terminal_outcome_shapes_and_serialized_lease_are_exact() -> None:
    request = _request()
    failure = TerminalFailure(code="failed", message="failed")
    common = {
        "request": request,
        "owner_id": NonEmptyId("worker"),
        "attempt_id": NonEmptyId("attempt"),
        "fence": 1,
    }
    with pytest.raises(ValidationError, match="succeeded terminal"):
        EffectTerminal.model_validate(
            {
                **common,
                "outcome": TerminalOutcome.SUCCEEDED,
                "result_ref": _result_ref(),
                "failure": failure,
            }
        )
    with pytest.raises(ValidationError, match="failed terminal"):
        EffectTerminal.model_validate(
            {
                **common,
                "outcome": TerminalOutcome.FAILED,
                "failure": failure,
            }
        )
    lease = EffectLease(
        request=request,
        owner_id=NonEmptyId("worker"),
        attempt_id=NonEmptyId("attempt"),
        fence=1,
        expires_at=_NOW + _LEASE_DURATION,
    )
    assert EffectLease.model_validate_json(lease.model_dump_json()) == lease
    with pytest.raises(ValidationError):
        EffectLease(
            request=request,
            owner_id=NonEmptyId("worker"),
            attempt_id=NonEmptyId("attempt"),
            fence=True,
            expires_at=lease.expires_at,
        )
    with pytest.raises(ValidationError):
        lease.attempt_id = NonEmptyId(  # ty: ignore[invalid-assignment]
            "changed"
        )


def test_heartbeat_renews_across_multiple_durations() -> None:
    clock = _FakeClock()
    renewal_wait = _ScriptedRenewalWait()
    authority = EffectAuthority.memory(
        clock=clock,
        _renewal_wait_strategy=renewal_wait,
    )
    acquired = _acquire(
        authority,
        _request(),
        duration=timedelta(milliseconds=90),
    )
    assert acquired.lease is not None
    initial_expiry = acquired.lease.expires_at

    with authority.maintain(
        acquired.lease,
        lease_duration=timedelta(milliseconds=90),
    ) as maintenance:
        intervals = renewal_wait.await_request_count(1)
        assert intervals == pytest.approx((0.03,))
        for tick in range(1, 5):
            clock.advance(timedelta(milliseconds=30))
            renewal_wait.release_one()
            intervals = renewal_wait.await_request_count(tick + 1)
            assert intervals == pytest.approx((0.03,) * (tick + 1))
            maintenance.check()
        maintained_lease = maintenance.lease
        assert maintained_lease.expires_at >= (
            initial_expiry + timedelta(milliseconds=90)
        )
        terminal = maintenance.succeed(result_ref=_result_ref("heartbeat"))
    assert terminal.outcome is TerminalOutcome.SUCCEEDED


def test_heartbeat_reports_lease_loss_on_clean_exit() -> None:
    clock = _FakeClock()
    renewal_wait = _ScriptedRenewalWait()
    authority = EffectAuthority.memory(
        clock=clock,
        _renewal_wait_strategy=renewal_wait,
    )
    acquired = _acquire(
        authority,
        _request(),
        duration=timedelta(milliseconds=60),
    )
    assert acquired.lease is not None

    with pytest.raises(StaleLeaseError):
        with authority.maintain(
            acquired.lease,
            lease_duration=timedelta(milliseconds=60),
        ) as maintenance:
            assert renewal_wait.await_request_count(1) == pytest.approx(
                (0.02,)
            )
            clock.advance(timedelta(milliseconds=61))
            renewal_wait.release_one()
            assert maintenance._stop.wait(timeout=1)


def test_maintained_success_waits_for_renewal_and_uses_latest_lease() -> None:
    clock = _FakeClock()
    renewal_wait = _ScriptedRenewalWait()
    base_authority = EffectAuthority.memory(
        clock=clock,
        _renewal_wait_strategy=renewal_wait,
    )
    authority = _CoordinatedAuthority(base_authority)
    acquired = _acquire(
        base_authority,
        _request(),
        duration=timedelta(milliseconds=90),
    )
    assert acquired.lease is not None
    initial_lease = acquired.lease
    terminals: list[EffectTerminal] = []
    errors: list[BaseException] = []

    with authority.maintain(
        initial_lease,
        lease_duration=timedelta(milliseconds=90),
    ) as maintenance:
        maintenance_thread = maintenance._thread
        assert renewal_wait.await_request_count(1) == pytest.approx((0.03,))
        clock.advance(timedelta(milliseconds=30))
        renewal_wait.release_one()
        assert authority.renewal_entered.wait(timeout=1)

        def publish() -> None:
            try:
                terminals.append(
                    maintenance.succeed(result_ref=_result_ref("renewal-won"))
                )
            except BaseException as exc:
                errors.append(exc)

        publisher = Thread(target=publish, name="test-terminal-publisher")
        publisher.start()
        try:
            assert maintenance._stop.wait(timeout=1)
            assert not authority.terminal_entered.is_set()
        finally:
            authority.release_renewal.set()
        publisher.join(timeout=2)
        assert not publisher.is_alive()
        assert not errors

    assert not maintenance_thread.is_alive()
    assert len(terminals) == 1
    terminal = terminals[0]
    assert authority.terminal_lease is not None
    assert authority.terminal_lease.expires_at > initial_lease.expires_at
    assert terminal == EffectTerminal(
        request=initial_lease.request,
        outcome=TerminalOutcome.SUCCEEDED,
        owner_id=initial_lease.owner_id,
        attempt_id=initial_lease.attempt_id,
        fence=initial_lease.fence,
        result_ref=_result_ref("renewal-won"),
    )


def test_maintained_success_stops_renewal_before_publish() -> None:
    base_authority = EffectAuthority.memory(clock=_FakeClock())
    authority = _CoordinatedAuthority(base_authority)
    acquired = _acquire(
        base_authority,
        _request(),
        duration=timedelta(seconds=30),
    )
    assert acquired.lease is not None

    with authority.maintain(
        acquired.lease,
        lease_duration=timedelta(seconds=30),
    ) as maintenance:
        maintenance_thread = maintenance._thread
        terminal = maintenance.succeed(result_ref=_result_ref("terminal-won"))

    assert terminal.outcome is TerminalOutcome.SUCCEEDED
    assert authority.renew_calls == 0
    assert not maintenance_thread.is_alive()


def test_maintained_failure_is_exact_and_rejects_double_publication() -> None:
    authority = EffectAuthority.memory(clock=_FakeClock())
    request = _request()
    acquired = _acquire(
        authority,
        request,
        duration=timedelta(seconds=30),
    )
    assert acquired.lease is not None
    failure = TerminalFailure(
        code="provider_rejected",
        message="provider rejected the request",
        details={"provider": "test"},
    )

    with authority.maintain(
        acquired.lease,
        lease_duration=timedelta(seconds=30),
    ) as maintenance:
        terminal = maintenance.fail(
            result_ref=_result_ref("failed-maintained-result"),
            failure=failure,
        )
        with pytest.raises(RuntimeError, match="already terminalized"):
            maintenance.succeed(result_ref=_result_ref("second-terminal"))

    assert terminal == EffectTerminal(
        request=request,
        outcome=TerminalOutcome.FAILED,
        owner_id=acquired.lease.owner_id,
        attempt_id=acquired.lease.attempt_id,
        fence=acquired.lease.fence,
        result_ref=_result_ref("failed-maintained-result"),
        failure=failure,
    )


def test_terminal_publication_requires_an_entered_live_context() -> None:
    authority = EffectAuthority.memory(clock=_FakeClock())
    acquired = _acquire(
        authority,
        _request(),
        duration=timedelta(seconds=30),
    )
    assert acquired.lease is not None
    maintenance = authority.maintain(
        acquired.lease,
        lease_duration=timedelta(seconds=30),
    )

    with pytest.raises(RuntimeError, match="active maintenance context"):
        maintenance.succeed(result_ref=_result_ref("before-enter"))

    with pytest.raises(LookupError, match="body failed"):
        with maintenance:
            raise LookupError("body failed")

    with pytest.raises(RuntimeError, match="active maintenance context"):
        maintenance.fail(
            result_ref=_result_ref("after-exit"),
            failure=TerminalFailure(code="failed", message="failed"),
        )


def test_clean_maintenance_exit_requires_terminal_publication() -> None:
    authority = EffectAuthority.memory(clock=_FakeClock())
    acquired = _acquire(
        authority,
        _request(),
        duration=timedelta(seconds=30),
    )
    assert acquired.lease is not None

    with pytest.raises(
        RuntimeError,
        match="clean lease maintenance exit requires terminal publication",
    ):
        with authority.maintain(
            acquired.lease,
            lease_duration=timedelta(seconds=30),
        ):
            pass


def test_exception_exit_stops_renewer_and_preserves_observed_loss() -> None:
    clock = _FakeClock()
    renewal_wait = _ScriptedRenewalWait()
    base_authority = EffectAuthority.memory(
        clock=clock,
        _renewal_wait_strategy=renewal_wait,
    )
    authority = _CoordinatedAuthority(base_authority)
    acquired = _acquire(
        base_authority,
        _request(),
        duration=timedelta(milliseconds=90),
    )
    assert acquired.lease is not None

    with pytest.raises(LookupError, match="body failed"):
        with authority.maintain(
            acquired.lease,
            lease_duration=timedelta(milliseconds=90),
        ) as maintenance:
            maintenance_thread = maintenance._thread
            assert renewal_wait.await_request_count(1) == pytest.approx(
                (0.03,)
            )
            clock.advance(timedelta(milliseconds=91))
            renewal_wait.release_one()
            assert authority.renewal_entered.wait(timeout=1)
            authority.release_renewal.set()
            assert maintenance._stop.wait(timeout=1)
            raise LookupError("body failed")

    assert not maintenance_thread.is_alive()
    with pytest.raises(StaleLeaseError):
        maintenance.check()
