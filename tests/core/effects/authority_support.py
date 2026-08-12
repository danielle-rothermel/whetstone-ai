from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Condition, Event, Lock

import pytest

from tests.optimization.sqlite_time import wait_for_sqlite_authority_after
from whetstone.core.effects.authority import (
    EffectAuthority,
    EffectLease,
    EffectRequest,
    EffectTerminal,
    ReplayPolicy,
    TerminalFailure,
)
from whetstone.core.identity import IdentityHash, OpaqueKey, TypedRef

_HASH_A = "a" * 64
_HASH_B = "b" * 64
_NOW = datetime(2026, 7, 31, 12, tzinfo=UTC)
_LEASE_DURATION = timedelta(milliseconds=100)


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
        "sqlite",
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


@pytest.fixture(
    name="timed_backend",
    params=(
        "memory",
        pytest.param("sqlite", marks=pytest.mark.sqlite_time_integration),
    ),
)
def timed_backend_fixture(
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> _Backend:
    if request.param == "memory":
        clock = _FakeClock()
        return _Backend(EffectAuthority.memory(clock=clock), clock, None)
    database = tmp_path / "timed-authority.sqlite"
    return _Backend(EffectAuthority.sqlite(database), None, database)


def _result_ref(label: str = "result") -> TypedRef:
    return TypedRef(
        schema_name=f"whetstone.test.{label}",
        content_hash="c" * 64,
    )


def _request(
    *,
    key: str = "evaluation:run-1:intent-1",
    identity_hash: str = _HASH_A,
    policy: ReplayPolicy = ReplayPolicy.IDEMPOTENT,
) -> EffectRequest:
    return EffectRequest(
        semantic_key=OpaqueKey(key),
        request_hash=IdentityHash(identity_hash),
        replay_policy=policy,
    )


def _acquire(
    authority: EffectAuthority,
    request: EffectRequest,
    *,
    owner: str = "worker-1",
    attempt: str = "attempt-1",
    duration: timedelta = _LEASE_DURATION,
):
    return authority.acquire(
        request,
        owner_id=owner,
        attempt_id=attempt,
        lease_duration=duration,
    )
