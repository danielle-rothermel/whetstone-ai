from __future__ import annotations

from datetime import UTC, datetime, timedelta
from threading import Condition, Event, Lock

from whetstone.core.effects.authority import (
    EffectAuthority,
    EffectRequest,
    ReplayPolicy,
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
