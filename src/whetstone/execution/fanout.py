"""Bounded concurrent fanout with deterministic result assembly."""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable, Hashable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

__all__ = [
    "DEFAULT_CONCURRENCY",
    "GUARD_MARGIN_SECONDS",
    "RUNNER_TIMEOUT_CODE",
    "CallSpec",
    "FanoutConfig",
    "FanoutResult",
    "PoolOutcome",
    "RateLimitController",
    "RunnerTimeout",
    "run_call_pool",
]

DEFAULT_CONCURRENCY = 5
GUARD_MARGIN_SECONDS = 15.0
RUNNER_TIMEOUT_CODE = "runner_timeout"


def _default_log(message: str) -> None:
    sys.stderr.write(message)


@dataclass(frozen=True, slots=True)
class FanoutConfig:
    """Concurrency and optional whole-operation deadline."""

    concurrency: int = DEFAULT_CONCURRENCY
    max_wall_seconds: float | None = None


class RunnerTimeout(RuntimeError):
    """A fanout unit exceeded its transport guard deadline."""


@dataclass(frozen=True, slots=True)
class CallSpec[K: Hashable, R]:
    """One stably keyed unit of fanout work."""

    key: K
    run: Callable[[], R]
    deadline_seconds: float


@dataclass(frozen=True, slots=True)
class FanoutResult[K: Hashable, R]:
    """The value or non-completion state for one fanout unit."""

    key: K
    value: R | None = None
    timed_out: bool = False
    not_dispatched: bool = False

    @property
    def completed(self) -> bool:
        return not self.timed_out and not self.not_dispatched


@dataclass(frozen=True, slots=True)
class PoolOutcome[K: Hashable, R]:
    """Terminal pool report, preserving caller input order."""

    results: tuple[FanoutResult[K, R], ...]
    effective_concurrency: int
    concurrency_halved: bool
    deadline_reached: bool
    guard_timeouts: int

    @property
    def not_dispatched(self) -> list[K]:
        return [result.key for result in self.results if result.not_dispatched]


class RateLimitController:
    """A shared concurrency gate that halves once after rate limiting."""

    def __init__(self, concurrency: int) -> None:
        self._initial = max(1, concurrency)
        self._effective = self._initial
        self._lock = threading.Lock()
        self._semaphore = threading.BoundedSemaphore(self._initial)
        self._halved = False

    @property
    def effective(self) -> int:
        with self._lock:
            return self._effective

    @property
    def halved(self) -> bool:
        with self._lock:
            return self._halved

    def acquire(self) -> None:
        self._semaphore.acquire()

    def release(self) -> None:
        try:
            self._semaphore.release()
        except ValueError:  # pragma: no cover - defensive over-release guard
            pass

    def note_rate_limited(self) -> None:
        """Permanently absorb permits down to half the initial capacity."""
        with self._lock:
            if self._halved:
                return
            self._halved = True
            target = max(1, self._effective // 2)
            to_absorb = self._effective - target
            self._effective = target
        for _ in range(to_absorb):
            threading.Thread(
                target=self._semaphore.acquire,
                daemon=True,
            ).start()


@dataclass(slots=True)
class _Dispatcher[K: Hashable, R]:
    specs: Sequence[CallSpec[K, R]]
    concurrency: int
    is_rate_limited: Callable[[R], bool]
    max_wall_seconds: float | None
    clock: Callable[[], float] = time.monotonic
    log: Callable[[str], None] = field(default=_default_log)

    def run(self) -> PoolOutcome[K, R]:
        controller = RateLimitController(self.concurrency)
        by_key: dict[K, FanoutResult[K, R]] = {}
        guard_timeouts = 0
        deadline_reached = False
        start = self.clock()
        deadline = (
            start + self.max_wall_seconds
            if self.max_wall_seconds is not None
            else None
        )

        def work(spec: CallSpec[K, R]) -> R:
            controller.acquire()
            try:
                return spec.run()
            finally:
                controller.release()

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            pending: dict[Future[R], CallSpec[K, R]] = {}
            for spec in self.specs:
                if deadline is not None and self.clock() >= deadline:
                    deadline_reached = True
                    by_key[spec.key] = FanoutResult(
                        key=spec.key,
                        not_dispatched=True,
                    )
                    continue
                pending[pool.submit(work, spec)] = spec

            for future, spec in pending.items():
                try:
                    value = future.result(timeout=spec.deadline_seconds)
                except TimeoutError:
                    guard_timeouts += 1
                    self.log(
                        "RUNNER GUARD TIMEOUT: call "
                        f"key={spec.key!r} exceeded "
                        f"{spec.deadline_seconds:.1f}s (transport policy "
                        "timeout + margin). The transport's OWN wall-clock "
                        "bound should have fired first -- it did NOT. "
                        "Recording typed "
                        f"{RUNNER_TIMEOUT_CODE} failure and continuing.\n"
                    )
                    by_key[spec.key] = FanoutResult(
                        key=spec.key,
                        timed_out=True,
                    )
                    continue
                if self.is_rate_limited(value):
                    controller.note_rate_limited()
                by_key[spec.key] = FanoutResult(key=spec.key, value=value)

        return PoolOutcome(
            results=tuple(by_key[spec.key] for spec in self.specs),
            effective_concurrency=controller.effective,
            concurrency_halved=controller.halved,
            deadline_reached=deadline_reached,
            guard_timeouts=guard_timeouts,
        )


def run_call_pool[K: Hashable, R](
    specs: Sequence[CallSpec[K, R]],
    *,
    concurrency: int = DEFAULT_CONCURRENCY,
    is_rate_limited: Callable[[R], bool],
    max_wall_seconds: float | None = None,
    clock: Callable[[], float] = time.monotonic,
    log: Callable[[str], None] | None = None,
) -> PoolOutcome[K, R]:
    """Run keyed work under bounded concurrency and deadline policies."""
    if concurrency < 1:
        raise ValueError("concurrency must be a positive integer")
    if max_wall_seconds is not None and max_wall_seconds < 0:
        raise ValueError("max_wall_seconds cannot be negative")
    if any(spec.deadline_seconds < 0 for spec in specs):
        raise ValueError("deadline_seconds cannot be negative")
    dispatcher: _Dispatcher[K, R] = _Dispatcher(
        specs=specs,
        concurrency=concurrency,
        is_rate_limited=is_rate_limited,
        max_wall_seconds=max_wall_seconds,
        clock=clock,
        log=log or _default_log,
    )
    return dispatcher.run()
