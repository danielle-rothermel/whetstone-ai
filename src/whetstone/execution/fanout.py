"""Bounded-concurrency worker pool for live evaluation fan-out.

The validation runner drives many provider calls per phase (a pilot's
instances x probes x repeats; a cell's official/internal splits x repeats).
Round-1/round-2 pilots ran them strictly sequentially, so a single slow call
stalled the whole phase and a crash lost every completed call. This module is
the shared fan-out primitive the pilot and cell phases both drive through:

* **Bounded concurrency** -- a thread pool runs at most ``concurrency``
  provider calls at once (default 5, ``--concurrency``). Provider calls are
  I/O-bound
  (a blocking HTTP round-trip inside dr-providers), so threads give real
  overlap without touching the pure driver.

* **Deterministic RECORDED assembly** -- every unit carries a stable ``key``
  ``(instance, probe/candidate, repeat)``. Results are collected into a
  key->outcome map and reassembled in the caller's INPUT order, so the recorded
  artifact is byte-identical no matter which worker finishes first. A test
  shuffles completion order with a fake transport and asserts identical output.

* **Runner-level call guard (belt-and-suspenders)** -- each call runs under a
  wall-clock deadline = the transport policy timeout + a 10s margin.
  dr-providers now enforces its own bound, so this should never fire; if it
  does, the
  transport bound failed, so we log LOUDLY and record a typed runner-timeout
  failure for that one call and CONTINUE (never abort the phase).

* **Shared rate-limit backpressure** -- all lanes share one key, so a 429 /
  rate-limit typed failure on ANY call halves the effective concurrency for the
  REST of the run (once). Simple and header-free: no adaptive parsing.

* **Whole-run deadline** -- an optional ``max_wall_seconds`` stops DISPATCHING
  new calls once breached; in-flight calls finish, and the un-dispatched units
  are reported so the caller can persist partials and exit with a halt note.

Nothing here makes a live paid call by itself: the work each unit performs is a
caller-supplied thunk, driven in tests through the same injected fake transport
as the rest of the runner.
"""

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

#: Default number of concurrent provider calls per phase.
DEFAULT_CONCURRENCY = 5

#: The extra margin (seconds) the runner-level guard adds on top of the
#: transport policy's absolute wall-clock CAP before it declares a
#: belt-and-suspenders breach. Aligned with the new transport semantics: the
#: transport bounds a SINGLE call at ``timeout_seconds`` (the absolute cap),
#: its own ~5s deadline margin, so the runner guard = cap + 15s sits just above
#: the transport's own single-call bound (was ``cap x max_attempts + 10`` under
#: the old flat-deadline model, which let 3 stacked semantic retries exceed the
#: guard and fire it before the transport bound could).
GUARD_MARGIN_SECONDS = 15.0

#: The failure ``code`` recorded for a call the runner-level guard timed out.
#: A DISTINCT code from the transport's own ``timeout``/``stalled_response`` so
#: a report can tell "the transport bound failed" apart from a transport-owned
#: timeout.
RUNNER_TIMEOUT_CODE = "runner_timeout"


def _default_log(message: str) -> None:
    """The default loud-log sink for guard timeouts (stderr, returns None)."""
    sys.stderr.write(message)


@dataclass(frozen=True, slots=True)
class FanoutConfig:
    """The bounded-concurrency knobs threaded through an evaluation phase.

    ``concurrency`` is the max simultaneous provider calls (default 5, halved
    once on a rate-limit failure). ``max_wall_seconds`` is the optional
    whole-phase deadline (``None`` = unbounded). A ``FanoutConfig`` with the
    defaults reproduces the pre-concurrency behavior closely enough that
    RECORDED artifacts are byte-identical (assembly is always by key in input
    order).
    """

    concurrency: int = DEFAULT_CONCURRENCY
    max_wall_seconds: float | None = None


class RunnerTimeout(RuntimeError):
    """The runner-level call guard breached its deadline for one call.

    A belt-and-suspenders signal only: the transport is meant to enforce its
    own wall-clock bound, so this firing means that bound failed. Recorded as a
    typed per-call failure (:data:`RUNNER_TIMEOUT_CODE`); never aborts a phase.
    """


@dataclass(frozen=True, slots=True)
class CallSpec[K: Hashable, R]:
    """One keyed unit of fan-out work.

    ``key`` is the stable RECORDED identity ``(instance, probe/candidate,
    repeat)``; the pool assembles results by this key in the caller's input
    order, so completion order never leaks into the artifact. ``run`` is a
    zero-arg thunk performing the (transport-injected) provider call and
    returning the caller's per-call result. ``deadline_seconds`` is the
    runner-level guard budget for this call (transport timeout + margin).
    """

    key: K
    run: Callable[[], R]
    deadline_seconds: float


@dataclass(frozen=True, slots=True)
class FanoutResult[K: Hashable, R]:
    """One unit's outcome: either a value, a guard timeout, or un-dispatched.

    Exactly one of ``value`` / ``timed_out`` / ``not_dispatched`` is
    meaningful. ``timed_out`` marks a runner-guard breach (the caller records a
    typed runner-timeout failure). ``not_dispatched`` marks a unit the
    whole-run deadline stopped before it started (finish in-flight, persist,
    halt).
    """

    key: K
    value: R | None = None
    timed_out: bool = False
    not_dispatched: bool = False

    @property
    def completed(self) -> bool:
        """True when the unit ran to a normal (non-timeout) result."""
        return not self.timed_out and not self.not_dispatched


@dataclass(slots=True)
class PoolOutcome[K: Hashable, R]:
    """The pool's terminal report over all units (input order preserved)."""

    results: list[FanoutResult[K, R]]
    effective_concurrency: int
    concurrency_halved: bool
    deadline_reached: bool
    guard_timeouts: int

    @property
    def not_dispatched(self) -> list[K]:
        return [r.key for r in self.results if r.not_dispatched]


class RateLimitController:
    """Shared, all-lanes-one-key concurrency gate that halves on rate limit.

    A worker acquires before its call and releases after. On the FIRST observed
    429 / rate-limit typed failure the effective permit count is halved (never
    below 1) for the rest of the run -- a single, header-free step-down. The
    halving is recorded so the run report can show it fired.
    """

    def __init__(self, concurrency: int) -> None:
        self._initial = max(1, concurrency)
        self._effective = self._initial
        self._lock = threading.Lock()
        # A bounded semaphore whose "budget" we shrink by absorbing permits.
        self._sem = threading.BoundedSemaphore(self._initial)
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
        self._sem.acquire()

    def release(self) -> None:
        # Absorbing permits (see halve) can push the semaphore's value above
        # its shrunk budget; releasing a permit that was already absorbed would
        # raise ValueError on a BoundedSemaphore, so guard it.
        try:
            self._sem.release()
        except ValueError:  # pragma: no cover - only if over-released
            pass

    def note_rate_limited(self) -> None:
        """Halve effective concurrency once, on the first rate-limit failure.

        Absorbs ``effective - target`` permits from the semaphore so no more
        than ``target`` workers proceed for the rest of the run. Absorption
        runs in a background waiter so a worker reporting a 429 never blocks
        holding a permit it still needs to release.
        """
        with self._lock:
            if self._halved:
                return
            self._halved = True
            target = max(1, self._effective // 2)
            to_absorb = self._effective - target
            self._effective = target
        for _ in range(to_absorb):
            # Consume a permit permanently (in a thread so we never deadlock
            # the reporting worker, which still holds its own permit).
            threading.Thread(
                target=self._sem.acquire, daemon=True
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

        def _work(spec: CallSpec[K, R]) -> R:
            controller.acquire()
            try:
                return spec.run()
            finally:
                controller.release()

        # A generous pool size: the controller (not the pool) is the true
        # concurrency gate, so the pool must not itself become the bottleneck
        # below the initial concurrency.
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            pending: dict[Future[R], CallSpec[K, R]] = {}
            for spec in self.specs:
                if deadline is not None and self.clock() >= deadline:
                    deadline_reached = True
                    by_key[spec.key] = FanoutResult(
                        key=spec.key, not_dispatched=True
                    )
                    continue
                pending[pool.submit(_work, spec)] = spec

            for future, spec in pending.items():
                remaining = spec.deadline_seconds
                try:
                    value = future.result(timeout=remaining)
                except TimeoutError:
                    guard_timeouts += 1
                    self.log(
                        "RUNNER GUARD TIMEOUT: call "
                        f"key={spec.key!r} exceeded "
                        f"{spec.deadline_seconds:.1f}s (transport policy "
                        "timeout + margin). The transport's OWN wall-clock "
                        "bound should have fired first -- it did NOT. This is "
                        "a transport-bound regression; recording a typed "
                        f"{RUNNER_TIMEOUT_CODE} failure and CONTINUING.\n"
                    )
                    by_key[spec.key] = FanoutResult(
                        key=spec.key, timed_out=True
                    )
                    continue
                if self.is_rate_limited(value):
                    controller.note_rate_limited()
                by_key[spec.key] = FanoutResult(key=spec.key, value=value)

        # Reassemble in the caller's INPUT order (completion order discarded).
        ordered = [by_key[spec.key] for spec in self.specs]
        return PoolOutcome(
            results=ordered,
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
    """Run ``specs`` through a bounded worker pool, assembling by key in order.

    Args:
        specs: the ordered keyed units. Assembly is by ``spec.key`` in this
            exact order, so completion order never affects the output.
        concurrency: max concurrent calls (default 5). Halved once on the first
            rate-limit failure for the rest of the run.
        is_rate_limited: predicate over one unit's result -- True marks a 429 /
            rate-limit typed failure that should halve concurrency.
        max_wall_seconds: optional whole-run deadline. Once breached, no new
            unit is dispatched; in-flight units finish; un-dispatched units are
            returned flagged ``not_dispatched`` for the caller to persist+halt.
        clock: injectable monotonic clock (tests drive it deterministically).
        log: injectable loud-log sink for guard timeouts (default stderr).

    Returns:
        A :class:`PoolOutcome` whose ``results`` are in input order.
    """
    if concurrency < 1:
        raise ValueError("concurrency must be a positive integer")
    dispatcher: _Dispatcher[K, R] = _Dispatcher(
        specs=specs,
        concurrency=concurrency,
        is_rate_limited=is_rate_limited,
        max_wall_seconds=max_wall_seconds,
        clock=clock,
        log=log or _default_log,
    )
    return dispatcher.run()
