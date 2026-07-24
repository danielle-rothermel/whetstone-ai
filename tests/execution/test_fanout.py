"""The bounded-concurrency worker pool: determinism, guard, rate-limit, halt.

The load-bearing property is DETERMINISTIC RECORDED assembly: results are keyed
and reassembled in the caller's INPUT order regardless of which worker finishes
first. These tests shuffle completion order (a thunk that sleeps a per-key
amount so later-submitted units finish first) and assert byte-identical output,
plus cover the runner guard, the rate-limit halving, and the run deadline.
"""

from __future__ import annotations

import threading
import time

from whetstone.execution.fanout import (
    RUNNER_TIMEOUT_CODE,
    CallSpec,
    RateLimitController,
    run_call_pool,
)


def _spec(key: int, *, sleep: float = 0.0, value: object = None,
          deadline: float = 30.0) -> CallSpec[int, object]:
    def _run() -> object:
        if sleep:
            time.sleep(sleep)
        return value if value is not None else key

    return CallSpec(key=key, run=_run, deadline_seconds=deadline)


def _never_rate_limited(_v: object) -> bool:
    return False


def test_assembly_is_input_order_despite_shuffled_completion() -> None:
    # Later-submitted units sleep LESS, so they finish FIRST -- completion
    # order is the reverse of input order. The assembled results must still be
    # input order (keys 0..9), so the artifact never leaks completion order.
    n = 10
    specs = [_spec(i, sleep=(n - i) * 0.01) for i in range(n)]
    outcome = run_call_pool(
        specs, concurrency=n, is_rate_limited=_never_rate_limited
    )
    assert [r.key for r in outcome.results] == list(range(n))
    assert [r.value for r in outcome.results] == list(range(n))


def test_determinism_across_repeated_shuffled_runs() -> None:
    # Two independent runs with shuffled completion produce identical output.
    def _run_once() -> list[tuple[int, object]]:
        specs = [_spec(i, sleep=((i * 7) % 5) * 0.005) for i in range(12)]
        out = run_call_pool(
            specs, concurrency=6, is_rate_limited=_never_rate_limited
        )
        return [(r.key, r.value) for r in out.results]

    assert _run_once() == _run_once()


def test_concurrency_is_bounded() -> None:
    # At most `concurrency` thunks run at once: a live-counter peaks at <= 3.
    peak = 0
    live = 0
    lock = threading.Lock()

    def _run() -> int:
        nonlocal peak, live
        with lock:
            live += 1
            peak = max(peak, live)
        time.sleep(0.02)
        with lock:
            live -= 1
        return 1

    specs = [
        CallSpec(key=i, run=_run, deadline_seconds=30.0) for i in range(12)
    ]
    run_call_pool(specs, concurrency=3, is_rate_limited=_never_rate_limited)
    assert peak <= 3


def test_runner_guard_times_out_and_continues() -> None:
    # One unit exceeds its deadline: it is flagged timed_out (a typed
    # runner-timeout the caller records) and the OTHER units still complete.
    logs: list[str] = []
    specs = [
        _spec(0, value="ok", deadline=30.0),
        _spec(1, sleep=1.0, deadline=0.05),  # breaches its 0.05s guard
        _spec(2, value="ok2", deadline=30.0),
    ]
    outcome = run_call_pool(
        specs,
        concurrency=3,
        is_rate_limited=_never_rate_limited,
        log=logs.append,
    )
    by_key = {r.key: r for r in outcome.results}
    assert by_key[0].value == "ok"
    assert by_key[2].value == "ok2"
    assert by_key[1].timed_out
    assert outcome.guard_timeouts == 1
    # The guard logs LOUDLY when it fires (transport bound regression signal).
    assert any(RUNNER_TIMEOUT_CODE in line for line in logs)
    assert any("did NOT" in line for line in logs)


def test_rate_limit_halves_effective_concurrency() -> None:
    # A single rate-limited result halves the effective concurrency (once).
    def _rl(v: object) -> bool:
        return v == "RL"

    specs = [_spec(0, value="RL")] + [_spec(i) for i in range(1, 6)]
    outcome = run_call_pool(specs, concurrency=4, is_rate_limited=_rl)
    assert outcome.concurrency_halved
    assert outcome.effective_concurrency == 2  # 4 // 2


def test_rate_limit_controller_never_below_one() -> None:
    controller = RateLimitController(1)
    controller.note_rate_limited()
    assert controller.effective == 1
    assert controller.halved


def test_whole_run_deadline_stops_dispatch() -> None:
    # A zero-budget deadline stops dispatch immediately: nothing runs, every
    # unit is flagged not_dispatched (the caller persists partials + halts).
    clock_value = [1000.0]

    def _clock() -> float:
        # First call (start) is 1000; every check after is already past the
        # zero budget, so no unit is dispatched.
        clock_value[0] += 10.0
        return clock_value[0]

    specs = [_spec(i, value=f"v{i}") for i in range(4)]
    outcome = run_call_pool(
        specs,
        concurrency=4,
        is_rate_limited=_never_rate_limited,
        max_wall_seconds=0.0,
        clock=_clock,
    )
    assert outcome.deadline_reached
    assert outcome.not_dispatched == [0, 1, 2, 3]
    # Order is still preserved for the not-dispatched units.
    assert [r.key for r in outcome.results] == [0, 1, 2, 3]
