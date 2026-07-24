"""Deterministic assembly, bounds, deadlines, and rate-limit behavior."""

from __future__ import annotations

import threading
import time

import pytest

from whetstone.execution.fanout import (
    RUNNER_TIMEOUT_CODE,
    CallSpec,
    RateLimitController,
    run_call_pool,
)


def _spec(
    key: int,
    *,
    sleep: float = 0.0,
    value: object = None,
    deadline: float = 30.0,
) -> CallSpec[int, object]:
    def run() -> object:
        if sleep:
            time.sleep(sleep)
        return value if value is not None else key

    return CallSpec(key=key, run=run, deadline_seconds=deadline)


def _never_rate_limited(_value: object) -> bool:
    return False


def test_assembly_preserves_input_order_across_completion_orders() -> None:
    count = 10
    specs = [
        _spec(index, sleep=(count - index) * 0.005) for index in range(count)
    ]
    outcome = run_call_pool(
        specs,
        concurrency=count,
        is_rate_limited=_never_rate_limited,
    )
    assert [result.key for result in outcome.results] == list(range(count))
    assert [result.value for result in outcome.results] == list(range(count))


def test_concurrency_is_bounded() -> None:
    peak = 0
    live = 0
    lock = threading.Lock()

    def run() -> int:
        nonlocal live, peak
        with lock:
            live += 1
            peak = max(peak, live)
        time.sleep(0.01)
        with lock:
            live -= 1
        return 1

    specs = [
        CallSpec(key=index, run=run, deadline_seconds=30.0)
        for index in range(12)
    ]
    run_call_pool(
        specs,
        concurrency=3,
        is_rate_limited=_never_rate_limited,
    )
    assert peak <= 3


def test_guard_timeout_is_classified_and_other_work_completes() -> None:
    logs: list[str] = []
    outcome = run_call_pool(
        [
            _spec(0, value="ok"),
            _spec(1, sleep=0.1, deadline=0.01),
            _spec(2, value="still-ok"),
        ],
        concurrency=3,
        is_rate_limited=_never_rate_limited,
        log=logs.append,
    )
    by_key = {result.key: result for result in outcome.results}
    assert by_key[0].value == "ok"
    assert by_key[1].timed_out
    assert by_key[2].value == "still-ok"
    assert outcome.guard_timeouts == 1
    assert any(RUNNER_TIMEOUT_CODE in line for line in logs)


def test_rate_limit_halves_effective_concurrency_once() -> None:
    outcome = run_call_pool(
        [_spec(0, value="limited"), *[_spec(i) for i in range(1, 6)]],
        concurrency=4,
        is_rate_limited=lambda value: value == "limited",
    )
    assert outcome.concurrency_halved
    assert outcome.effective_concurrency == 2

    controller = RateLimitController(1)
    controller.note_rate_limited()
    controller.note_rate_limited()
    assert controller.effective == 1
    assert controller.halved


def test_whole_operation_deadline_stops_dispatch_in_input_order() -> None:
    tick = [1000.0]

    def clock() -> float:
        tick[0] += 10.0
        return tick[0]

    outcome = run_call_pool(
        [_spec(index) for index in range(4)],
        concurrency=4,
        is_rate_limited=_never_rate_limited,
        max_wall_seconds=0.0,
        clock=clock,
    )
    assert outcome.deadline_reached
    assert outcome.not_dispatched == [0, 1, 2, 3]
    assert [result.key for result in outcome.results] == [0, 1, 2, 3]


def test_invalid_bounds_fail_before_dispatch() -> None:
    with pytest.raises(ValueError, match="positive"):
        run_call_pool([], concurrency=0, is_rate_limited=lambda _value: False)
    with pytest.raises(ValueError, match="cannot be negative"):
        run_call_pool(
            [],
            max_wall_seconds=-1,
            is_rate_limited=lambda _value: False,
        )
