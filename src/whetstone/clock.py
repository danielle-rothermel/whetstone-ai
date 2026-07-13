"""Whetstone-owned clock boundary for deterministic domain work."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

type Clock = Callable[[], datetime]


def system_clock() -> datetime:
    return datetime.now(UTC)


_clock: Clock = system_clock


def now() -> datetime:
    return _clock()


def set_clock(clock: Clock) -> None:
    global _clock
    _clock = clock
