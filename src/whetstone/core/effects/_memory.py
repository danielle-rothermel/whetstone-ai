"""Process-local effect authority storage."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from threading import RLock

from whetstone.core.effects._storage import _T, _Transition
from whetstone.core.effects.models import _EffectRow, _require_utc


class _MemoryStore:
    def __init__(self, clock: Callable[[], datetime]) -> None:
        self._rows: dict[str, _EffectRow] = {}
        self._lock = RLock()
        self._clock = clock

    def initialize(self) -> None:
        pass

    def validate_lease_duration(self, duration: timedelta) -> timedelta:
        return duration

    def transaction(
        self,
        semantic_key: str,
        transition: _Transition[_T],
    ) -> _T:
        with self._lock:
            now = _require_utc(self._clock(), field="authority clock")
            updated, result = transition(self._rows.get(semantic_key), now)
            if updated is None:
                self._rows.pop(semantic_key, None)
            else:
                self._rows[semantic_key] = updated
            return result

    def close(self) -> None:
        pass
