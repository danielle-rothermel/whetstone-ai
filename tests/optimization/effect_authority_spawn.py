"""Spawn-safe workers for effect-authority process tests."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

from whetstone.optimization.effect_authority import (
    EffectAuthority,
    EffectRequest,
)


class _TransactionSignals:
    def __init__(self, attempted: Any, acquired: Any) -> None:
        self._attempted = attempted
        self._acquired = acquired

    def transaction_attempted(self) -> None:
        self._attempted.set()

    def transaction_acquired(self) -> None:
        self._acquired.set()


def race_acquire(
    database_path: str,
    request_payload: dict[str, Any],
    owner_id: str,
    attempt_id: str,
    lease_seconds: float,
    ready: Any,
    start: Any,
    attempted: Any,
    acquired: Any,
    output: Any,
) -> None:
    """Acquire from an independently constructed SQLite authority."""
    try:
        authority = EffectAuthority.sqlite(
            Path(database_path),
            _transaction_observer=_TransactionSignals(attempted, acquired),
        )
        request = EffectRequest.model_validate(request_payload)
        ready.set()
        if not start.wait(timeout=60):
            raise TimeoutError("authority race worker was not released")
        result = authority.acquire(
            request,
            owner_id=owner_id,
            attempt_id=attempt_id,
            lease_duration=timedelta(seconds=lease_seconds),
        )
        output.put(result.model_dump(mode="json"))
    except BaseException as exc:
        output.put({"error": f"{type(exc).__name__}: {exc}"})


def acquire_then_exit(
    database_path: str,
    request_payload: dict[str, Any],
    owner_id: str,
    attempt_id: str,
    lease_seconds: float,
    output: Any,
) -> None:
    """Acquire a lease and exit without terminalizing it."""
    authority = EffectAuthority.sqlite(Path(database_path))
    result = authority.acquire(
        EffectRequest.model_validate(request_payload),
        owner_id=owner_id,
        attempt_id=attempt_id,
        lease_duration=timedelta(seconds=lease_seconds),
    )
    output.put(result.model_dump(mode="json"))
