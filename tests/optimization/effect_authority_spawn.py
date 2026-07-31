"""Spawn-safe workers for effect-authority process tests."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

from whetstone.optimization.effect_authority import (
    EffectAuthority,
    EffectRequest,
)


def race_acquire(
    database_path: str,
    request_payload: dict[str, Any],
    owner_id: str,
    attempt_id: str,
    lease_seconds: float,
    start: Any,
    output: Any,
) -> None:
    """Acquire from an independently constructed SQLite authority."""
    authority = EffectAuthority.sqlite(Path(database_path))
    start.wait()
    result = authority.acquire(
        EffectRequest.model_validate(request_payload),
        owner_id=owner_id,
        attempt_id=attempt_id,
        lease_duration=timedelta(seconds=lease_seconds),
    )
    output.put(result.model_dump(mode="json"))


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
