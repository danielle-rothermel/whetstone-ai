from __future__ import annotations

import sqlite3
import time
from datetime import datetime
from pathlib import Path
from threading import Event

_AUTHORITY_NOW = """
SELECT strftime('%Y-%m-%dT%H:%M:%f000+00:00', 'now')
"""
_POLL_INTERVAL_SECONDS = 0.005
_WATCHDOG_SECONDS = 10.0
_POLL_GATE = Event()


def sqlite_authority_now(database: Path) -> datetime:
    with sqlite3.connect(database) as connection:
        row = connection.execute(_AUTHORITY_NOW).fetchone()
    if row is None or type(row[0]) is not str:
        raise AssertionError("SQLite did not return text authority time")
    return datetime.fromisoformat(row[0])


def wait_for_sqlite_authority_after(
    database: Path,
    instant: datetime,
    *,
    watchdog_seconds: float = _WATCHDOG_SECONDS,
) -> datetime:
    deadline = time.monotonic() + watchdog_seconds
    while True:
        now = sqlite_authority_now(database)
        if now > instant:
            return now
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AssertionError(
                f"SQLite authority time did not pass {instant.isoformat()}"
            )
        # This wait only throttles queries; the database predicate proves
        # success.
        _POLL_GATE.wait(timeout=min(_POLL_INTERVAL_SECONDS, remaining))
