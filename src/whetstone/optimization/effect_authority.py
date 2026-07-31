"""Atomic lease and terminal-outcome authority for effectful work.

An authority prevents concurrent active workers and fences authoritative
persistence. It does not make an external effect exactly once: callers must
still use the stable semantic key at an idempotent provider boundary or place
the physical attempt inside a durable workflow.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import UNIQUE, StrEnum, verify
from pathlib import Path
from threading import Event, Lock, RLock, Thread
from typing import Any, Protocol, TypeVar, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictInt,
    field_validator,
    model_validator,
)

from whetstone.optimization.identity import (
    IdentityHash,
    NonEmptyId,
    OpaqueKey,
    TerminalFailure,
    TypedRef,
)

__all__ = [
    "AcquireOutcome",
    "AcquireResult",
    "EffectAuthority",
    "EffectAuthorityError",
    "EffectAuthoritySchemaMismatchError",
    "EffectLease",
    "EffectRequest",
    "EffectTerminal",
    "LeaseMaintenance",
    "ReplayPolicy",
    "StaleLeaseError",
    "TerminalConflictError",
    "TerminalFailure",
    "TerminalOutcome",
]

_TABLE_NAME = "whetstone_effect_authority"
_METADATA_TABLE_NAME = "whetstone_effect_authority_metadata"
_SCHEMA_VERSION = 2
_MAX_FENCE = (1 << 63) - 1
_SQLITE_MINIMUM_LEASE_DURATION = timedelta(milliseconds=1)
_RECOVERY_MESSAGE = (
    "the non-redrivable effect lease expired without a terminal outcome"
)


def _require_text(value: str, *, field: str, maximum: int = 1024) -> str:
    if not value or not value.strip():
        raise ValueError(f"{field} must be non-empty")
    if "\x00" in value:
        raise ValueError(f"{field} cannot contain NUL")
    if len(value) > maximum:
        raise ValueError(f"{field} cannot exceed {maximum} characters")
    return value


def _require_utc(value: datetime, *, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    if value.utcoffset() != UTC.utcoffset(value):
        raise ValueError(f"{field} must use UTC")
    return value


def _require_lease_duration(value: timedelta) -> timedelta:
    if not isinstance(value, timedelta):
        raise TypeError("lease_duration must be a timedelta")
    seconds = value.total_seconds()
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("lease_duration must be positive")
    return value


@verify(UNIQUE)
class ReplayPolicy(StrEnum):
    """Whether an expired effect may be assigned to a new physical attempt.

    These values are persisted contract literals. Never iterate over this
    enum to construct a persisted payload.
    """

    IDEMPOTENT = "idempotent"
    DURABLE_WORKFLOW = "durable_workflow"
    NO_REDRIVE = "no_redrive"


@verify(UNIQUE)
class TerminalOutcome(StrEnum):
    """Persisted terminal state; never iterate it to construct a payload."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    RECOVERY_REQUIRED = "recovery_required"


@verify(UNIQUE)
class AcquireOutcome(StrEnum):
    """Serialized acquisition result; never iterate it to build a payload."""

    ACQUIRED = "acquired"
    BUSY = "busy"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REQUEST_CONFLICT = "request_conflict"
    RECOVERY_REQUIRED = "recovery_required"


@verify(UNIQUE)
class _StoredState(StrEnum):
    LEASED = "leased"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    RECOVERY_REQUIRED = "recovery_required"


class EffectRequest(BaseModel):
    """Immutable identity of one semantic effect."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    semantic_key: OpaqueKey
    request_identity_hash: IdentityHash
    replay_policy: ReplayPolicy

    @model_validator(mode="after")
    def _validate(self) -> EffectRequest:
        _require_text(self.semantic_key, field="semantic_key")
        return self


class EffectLease(BaseModel):
    """Exact authority token required for renewal and terminalization."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    request: EffectRequest
    owner_id: NonEmptyId
    attempt_id: NonEmptyId
    fence: StrictInt
    expires_at: datetime

    @field_validator("expires_at")
    @classmethod
    def _validate_expires_at(cls, value: datetime) -> datetime:
        return _require_utc(value, field="expires_at")

    @model_validator(mode="after")
    def _validate(self) -> EffectLease:
        _require_text(self.owner_id, field="owner_id", maximum=255)
        _require_text(self.attempt_id, field="attempt_id", maximum=255)
        if not 1 <= self.fence <= _MAX_FENCE:
            raise ValueError("fence must be a positive signed 64-bit integer")
        return self


class EffectTerminal(BaseModel):
    """Immutable authoritative terminal outcome."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    request: EffectRequest
    outcome: TerminalOutcome
    owner_id: NonEmptyId
    attempt_id: NonEmptyId
    fence: StrictInt
    result_ref: TypedRef | None = None
    failure: TerminalFailure | None = None

    @model_validator(mode="after")
    def _validate(self) -> EffectTerminal:
        _require_text(self.owner_id, field="owner_id", maximum=255)
        _require_text(self.attempt_id, field="attempt_id", maximum=255)
        if not 1 <= self.fence <= _MAX_FENCE:
            raise ValueError("fence must be a positive signed 64-bit integer")
        if self.outcome is TerminalOutcome.SUCCEEDED:
            if self.result_ref is None or self.failure is not None:
                raise ValueError(
                    "a succeeded terminal requires only result_ref"
                )
        elif self.outcome is TerminalOutcome.FAILED:
            if self.result_ref is None or self.failure is None:
                raise ValueError(
                    "a failed terminal requires result_ref and failure"
                )
        elif self.failure is None or self.result_ref is not None:
            raise ValueError(
                "a recovery-required terminal requires only failure"
            )
        return self


class AcquireResult(BaseModel):
    """Typed result of acquiring or replaying one semantic effect."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    request: EffectRequest
    outcome: AcquireOutcome
    lease: EffectLease | None = None
    terminal: EffectTerminal | None = None
    busy_expires_at: datetime | None = None
    existing_request_identity_hash: IdentityHash | None = None
    existing_replay_policy: ReplayPolicy | None = None

    @field_validator("busy_expires_at")
    @classmethod
    def _validate_busy_expires_at(
        cls, value: datetime | None
    ) -> datetime | None:
        if value is not None:
            _require_utc(value, field="busy_expires_at")
        return value

    @model_validator(mode="after")
    def _validate(self) -> AcquireResult:
        populated = (
            self.lease is not None,
            self.terminal is not None,
            self.busy_expires_at is not None,
            self.existing_request_identity_hash is not None,
            self.existing_replay_policy is not None,
        )
        if self.outcome is AcquireOutcome.ACQUIRED:
            if populated != (True, False, False, False, False):
                raise ValueError("ACQUIRED requires only a lease")
            if self.lease is not None and self.lease.request != self.request:
                raise ValueError("acquired lease must match the request")
        elif self.outcome is AcquireOutcome.BUSY:
            if populated != (False, False, True, False, False):
                raise ValueError("BUSY requires only busy_expires_at")
        elif self.outcome is AcquireOutcome.REQUEST_CONFLICT:
            if populated != (False, False, False, True, True):
                raise ValueError(
                    "REQUEST_CONFLICT requires the existing identity "
                    "and policy"
                )
        else:
            if populated != (False, True, False, False, False):
                raise ValueError(
                    "terminal acquisition outcomes require only terminal"
                )
            expected = {
                AcquireOutcome.SUCCEEDED: TerminalOutcome.SUCCEEDED,
                AcquireOutcome.FAILED: TerminalOutcome.FAILED,
                AcquireOutcome.RECOVERY_REQUIRED: (
                    TerminalOutcome.RECOVERY_REQUIRED
                ),
            }[self.outcome]
            if self.terminal is not None:
                if self.terminal.request != self.request:
                    raise ValueError("terminal must match the request")
                if self.terminal.outcome is not expected:
                    raise ValueError("terminal outcome does not match result")
        return self


class EffectAuthorityError(RuntimeError):
    """Base error for non-acquisition authority transitions."""


class EffectAuthoritySchemaMismatchError(EffectAuthorityError):
    """The durable authority schema does not match its exact contract."""


class StaleLeaseError(EffectAuthorityError):
    """The supplied owner/fence is no longer authorized."""


class TerminalConflictError(EffectAuthorityError):
    """A different immutable terminal outcome is already authoritative."""


class _AuthorityCorruptionError(EffectAuthorityError):
    pass


@dataclass(frozen=True, slots=True)
class _EffectRow:
    request: EffectRequest
    state: _StoredState
    owner_id: NonEmptyId
    attempt_id: NonEmptyId
    fence: int
    expires_at: datetime | None
    terminal: EffectTerminal | None

    @classmethod
    def leased(cls, lease: EffectLease) -> _EffectRow:
        return cls(
            request=lease.request,
            state=_StoredState.LEASED,
            owner_id=lease.owner_id,
            attempt_id=lease.attempt_id,
            fence=lease.fence,
            expires_at=lease.expires_at,
            terminal=None,
        )

    @classmethod
    def terminalized(cls, terminal: EffectTerminal) -> _EffectRow:
        return cls(
            request=terminal.request,
            state=_StoredState(terminal.outcome.value),
            owner_id=terminal.owner_id,
            attempt_id=terminal.attempt_id,
            fence=terminal.fence,
            expires_at=None,
            terminal=terminal,
        )

    def lease(self) -> EffectLease:
        if self.state is not _StoredState.LEASED or self.expires_at is None:
            raise _AuthorityCorruptionError("row is not an active lease")
        return EffectLease(
            request=self.request,
            owner_id=self.owner_id,
            attempt_id=self.attempt_id,
            fence=self.fence,
            expires_at=self.expires_at,
        )


_T = TypeVar("_T")
_Transition = Callable[
    [_EffectRow | None, datetime], tuple[_EffectRow | None, _T]
]


class _Store(Protocol):
    def initialize(self) -> None: ...

    def validate_lease_duration(self, duration: timedelta) -> timedelta: ...

    def transaction(
        self,
        semantic_key: str,
        transition: _Transition[_T],
    ) -> _T: ...

    def close(self) -> None: ...


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


_SQLITE_CREATE_TABLE = f"""
CREATE TABLE {_TABLE_NAME} (
    semantic_key TEXT PRIMARY KEY CHECK (typeof(semantic_key) = 'text'),
    request_identity_hash TEXT NOT NULL CHECK (
        typeof(request_identity_hash) = 'text'
    ),
    replay_policy TEXT NOT NULL CHECK (
        typeof(replay_policy) = 'text'
        AND replay_policy IN ('idempotent', 'durable_workflow', 'no_redrive')
    ),
    state TEXT NOT NULL CHECK (
        typeof(state) = 'text'
        AND state IN (
            'leased', 'succeeded', 'failed', 'recovery_required'
        )
    ),
    owner_id TEXT NOT NULL CHECK (typeof(owner_id) = 'text'),
    attempt_id TEXT NOT NULL CHECK (typeof(attempt_id) = 'text'),
    fence INTEGER NOT NULL CHECK (typeof(fence) = 'integer' AND fence > 0),
    expires_at TEXT CHECK (
        expires_at IS NULL OR typeof(expires_at) = 'text'
    ),
    terminal_json TEXT CHECK (
        terminal_json IS NULL OR typeof(terminal_json) = 'text'
    ),
    CHECK (
        (state = 'leased' AND expires_at IS NOT NULL
            AND terminal_json IS NULL)
        OR
        (state != 'leased' AND expires_at IS NULL
            AND terminal_json IS NOT NULL)
    )
)
"""

_SQLITE_CREATE_METADATA_TABLE = f"""
CREATE TABLE {_METADATA_TABLE_NAME} (
    singleton INTEGER PRIMARY KEY CHECK (
        typeof(singleton) = 'integer' AND singleton = 1
    ),
    schema_version INTEGER NOT NULL CHECK (
        typeof(schema_version) = 'integer'
    )
)
"""

_SQLITE_TABLE_COLUMNS = (
    ("semantic_key", "TEXT", 0, None, 1),
    ("request_identity_hash", "TEXT", 1, None, 0),
    ("replay_policy", "TEXT", 1, None, 0),
    ("state", "TEXT", 1, None, 0),
    ("owner_id", "TEXT", 1, None, 0),
    ("attempt_id", "TEXT", 1, None, 0),
    ("fence", "INTEGER", 1, None, 0),
    ("expires_at", "TEXT", 0, None, 0),
    ("terminal_json", "TEXT", 0, None, 0),
)
_SQLITE_METADATA_COLUMNS = (
    ("singleton", "INTEGER", 0, None, 1),
    ("schema_version", "INTEGER", 1, None, 0),
)

_SELECT_ROW_SQLITE = f"""
SELECT request_identity_hash, replay_policy, state, owner_id, attempt_id,
       fence, expires_at, terminal_json
FROM {_TABLE_NAME}
WHERE semantic_key = ?
"""

_INSERT_ROW_SQLITE = f"""
INSERT INTO {_TABLE_NAME} (
    semantic_key, request_identity_hash, replay_policy, state, owner_id,
    attempt_id, fence, expires_at, terminal_json
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_UPDATE_ROW_SQLITE = f"""
UPDATE {_TABLE_NAME}
SET request_identity_hash = ?, replay_policy = ?, state = ?, owner_id = ?,
    attempt_id = ?, fence = ?, expires_at = ?, terminal_json = ?
WHERE semantic_key = ?
  AND request_identity_hash = ?
  AND replay_policy = ?
  AND state = ?
  AND owner_id = ?
  AND attempt_id = ?
  AND fence = ?
  AND expires_at IS ?
  AND terminal_json IS ?
"""

_SQLITE_SELECT_NOW = """
SELECT strftime('%Y-%m-%dT%H:%M:%f000+00:00', 'now')
"""


def _normalized_sql(value: str) -> str:
    return " ".join(value.split())


def _sqlite_schema_sql(
    connection: sqlite3.Connection, table_name: str
) -> str | None:
    raw = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return None if raw is None else str(raw[0])


def _sqlite_columns(
    connection: sqlite3.Connection, table_name: str
) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (name, column_type, not_null, default, primary_key)
        for (
            _column_id,
            name,
            column_type,
            not_null,
            default,
            primary_key,
        ) in connection.execute(f"PRAGMA table_info({table_name})")
    )


def _verify_sqlite_schema(connection: sqlite3.Connection) -> None:
    table_sql = _sqlite_schema_sql(connection, _TABLE_NAME)
    metadata_sql = _sqlite_schema_sql(connection, _METADATA_TABLE_NAME)
    if (
        table_sql is None
        or metadata_sql is None
        or _normalized_sql(table_sql) != _normalized_sql(_SQLITE_CREATE_TABLE)
        or _normalized_sql(metadata_sql)
        != _normalized_sql(_SQLITE_CREATE_METADATA_TABLE)
        or _sqlite_columns(connection, _TABLE_NAME) != _SQLITE_TABLE_COLUMNS
        or _sqlite_columns(connection, _METADATA_TABLE_NAME)
        != _SQLITE_METADATA_COLUMNS
    ):
        raise EffectAuthoritySchemaMismatchError(
            "incompatible SQLite effect-authority schema"
        )
    versions = connection.execute(
        f"SELECT singleton, schema_version FROM {_METADATA_TABLE_NAME}"
    ).fetchall()
    if (
        len(versions) != 1
        or len(versions[0]) != 2
        or type(versions[0][0]) is not int
        or type(versions[0][1]) is not int
        or versions[0] != (1, _SCHEMA_VERSION)
    ):
        raise EffectAuthoritySchemaMismatchError(
            "incompatible SQLite effect-authority schema version"
        )


class _SQLiteStore:
    def __init__(self, path: str | Path) -> None:
        raw_path = str(path)
        if not raw_path:
            raise ValueError("SQLite path must be non-empty")
        if raw_path == ":memory:":
            raise ValueError(
                "use EffectAuthority.memory() for process-local memory"
            )
        self._path = raw_path

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self._path,
            timeout=30.0,
            isolation_level=None,
        )
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def initialize(self) -> None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            table_sql = _sqlite_schema_sql(connection, _TABLE_NAME)
            metadata_sql = _sqlite_schema_sql(connection, _METADATA_TABLE_NAME)
            if table_sql is None and metadata_sql is None:
                connection.execute(_SQLITE_CREATE_TABLE)
                connection.execute(_SQLITE_CREATE_METADATA_TABLE)
                connection.execute(
                    f"""
                    INSERT INTO {_METADATA_TABLE_NAME} (
                        singleton, schema_version
                    ) VALUES (1, ?)
                    """,
                    (_SCHEMA_VERSION,),
                )
            elif table_sql is None or metadata_sql is None:
                raise EffectAuthoritySchemaMismatchError(
                    "incomplete SQLite effect-authority schema"
                )
            _verify_sqlite_schema(connection)
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def validate_lease_duration(self, duration: timedelta) -> timedelta:
        if duration < _SQLITE_MINIMUM_LEASE_DURATION:
            raise ValueError(
                "lease_duration must be at least 1 millisecond for SQLite "
                "authority clock precision"
            )
        return duration

    def transaction(
        self,
        semantic_key: str,
        transition: _Transition[_T],
    ) -> _T:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            now_raw = connection.execute(_SQLITE_SELECT_NOW).fetchone()
            if now_raw is None:
                raise _AuthorityCorruptionError(
                    "SQLite did not return authority time"
                )
            now_text = _require_persisted_text(
                now_raw[0], field="SQLite authority time"
            )
            now = _require_utc(
                datetime.fromisoformat(now_text),
                field="SQLite authority time",
            )
            raw = connection.execute(
                _SELECT_ROW_SQLITE, (semantic_key,)
            ).fetchone()
            original = _decode_row(semantic_key, raw)
            updated, result = transition(original, now)
            if updated != original:
                if original is None:
                    if updated is None:
                        raise _AuthorityCorruptionError(
                            "transition removed an absent row"
                        )
                    connection.execute(
                        _INSERT_ROW_SQLITE,
                        _row_insert_values(updated),
                    )
                elif updated is None:
                    raise _AuthorityCorruptionError(
                        "authority rows cannot be deleted"
                    )
                else:
                    cursor = connection.execute(
                        _UPDATE_ROW_SQLITE,
                        (
                            *_row_update_values(updated),
                            semantic_key,
                            *_row_match_values(original),
                        ),
                    )
                    if cursor.rowcount != 1:
                        raise _AuthorityCorruptionError(
                            "conditional SQLite authority update lost"
                        )
            connection.commit()
            return result
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def close(self) -> None:
        pass


_POSTGRES_CREATE_TABLE = f"""
CREATE TABLE {_TABLE_NAME} (
    semantic_key TEXT COLLATE "C" PRIMARY KEY,
    request_identity_hash TEXT COLLATE "C" NOT NULL,
    replay_policy TEXT COLLATE "C" NOT NULL CHECK (
        replay_policy IN ('idempotent', 'durable_workflow', 'no_redrive')
    ),
    state TEXT COLLATE "C" NOT NULL CHECK (
        state IN (
            'leased', 'succeeded', 'failed', 'recovery_required'
        )
    ),
    owner_id TEXT COLLATE "C" NOT NULL,
    attempt_id TEXT COLLATE "C" NOT NULL,
    fence BIGINT NOT NULL CHECK (fence > 0),
    expires_at TEXT COLLATE "C",
    terminal_json TEXT COLLATE "C",
    CHECK (
        (state = 'leased' AND expires_at IS NOT NULL
            AND terminal_json IS NULL)
        OR
        (state != 'leased' AND expires_at IS NULL
            AND terminal_json IS NOT NULL)
    )
)
"""

_POSTGRES_CREATE_METADATA_TABLE = f"""
CREATE TABLE {_METADATA_TABLE_NAME} (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    schema_version INTEGER NOT NULL
)
"""

_POSTGRES_TABLE_COLUMNS = (
    ("semantic_key", "text", "NO", 1, "pg_catalog", "C", "c", True, -1),
    (
        "request_identity_hash",
        "text",
        "NO",
        2,
        "pg_catalog",
        "C",
        "c",
        True,
        -1,
    ),
    ("replay_policy", "text", "NO", 3, "pg_catalog", "C", "c", True, -1),
    ("state", "text", "NO", 4, "pg_catalog", "C", "c", True, -1),
    ("owner_id", "text", "NO", 5, "pg_catalog", "C", "c", True, -1),
    ("attempt_id", "text", "NO", 6, "pg_catalog", "C", "c", True, -1),
    ("fence", "bigint", "NO", 7, None, None, None, None, None),
    ("expires_at", "text", "YES", 8, "pg_catalog", "C", "c", True, -1),
    ("terminal_json", "text", "YES", 9, "pg_catalog", "C", "c", True, -1),
)
_POSTGRES_METADATA_COLUMNS = (
    ("singleton", "integer", "NO", 1, None, None, None, None, None),
    ("schema_version", "integer", "NO", 2, None, None, None, None, None),
)

_PostgreSQLConstraint = tuple[
    str,
    str,
    tuple[str, ...],
    str | None,
    bool,
    bool,
    bool,
    bool,
]
_POSTGRES_CONSTRAINTS: tuple[_PostgreSQLConstraint, ...] = (
    (
        _TABLE_NAME,
        "c",
        ("state", "expires_at", "terminal_json"),
        "state = 'leased'::text AND expires_at IS NOT NULL AND "
        "terminal_json IS NULL OR state <> 'leased'::text AND "
        "expires_at IS NULL AND terminal_json IS NOT NULL",
        False,
        False,
        True,
        False,
    ),
    (
        _TABLE_NAME,
        "c",
        ("fence",),
        "fence > 0",
        False,
        False,
        True,
        False,
    ),
    (
        _TABLE_NAME,
        "c",
        ("replay_policy",),
        "replay_policy = ANY (ARRAY['idempotent'::text, "
        "'durable_workflow'::text, 'no_redrive'::text])",
        False,
        False,
        True,
        False,
    ),
    (
        _TABLE_NAME,
        "c",
        ("state",),
        "state = ANY (ARRAY['leased'::text, 'succeeded'::text, "
        "'failed'::text, 'recovery_required'::text])",
        False,
        False,
        True,
        False,
    ),
    (
        _TABLE_NAME,
        "p",
        ("semantic_key",),
        None,
        False,
        False,
        True,
        True,
    ),
    (
        _METADATA_TABLE_NAME,
        "c",
        ("singleton",),
        "singleton = 1",
        False,
        False,
        True,
        False,
    ),
    (
        _METADATA_TABLE_NAME,
        "p",
        ("singleton",),
        None,
        False,
        False,
        True,
        True,
    ),
)

_SELECT_ROW_POSTGRES = f"""
SELECT request_identity_hash, replay_policy, state, owner_id, attempt_id,
       fence, expires_at, terminal_json
FROM {_TABLE_NAME}
WHERE semantic_key = %s
FOR UPDATE
"""

_INSERT_ROW_POSTGRES = f"""
INSERT INTO {_TABLE_NAME} (
    semantic_key, request_identity_hash, replay_policy, state, owner_id,
    attempt_id, fence, expires_at, terminal_json
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (semantic_key) DO NOTHING
RETURNING 1
"""

_UPDATE_ROW_POSTGRES = f"""
UPDATE {_TABLE_NAME}
SET request_identity_hash = %s, replay_policy = %s, state = %s,
    owner_id = %s, attempt_id = %s, fence = %s, expires_at = %s,
    terminal_json = %s
WHERE semantic_key = %s
  AND request_identity_hash = %s
  AND replay_policy = %s
  AND state = %s
  AND owner_id = %s
  AND attempt_id = %s
  AND fence = %s
  AND expires_at IS NOT DISTINCT FROM %s
  AND terminal_json IS NOT DISTINCT FROM %s
"""

_POSTGRES_SELECT_NOW = """
SELECT to_char(
    clock_timestamp() AT TIME ZONE 'UTC',
    'YYYY-MM-DD"T"HH24:MI:SS.US'
) || '+00:00'
"""

_POSTGRES_SELECT_TABLES = """
SELECT table_name
FROM information_schema.tables
WHERE table_schema = current_schema()
  AND table_name IN (%s, %s)
ORDER BY table_name
"""

_POSTGRES_SELECT_COLUMNS = """
SELECT column_record.table_name,
       column_record.column_name,
       column_record.data_type,
       column_record.is_nullable,
       column_record.ordinal_position,
       collation_namespace.nspname,
       collation_record.collname,
       collation_record.collprovider,
       collation_record.collisdeterministic,
       collation_record.collencoding
FROM information_schema.columns AS column_record
JOIN pg_catalog.pg_namespace AS table_namespace
  ON table_namespace.nspname = column_record.table_schema
JOIN pg_catalog.pg_class AS table_record
  ON table_record.relnamespace = table_namespace.oid
 AND table_record.relname = column_record.table_name
JOIN pg_catalog.pg_attribute AS attribute_record
  ON attribute_record.attrelid = table_record.oid
 AND attribute_record.attname = column_record.column_name
LEFT JOIN pg_catalog.pg_collation AS collation_record
  ON collation_record.oid = attribute_record.attcollation
LEFT JOIN pg_catalog.pg_namespace AS collation_namespace
  ON collation_namespace.oid = collation_record.collnamespace
WHERE column_record.table_schema = current_schema()
  AND column_record.table_name IN (%s, %s)
ORDER BY column_record.table_name, column_record.ordinal_position
"""

_POSTGRES_SELECT_SERVER_ENCODING = "SHOW server_encoding"

_POSTGRES_SELECT_CONSTRAINTS = """
SELECT cls.relname, constraint_record.contype,
       COALESCE(
           array_agg(attribute_record.attname::text ORDER BY key.ordinality)
               FILTER (WHERE attribute_record.attname IS NOT NULL),
           ARRAY[]::text[]
       ),
       CASE
           WHEN constraint_record.contype = 'c'
           THEN pg_get_expr(
               constraint_record.conbin,
               constraint_record.conrelid,
               true
           )
           ELSE NULL
       END,
       constraint_record.condeferrable,
       constraint_record.condeferred,
       constraint_record.convalidated,
       constraint_record.connoinherit
FROM pg_catalog.pg_constraint AS constraint_record
JOIN pg_catalog.pg_class AS cls
  ON cls.oid = constraint_record.conrelid
JOIN pg_catalog.pg_namespace AS namespace_record
  ON namespace_record.oid = cls.relnamespace
LEFT JOIN LATERAL unnest(constraint_record.conkey)
    WITH ORDINALITY AS key(attnum, ordinality)
  ON true
LEFT JOIN pg_catalog.pg_attribute AS attribute_record
  ON attribute_record.attrelid = constraint_record.conrelid
 AND attribute_record.attnum = key.attnum
WHERE namespace_record.nspname = current_schema()
  AND cls.relname IN (%s, %s)
  AND constraint_record.contype IN ('p', 'c')
GROUP BY cls.relname,
         constraint_record.contype,
         constraint_record.conname,
         constraint_record.conbin,
         constraint_record.conrelid,
         constraint_record.condeferrable,
         constraint_record.condeferred,
         constraint_record.convalidated,
         constraint_record.connoinherit
ORDER BY cls.relname, constraint_record.contype, constraint_record.conname
"""

_POSTGRES_INIT_LOCK = """
SELECT pg_advisory_xact_lock(1465141076, 1)
"""
# This stable two-key namespace serializes first-use schema creation. Changing
# it would let different releases initialize the same table concurrently.


class _Cursor(Protocol):
    rowcount: int

    def execute(
        self, query: str, params: tuple[Any, ...] | None = None
    ) -> Any: ...

    def fetchone(self) -> tuple[Any, ...] | None: ...

    def fetchall(self) -> list[tuple[Any, ...]]: ...

    def __enter__(self) -> _Cursor: ...

    def __exit__(self, *args: object) -> None: ...


class _Connection(Protocol):
    def cursor(self) -> _Cursor: ...

    def __enter__(self) -> _Connection: ...

    def __exit__(self, *args: object) -> None: ...


_Connect = Callable[[str], AbstractContextManager[_Connection]]


def _postgresql_constraints(
    rows: list[tuple[Any, ...]],
) -> tuple[_PostgreSQLConstraint, ...]:
    constraints: list[_PostgreSQLConstraint] = []
    for row in rows:
        if len(row) != 8:
            raise EffectAuthoritySchemaMismatchError(
                "PostgreSQL effect-authority constraint catalog returned "
                f"an unexpected row shape: {row!r}"
            )
        (
            table_name,
            constraint_type,
            columns,
            expression,
            deferrable,
            deferred,
            validated,
            no_inherit,
        ) = row
        if not isinstance(columns, (list, tuple)) or not all(
            isinstance(column, str) for column in columns
        ):
            raise EffectAuthoritySchemaMismatchError(
                "PostgreSQL effect-authority constraint catalog returned "
                f"invalid constrained columns: {columns!r}"
            )
        flags = (deferrable, deferred, validated, no_inherit)
        if not all(isinstance(flag, bool) for flag in flags):
            raise EffectAuthoritySchemaMismatchError(
                "PostgreSQL effect-authority constraint catalog returned "
                f"invalid constraint flags: {flags!r}"
            )
        constraints.append(
            (
                str(table_name),
                str(constraint_type),
                tuple(columns),
                None if expression is None else str(expression),
                deferrable,
                deferred,
                validated,
                no_inherit,
            )
        )
    return tuple(constraints)


def _describe_postgresql_constraint(
    constraint: _PostgreSQLConstraint,
) -> str:
    (
        table_name,
        constraint_type,
        columns,
        expression,
        deferrable,
        deferred,
        validated,
        no_inherit,
    ) = constraint
    column_text = ", ".join(columns)
    if constraint_type == "p":
        definition = f"PRIMARY KEY ({column_text})"
    else:
        definition = f"CHECK ({expression}) on columns ({column_text})"
    flags = (
        f"deferrable={deferrable}, deferred={deferred}, "
        f"validated={validated}, no_inherit={no_inherit}"
    )
    return f"{table_name} {definition} [{flags}]"


def _verify_postgresql_constraints(rows: list[tuple[Any, ...]]) -> None:
    remaining = list(_postgresql_constraints(rows))
    missing: list[_PostgreSQLConstraint] = []
    for expected in _POSTGRES_CONSTRAINTS:
        if expected in remaining:
            remaining.remove(expected)
        else:
            missing.append(expected)
    if not missing and not remaining:
        return
    mismatch_parts = []
    if missing:
        mismatch_parts.append(
            "missing "
            + "; ".join(
                _describe_postgresql_constraint(constraint)
                for constraint in missing
            )
        )
    if remaining:
        mismatch_parts.append(
            "unexpected "
            + "; ".join(
                _describe_postgresql_constraint(constraint)
                for constraint in remaining
            )
        )
    raise EffectAuthoritySchemaMismatchError(
        "PostgreSQL effect-authority constraint mismatch: "
        + "; ".join(mismatch_parts)
    )


class _PostgreSQLStore:
    def __init__(
        self,
        dsn: str,
        *,
        connect: _Connect | None = None,
    ) -> None:
        _require_text(dsn, field="PostgreSQL DSN", maximum=16_384)
        self._dsn = dsn
        if connect is None:
            from psycopg import connect as psycopg_connect

            self._connect = cast(_Connect, psycopg_connect)
        else:
            self._connect = connect

    def initialize(self) -> None:
        with self._connect(self._dsn) as connection:
            with connection.cursor() as cursor:
                cursor.execute(_POSTGRES_SELECT_SERVER_ENCODING)
                encoding = cursor.fetchone()
                if encoding != ("UTF8",):
                    raise EffectAuthoritySchemaMismatchError(
                        "PostgreSQL effect-authority requires exact "
                        f"server_encoding 'UTF8', found {encoding!r}"
                    )
                cursor.execute(_POSTGRES_INIT_LOCK)
                cursor.execute(
                    _POSTGRES_SELECT_TABLES,
                    (_TABLE_NAME, _METADATA_TABLE_NAME),
                )
                tables = {str(row[0]) for row in cursor.fetchall()}
                if not tables:
                    cursor.execute(_POSTGRES_CREATE_TABLE)
                    cursor.execute(_POSTGRES_CREATE_METADATA_TABLE)
                    cursor.execute(
                        f"""
                        INSERT INTO {_METADATA_TABLE_NAME} (
                            singleton, schema_version
                        ) VALUES (%s, %s)
                        """,
                        (1, _SCHEMA_VERSION),
                    )
                elif tables != {_TABLE_NAME, _METADATA_TABLE_NAME}:
                    raise EffectAuthoritySchemaMismatchError(
                        "incomplete PostgreSQL effect-authority schema: "
                        "expected both authority tables, found "
                        f"{sorted(tables)}"
                    )
                cursor.execute(
                    _POSTGRES_SELECT_COLUMNS,
                    (_TABLE_NAME, _METADATA_TABLE_NAME),
                )
                columns_by_table: dict[str, list[tuple[object, ...]]] = {
                    _TABLE_NAME: [],
                    _METADATA_TABLE_NAME: [],
                }
                for (
                    table_name,
                    column_name,
                    data_type,
                    is_nullable,
                    ordinal_position,
                    collation_schema,
                    collation_name,
                    collation_provider,
                    collation_is_deterministic,
                    collation_encoding,
                ) in cursor.fetchall():
                    columns_by_table[str(table_name)].append(
                        (
                            str(column_name),
                            str(data_type),
                            str(is_nullable),
                            int(ordinal_position),
                            (
                                None
                                if collation_schema is None
                                else str(collation_schema)
                            ),
                            (
                                None
                                if collation_name is None
                                else str(collation_name)
                            ),
                            (
                                None
                                if collation_provider is None
                                else str(collation_provider)
                            ),
                            (
                                None
                                if collation_is_deterministic is None
                                else bool(collation_is_deterministic)
                            ),
                            (
                                None
                                if collation_encoding is None
                                else int(collation_encoding)
                            ),
                        )
                    )
                if (
                    tuple(columns_by_table[_TABLE_NAME])
                    != _POSTGRES_TABLE_COLUMNS
                    or tuple(columns_by_table[_METADATA_TABLE_NAME])
                    != _POSTGRES_METADATA_COLUMNS
                ):
                    raise EffectAuthoritySchemaMismatchError(
                        "incompatible PostgreSQL effect-authority columns: "
                        f"expected {_POSTGRES_TABLE_COLUMNS!r} and "
                        f"{_POSTGRES_METADATA_COLUMNS!r}; found "
                        f"{tuple(columns_by_table[_TABLE_NAME])!r} and "
                        f"{tuple(columns_by_table[_METADATA_TABLE_NAME])!r}"
                    )
                cursor.execute(
                    _POSTGRES_SELECT_CONSTRAINTS,
                    (_TABLE_NAME, _METADATA_TABLE_NAME),
                )
                _verify_postgresql_constraints(cursor.fetchall())
                cursor.execute(
                    f"""
                    SELECT singleton, schema_version
                    FROM {_METADATA_TABLE_NAME}
                    """
                )
                if cursor.fetchall() != [(1, _SCHEMA_VERSION)]:
                    raise EffectAuthoritySchemaMismatchError(
                        "incompatible PostgreSQL effect-authority "
                        f"schema version: expected {_SCHEMA_VERSION}"
                    )

    def validate_lease_duration(self, duration: timedelta) -> timedelta:
        return duration

    def transaction(
        self,
        semantic_key: str,
        transition: _Transition[_T],
    ) -> _T:
        with self._connect(self._dsn) as connection:
            with connection.cursor() as cursor:
                cursor.execute(_SELECT_ROW_POSTGRES, (semantic_key,))
                raw = cursor.fetchone()
                original = _decode_row(semantic_key, raw)
                now = self._database_now(cursor)
                updated, result = transition(original, now)
                if updated == original:
                    return result
                if original is None:
                    if updated is None:
                        raise _AuthorityCorruptionError(
                            "transition removed an absent row"
                        )
                    cursor.execute(
                        _INSERT_ROW_POSTGRES,
                        _row_insert_values(updated),
                    )
                    if cursor.fetchone() is not None:
                        return result
                    cursor.execute(_SELECT_ROW_POSTGRES, (semantic_key,))
                    original = _decode_row(semantic_key, cursor.fetchone())
                    if original is None:
                        raise _AuthorityCorruptionError(
                            "conflicting PostgreSQL row disappeared"
                        )
                    now = self._database_now(cursor)
                    updated, result = transition(original, now)
                    if updated == original:
                        return result
                if original is None or updated is None:
                    raise _AuthorityCorruptionError(
                        "authority rows cannot be deleted"
                    )
                cursor.execute(
                    _UPDATE_ROW_POSTGRES,
                    (
                        *_row_update_values(updated),
                        semantic_key,
                        *_row_match_values(original),
                    ),
                )
                if cursor.rowcount != 1:
                    raise _AuthorityCorruptionError(
                        "conditional PostgreSQL authority update lost"
                    )
                return result

    @staticmethod
    def _database_now(cursor: _Cursor) -> datetime:
        cursor.execute(_POSTGRES_SELECT_NOW)
        raw = cursor.fetchone()
        if raw is None:
            raise _AuthorityCorruptionError(
                "PostgreSQL did not return authority time"
            )
        now_text = _require_persisted_text(
            raw[0], field="PostgreSQL authority time"
        )
        return _require_utc(
            datetime.fromisoformat(now_text),
            field="PostgreSQL authority time",
        )

    def close(self) -> None:
        pass


def _timestamp_text(value: datetime | None) -> str | None:
    if value is None:
        return None
    _require_utc(value, field="persisted timestamp")
    return value.isoformat(timespec="microseconds")


def _terminal_text(value: EffectTerminal | None) -> str | None:
    if value is None:
        return None
    return json.dumps(
        value.model_dump(mode="json"),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _row_insert_values(row: _EffectRow) -> tuple[Any, ...]:
    return (
        str(row.request.semantic_key),
        str(row.request.request_identity_hash),
        row.request.replay_policy.value,
        row.state.value,
        str(row.owner_id),
        str(row.attempt_id),
        row.fence,
        _timestamp_text(row.expires_at),
        _terminal_text(row.terminal),
    )


def _row_update_values(row: _EffectRow) -> tuple[Any, ...]:
    return _row_insert_values(row)[1:]


def _row_match_values(row: _EffectRow) -> tuple[Any, ...]:
    return (
        str(row.request.request_identity_hash),
        row.request.replay_policy.value,
        row.state.value,
        str(row.owner_id),
        str(row.attempt_id),
        row.fence,
        _timestamp_text(row.expires_at),
        _terminal_text(row.terminal),
    )


def _require_persisted_text(value: object, *, field: str) -> str:
    if type(value) is not str:
        raise _AuthorityCorruptionError(
            f"persisted {field} must have text storage"
        )
    return value


def _require_persisted_integer(value: object, *, field: str) -> int:
    if type(value) is not int:
        raise _AuthorityCorruptionError(
            f"persisted {field} must have integer storage"
        )
    return value


def _decode_row(
    semantic_key: str,
    raw: tuple[Any, ...] | None,
) -> _EffectRow | None:
    if raw is None:
        return None
    (
        request_identity_hash,
        replay_policy,
        state,
        owner_id,
        attempt_id,
        fence,
        expires_at,
        terminal_json,
    ) = raw
    request_identity_hash_text = _require_persisted_text(
        request_identity_hash,
        field="request_identity_hash",
    )
    replay_policy_text = _require_persisted_text(
        replay_policy, field="replay_policy"
    )
    state_text = _require_persisted_text(state, field="state")
    owner_id_text = _require_persisted_text(owner_id, field="owner_id")
    attempt_id_text = _require_persisted_text(attempt_id, field="attempt_id")
    fence_integer = _require_persisted_integer(fence, field="fence")
    expires_at_text = (
        None
        if expires_at is None
        else _require_persisted_text(expires_at, field="expires_at")
    )
    terminal_text = (
        None
        if terminal_json is None
        else _require_persisted_text(terminal_json, field="terminal_json")
    )
    request = EffectRequest.model_validate(
        {
            "semantic_key": semantic_key,
            "request_identity_hash": request_identity_hash_text,
            "replay_policy": ReplayPolicy(replay_policy_text),
        }
    )
    parsed_expires_at = (
        None
        if expires_at_text is None
        else datetime.fromisoformat(expires_at_text)
    )
    terminal = (
        None
        if terminal_text is None
        else EffectTerminal.model_validate_json(terminal_text)
    )
    row = _EffectRow(
        request=request,
        state=_StoredState(state_text),
        owner_id=NonEmptyId(owner_id_text),
        attempt_id=NonEmptyId(attempt_id_text),
        fence=fence_integer,
        expires_at=parsed_expires_at,
        terminal=terminal,
    )
    if row.state is _StoredState.LEASED:
        if terminal is not None:
            raise _AuthorityCorruptionError(
                "persisted lease contains a terminal record"
            )
        row.lease()
    else:
        if row.expires_at is not None:
            raise _AuthorityCorruptionError(
                "persisted terminal contains a lease expiration"
            )
        if terminal is None or terminal.outcome.value != row.state.value:
            raise _AuthorityCorruptionError(
                "persisted terminal state and record disagree"
            )
    if terminal is not None:
        if (
            terminal.request != request
            or terminal.owner_id != row.owner_id
            or terminal.attempt_id != row.attempt_id
            or terminal.fence != row.fence
        ):
            raise _AuthorityCorruptionError(
                "persisted terminal metadata and row disagree"
            )
    return row


def _recovery_failure(row: _EffectRow) -> TerminalFailure:
    if row.expires_at is None:
        raise _AuthorityCorruptionError("expired lease has no expiration")
    payload = {
        "semantic_key": row.request.semantic_key,
        "request_identity_hash": row.request.request_identity_hash,
        "owner_id": row.owner_id,
        "attempt_id": row.attempt_id,
        "fence": row.fence,
        "expires_at": _timestamp_text(row.expires_at),
    }
    digest = hashlib.sha256(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return TerminalFailure(
        code=f"effect-recovery:{digest}",
        message=_RECOVERY_MESSAGE,
        details=payload,
    )


def _acquire_terminal(
    request: EffectRequest, terminal: EffectTerminal
) -> AcquireResult:
    return AcquireResult(
        request=request,
        outcome=AcquireOutcome(terminal.outcome.value),
        terminal=terminal,
    )


class LeaseMaintenance:
    """Keep a lease live while ordinary long-running work proceeds.

    The helper cannot cancel arbitrary external work. A clean context exit
    proves that renewal did not observe lease loss. Long-running callers must
    publish through ``succeed`` or ``fail`` inside the context so renewal is
    stopped before the exact latest lease is terminalized.
    """

    def __init__(
        self,
        authority: EffectAuthority,
        lease: EffectLease,
        lease_duration: timedelta,
    ) -> None:
        self._authority = authority
        self._lease = lease
        self._lease_duration = authority._validate_lease_duration(
            lease_duration
        )
        self._interval_seconds = self._lease_duration.total_seconds() / 3
        self._stop = Event()
        self._lock = Lock()
        self._loss: BaseException | None = None
        self._thread = Thread(
            target=self._run,
            name=f"effect-lease-{lease.attempt_id}",
            daemon=True,
        )
        self._entered = False
        self._exited = False
        self._terminalization_started = False
        self._terminalized = False

    @property
    def lease(self) -> EffectLease:
        """Return the latest exact lease token, raising observed loss."""
        self.check()
        with self._lock:
            return self._lease

    def check(self) -> None:
        """Raise when the renewer has observed lease loss or failure."""
        with self._lock:
            loss = self._loss
        if loss is None:
            return
        if isinstance(loss, EffectAuthorityError):
            raise loss
        raise EffectAuthorityError("lease maintenance failed") from loss

    def _lease_for_terminalization(self) -> EffectLease:
        with self._lock:
            if self._terminalization_started:
                raise RuntimeError("lease maintenance is already terminalized")
            loss = self._loss
            if loss is None and (
                not self._entered or self._exited or self._stop.is_set()
            ):
                raise RuntimeError(
                    "terminal publication requires an active maintenance "
                    "context"
                )
            if loss is None:
                self._terminalization_started = True
                self._stop.set()

        if loss is not None:
            self.check()
            raise AssertionError("lease loss check returned unexpectedly")

        # A renewal may already be inside authority I/O. Join without holding
        # the state lock so it can publish its renewed lease or recorded loss.
        self._thread.join()
        self.check()
        with self._lock:
            return self._lease

    def succeed(self, *, result_ref: TypedRef) -> EffectTerminal:
        """Stop renewal and publish the exact successful terminal."""
        lease = self._lease_for_terminalization()
        terminal = self._authority.succeed(lease, result_ref=result_ref)
        with self._lock:
            self._terminalized = True
        return terminal

    def fail(
        self,
        *,
        result_ref: TypedRef,
        failure: TerminalFailure,
    ) -> EffectTerminal:
        """Stop renewal and publish the exact failed terminal."""
        lease = self._lease_for_terminalization()
        terminal = self._authority.fail(
            lease,
            result_ref=result_ref,
            failure=failure,
        )
        with self._lock:
            self._terminalized = True
        return terminal

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            with self._lock:
                current = self._lease
            try:
                renewed = self._authority.renew(
                    current,
                    lease_duration=self._lease_duration,
                )
            except BaseException as exc:
                with self._lock:
                    self._loss = exc
                self._stop.set()
                return
            with self._lock:
                self._lease = renewed

    def __enter__(self) -> LeaseMaintenance:
        with self._lock:
            if self._entered:
                raise RuntimeError("lease maintenance cannot be re-entered")
            self._entered = True
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        with self._lock:
            self._exited = True
            self._stop.set()
        self._thread.join()
        with self._lock:
            terminalized = self._terminalized
        if not args or args[0] is None:
            self.check()
            if not terminalized:
                raise RuntimeError(
                    "clean lease maintenance exit requires terminal "
                    "publication"
                )


class EffectAuthority:
    """One authority contract backed by memory, SQLite, or PostgreSQL."""

    def __init__(self, store: _Store) -> None:
        self._store = store
        self._store.initialize()

    @classmethod
    def memory(
        cls,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> EffectAuthority:
        """Open a process-local authority owning one injectable UTC clock."""
        authority_clock = (
            (lambda: datetime.now(UTC)) if clock is None else clock
        )
        return cls(_MemoryStore(authority_clock))

    @classmethod
    def sqlite(cls, path: str | Path) -> EffectAuthority:
        return cls(_SQLiteStore(path))

    @classmethod
    def postgresql(
        cls,
        dsn: str,
        *,
        _connect: _Connect | None = None,
    ) -> EffectAuthority:
        """Open a PostgreSQL authority.

        ``_connect`` exists for transaction-adapter tests. Production callers
        use the psycopg-backed default.
        """
        return cls(_PostgreSQLStore(dsn, connect=_connect))

    def acquire(
        self,
        request: EffectRequest,
        *,
        owner_id: str,
        attempt_id: str,
        lease_duration: timedelta,
    ) -> AcquireResult:
        """Acquire, take over, or replay an exact semantic effect."""
        _require_text(owner_id, field="owner_id", maximum=255)
        _require_text(attempt_id, field="attempt_id", maximum=255)
        validated_owner_id = NonEmptyId(owner_id)
        validated_attempt_id = NonEmptyId(attempt_id)
        validated_duration = self._validate_lease_duration(lease_duration)

        def transition(
            row: _EffectRow | None, now: datetime
        ) -> tuple[_EffectRow, AcquireResult]:
            lease_expires_at = now + validated_duration
            if row is None:
                lease = EffectLease(
                    request=request,
                    owner_id=validated_owner_id,
                    attempt_id=validated_attempt_id,
                    fence=1,
                    expires_at=lease_expires_at,
                )
                return (
                    _EffectRow.leased(lease),
                    AcquireResult(
                        request=request,
                        outcome=AcquireOutcome.ACQUIRED,
                        lease=lease,
                    ),
                )
            if row.request != request:
                return (
                    row,
                    AcquireResult(
                        request=request,
                        outcome=AcquireOutcome.REQUEST_CONFLICT,
                        existing_request_identity_hash=(
                            row.request.request_identity_hash
                        ),
                        existing_replay_policy=row.request.replay_policy,
                    ),
                )
            if row.terminal is not None:
                return row, _acquire_terminal(request, row.terminal)
            lease = row.lease()
            if lease.expires_at > now:
                if (
                    lease.owner_id == validated_owner_id
                    and lease.attempt_id == validated_attempt_id
                ):
                    return (
                        row,
                        AcquireResult(
                            request=request,
                            outcome=AcquireOutcome.ACQUIRED,
                            lease=lease,
                        ),
                    )
                return (
                    row,
                    AcquireResult(
                        request=request,
                        outcome=AcquireOutcome.BUSY,
                        busy_expires_at=lease.expires_at,
                    ),
                )
            if request.replay_policy is ReplayPolicy.NO_REDRIVE:
                terminal = EffectTerminal(
                    request=request,
                    outcome=TerminalOutcome.RECOVERY_REQUIRED,
                    owner_id=lease.owner_id,
                    attempt_id=lease.attempt_id,
                    fence=lease.fence,
                    failure=_recovery_failure(row),
                )
                return (
                    _EffectRow.terminalized(terminal),
                    _acquire_terminal(request, terminal),
                )
            if lease.fence == _MAX_FENCE:
                raise EffectAuthorityError(
                    "effect fence exhausted the signed 64-bit range"
                )
            takeover = EffectLease(
                request=request,
                owner_id=validated_owner_id,
                attempt_id=validated_attempt_id,
                fence=lease.fence + 1,
                expires_at=lease_expires_at,
            )
            return (
                _EffectRow.leased(takeover),
                AcquireResult(
                    request=request,
                    outcome=AcquireOutcome.ACQUIRED,
                    lease=takeover,
                ),
            )

        return self._store.transaction(request.semantic_key, transition)

    def renew(
        self,
        lease: EffectLease,
        *,
        lease_duration: timedelta,
    ) -> EffectLease:
        """Extend one still-live exact owner/fence lease."""
        validated_duration = self._validate_lease_duration(lease_duration)

        def transition(
            row: _EffectRow | None, now: datetime
        ) -> tuple[_EffectRow | None, EffectLease]:
            if row is None or row.request != lease.request:
                raise StaleLeaseError("effect lease no longer exists")
            current = row.lease() if row.terminal is None else None
            if (
                current is None
                or current.owner_id != lease.owner_id
                or current.attempt_id != lease.attempt_id
                or current.fence != lease.fence
                or current.expires_at != lease.expires_at
                or current.expires_at <= now
            ):
                raise StaleLeaseError("effect lease is stale")
            lease_expires_at = now + validated_duration
            if lease_expires_at <= current.expires_at:
                return row, current
            renewed = EffectLease(
                request=current.request,
                owner_id=current.owner_id,
                attempt_id=current.attempt_id,
                fence=current.fence,
                expires_at=lease_expires_at,
            )
            return _EffectRow.leased(renewed), renewed

        return self._store.transaction(lease.request.semantic_key, transition)

    def succeed(
        self,
        lease: EffectLease,
        *,
        result_ref: TypedRef,
    ) -> EffectTerminal:
        """Persist or replay the exact successful terminal outcome."""
        terminal = EffectTerminal(
            request=lease.request,
            outcome=TerminalOutcome.SUCCEEDED,
            owner_id=lease.owner_id,
            attempt_id=lease.attempt_id,
            fence=lease.fence,
            result_ref=result_ref,
        )
        return self._terminalize(lease, terminal=terminal)

    def fail(
        self,
        lease: EffectLease,
        *,
        result_ref: TypedRef,
        failure: TerminalFailure,
    ) -> EffectTerminal:
        """Persist or replay the exact failed terminal outcome."""
        terminal = EffectTerminal(
            request=lease.request,
            outcome=TerminalOutcome.FAILED,
            owner_id=lease.owner_id,
            attempt_id=lease.attempt_id,
            fence=lease.fence,
            result_ref=result_ref,
            failure=failure,
        )
        return self._terminalize(lease, terminal=terminal)

    def verify_terminal(self, terminal: EffectTerminal) -> EffectTerminal:
        """Return an exact terminal only when this authority owns it."""
        validated = EffectTerminal.model_validate_json(
            terminal.model_dump_json()
        )

        def transition(
            row: _EffectRow | None, now: datetime
        ) -> tuple[_EffectRow | None, EffectTerminal]:
            del now
            authoritative = None if row is None else row.terminal
            if authoritative is None or authoritative != validated:
                raise TerminalConflictError(
                    "the supplied effect terminal is not authoritative"
                )
            return row, authoritative

        return self._store.transaction(
            validated.request.semantic_key, transition
        )

    def _terminalize(
        self,
        lease: EffectLease,
        *,
        terminal: EffectTerminal,
    ) -> EffectTerminal:
        def transition(
            row: _EffectRow | None, now: datetime
        ) -> tuple[_EffectRow | None, EffectTerminal]:
            if row is None or row.request != lease.request:
                raise StaleLeaseError("effect lease no longer exists")
            if row.terminal is not None:
                if row.terminal == terminal:
                    return row, row.terminal
                raise TerminalConflictError(
                    "a different terminal outcome is already authoritative"
                )
            current = row.lease()
            if (
                current.owner_id != lease.owner_id
                or current.attempt_id != lease.attempt_id
                or current.fence != lease.fence
                or current.expires_at != lease.expires_at
                or current.expires_at <= now
            ):
                raise StaleLeaseError("effect lease is stale")
            return _EffectRow.terminalized(terminal), terminal

        return self._store.transaction(lease.request.semantic_key, transition)

    def maintain(
        self,
        lease: EffectLease,
        *,
        lease_duration: timedelta,
    ) -> LeaseMaintenance:
        """Maintain a lease whose terminal is published inside the context."""
        return LeaseMaintenance(self, lease, lease_duration)

    def _validate_lease_duration(self, value: timedelta) -> timedelta:
        return self._store.validate_lease_duration(
            _require_lease_duration(value)
        )

    def close(self) -> None:
        self._store.close()

    def __enter__(self) -> EffectAuthority:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
