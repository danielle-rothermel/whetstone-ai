from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from whetstone.core.effects._storage import (
    _T,
    _decode_row,
    _require_persisted_text,
    _row_insert_values,
    _row_match_values,
    _row_update_values,
    _SQLiteTransactionObserver,
    _Transition,
)
from whetstone.core.effects.models import (
    EffectAuthoritySchemaMismatchError,
    _AuthorityCorruptionError,
    _require_utc,
)

_TABLE_NAME = "whetstone_effect_authority"
_METADATA_TABLE_NAME = "whetstone_effect_authority_metadata"
_SCHEMA_VERSION = 2
_SQLITE_MINIMUM_LEASE_DURATION = timedelta(milliseconds=1)

_SQLITE_CREATE_TABLE = f"""
CREATE TABLE {_TABLE_NAME} (
    semantic_key TEXT PRIMARY KEY CHECK (typeof(semantic_key) = 'text'),
    request_hash TEXT NOT NULL CHECK (
        typeof(request_hash) = 'text'
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
    ("request_hash", "TEXT", 1, None, 0),
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
SELECT request_hash, replay_policy, state, owner_id, attempt_id,
       fence, expires_at, terminal_json
FROM {_TABLE_NAME}
WHERE semantic_key = ?
"""

_INSERT_ROW_SQLITE = f"""
INSERT INTO {_TABLE_NAME} (
    semantic_key, request_hash, replay_policy, state, owner_id,
    attempt_id, fence, expires_at, terminal_json
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_UPDATE_ROW_SQLITE = f"""
UPDATE {_TABLE_NAME}
SET request_hash = ?, replay_policy = ?, state = ?, owner_id = ?,
    attempt_id = ?, fence = ?, expires_at = ?, terminal_json = ?
WHERE semantic_key = ?
  AND request_hash = ?
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
    def __init__(
        self,
        path: str | Path,
        *,
        transaction_observer: _SQLiteTransactionObserver | None = None,
    ) -> None:
        raw_path = str(path)
        if not raw_path:
            raise ValueError("SQLite path must be non-empty")
        if raw_path == ":memory:":
            raise ValueError(
                "use EffectAuthority.memory() for process-local memory"
            )
        self._path = raw_path
        self._transaction_observer = transaction_observer

    def _connect(
        self, *, observe_transaction: bool = False
    ) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self._path,
            timeout=30.0,
            isolation_level=None,
        )
        connection.execute("PRAGMA busy_timeout = 30000")
        if observe_transaction and self._transaction_observer is not None:
            observer = self._transaction_observer

            def authorize(
                action_code: int,
                argument_1: str | None,
                _argument_2: str | None,
                _database_name: str | None,
                _trigger_name: str | None,
            ) -> int:
                if (
                    action_code == sqlite3.SQLITE_TRANSACTION
                    and argument_1 == "BEGIN"
                ):
                    observer.transaction_attempted()
                return sqlite3.SQLITE_OK

            connection.set_authorizer(authorize)
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
        connection = self._connect(observe_transaction=True)
        try:
            connection.execute("BEGIN IMMEDIATE")
            if self._transaction_observer is not None:
                self._transaction_observer.transaction_acquired()
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
