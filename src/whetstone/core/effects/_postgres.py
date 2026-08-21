from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, cast

from dr_store.relational import (
    ConnectFactory,
    RelationalContractMismatchError,
    verify_postgres_table,
)

from whetstone.core.effects._storage import (
    _T,
    _decode_row,
    _persisted_text,
    _reraise_schema_mismatch,
    _row_insert_values,
    _row_match_values,
    _row_update_values,
    _Transition,
)
from whetstone.core.effects.models import (
    EffectAuthoritySchemaMismatchError,
    _AuthorityCorruptionError,
    _require_text,
    _require_utc,
)

_TABLE_NAME = "whetstone_effect_authority"
_METADATA_TABLE_NAME = "whetstone_effect_authority_metadata"
_SCHEMA_VERSION = 2

_POSTGRES_CREATE_TABLE = f"""
CREATE TABLE {_TABLE_NAME} (
    semantic_key TEXT COLLATE "C" PRIMARY KEY,
    request_hash TEXT COLLATE "C" NOT NULL,
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

type _PostgreSQLColumnContract = tuple[
    str,
    str,
    bool,
    int,
    str | None,
    str | None,
    str | None,
    bool | None,
    int | None,
]

type _PostgreSQLConstraint = tuple[
    str,
    str,
    tuple[str, ...],
    str | None,
    bool,
    bool,
    bool,
    bool,
]

_POSTGRES_TABLE_COLUMNS: tuple[_PostgreSQLColumnContract, ...] = (
    ("semantic_key", "text", True, 1, "pg_catalog", "C", "c", True, -1),
    ("request_hash", "text", True, 2, "pg_catalog", "C", "c", True, -1),
    ("replay_policy", "text", True, 3, "pg_catalog", "C", "c", True, -1),
    ("state", "text", True, 4, "pg_catalog", "C", "c", True, -1),
    ("owner_id", "text", True, 5, "pg_catalog", "C", "c", True, -1),
    ("attempt_id", "text", True, 6, "pg_catalog", "C", "c", True, -1),
    ("fence", "bigint", True, 7, None, None, None, None, None),
    ("expires_at", "text", False, 8, "pg_catalog", "C", "c", True, -1),
    ("terminal_json", "text", False, 9, "pg_catalog", "C", "c", True, -1),
)
_POSTGRES_METADATA_COLUMNS: tuple[_PostgreSQLColumnContract, ...] = (
    ("singleton", "integer", True, 1, None, None, None, None, None),
    ("schema_version", "integer", True, 2, None, None, None, None, None),
)

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
SELECT request_hash, replay_policy, state, owner_id, attempt_id,
       fence, expires_at, terminal_json
FROM {_TABLE_NAME}
WHERE semantic_key = %s
FOR UPDATE
"""

_INSERT_ROW_POSTGRES = f"""
INSERT INTO {_TABLE_NAME} (
    semantic_key, request_hash, replay_policy, state, owner_id,
    attempt_id, fence, expires_at, terminal_json
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (semantic_key) DO NOTHING
RETURNING 1
"""

_UPDATE_ROW_POSTGRES = f"""
UPDATE {_TABLE_NAME}
SET request_hash = %s, replay_policy = %s, state = %s,
    owner_id = %s, attempt_id = %s, fence = %s, expires_at = %s,
    terminal_json = %s
WHERE semantic_key = %s
  AND request_hash = %s
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

_POSTGRES_SELECT_SERVER_ENCODING = "SHOW server_encoding"

_POSTGRES_INIT_LOCK = """
SELECT pg_advisory_xact_lock(1465141076, 1)
"""

type _Connect = ConnectFactory


def _verify_postgresql_schema(connection: Any) -> None:
    try:
        verify_postgres_table(
            connection,
            table=_TABLE_NAME,
            columns=_POSTGRES_TABLE_COLUMNS,
            constraints=tuple(
                constraint
                for constraint in _POSTGRES_CONSTRAINTS
                if constraint[0] == _TABLE_NAME
            ),
        )
        verify_postgres_table(
            connection,
            table=_METADATA_TABLE_NAME,
            columns=_POSTGRES_METADATA_COLUMNS,
            constraints=tuple(
                constraint
                for constraint in _POSTGRES_CONSTRAINTS
                if constraint[0] == _METADATA_TABLE_NAME
            ),
        )
    except RelationalContractMismatchError as exc:
        _reraise_schema_mismatch(exc)


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
            try:
                from psycopg import connect as psycopg_connect
            except ImportError as exc:
                raise ImportError(
                    "PostgreSQL effects require the optional postgres extra: "
                    "pip install 'whetstone-ai[postgres]'"
                ) from exc

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
                        table="<database>",
                        aspect="server_encoding",
                        expected="UTF8",
                        actual=encoding,
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
                        table=(
                            _TABLE_NAME
                            if _TABLE_NAME not in tables
                            else _METADATA_TABLE_NAME
                        ),
                        aspect="owned table inventory",
                        expected={_TABLE_NAME, _METADATA_TABLE_NAME},
                        actual=tables,
                    )
            _verify_postgresql_schema(connection)
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT singleton, schema_version
                    FROM {_METADATA_TABLE_NAME}
                    """
                )
                versions = cursor.fetchall()
                if versions != [(1, _SCHEMA_VERSION)]:
                    raise EffectAuthoritySchemaMismatchError(
                        table=_METADATA_TABLE_NAME,
                        aspect="schema version",
                        expected=[(1, _SCHEMA_VERSION)],
                        actual=versions,
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
    def _database_now(cursor: Any) -> datetime:
        cursor.execute(_POSTGRES_SELECT_NOW)
        raw = cursor.fetchone()
        if raw is None:
            raise _AuthorityCorruptionError(
                "PostgreSQL did not return authority time"
            )
        now_text = _persisted_text(
            raw[0], field="PostgreSQL authority time"
        )
        return _require_utc(
            datetime.fromisoformat(now_text),
            field="PostgreSQL authority time",
        )

    def close(self) -> None:
        pass
