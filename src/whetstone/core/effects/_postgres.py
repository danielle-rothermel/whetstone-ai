"""PostgreSQL effect authority storage."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import datetime, timedelta
from typing import Any, Protocol, cast

from whetstone.core.effects._storage import (
    _T,
    _decode_row,
    _require_persisted_text,
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
