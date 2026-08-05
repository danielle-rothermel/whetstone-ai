"""PostgreSQL tool admission backend."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any, Protocol, cast

from whetstone.optimization.tools.admission import (
    _CAPACITY_TABLE,
    _ENTRY_TABLE,
    _SCHEMA_COMPONENT,
    _SCHEMA_TABLE,
    _SCHEMA_VERSION,
    ToolAdmissionSchemaMismatchError,
    ToolCallStoreEntry,
    _accepted_with_ordinal,
    _backend_scope_id,
    _complete_transition,
    _decode_entry,
    _decode_persisted_count,
    _entry_key,
    _entry_text,
    _EntryKey,
    _is_exact_schema_metadata,
    _is_exact_schema_version_row,
    _raise_owned_table_inventory_mismatch,
    _replay_or_conflict,
    _scope_key,
)
from whetstone.optimization.tools.contracts import (
    ToolCapacityScope,
)

_ENTRY_LOCK_DOMAIN = "whetstone.tool_admission.entry_lock.v1"

type _PostgreSQLColumnContract = tuple[
    str,
    str,
    bool,
    str | None,
    str | None,
    str | None,
    bool | None,
    int | None,
]

_POSTGRES_SCHEMA_COLUMNS: tuple[_PostgreSQLColumnContract, ...] = (
    ("component", "text", True, "pg_catalog", "C", "c", True, -1),
    ("version", "bigint", True, None, None, None, None, None),
)
_POSTGRES_ENTRY_COLUMNS: tuple[_PostgreSQLColumnContract, ...] = (
    (
        "store_namespace_key",
        "text",
        True,
        "pg_catalog",
        "C",
        "c",
        True,
        -1,
    ),
    ("call_id", "text", True, "pg_catalog", "C", "c", True, -1),
    ("entry_json", "text", True, "pg_catalog", "C", "c", True, -1),
)
_POSTGRES_CAPACITY_COLUMNS: tuple[_PostgreSQLColumnContract, ...] = (
    (
        "store_namespace_key",
        "text",
        True,
        "pg_catalog",
        "C",
        "c",
        True,
        -1,
    ),
    (
        "tool_config_hash",
        "text",
        True,
        "pg_catalog",
        "C",
        "c",
        True,
        -1,
    ),
    ("capacity_scope", "text", True, "pg_catalog", "C", "c", True, -1),
    ("capacity_scope_id", "text", True, "pg_catalog", "C", "c", True, -1),
    ("max_accepted_calls", "bigint", True, None, None, None, None, None),
    ("consumed", "bigint", True, None, None, None, None, None),
)

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

_POSTGRES_CONSTRAINTS: tuple[_PostgreSQLConstraint, ...] = (
    (
        _SCHEMA_TABLE,
        "c",
        ("version",),
        "version > 0",
        False,
        False,
        True,
        False,
    ),
    (
        _SCHEMA_TABLE,
        "p",
        ("component",),
        None,
        False,
        False,
        True,
        True,
    ),
    (
        _ENTRY_TABLE,
        "p",
        ("store_namespace_key", "call_id"),
        None,
        False,
        False,
        True,
        True,
    ),
    (
        _CAPACITY_TABLE,
        "c",
        ("capacity_scope",),
        "capacity_scope = ANY (ARRAY['global'::text, 'run'::text, "
        "'step'::text])",
        False,
        False,
        True,
        False,
    ),
    (
        _CAPACITY_TABLE,
        "c",
        ("consumed", "max_accepted_calls"),
        "consumed >= 0 AND consumed <= max_accepted_calls",
        False,
        False,
        True,
        False,
    ),
    (
        _CAPACITY_TABLE,
        "c",
        ("max_accepted_calls",),
        "max_accepted_calls >= 0",
        False,
        False,
        True,
        False,
    ),
    (
        _CAPACITY_TABLE,
        "p",
        (
            "store_namespace_key",
            "tool_config_hash",
            "capacity_scope",
            "capacity_scope_id",
        ),
        None,
        False,
        False,
        True,
        True,
    ),
)


_POSTGRES_CREATE_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {_SCHEMA_TABLE} (
    component TEXT COLLATE "C" NOT NULL PRIMARY KEY,
    version BIGINT NOT NULL CHECK (version > 0)
)
"""
_POSTGRES_CREATE_ENTRY = f"""
CREATE TABLE IF NOT EXISTS {_ENTRY_TABLE} (
    store_namespace_key TEXT COLLATE "C" NOT NULL,
    call_id TEXT COLLATE "C" NOT NULL,
    entry_json TEXT COLLATE "C" NOT NULL,
    PRIMARY KEY (store_namespace_key, call_id)
)
"""
_POSTGRES_CREATE_CAPACITY = f"""
CREATE TABLE IF NOT EXISTS {_CAPACITY_TABLE} (
    store_namespace_key TEXT COLLATE "C" NOT NULL,
    tool_config_hash TEXT COLLATE "C" NOT NULL,
    capacity_scope TEXT COLLATE "C" NOT NULL CHECK (
        capacity_scope IN ('global', 'run', 'step')
    ),
    capacity_scope_id TEXT COLLATE "C" NOT NULL,
    max_accepted_calls BIGINT NOT NULL CHECK (max_accepted_calls >= 0),
    consumed BIGINT NOT NULL CHECK (
        consumed >= 0 AND consumed <= max_accepted_calls
    ),
    PRIMARY KEY (
        store_namespace_key, tool_config_hash, capacity_scope,
        capacity_scope_id
    )
)
"""
_POSTGRES_INIT_LOCK = "SELECT pg_advisory_xact_lock(1465141076, 2)"
# This stable two-key namespace serializes schema verification and migration.
_POSTGRES_ENTRY_LOCK_SQL = "SELECT pg_advisory_xact_lock(%s)"
_POSTGRES_TABLES_SQL = """
SELECT table_name
FROM information_schema.tables
WHERE table_schema = current_schema()
  AND table_name IN (%s, %s, %s)
ORDER BY table_name
"""


def _entry_lock_key(key: _EntryKey) -> int:
    material = json.dumps(
        [_ENTRY_LOCK_DOMAIN, key[0], key[1]],
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(material).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=True)


_POSTGRES_COLUMNS_SQL = """
SELECT column_record.column_name,
       column_record.data_type,
       column_record.is_nullable,
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
  AND column_record.table_name = %s
ORDER BY column_record.ordinal_position
"""
_POSTGRES_SERVER_ENCODING_SQL = "SHOW server_encoding"
_POSTGRES_CONSTRAINTS_SQL = """
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
  AND cls.relname IN (%s, %s, %s)
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


def _postgres_columns(
    cursor: _Cursor,
    table: str,
) -> tuple[_PostgreSQLColumnContract, ...]:
    cursor.execute(_POSTGRES_COLUMNS_SQL, (table,))
    return tuple(
        (
            str(name),
            str(column_type),
            str(is_nullable) == "NO",
            (None if collation_schema is None else str(collation_schema)),
            None if collation_name is None else str(collation_name),
            (None if collation_provider is None else str(collation_provider)),
            (
                None
                if collation_is_deterministic is None
                else bool(collation_is_deterministic)
            ),
            (None if collation_encoding is None else int(collation_encoding)),
        )
        for (
            name,
            column_type,
            is_nullable,
            collation_schema,
            collation_name,
            collation_provider,
            collation_is_deterministic,
            collation_encoding,
        ) in cursor.fetchall()
    )


def _postgres_constraints(
    rows: list[tuple[Any, ...]],
) -> tuple[_PostgreSQLConstraint, ...]:
    constraints: list[_PostgreSQLConstraint] = []
    for row in rows:
        if len(row) != 8:
            raise ToolAdmissionSchemaMismatchError(
                table="<catalog>",
                aspect="constraint row shape",
                expected=8,
                actual=row,
            )
        (
            table,
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
            raise ToolAdmissionSchemaMismatchError(
                table=str(table),
                aspect="constrained columns",
                expected="a sequence of column names",
                actual=columns,
            )
        flags = (deferrable, deferred, validated, no_inherit)
        if not all(isinstance(flag, bool) for flag in flags):
            raise ToolAdmissionSchemaMismatchError(
                table=str(table),
                aspect="constraint flags",
                expected="four booleans",
                actual=flags,
            )
        constraints.append(
            (
                str(table),
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


def _describe_postgres_constraint(
    constraint: _PostgreSQLConstraint,
) -> str:
    (
        table,
        constraint_type,
        columns,
        expression,
        deferrable,
        deferred,
        validated,
        no_inherit,
    ) = constraint
    if constraint_type == "p":
        definition = f"PRIMARY KEY ({', '.join(columns)})"
    else:
        definition = f"CHECK ({expression}) on columns ({', '.join(columns)})"
    return (
        f"{table} {definition} [deferrable={deferrable}, "
        f"deferred={deferred}, validated={validated}, "
        f"no_inherit={no_inherit}]"
    )


def _verify_postgres_constraints(rows: list[tuple[Any, ...]]) -> None:
    remaining = list(_postgres_constraints(rows))
    missing: list[_PostgreSQLConstraint] = []
    for expected in _POSTGRES_CONSTRAINTS:
        if expected in remaining:
            remaining.remove(expected)
        else:
            missing.append(expected)
    if not missing and not remaining:
        return
    details: list[str] = []
    if missing:
        details.append(
            "missing "
            + "; ".join(
                _describe_postgres_constraint(constraint)
                for constraint in missing
            )
        )
    if remaining:
        details.append(
            "unexpected "
            + "; ".join(
                _describe_postgres_constraint(constraint)
                for constraint in remaining
            )
        )
    affected_tables = sorted(
        {constraint[0] for constraint in (*missing, *remaining)}
    )
    raise ToolAdmissionSchemaMismatchError(
        table=(
            affected_tables[0]
            if len(affected_tables) == 1
            else "<constraint catalog>"
        ),
        aspect="PRIMARY KEY and CHECK constraints",
        expected="; ".join(
            _describe_postgres_constraint(constraint)
            for constraint in _POSTGRES_CONSTRAINTS
        ),
        actual="; ".join(details),
    )


def _verify_postgres_schema(cursor: _Cursor) -> None:
    for table, expected_columns in (
        (_SCHEMA_TABLE, _POSTGRES_SCHEMA_COLUMNS),
        (_ENTRY_TABLE, _POSTGRES_ENTRY_COLUMNS),
        (_CAPACITY_TABLE, _POSTGRES_CAPACITY_COLUMNS),
    ):
        actual = _postgres_columns(cursor, table)
        if actual != expected_columns:
            raise ToolAdmissionSchemaMismatchError(
                table=table,
                aspect="columns",
                expected=expected_columns,
                actual=actual,
            )
    cursor.execute(
        _POSTGRES_CONSTRAINTS_SQL,
        (_SCHEMA_TABLE, _ENTRY_TABLE, _CAPACITY_TABLE),
    )
    _verify_postgres_constraints(cursor.fetchall())


class _PostgreSQLAdmissionBackend:
    def __init__(self, dsn: str, *, connect: _Connect | None = None) -> None:
        if not dsn:
            raise ValueError("PostgreSQL DSN must be non-empty")
        self._dsn = dsn
        if connect is None:
            from psycopg import connect as psycopg_connect

            self._connect = cast(_Connect, psycopg_connect)
        else:
            self._connect = connect

    def initialize(self) -> None:
        with self._connect(self._dsn) as connection:
            with connection.cursor() as cursor:
                cursor.execute(_POSTGRES_SERVER_ENCODING_SQL)
                encoding = cursor.fetchone()
                if encoding != ("UTF8",):
                    raise ToolAdmissionSchemaMismatchError(
                        table="<database>",
                        aspect="server_encoding",
                        expected="UTF8",
                        actual=encoding,
                    )
                cursor.execute(_POSTGRES_INIT_LOCK)
                cursor.execute(
                    _POSTGRES_TABLES_SQL,
                    (_SCHEMA_TABLE, _ENTRY_TABLE, _CAPACITY_TABLE),
                )
                table_rows = cursor.fetchall()
                if not all(
                    len(row) == 1 and type(row[0]) is str for row in table_rows
                ):
                    raise ToolAdmissionSchemaMismatchError(
                        table="<catalog>",
                        aspect="owned table inventory",
                        expected="text table names",
                        actual=table_rows,
                    )
                tables = {row[0] for row in table_rows}
                unversioned = {_ENTRY_TABLE, _CAPACITY_TABLE}
                versioned = {_SCHEMA_TABLE, _ENTRY_TABLE, _CAPACITY_TABLE}
                if not tables:
                    cursor.execute(_POSTGRES_CREATE_SCHEMA)
                    cursor.execute(_POSTGRES_CREATE_ENTRY)
                    cursor.execute(_POSTGRES_CREATE_CAPACITY)
                    insert_metadata = True
                elif tables == unversioned:
                    cursor.execute(_POSTGRES_CREATE_SCHEMA)
                    insert_metadata = True
                elif tables == versioned:
                    insert_metadata = False
                else:
                    _raise_owned_table_inventory_mismatch(tables)
                _verify_postgres_schema(cursor)
                cursor.execute(
                    f"""
                    SELECT version FROM {_SCHEMA_TABLE}
                    WHERE component = %s
                    """,
                    (_SCHEMA_COMPONENT,),
                )
                row = cursor.fetchone()
                if insert_metadata:
                    if row is not None:
                        raise ToolAdmissionSchemaMismatchError(
                            table=_SCHEMA_TABLE,
                            aspect="schema metadata",
                            expected=None,
                            actual=row,
                        )
                    cursor.execute(
                        f"""
                        INSERT INTO {_SCHEMA_TABLE} (component, version)
                        VALUES (%s, %s)
                        """,
                        (_SCHEMA_COMPONENT, _SCHEMA_VERSION),
                    )
                elif not _is_exact_schema_version_row(row):
                    raise ToolAdmissionSchemaMismatchError(
                        table=_SCHEMA_TABLE,
                        aspect="schema version",
                        expected=_SCHEMA_VERSION,
                        actual=None if row is None else row[0],
                    )
                cursor.execute(
                    f"""
                    SELECT component, version FROM {_SCHEMA_TABLE}
                    ORDER BY component
                    """
                )
                metadata = cursor.fetchall()
                if not _is_exact_schema_metadata(metadata):
                    raise ToolAdmissionSchemaMismatchError(
                        table=_SCHEMA_TABLE,
                        aspect="schema metadata",
                        expected=[(_SCHEMA_COMPONENT, _SCHEMA_VERSION)],
                        actual=metadata,
                    )

    @staticmethod
    def _lock_entry(cursor: _Cursor, key: _EntryKey) -> None:
        cursor.execute(_POSTGRES_ENTRY_LOCK_SQL, (_entry_lock_key(key),))

    @staticmethod
    def _load(cursor: _Cursor, key: _EntryKey) -> ToolCallStoreEntry | None:
        cursor.execute(
            f"""
            SELECT entry_json FROM {_ENTRY_TABLE}
            WHERE store_namespace_key = %s AND call_id = %s
            """,
            key,
        )
        raw = cursor.fetchone()
        return None if raw is None else _decode_entry(raw[0])

    def admit(
        self,
        *,
        accepted: ToolCallStoreEntry,
        refused: ToolCallStoreEntry,
        max_accepted_calls: int,
    ) -> ToolCallStoreEntry:
        with self._connect(self._dsn) as connection:
            with connection.cursor() as cursor:
                key = _entry_key(accepted)
                self._lock_entry(cursor, key)
                existing = self._load(cursor, key)
                if existing is not None:
                    return _replay_or_conflict(existing, accepted)
                scope = _scope_key(accepted)
                cursor.execute(
                    f"""
                    INSERT INTO {_CAPACITY_TABLE} (
                        store_namespace_key, tool_config_hash, capacity_scope,
                        capacity_scope_id, max_accepted_calls, consumed
                    ) VALUES (%s, %s, %s, %s, %s, 0)
                    ON CONFLICT (
                        store_namespace_key, tool_config_hash, capacity_scope,
                        capacity_scope_id
                    ) DO NOTHING
                    """,
                    (*scope, max_accepted_calls),
                )
                cursor.execute(
                    f"""
                    SELECT max_accepted_calls, consumed
                    FROM {_CAPACITY_TABLE}
                    WHERE store_namespace_key = %s
                      AND tool_config_hash = %s
                      AND capacity_scope = %s
                      AND capacity_scope_id = %s
                    FOR UPDATE
                    """,
                    scope,
                )
                capacity = cursor.fetchone()
                if capacity is None:
                    raise RuntimeError("capacity scope disappeared")
                maximum = _decode_persisted_count(
                    capacity[0], field="max_accepted_calls"
                )
                consumed = _decode_persisted_count(
                    capacity[1], field="consumed"
                )
                if maximum != max_accepted_calls:
                    raise RuntimeError(
                        "capacity maximum changed within one scope"
                    )
                if consumed < maximum:
                    ordinal = consumed + 1
                    decision = _accepted_with_ordinal(accepted, ordinal)
                    cursor.execute(
                        f"""
                        UPDATE {_CAPACITY_TABLE} SET consumed = %s
                        WHERE store_namespace_key = %s
                          AND tool_config_hash = %s
                          AND capacity_scope = %s
                          AND capacity_scope_id = %s
                        """,
                        (ordinal, *scope),
                    )
                else:
                    decision = refused
                cursor.execute(
                    f"""
                    INSERT INTO {_ENTRY_TABLE} (
                        store_namespace_key, call_id, entry_json
                    ) VALUES (%s, %s, %s)
                    """,
                    (*key, _entry_text(decision)),
                )
                return decision

    def refuse(self, entry: ToolCallStoreEntry) -> ToolCallStoreEntry:
        with self._connect(self._dsn) as connection:
            with connection.cursor() as cursor:
                key = _entry_key(entry)
                self._lock_entry(cursor, key)
                existing = self._load(cursor, key)
                if existing is not None:
                    return _replay_or_conflict(existing, entry)
                cursor.execute(
                    f"""
                    INSERT INTO {_ENTRY_TABLE} (
                        store_namespace_key, call_id, entry_json
                    ) VALUES (%s, %s, %s)
                    """,
                    (*key, _entry_text(entry)),
                )
                return entry

    def get(
        self, store_namespace_key: str, call_id: str
    ) -> ToolCallStoreEntry | None:
        with self._connect(self._dsn) as connection:
            with connection.cursor() as cursor:
                return self._load(cursor, (store_namespace_key, call_id))

    def complete(self, entry: ToolCallStoreEntry) -> ToolCallStoreEntry:
        with self._connect(self._dsn) as connection:
            with connection.cursor() as cursor:
                key = _entry_key(entry)
                self._lock_entry(cursor, key)
                completed = _complete_transition(
                    self._load(cursor, key), entry
                )
                cursor.execute(
                    f"""
                    UPDATE {_ENTRY_TABLE} SET entry_json = %s
                    WHERE store_namespace_key = %s AND call_id = %s
                    """,
                    (_entry_text(completed), *key),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError(
                        "Tool admission completion row disappeared"
                    )
                return completed

    def accepted_count(
        self,
        *,
        store_namespace_key: str,
        tool_config_hash: str,
        capacity_scope: ToolCapacityScope,
        capacity_scope_id: str,
    ) -> int:
        with self._connect(self._dsn) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT consumed FROM {_CAPACITY_TABLE}
                    WHERE store_namespace_key = %s
                      AND tool_config_hash = %s
                      AND capacity_scope = %s
                      AND capacity_scope_id = %s
                    """,
                    (
                        store_namespace_key,
                        tool_config_hash,
                        capacity_scope.value,
                        _backend_scope_id(capacity_scope, capacity_scope_id),
                    ),
                )
                raw = cursor.fetchone()
                return (
                    0
                    if raw is None
                    else _decode_persisted_count(raw[0], field="consumed")
                )

    def close(self) -> None:
        pass
