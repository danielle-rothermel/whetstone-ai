from __future__ import annotations

import sqlite3
from pathlib import Path

from dr_store.relational import (
    RelationalContractMismatchError,
    TransactionObserver,
    connect_sqlite,
    sqlite_owned_tables,
    verify_sqlite_table,
)

from whetstone.optim.tools.admission import (
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
    _entries_in_scope,
    _entry_key,
    _entry_text,
    _EntryKey,
    _is_exact_schema_metadata,
    _is_exact_schema_version_row,
    _raise_owned_table_inventory_mismatch,
    _replay_or_conflict,
    _reraise_schema_mismatch,
    _scope_key,
)
from whetstone.optim.tools.contracts import (
    ToolCapacityScope,
)

_SQLITE_SCHEMA_COLUMNS = (
    ("component", "TEXT", True, 1),
    ("version", "INTEGER", True, 0),
)
_SQLITE_ENTRY_COLUMNS = (
    ("store_namespace_key", "TEXT", True, 1),
    ("call_id", "TEXT", True, 2),
    ("entry_json", "TEXT", True, 0),
)
_SQLITE_CAPACITY_COLUMNS = (
    ("store_namespace_key", "TEXT", True, 1),
    ("tool_config_hash", "TEXT", True, 2),
    ("capacity_scope", "TEXT", True, 3),
    ("capacity_scope_id", "TEXT", True, 4),
    ("max_accepted_calls", "INTEGER", True, 0),
    ("consumed", "INTEGER", True, 0),
)

_SQLITE_CREATE_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {_SCHEMA_TABLE} (
    component TEXT NOT NULL PRIMARY KEY CHECK (
        typeof(component) = 'text'
    ),
    version INTEGER NOT NULL CHECK (
        typeof(version) = 'integer' AND version > 0
    )
)
"""

_SQLITE_CREATE_ENTRY = f"""
CREATE TABLE IF NOT EXISTS {_ENTRY_TABLE} (
    store_namespace_key TEXT NOT NULL CHECK (
        typeof(store_namespace_key) = 'text'
    ),
    call_id TEXT NOT NULL CHECK (typeof(call_id) = 'text'),
    entry_json TEXT NOT NULL CHECK (typeof(entry_json) = 'text'),
    PRIMARY KEY (store_namespace_key, call_id)
)
"""

_SQLITE_CREATE_CAPACITY = f"""
CREATE TABLE IF NOT EXISTS {_CAPACITY_TABLE} (
    store_namespace_key TEXT NOT NULL CHECK (
        typeof(store_namespace_key) = 'text'
    ),
    tool_config_hash TEXT NOT NULL CHECK (
        typeof(tool_config_hash) = 'text'
    ),
    capacity_scope TEXT NOT NULL CHECK (
        typeof(capacity_scope) = 'text'
        AND capacity_scope IN ('global', 'run', 'step')
    ),
    capacity_scope_id TEXT NOT NULL CHECK (
        typeof(capacity_scope_id) = 'text'
    ),
    max_accepted_calls INTEGER NOT NULL CHECK (
        typeof(max_accepted_calls) = 'integer'
        AND max_accepted_calls >= 0
    ),
    consumed INTEGER NOT NULL CHECK (
        typeof(consumed) = 'integer'
        AND consumed >= 0 AND consumed <= max_accepted_calls
    ),
    PRIMARY KEY (
        store_namespace_key, tool_config_hash, capacity_scope,
        capacity_scope_id
    )
)
"""


def _owned_tables(connection: sqlite3.Connection) -> set[str]:
    try:
        return sqlite_owned_tables(
            connection,
            (_SCHEMA_TABLE, _ENTRY_TABLE, _CAPACITY_TABLE),
        )
    except RelationalContractMismatchError as exc:
        _reraise_schema_mismatch(exc)


def _verify_sqlite_schema(connection: sqlite3.Connection) -> None:
    for table, expected_columns, create_sql in (
        (_SCHEMA_TABLE, _SQLITE_SCHEMA_COLUMNS, _SQLITE_CREATE_SCHEMA),
        (_ENTRY_TABLE, _SQLITE_ENTRY_COLUMNS, _SQLITE_CREATE_ENTRY),
        (
            _CAPACITY_TABLE,
            _SQLITE_CAPACITY_COLUMNS,
            _SQLITE_CREATE_CAPACITY,
        ),
    ):
        try:
            verify_sqlite_table(
                connection,
                table=table,
                create_sql=create_sql,
                columns=expected_columns,
            )
        except RelationalContractMismatchError as exc:
            _reraise_schema_mismatch(exc)


class _SQLiteAdmissionBackend:
    def __init__(
        self,
        path: str | Path,
        *,
        transaction_observer: TransactionObserver | None = None,
    ) -> None:
        self._path = str(path)
        if not self._path:
            raise ValueError("SQLite path must be non-empty")
        if self._path == ":memory:":
            raise ValueError(
                "use ToolAdmissionAuthority.memory() for process-local memory"
            )
        self._transaction_observer = transaction_observer

    def _connect(
        self, *, observe_transaction: bool = False
    ) -> sqlite3.Connection:
        return connect_sqlite(
            self._path,
            observer=(
                self._transaction_observer if observe_transaction else None
            ),
        )

    def initialize(self) -> None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            tables = _owned_tables(connection)
            unversioned = {_ENTRY_TABLE, _CAPACITY_TABLE}
            versioned = {_SCHEMA_TABLE, _ENTRY_TABLE, _CAPACITY_TABLE}
            if not tables:
                connection.execute(_SQLITE_CREATE_SCHEMA)
                connection.execute(_SQLITE_CREATE_ENTRY)
                connection.execute(_SQLITE_CREATE_CAPACITY)
                insert_metadata = True
            elif tables == unversioned:
                connection.execute(_SQLITE_CREATE_SCHEMA)
                insert_metadata = True
            elif tables == versioned:
                insert_metadata = False
            else:
                _raise_owned_table_inventory_mismatch(tables)
            _verify_sqlite_schema(connection)
            row = connection.execute(
                f"""
                SELECT version FROM {_SCHEMA_TABLE}
                WHERE component = ?
                """,
                (_SCHEMA_COMPONENT,),
            ).fetchone()
            if insert_metadata:
                if row is not None:
                    raise ToolAdmissionSchemaMismatchError(
                        table=_SCHEMA_TABLE,
                        aspect="schema metadata",
                        expected=None,
                        actual=row,
                    )
                connection.execute(
                    f"""
                    INSERT INTO {_SCHEMA_TABLE} (component, version)
                    VALUES (?, ?)
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
            metadata = connection.execute(
                f"""
                SELECT component, version FROM {_SCHEMA_TABLE}
                ORDER BY component
                """
            ).fetchall()
            if not _is_exact_schema_metadata(metadata):
                raise ToolAdmissionSchemaMismatchError(
                    table=_SCHEMA_TABLE,
                    aspect="schema metadata",
                    expected=[(_SCHEMA_COMPONENT, _SCHEMA_VERSION)],
                    actual=metadata,
                )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _load(
        self, connection: sqlite3.Connection, key: _EntryKey
    ) -> ToolCallStoreEntry | None:
        raw = connection.execute(
            f"""
            SELECT entry_json FROM {_ENTRY_TABLE}
            WHERE store_namespace_key = ? AND call_id = ?
            """,
            key,
        ).fetchone()
        return None if raw is None else _decode_entry(raw[0])

    def admit(
        self,
        *,
        accepted: ToolCallStoreEntry,
        refused: ToolCallStoreEntry,
        max_accepted_calls: int,
    ) -> ToolCallStoreEntry:
        connection = self._connect(observe_transaction=True)
        try:
            connection.execute("BEGIN IMMEDIATE")
            if self._transaction_observer is not None:
                self._transaction_observer.transaction_acquired()
            key = _entry_key(accepted)
            existing = self._load(connection, key)
            if existing is not None:
                result = _replay_or_conflict(existing, accepted)
                connection.commit()
                return result
            scope = _scope_key(accepted)
            connection.execute(
                f"""
                INSERT OR IGNORE INTO {_CAPACITY_TABLE} (
                    store_namespace_key, tool_config_hash, capacity_scope,
                    capacity_scope_id, max_accepted_calls, consumed
                ) VALUES (?, ?, ?, ?, ?, 0)
                """,
                (*scope, max_accepted_calls),
            )
            capacity = connection.execute(
                f"""
                SELECT max_accepted_calls, consumed FROM {_CAPACITY_TABLE}
                WHERE store_namespace_key = ? AND tool_config_hash = ?
                  AND capacity_scope = ? AND capacity_scope_id = ?
                """,
                scope,
            ).fetchone()
            if capacity is None:
                raise RuntimeError("capacity scope disappeared")
            maximum = _decode_persisted_count(
                capacity[0], field="max_accepted_calls"
            )
            consumed = _decode_persisted_count(capacity[1], field="consumed")
            if maximum != max_accepted_calls:
                raise RuntimeError("capacity maximum changed within one scope")
            if consumed < maximum:
                ordinal = consumed + 1
                decision = _accepted_with_ordinal(accepted, ordinal)
                connection.execute(
                    f"""
                    UPDATE {_CAPACITY_TABLE} SET consumed = ?
                    WHERE store_namespace_key = ? AND tool_config_hash = ?
                      AND capacity_scope = ? AND capacity_scope_id = ?
                    """,
                    (ordinal, *scope),
                )
            else:
                decision = refused
            connection.execute(
                f"""
                INSERT INTO {_ENTRY_TABLE} (
                    store_namespace_key, call_id, entry_json
                ) VALUES (?, ?, ?)
                """,
                (*key, _entry_text(decision)),
            )
            connection.commit()
            return decision
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def refuse(self, entry: ToolCallStoreEntry) -> ToolCallStoreEntry:
        connection = self._connect(observe_transaction=True)
        try:
            connection.execute("BEGIN IMMEDIATE")
            if self._transaction_observer is not None:
                self._transaction_observer.transaction_acquired()
            key = _entry_key(entry)
            existing = self._load(connection, key)
            if existing is not None:
                result = _replay_or_conflict(existing, entry)
                connection.commit()
                return result
            connection.execute(
                f"""
                INSERT INTO {_ENTRY_TABLE} (
                    store_namespace_key, call_id, entry_json
                ) VALUES (?, ?, ?)
                """,
                (*key, _entry_text(entry)),
            )
            connection.commit()
            return entry
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def get(
        self, store_namespace_key: str, call_id: str
    ) -> ToolCallStoreEntry | None:
        connection = self._connect()
        try:
            return self._load(connection, (store_namespace_key, call_id))
        finally:
            connection.close()

    def complete(self, entry: ToolCallStoreEntry) -> ToolCallStoreEntry:
        connection = self._connect(observe_transaction=True)
        try:
            connection.execute("BEGIN IMMEDIATE")
            if self._transaction_observer is not None:
                self._transaction_observer.transaction_acquired()
            key = _entry_key(entry)
            completed = _complete_transition(
                self._load(connection, key), entry
            )
            connection.execute(
                f"""
                UPDATE {_ENTRY_TABLE} SET entry_json = ?
                WHERE store_namespace_key = ? AND call_id = ?
                """,
                (_entry_text(completed), *key),
            )
            connection.commit()
            return completed
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def admitted_entries(
        self,
        *,
        store_namespace_key: str,
        tool_config_hash: str,
        capacity_scope: ToolCapacityScope,
        capacity_scope_id: str,
    ) -> tuple[ToolCallStoreEntry, ...]:
        """Every entry this scope debited capacity for, in debit order.

        The entry rows are keyed by call id, so the scope columns are not
        indexed; the namespace narrows the scan and the decoded entry is
        the authority on which scope it belongs to. Callers use this to
        reconcile against ``accepted_count``, which counts the same
        debits from the capacity row.
        """
        scope_id = _backend_scope_id(capacity_scope, capacity_scope_id)
        connection = self._connect()
        try:
            rows = connection.execute(
                f"""
                SELECT entry_json FROM {_ENTRY_TABLE}
                WHERE store_namespace_key = ?
                """,
                (store_namespace_key,),
            ).fetchall()
        finally:
            connection.close()
        return _entries_in_scope(
            (_decode_entry(row[0]) for row in rows),
            store_namespace_key=store_namespace_key,
            tool_config_hash=tool_config_hash,
            capacity_scope=capacity_scope,
            capacity_scope_id=scope_id,
        )

    def accepted_count(
        self,
        *,
        store_namespace_key: str,
        tool_config_hash: str,
        capacity_scope: ToolCapacityScope,
        capacity_scope_id: str,
    ) -> int:
        connection = self._connect()
        try:
            raw = connection.execute(
                f"""
                SELECT consumed FROM {_CAPACITY_TABLE}
                WHERE store_namespace_key = ? AND tool_config_hash = ?
                  AND capacity_scope = ? AND capacity_scope_id = ?
                """,
                (
                    store_namespace_key,
                    tool_config_hash,
                    capacity_scope.value,
                    _backend_scope_id(capacity_scope, capacity_scope_id),
                ),
            ).fetchone()
            return (
                0
                if raw is None
                else _decode_persisted_count(raw[0], field="consumed")
            )
        finally:
            connection.close()

    def close(self) -> None:
        pass
