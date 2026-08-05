"""Atomic admission and terminal persistence for exact Tool Calls."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable
from contextlib import AbstractContextManager
from enum import UNIQUE, StrEnum, verify
from pathlib import Path
from threading import RLock
from typing import Any, Protocol, cast

from dr_store import ObjectStore
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from whetstone.optimization.effect_authority import (
    EffectAuthority,
    EffectRequest,
    EffectTerminal,
    ReplayPolicy,
    TerminalOutcome,
)
from whetstone.optimization.identity import (
    IdentityHash,
    NonEmptyId,
    NonNegativeInt,
    OpaqueKey,
    TypedRef,
    compute_identity_hash,
    typed_ref_for_record,
)
from whetstone.optimization.reward import REWARD_SCHEMA
from whetstone.optimization.tools import (
    GLOBAL_CAPACITY_SCOPE_ID,
    TOOL_CALL_SCHEMA,
    TOOL_CONFIG_SCHEMA,
    TOOL_DEFINITION_SCHEMA,
    TOOL_RESULT_SCHEMA,
    RefusalClass,
    ToolCall,
    ToolCallRef,
    ToolCapacityBinding,
    ToolCapacityScope,
    ToolConfig,
    ToolConfigRef,
    ToolRefusal,
    ToolResult,
    tool_call_reference,
    tool_config_reference,
    tool_result_reference,
)

__all__ = [
    "TOOL_CALL_ENTRY_SCHEMA",
    "ToolAdmissionAuthority",
    "ToolAdmissionSchemaMismatchError",
    "ToolCallState",
    "ToolCallStore",
    "ToolCallStoreConflictError",
    "ToolCallStoreEntry",
    "tool_effect_request",
]

TOOL_CALL_ENTRY_SCHEMA = "whetstone.tool_call_store_entry"

_SCHEMA_TABLE = "whetstone_tool_admission_schema"
_ENTRY_TABLE = "whetstone_tool_admission_entry"
_CAPACITY_TABLE = "whetstone_tool_admission_capacity"
_SCHEMA_COMPONENT = "tool_admission"
_SCHEMA_VERSION = 2
_ENTRY_LOCK_DOMAIN = "whetstone.tool_admission.entry_lock.v1"
_TOOL_EFFECT_SCHEMA = "whetstone.tool_execution_effect"
_TOOL_EFFECT_SCHEMA_VERSION = 1
_TOOL_EFFECT_KEY_SCHEMA = "whetstone.tool_execution_effect_key"
_TOOL_EFFECT_KEY_SCHEMA_VERSION = 1
_TOOL_EFFECT_KEY_PREFIX = "whetstone.tool_execution:"

type _ColumnContract = tuple[str, str, bool, int]

_SQLITE_SCHEMA_COLUMNS: tuple[_ColumnContract, ...] = (
    ("component", "TEXT", True, 1),
    ("version", "INTEGER", True, 0),
)
_SQLITE_ENTRY_COLUMNS: tuple[_ColumnContract, ...] = (
    ("store_namespace_key", "TEXT", True, 1),
    ("call_id", "TEXT", True, 2),
    ("entry_json", "TEXT", True, 0),
)
_SQLITE_CAPACITY_COLUMNS: tuple[_ColumnContract, ...] = (
    ("store_namespace_key", "TEXT", True, 1),
    ("tool_config_hash", "TEXT", True, 2),
    ("capacity_scope", "TEXT", True, 3),
    ("capacity_scope_id", "TEXT", True, 4),
    ("max_accepted_calls", "INTEGER", True, 0),
    ("consumed", "INTEGER", True, 0),
)
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


class ToolAdmissionSchemaMismatchError(RuntimeError):
    """The durable Tool admission schema is not the exact owned contract."""

    def __init__(
        self,
        *,
        table: str,
        aspect: str,
        expected: object,
        actual: object,
    ) -> None:
        self.table = table
        self.aspect = aspect
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"incompatible Tool admission table {table!r}: expected exact "
            f"{aspect} {expected!r}, found {actual!r}; apply the Tool "
            "admission schema migration before constructing "
            "ToolAdmissionAuthority"
        )


@verify(UNIQUE)
class ToolCallState(StrEnum):
    """Persisted Tool Call lifecycle state.

    These values are persisted contract literals. Never iterate over this
    enum to construct a persisted payload.
    """

    ACCEPTED = "accepted"
    REFUSED = "refused"
    COMPLETED = "completed"


def tool_effect_request(call: ToolCall) -> EffectRequest:
    """Return the one effect request authorized by an exact Tool Call."""
    exact = ToolCall.model_validate(call.model_dump(mode="json"))
    replay_policy = (
        ReplayPolicy.IDEMPOTENT
        if exact.tool_config.record.idempotent_replay
        else ReplayPolicy.NO_REDRIVE
    )
    # Persisted-format contract: schema, version, prefix, and payload keys are
    # pinned by golden tests. Field names must never be derived from models.
    semantic_key_hash = compute_identity_hash(
        schema=_TOOL_EFFECT_KEY_SCHEMA,
        schema_version=_TOOL_EFFECT_KEY_SCHEMA_VERSION,
        payload={
            "store_namespace_key": exact.store_namespace_key,
            "call_id": exact.call_id,
        },
    )
    payload = {
        "tool_call": exact.record_content(),
        "tool_config_record_ref": exact.tool_config.record_ref.model_dump(
            mode="json"
        ),
        "store_namespace_key": exact.store_namespace_key,
        "capacity_scope": exact.capacity_scope,
        "capacity_scope_id": exact.capacity_scope_id,
    }
    return EffectRequest(
        semantic_key=OpaqueKey(
            f"{_TOOL_EFFECT_KEY_PREFIX}{semantic_key_hash}"
        ),
        request_identity_hash=compute_identity_hash(
            schema=_TOOL_EFFECT_SCHEMA,
            schema_version=_TOOL_EFFECT_SCHEMA_VERSION,
            payload=payload,
        ),
        replay_policy=replay_policy,
    )


class ToolCallStoreEntry(BaseModel):
    """One exact admission decision and its optional terminal result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_call: ToolCallRef
    tool_config: ToolConfigRef
    store_namespace_key: OpaqueKey
    capacity_scope: ToolCapacityScope
    capacity_scope_id: NonEmptyId
    state: ToolCallState
    capacity_debit_ordinal: NonNegativeInt | None = None
    refusal: ToolRefusal | None = None
    tool_result_ref: TypedRef | None = None
    effect_terminal: EffectTerminal | None = None

    @field_validator("effect_terminal", mode="before")
    @classmethod
    def _parse_effect_terminal(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return EffectTerminal.model_validate_json(
                json.dumps(
                    value,
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
        return value

    @model_validator(mode="after")
    def _validate(self) -> ToolCallStoreEntry:
        call = self.tool_call.record
        if self.tool_config != call.tool_config:
            raise ValueError(
                "Tool Call Store entry must cite the call's exact Tool Config"
            )
        if self.store_namespace_key != call.store_namespace_key:
            raise ValueError(
                "Tool Call Store entry namespace must match the exact call"
            )
        if self.capacity_scope is not call.capacity_scope:
            raise ValueError(
                "Tool Call Store entry scope must match the exact call"
            )
        if self.capacity_scope_id != call.capacity_scope_id:
            raise ValueError(
                "Tool Call Store entry scope ID must match the exact call"
            )
        _capacity_scope_key(call.capacity_binding)
        if self.state is ToolCallState.ACCEPTED:
            if self.capacity_debit_ordinal is None:
                raise ValueError("an accepted entry must record an ordinal")
            if self.capacity_debit_ordinal == 0:
                raise ValueError("capacity debit ordinals are one-based")
            if (
                self.refusal is not None
                or self.tool_result_ref is not None
                or self.effect_terminal is not None
            ):
                raise ValueError(
                    "an accepted entry has no refusal or terminal result"
                )
        elif self.state is ToolCallState.REFUSED:
            if self.capacity_debit_ordinal is not None:
                raise ValueError("a refused entry consumes no capacity")
            if (
                self.refusal is None
                or self.tool_result_ref is None
                or self.effect_terminal is not None
            ):
                raise ValueError(
                    "a refused entry requires its exact refusal and result"
                )
        else:
            if self.capacity_debit_ordinal is None:
                raise ValueError(
                    "a completed entry retains its capacity ordinal"
                )
            if self.capacity_debit_ordinal == 0:
                raise ValueError("capacity debit ordinals are one-based")
            if (
                self.refusal is not None
                or self.tool_result_ref is None
                or self.effect_terminal is None
            ):
                raise ValueError(
                    "a completed entry requires its exact effect terminal "
                    "and result"
                )
        if (
            self.tool_result_ref is not None
            and self.tool_result_ref.schema_name != TOOL_RESULT_SCHEMA
        ):
            raise ValueError("terminal entry must reference a Tool Result")
        if self.effect_terminal is not None:
            if self.effect_terminal.request != tool_effect_request(call):
                raise ValueError(
                    "completed entry effect terminal belongs to another "
                    "exact Tool request"
                )
            if self.effect_terminal.outcome not in (
                TerminalOutcome.SUCCEEDED,
                TerminalOutcome.FAILED,
            ):
                raise ValueError(
                    "recovery-required effects have no completed Tool Result"
                )
            if self.effect_terminal.result_ref != self.tool_result_ref:
                raise ValueError(
                    "completed entry effect terminal references another "
                    "exact Tool Result"
                )
        return self

    @property
    def tool_call_ref(self) -> TypedRef:
        """The exact call reference consumed by Tool Evidence."""
        return self.tool_call.record_ref

    @property
    def tool_config_hash(self) -> IdentityHash:
        return self.tool_config.identity_hash

    @property
    def call_id(self) -> NonEmptyId:
        return self.tool_call.record.call_id


class ToolCallStoreConflictError(RuntimeError):
    """A divergent request or transition lost to an immutable decision."""

    def __init__(
        self,
        *,
        existing: ToolCallStoreEntry,
        attempted_state: ToolCallState,
        detail: str,
    ) -> None:
        self.existing = existing
        self.attempted_state = attempted_state
        self.tool_config_hash = str(existing.tool_config_hash)
        self.call_id = str(existing.call_id)
        super().__init__(
            "Tool Call Store key "
            f"({existing.store_namespace_key}, {existing.call_id}) is in "
            f"state {existing.state.value!r}; refusing divergent transition "
            f"to {attempted_state.value!r}: {detail}"
        )


def _entry_text(entry: ToolCallStoreEntry) -> str:
    return json.dumps(
        entry.model_dump(mode="json"),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _decode_entry(raw: object) -> ToolCallStoreEntry:
    if type(raw) is not str:
        raise RuntimeError("persisted Tool admission entry is not JSON text")
    return ToolCallStoreEntry.model_validate_json(raw)


def _decode_persisted_count(raw: object, *, field: str) -> int:
    if type(raw) is not int:
        raise RuntimeError(
            f"persisted Tool admission {field} is not an integer"
        )
    return raw


def _is_exact_schema_version_row(row: tuple[Any, ...] | None) -> bool:
    return (
        row is not None
        and len(row) == 1
        and type(row[0]) is int
        and row[0] == _SCHEMA_VERSION
    )


def _is_exact_schema_metadata(rows: list[tuple[Any, ...]]) -> bool:
    return (
        len(rows) == 1
        and len(rows[0]) == 2
        and type(rows[0][0]) is str
        and type(rows[0][1]) is int
        and rows[0] == (_SCHEMA_COMPONENT, _SCHEMA_VERSION)
    )


def _replay_or_conflict(
    existing: ToolCallStoreEntry,
    attempted: ToolCallStoreEntry,
) -> ToolCallStoreEntry:
    if existing.tool_call != attempted.tool_call:
        raise ToolCallStoreConflictError(
            existing=existing,
            attempted_state=attempted.state,
            detail="the existing key cites a different exact Tool Call",
        )
    if attempted.state is ToolCallState.ACCEPTED:
        return existing
    if existing == attempted:
        return existing
    raise ToolCallStoreConflictError(
        existing=existing,
        attempted_state=attempted.state,
        detail="the existing key has a different immutable decision",
    )


def _complete_transition(
    existing: ToolCallStoreEntry | None,
    completed: ToolCallStoreEntry,
) -> ToolCallStoreEntry:
    if existing is None:
        raise ToolCallStoreConflictError(
            existing=completed,
            attempted_state=ToolCallState.COMPLETED,
            detail="the exact Tool Call was never admitted",
        )
    if existing.tool_call != completed.tool_call:
        raise ToolCallStoreConflictError(
            existing=existing,
            attempted_state=ToolCallState.COMPLETED,
            detail="the terminal result belongs to a different Tool Call",
        )
    if existing.state is ToolCallState.COMPLETED:
        return _replay_or_conflict(existing, completed)
    if existing.state is not ToolCallState.ACCEPTED:
        raise ToolCallStoreConflictError(
            existing=existing,
            attempted_state=ToolCallState.COMPLETED,
            detail="a refused call cannot become completed",
        )
    if existing.capacity_debit_ordinal != completed.capacity_debit_ordinal:
        raise ToolCallStoreConflictError(
            existing=existing,
            attempted_state=ToolCallState.COMPLETED,
            detail="completion changed the accepted capacity ordinal",
        )
    return completed


class _AdmissionBackend(Protocol):
    def initialize(self) -> None: ...

    def admit(
        self,
        *,
        accepted: ToolCallStoreEntry,
        refused: ToolCallStoreEntry,
        max_accepted_calls: int,
    ) -> ToolCallStoreEntry: ...

    def refuse(self, entry: ToolCallStoreEntry) -> ToolCallStoreEntry: ...

    def get(
        self, store_namespace_key: str, call_id: str
    ) -> ToolCallStoreEntry | None: ...

    def complete(self, entry: ToolCallStoreEntry) -> ToolCallStoreEntry: ...

    def accepted_count(
        self,
        *,
        store_namespace_key: str,
        tool_config_hash: str,
        capacity_scope: ToolCapacityScope,
        capacity_scope_id: str,
    ) -> int: ...

    def close(self) -> None: ...


class _SQLiteTransactionObserver(Protocol):
    def transaction_attempted(self) -> None: ...

    def transaction_acquired(self) -> None: ...


type _EntryKey = tuple[str, str]
type _ScopeKey = tuple[str, str, str, str]


def _capacity_scope_key(
    binding: ToolCapacityBinding,
) -> tuple[ToolCapacityScope, str]:
    exact = ToolCapacityBinding.model_validate(binding.model_dump(mode="json"))
    return exact.scope, str(exact.capacity_scope_id)


def _backend_scope_id(
    capacity_scope: ToolCapacityScope,
    capacity_scope_id: str,
) -> str:
    scope_id = str(capacity_scope_id)
    if (
        capacity_scope is ToolCapacityScope.GLOBAL
        and scope_id != GLOBAL_CAPACITY_SCOPE_ID
    ):
        raise ValueError(
            "GLOBAL Tool Capacity requires capacity_scope_id "
            f"{GLOBAL_CAPACITY_SCOPE_ID!r}"
        )
    return scope_id


def _entry_key(entry: ToolCallStoreEntry) -> _EntryKey:
    return (str(entry.store_namespace_key), str(entry.call_id))


def _scope_key(entry: ToolCallStoreEntry) -> _ScopeKey:
    capacity_scope, capacity_scope_id = _capacity_scope_key(
        entry.tool_call.record.capacity_binding
    )
    return (
        str(entry.store_namespace_key),
        str(entry.tool_config_hash),
        capacity_scope.value,
        capacity_scope_id,
    )


def _accepted_with_ordinal(
    accepted: ToolCallStoreEntry, ordinal: int
) -> ToolCallStoreEntry:
    content = accepted.model_dump(mode="json")
    content["capacity_debit_ordinal"] = ordinal
    return ToolCallStoreEntry.model_validate(content)


def _validate_admission_attempt(
    *,
    accepted: ToolCallStoreEntry,
    refused: ToolCallStoreEntry,
    max_accepted_calls: int,
) -> None:
    if accepted.state is not ToolCallState.ACCEPTED:
        raise ValueError("admission accepted candidate must be accepted")
    if refused.state is not ToolCallState.REFUSED:
        raise ValueError("admission refused candidate must be refused")
    if accepted.tool_call != refused.tool_call:
        raise ValueError("admission candidates must cite one exact Tool Call")
    if (
        refused.refusal is None
        or refused.refusal.refusal_class is not RefusalClass.CAPACITY
    ):
        raise ValueError("admission refusal must be a capacity refusal")
    if max_accepted_calls < 0:
        raise ValueError("max_accepted_calls must be non-negative")


class _MemoryAdmissionBackend:
    def __init__(self) -> None:
        self._entries: dict[_EntryKey, ToolCallStoreEntry] = {}
        self._capacity: dict[_ScopeKey, tuple[int, int]] = {}
        self._lock = RLock()

    def initialize(self) -> None:
        pass

    def admit(
        self,
        *,
        accepted: ToolCallStoreEntry,
        refused: ToolCallStoreEntry,
        max_accepted_calls: int,
    ) -> ToolCallStoreEntry:
        with self._lock:
            key = _entry_key(accepted)
            existing = self._entries.get(key)
            if existing is not None:
                return _replay_or_conflict(existing, accepted)
            scope_key = _scope_key(accepted)
            maximum, consumed = self._capacity.get(
                scope_key, (max_accepted_calls, 0)
            )
            if maximum != max_accepted_calls:
                raise RuntimeError("capacity maximum changed within one scope")
            if consumed < maximum:
                ordinal = consumed + 1
                decision = _accepted_with_ordinal(accepted, ordinal)
                self._capacity[scope_key] = (maximum, ordinal)
            else:
                decision = refused
            self._entries[key] = decision
            return decision

    def refuse(self, entry: ToolCallStoreEntry) -> ToolCallStoreEntry:
        with self._lock:
            key = _entry_key(entry)
            existing = self._entries.get(key)
            if existing is not None:
                return _replay_or_conflict(existing, entry)
            self._entries[key] = entry
            return entry

    def get(
        self, store_namespace_key: str, call_id: str
    ) -> ToolCallStoreEntry | None:
        with self._lock:
            return self._entries.get((store_namespace_key, call_id))

    def complete(self, entry: ToolCallStoreEntry) -> ToolCallStoreEntry:
        with self._lock:
            key = _entry_key(entry)
            completed = _complete_transition(self._entries.get(key), entry)
            self._entries[key] = completed
            return completed

    def accepted_count(
        self,
        *,
        store_namespace_key: str,
        tool_config_hash: str,
        capacity_scope: ToolCapacityScope,
        capacity_scope_id: str,
    ) -> int:
        with self._lock:
            return self._capacity.get(
                (
                    store_namespace_key,
                    tool_config_hash,
                    capacity_scope.value,
                    _backend_scope_id(capacity_scope, capacity_scope_id),
                ),
                (0, 0),
            )[1]

    def close(self) -> None:
        pass


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


def _sqlite_columns(
    connection: sqlite3.Connection,
    table: str,
) -> tuple[_ColumnContract, ...]:
    rows = connection.execute(f'PRAGMA table_info("{table}")').fetchall()
    return tuple(
        (str(name), str(column_type), bool(not_null), int(primary_key))
        for _, name, column_type, not_null, _, primary_key in rows
    )


def _normalized_sql(value: str) -> str:
    return " ".join(value.split())


def _sqlite_owned_tables(connection: sqlite3.Connection) -> set[str]:
    rows = connection.execute(
        """
        SELECT name FROM sqlite_master
        WHERE type = 'table' AND name IN (?, ?, ?)
        ORDER BY name
        """,
        (_SCHEMA_TABLE, _ENTRY_TABLE, _CAPACITY_TABLE),
    ).fetchall()
    if not all(len(row) == 1 and type(row[0]) is str for row in rows):
        raise ToolAdmissionSchemaMismatchError(
            table="<catalog>",
            aspect="owned table inventory",
            expected="text table names",
            actual=rows,
        )
    return {row[0] for row in rows}


def _raise_owned_table_inventory_mismatch(tables: set[str]) -> None:
    raise ToolAdmissionSchemaMismatchError(
        table="<database>",
        aspect="owned table inventory",
        expected=(
            set(),
            {_ENTRY_TABLE, _CAPACITY_TABLE},
            {_SCHEMA_TABLE, _ENTRY_TABLE, _CAPACITY_TABLE},
        ),
        actual=tables,
    )


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
        actual = _sqlite_columns(connection, table)
        if actual != expected_columns:
            raise ToolAdmissionSchemaMismatchError(
                table=table,
                aspect="columns",
                expected=expected_columns,
                actual=actual,
            )
        raw_sql = connection.execute(
            """
            SELECT sql FROM sqlite_master
            WHERE type = 'table' AND name = ?
            """,
            (table,),
        ).fetchone()
        actual_sql = (
            None
            if raw_sql is None or not isinstance(raw_sql[0], str)
            else _normalized_sql(raw_sql[0])
        )
        expected_sql = _normalized_sql(create_sql).replace(
            "CREATE TABLE IF NOT EXISTS",
            "CREATE TABLE",
            1,
        )
        if actual_sql != expected_sql:
            raise ToolAdmissionSchemaMismatchError(
                table=table,
                aspect="table definition",
                expected=expected_sql,
                actual=actual_sql,
            )


class _SQLiteAdmissionBackend:
    def __init__(
        self,
        path: str | Path,
        *,
        transaction_observer: _SQLiteTransactionObserver | None = None,
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
        connection = sqlite3.connect(
            self._path, timeout=30.0, isolation_level=None
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
            tables = _sqlite_owned_tables(connection)
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


class ToolAdmissionAuthority:
    """Atomic call-decision and scoped-capacity authority."""

    def __init__(self, backend: _AdmissionBackend) -> None:
        self._backend = backend
        self._backend.initialize()

    @classmethod
    def memory(cls) -> ToolAdmissionAuthority:
        return cls(_MemoryAdmissionBackend())

    @classmethod
    def sqlite(
        cls,
        path: str | Path,
        *,
        _transaction_observer: _SQLiteTransactionObserver | None = None,
    ) -> ToolAdmissionAuthority:
        return cls(
            _SQLiteAdmissionBackend(
                path,
                transaction_observer=_transaction_observer,
            )
        )

    @classmethod
    def postgresql(
        cls,
        dsn: str,
        *,
        _connect: _Connect | None = None,
    ) -> ToolAdmissionAuthority:
        return cls(_PostgreSQLAdmissionBackend(dsn, connect=_connect))

    def admit(
        self,
        *,
        accepted: ToolCallStoreEntry,
        refused: ToolCallStoreEntry,
        max_accepted_calls: int,
    ) -> ToolCallStoreEntry:
        _validate_admission_attempt(
            accepted=accepted,
            refused=refused,
            max_accepted_calls=max_accepted_calls,
        )
        return self._backend.admit(
            accepted=accepted,
            refused=refused,
            max_accepted_calls=max_accepted_calls,
        )

    def refuse(self, entry: ToolCallStoreEntry) -> ToolCallStoreEntry:
        if entry.state is not ToolCallState.REFUSED:
            raise ValueError("refusal candidate must be refused")
        if (
            entry.refusal is not None
            and entry.refusal.refusal_class is RefusalClass.CAPACITY
        ):
            raise ValueError("capacity refusal is owned by admission")
        return self._backend.refuse(entry)

    def get(
        self, store_namespace_key: str, call_id: str
    ) -> ToolCallStoreEntry | None:
        return self._backend.get(store_namespace_key, call_id)

    def complete(self, entry: ToolCallStoreEntry) -> ToolCallStoreEntry:
        if entry.state is not ToolCallState.COMPLETED:
            raise ValueError("completion candidate must be completed")
        return self._backend.complete(entry)

    def accepted_count(
        self,
        *,
        store_namespace_key: str,
        tool_config_hash: str,
        capacity_scope: ToolCapacityScope,
        capacity_scope_id: str,
    ) -> int:
        return self._backend.accepted_count(
            store_namespace_key=store_namespace_key,
            tool_config_hash=tool_config_hash,
            capacity_scope=capacity_scope,
            capacity_scope_id=capacity_scope_id,
        )

    def close(self) -> None:
        self._backend.close()


class ToolCallStore:
    """Persist exact Tool records and delegate mutable decisions atomically."""

    def __init__(
        self,
        store: ObjectStore,
        admission_authority: ToolAdmissionAuthority,
        effect_authority: EffectAuthority,
    ) -> None:
        self._store = store
        self._admission = admission_authority
        self._effect_authority = effect_authority

    def _put_exact(self, schema: str, content: dict[str, Any]) -> TypedRef:
        expected = typed_ref_for_record(schema, content)
        reference, _status = self._store.put(schema, content)
        persisted = TypedRef(
            schema_name=reference.schema,
            content_hash=reference.content_hash,
        )
        if persisted != expected:
            raise ValueError(f"persisted {schema} ref failed validation")
        return persisted

    def _persist_call_chain(
        self, call: ToolCall, config: ToolConfig
    ) -> ToolCallRef:
        validated_config = ToolConfig.model_validate(
            config.model_dump(mode="json")
        )
        validated_call = ToolCall.model_validate(call.model_dump(mode="json"))
        _capacity_scope_key(validated_call.capacity_binding)
        config_ref = tool_config_reference(validated_config)
        if validated_call.tool_config != config_ref:
            raise ValueError(
                "Tool Call must cite the exact supplied Tool Config"
            )
        definition = config_ref.record.definition
        if (
            self._put_exact(
                TOOL_DEFINITION_SCHEMA, definition.record.record_content()
            )
            != definition.record_ref
        ):
            raise ValueError("persisted Tool Definition ref failed validation")
        if (
            self._put_exact(
                TOOL_CONFIG_SCHEMA, validated_config.record_content()
            )
            != config_ref.record_ref
        ):
            raise ValueError("persisted Tool Config ref failed validation")
        call_ref = tool_call_reference(validated_call)
        if (
            self._put_exact(TOOL_CALL_SCHEMA, validated_call.record_content())
            != call_ref.record_ref
        ):
            raise ValueError("persisted Tool Call ref failed validation")
        return call_ref

    def get(self, call: ToolCall) -> ToolCallStoreEntry | None:
        validated = ToolCall.model_validate(call.model_dump(mode="json"))
        _capacity_scope_key(validated.capacity_binding)
        existing = self._admission.get(
            str(validated.store_namespace_key), str(validated.call_id)
        )
        if existing is not None and existing.tool_call != tool_call_reference(
            validated
        ):
            raise ToolCallStoreConflictError(
                existing=existing,
                attempted_state=ToolCallState.ACCEPTED,
                detail="the existing key cites a different exact Tool Call",
            )
        return existing

    def admit(self, call: ToolCall, config: ToolConfig) -> ToolCallStoreEntry:
        call_ref = self._persist_call_chain(call, config)
        config_ref = tool_config_reference(config)
        common: dict[str, Any] = {
            "tool_call": call_ref,
            "tool_config": config_ref,
            "store_namespace_key": call_ref.record.store_namespace_key,
            "capacity_scope": call_ref.record.capacity_scope,
            "capacity_scope_id": call_ref.record.capacity_scope_id,
        }
        accepted = ToolCallStoreEntry(
            **common,
            state=ToolCallState.ACCEPTED,
            capacity_debit_ordinal=1,
        )
        refusal = ToolRefusal(
            refusal_class=RefusalClass.CAPACITY,
            reason=(
                "Tool Capacity exhausted: "
                f"{config.capacity.max_accepted_calls}/"
                f"{config.capacity.max_accepted_calls} accepted calls consumed"
            ),
        )
        refused_result = ToolResult(call=call_ref, refusal=refusal)
        refused_ref = self.persist_result(refused_result)
        refused = ToolCallStoreEntry(
            **common,
            state=ToolCallState.REFUSED,
            refusal=refusal,
            tool_result_ref=refused_ref,
        )
        return self._admission.admit(
            accepted=accepted,
            refused=refused,
            max_accepted_calls=int(config.capacity.max_accepted_calls),
        )

    def refuse(
        self,
        call: ToolCall,
        config: ToolConfig,
        *,
        refusal: ToolRefusal,
    ) -> ToolCallStoreEntry:
        if refusal.refusal_class is RefusalClass.CAPACITY:
            raise ValueError("capacity refusal is owned by admission")
        call_ref = self._persist_call_chain(call, config)
        result = ToolResult(call=call_ref, refusal=refusal)
        result_ref = self.persist_result(result)
        entry = ToolCallStoreEntry(
            tool_call=call_ref,
            tool_config=call_ref.record.tool_config,
            store_namespace_key=call_ref.record.store_namespace_key,
            capacity_scope=call_ref.record.capacity_scope,
            capacity_scope_id=call_ref.record.capacity_scope_id,
            state=ToolCallState.REFUSED,
            refusal=refusal,
            tool_result_ref=result_ref,
        )
        return self._admission.refuse(entry)

    def persist_result(self, result: ToolResult) -> TypedRef:
        validated = ToolResult.model_validate(result.model_dump(mode="json"))
        if validated.reward is not None:
            persisted_reward = self._put_exact(
                REWARD_SCHEMA,
                validated.reward.record.record_content(),
            )
            if persisted_reward != validated.reward.record_ref:
                raise ValueError("persisted Reward ref failed validation")
        expected = tool_result_reference(validated)
        persisted = self._put_exact(
            TOOL_RESULT_SCHEMA, validated.record_content()
        )
        if persisted != expected.record_ref:
            raise ValueError("persisted Tool Result ref failed validation")
        return persisted

    def complete(
        self,
        result: ToolResult,
        *,
        terminal: EffectTerminal | None = None,
    ) -> ToolCallStoreEntry:
        validated = ToolResult.model_validate(result.model_dump(mode="json"))
        if validated.refusal is not None:
            raise ValueError("a refused Tool Result is terminal at admission")
        existing = self.get(validated.call.record)
        if existing is None:
            raise ValueError("the exact Tool Call was never admitted")
        self._validate_result_authority(validated, existing)
        result_ref = tool_result_reference(validated).record_ref
        if terminal is None:
            raise ValueError(
                "completion requires an exact authoritative "
                "EffectTerminal proof"
            )
        exact_terminal = EffectTerminal.model_validate_json(
            terminal.model_dump_json()
        )
        expected_request = tool_effect_request(validated.call.record)
        if exact_terminal.request != expected_request:
            raise ValueError(
                "effect terminal belongs to another exact Tool request"
            )
        if exact_terminal.result_ref != result_ref:
            raise ValueError(
                "effect terminal references another exact Tool Result"
            )
        if (
            exact_terminal.outcome is TerminalOutcome.SUCCEEDED
            and validated.terminal_failure is not None
        ):
            raise ValueError(
                "succeeded effect terminal references a failed Tool Result"
            )
        if exact_terminal.outcome is TerminalOutcome.FAILED:
            if exact_terminal.failure != validated.terminal_failure:
                raise ValueError(
                    "failed effect terminal and Tool Result disagree"
                )
        elif exact_terminal.outcome is not TerminalOutcome.SUCCEEDED:
            raise ValueError(
                "recovery-required effects have no completed Tool Result"
            )
        authoritative_terminal = self._effect_authority.verify_terminal(
            exact_terminal
        )
        self.persist_result(validated)
        completed = ToolCallStoreEntry(
            tool_call=validated.call,
            tool_config=validated.tool_config,
            store_namespace_key=validated.store_namespace_key,
            capacity_scope=validated.call.record.capacity_scope,
            capacity_scope_id=validated.call.record.capacity_scope_id,
            state=ToolCallState.COMPLETED,
            capacity_debit_ordinal=existing.capacity_debit_ordinal,
            tool_result_ref=result_ref,
            effect_terminal=authoritative_terminal,
        )
        return self._admission.complete(completed)

    def load_terminal_result(self, entry: ToolCallStoreEntry) -> ToolResult:
        if entry.tool_result_ref is None:
            raise ValueError("entry has no terminal Tool Result")
        durable_entry = self._admission.get(
            str(entry.store_namespace_key),
            str(entry.call_id),
        )
        if durable_entry != entry:
            raise ValueError(
                "terminal entry does not match the durable admission decision"
            )
        if entry.state is ToolCallState.COMPLETED:
            if entry.effect_terminal is None:
                raise ValueError(
                    "completed entry has no exact effect terminal"
                )
            self._effect_authority.verify_terminal(entry.effect_terminal)
        content = self._store.get(entry.tool_result_ref.reference)
        result = ToolResult.model_validate(content)
        expected = tool_result_reference(result)
        if expected.record_ref != entry.tool_result_ref:
            raise ValueError(
                "persisted Tool Result ref does not match the terminal entry"
            )
        if result.call != entry.tool_call:
            raise ValueError(
                "persisted Tool Result belongs to a different exact Tool Call"
            )
        if entry.state is ToolCallState.REFUSED:
            if result.refusal != entry.refusal:
                raise ValueError(
                    "persisted Tool Result and admission refusal disagree"
                )
        elif entry.state is ToolCallState.COMPLETED:
            terminal = entry.effect_terminal
            if terminal is None:
                raise AssertionError(
                    "completed entry validation lost its effect terminal"
                )
            expected_outcome = (
                TerminalOutcome.FAILED
                if result.terminal_failure is not None
                else TerminalOutcome.SUCCEEDED
            )
            if terminal.outcome is not expected_outcome:
                raise ValueError(
                    "persisted Tool Result and effect terminal outcome "
                    "disagree"
                )
            if terminal.failure != result.terminal_failure:
                raise ValueError(
                    "persisted Tool Result and effect terminal failure "
                    "disagree"
                )
            self._validate_result_authority(result, entry)
        return result

    def _validate_result_authority(
        self,
        result: ToolResult,
        entry: ToolCallStoreEntry,
    ) -> None:
        if result.refusal is not None:
            raise ValueError(
                "a refused Tool Result has no capacity provenance"
            )
        ordinal = entry.capacity_debit_ordinal
        if ordinal is None:
            raise ValueError(
                "a non-refused Tool Result requires an admitted capacity "
                "ordinal"
            )
        capacity_scope, capacity_scope_id = _capacity_scope_key(
            entry.tool_call.record.capacity_binding
        )
        accepted_count = self._admission.accepted_count(
            store_namespace_key=str(entry.store_namespace_key),
            tool_config_hash=str(entry.tool_config_hash),
            capacity_scope=capacity_scope,
            capacity_scope_id=capacity_scope_id,
        )
        if not 1 <= ordinal <= accepted_count:
            raise ValueError(
                "completed entry capacity ordinal is outside the durable "
                "admission projection"
            )
        result_ordinal = result.provenance_ordinal
        if result_ordinal is None or result_ordinal == 0:
            raise ValueError(
                "a non-refused Tool Result requires a positive provenance "
                "ordinal"
            )
        if result_ordinal != ordinal:
            raise ValueError(
                "Tool Result provenance ordinal disagrees with the durable "
                "admission capacity ordinal"
            )

    def load_result(
        self, result_ref: TypedRef, *, expected_call: ToolCall
    ) -> ToolResult:
        """Load one exact Tool Result and validate its complete call chain."""
        if result_ref.schema_name != TOOL_RESULT_SCHEMA:
            raise ValueError("effect terminal references a non-Tool Result")
        content = self._store.get(result_ref.reference)
        result = ToolResult.model_validate(content)
        if tool_result_reference(result).record_ref != result_ref:
            raise ValueError("effect terminal Tool Result ref is not exact")
        if result.call != tool_call_reference(expected_call):
            raise ValueError(
                "effect terminal Tool Result belongs to another Tool Call"
            )
        return result

    def accepted_count(
        self,
        config: ToolConfig,
        binding: ToolCapacityBinding,
    ) -> int:
        validated = ToolConfig.model_validate(config.model_dump(mode="json"))
        exact_binding = ToolCapacityBinding.model_validate(
            binding.model_dump(mode="json")
        )
        capacity_scope, capacity_scope_id = _capacity_scope_key(exact_binding)
        if capacity_scope is not validated.capacity.scope:
            raise ValueError(
                "Tool Capacity binding scope must match the exact Tool Config"
            )
        return self._admission.accepted_count(
            store_namespace_key=str(validated.store_namespace_key),
            tool_config_hash=str(validated.identity_hash()),
            capacity_scope=capacity_scope,
            capacity_scope_id=capacity_scope_id,
        )

    @property
    def effect_authority(self) -> EffectAuthority:
        """The sole effect authority allowed to terminalize this store."""
        return self._effect_authority

    def close(self) -> None:
        self._admission.close()
