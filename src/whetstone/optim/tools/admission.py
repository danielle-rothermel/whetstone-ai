from __future__ import annotations

import json
from enum import UNIQUE, StrEnum, verify
from typing import Any, NoReturn, Protocol

from dr_serialize import decode_strict_json_bytes
from dr_store.relational import (
    RelationalContractMismatchError,
    raise_owned_table_inventory_mismatch,
    require_persisted_integer,
)
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from whetstone.core.leasing import (
    EffectRequest,
    EffectTerminal,
    ReplayPolicy,
    TerminalOutcome,
)
from whetstone.core.identity import (
    IdentityHash,
    NonEmptyId,
    NonNegativeInt,
    OpaqueKey,
    TypedRef,
    compute_identity_hash,
    compute_prefixed_identity_key,
)
from whetstone.optim.tools.contracts import (
    GLOBAL_CAPACITY_SCOPE_ID,
    TOOL_RESULT_SCHEMA,
    RefusalClass,
    ToolCall,
    ToolCallRef,
    ToolCapacityBinding,
    ToolCapacityScope,
    ToolConfigRef,
    ToolRefusal,
)

TOOL_CALL_ENTRY_SCHEMA = "whetstone.tool_call_store_entry"

_SCHEMA_TABLE = "whetstone_tool_admission_schema"
_ENTRY_TABLE = "whetstone_tool_admission_entry"
_CAPACITY_TABLE = "whetstone_tool_admission_capacity"
_SCHEMA_COMPONENT = "tool_admission"
_SCHEMA_VERSION = 2
_TOOL_EFFECT_SCHEMA = "whetstone.tool_execution_effect"
_TOOL_EFFECT_SCHEMA_VERSION = 1
_TOOL_EFFECT_KEY_SCHEMA = "whetstone.tool_execution_effect_key"
_TOOL_EFFECT_KEY_SCHEMA_VERSION = 1
_TOOL_EFFECT_KEY_PREFIX = "whetstone.tool_execution:"


class ToolAdmissionSchemaMismatchError(RuntimeError):
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


def _reraise_schema_mismatch(
    exc: RelationalContractMismatchError,
) -> NoReturn:
    raise ToolAdmissionSchemaMismatchError(
        table=exc.table,
        aspect=exc.aspect,
        expected=exc.expected,
        actual=exc.actual,
    ) from exc


@verify(UNIQUE)
class ToolCallState(StrEnum):
    ACCEPTED = "accepted"
    REFUSED = "refused"
    COMPLETED = "completed"


def tool_effect_request(call: ToolCall) -> EffectRequest:
    exact = ToolCall.model_validate(call.model_dump(mode="json"))
    replay_policy = (
        ReplayPolicy.IDEMPOTENT
        if exact.tool_config.record.idempotent_replay
        else ReplayPolicy.NO_REDRIVE
    )

    semantic_key = compute_prefixed_identity_key(
        schema=_TOOL_EFFECT_KEY_SCHEMA,
        schema_version=_TOOL_EFFECT_KEY_SCHEMA_VERSION,
        prefix=_TOOL_EFFECT_KEY_PREFIX,
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
        semantic_key=semantic_key,
        request_hash=compute_identity_hash(
            schema=_TOOL_EFFECT_SCHEMA,
            schema_version=_TOOL_EFFECT_SCHEMA_VERSION,
            payload=payload,
        ),
        replay_policy=replay_policy,
    )


class ToolCallStoreEntry(BaseModel):
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
        return self.tool_call.record_ref

    @property
    def tool_config_hash(self) -> IdentityHash:
        return self.tool_config.config_hash

    @property
    def call_id(self) -> NonEmptyId:
        return self.tool_call.record.call_id


class ToolCallStoreConflictError(RuntimeError):
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
    encoded = raw.encode()
    decode_strict_json_bytes(
        encoded,
        max_bytes=len(encoded),
        max_depth=len(encoded),
    )
    return ToolCallStoreEntry.model_validate_json(encoded)


def _decode_persisted_count(raw: object, *, field: str) -> int:
    try:
        return require_persisted_integer(raw, field=field)
    except RelationalContractMismatchError as exc:
        raise RuntimeError(
            f"persisted Tool admission {field} is not an integer"
        ) from exc


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


def _raise_owned_table_inventory_mismatch(tables: set[str]) -> NoReturn:
    try:
        raise_owned_table_inventory_mismatch(
            tables=tables,
            allowed=(
                set(),
                {_ENTRY_TABLE, _CAPACITY_TABLE},
                {_SCHEMA_TABLE, _ENTRY_TABLE, _CAPACITY_TABLE},
            ),
        )
    except RelationalContractMismatchError as exc:
        _reraise_schema_mismatch(exc)


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


__all__ = [
    "TOOL_CALL_ENTRY_SCHEMA",
    "ToolAdmissionSchemaMismatchError",
    "ToolCallState",
    "ToolCallStoreConflictError",
    "ToolCallStoreEntry",
    "tool_effect_request",
]
