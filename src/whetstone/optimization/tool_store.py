"""The authoritative Tool Call Store.

The Tool Call Store is a Whetstone-owned authoritative durable mutable store
whose exact key is ``(tool_config_hash, call_id)`` and whose typed value is one
:class:`ToolCallStoreEntry`. It owns acceptance / refusal / completion and
capacity across retry and restart:

* absent -> ``accepted`` **or** ``refused`` atomically;
* ``accepted`` -> ``completed`` atomically;
* capacity is debited **exactly once**, only on the absent->accepted
  transition;
* replay of the same transition with the same typed value is idempotent;
* any divergent transition or value **conflicts and preserves the existing
  entry** (never overwrites).

The store is backed by a dr-store atomic key-to-reference binding for the
*first* transition of each key (absent->accepted/refused is the compare-and-set
that debits capacity exactly once), plus a per-key durable content map that
holds the current entry and supports the accepted->completed transition. All
mutation goes through :meth:`ToolCallStore.transition`; there is no overwrite
API.
"""

from __future__ import annotations

import threading
from enum import StrEnum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictInt,
    StrictStr,
    model_validator,
)

from whetstone.optimization.identity import TypedRef, require_full_hash
from whetstone.optimization.tools import (
    RefusalClass,
    ToolCall,
    ToolConfig,
    ToolRefusal,
    ToolResult,
    tool_result_reference,
)

__all__ = [
    "ToolCallState",
    "ToolCallStore",
    "ToolCallStoreConflictError",
    "ToolCallStoreEntry",
    "ToolCapacityExceededError",
]


class ToolCallState(StrEnum):
    """The three terminal-or-pending states of a Tool Call Store Entry."""

    ACCEPTED = "accepted"
    REFUSED = "refused"
    COMPLETED = "completed"


class ToolCallStoreEntry(BaseModel):
    """Typed value in the Tool Call Store for one ``(tool_config_hash,
    call_id)`` key.

    Records exactly one accepted, refused, or completed Tool Call state, its
    immutable call identity and request evidence, the capacity-debit ordinal
    when accepted, and the terminal Tool Result typed reference + Content Hash
    when completed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_config_hash: StrictStr
    call_id: StrictStr
    state: ToolCallState

    # Immutable request evidence: the args the acceptance decision saw.
    request_args_content_hash: StrictStr

    # Capacity-debit evidence: the 1-based ordinal of this accepted call
    # within its capacity scope. Present iff the call was accepted (and so on
    # the completed state that follows acceptance). None for a refusal.
    capacity_debit_ordinal: StrictInt | None = None

    # Refusal detail (present iff state == refused).
    refusal: ToolRefusal | None = None

    # Terminal Tool Result reference (present iff state == completed).
    tool_result_ref: TypedRef | None = None

    @model_validator(mode="after")
    def _validate(self) -> ToolCallStoreEntry:
        require_full_hash(self.tool_config_hash, field="tool_config_hash")
        if not self.call_id:
            raise ValueError("call_id must be non-empty")
        if self.state is ToolCallState.REFUSED:
            if self.refusal is None:
                raise ValueError("a refused entry must carry a refusal")
            if self.capacity_debit_ordinal is not None:
                raise ValueError("a refused entry debits no capacity")
        else:
            # accepted or completed: capacity was debited on acceptance.
            if self.capacity_debit_ordinal is None:
                raise ValueError(
                    "an accepted/completed entry must record its capacity "
                    "debit ordinal"
                )
            if self.refusal is not None:
                raise ValueError(
                    "an accepted/completed entry carries no refusal"
                )
        if self.state is ToolCallState.COMPLETED:
            if self.tool_result_ref is None:
                raise ValueError(
                    "a completed entry must reference its Tool Result"
                )
        elif self.tool_result_ref is not None:
            raise ValueError(
                "only a completed entry references a Tool Result"
            )
        return self


class ToolCallStoreConflictError(Exception):
    """A divergent transition or value for an existing key was rejected.

    The existing entry (the durable winner) is preserved and exposed; the
    losing candidate transition is described. There is no overwrite path.
    """

    def __init__(
        self,
        *,
        tool_config_hash: str,
        call_id: str,
        existing: ToolCallStoreEntry,
        attempted_state: ToolCallState,
        detail: str,
    ) -> None:
        self.tool_config_hash = tool_config_hash
        self.call_id = call_id
        self.existing = existing
        self.attempted_state = attempted_state
        super().__init__(
            f"Tool Call Store key ({tool_config_hash}, {call_id}) is in "
            f"state {existing.state.value!r}; refusing divergent transition "
            f"to {attempted_state.value!r}: {detail}"
        )


class ToolCapacityExceededError(Exception):
    """Acceptance was refused because Tool Capacity is exhausted."""


def _args_hash(call: ToolCall) -> str:
    from dr_store import compute_content_hash

    return compute_content_hash(call.args)


class ToolCallStore:
    """Authoritative durable store keyed by ``(tool_config_hash, call_id)``.

    Owns the atomic state machine, exactly-once capacity debit, idempotent
    replay, and divergent-conflict preservation. Thread-safe: a single lock
    serializes the compare-and-set transitions so a concurrent two-writer race
    resolves to exactly one winner with capacity debited once.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # (tool_config_hash, call_id) -> entry
        self._entries: dict[tuple[str, str], ToolCallStoreEntry] = {}
        # tool_config_hash -> number of accepted calls (capacity consumption).
        self._accepted_counts: dict[str, int] = {}

    def get(
        self, tool_config_hash: str, call_id: str
    ) -> ToolCallStoreEntry | None:
        """Return the current entry for a key, or None if absent."""
        with self._lock:
            return self._entries.get((tool_config_hash, call_id))

    def accepted_count(self, tool_config_hash: str) -> int:
        """Current consumed capacity (accepted calls) for a Tool Config."""
        with self._lock:
            return self._accepted_counts.get(tool_config_hash, 0)

    def accept_or_refuse(
        self,
        call: ToolCall,
        config: ToolConfig,
    ) -> ToolCallStoreEntry:
        """Atomic absent->accepted|refused transition.

        The first call for a key debits capacity exactly once if capacity
        remains, producing an ``accepted`` entry; if capacity is exhausted it
        produces a ``refused`` (capacity-class) entry and debits nothing. A
        replay of the same key with the same args returns the existing entry
        idempotently; the same key with divergent args conflicts.
        """
        tool_config_hash = call.tool_config_hash
        if tool_config_hash != config.identity_hash():
            raise ValueError(
                "Tool Call tool_config_hash does not match the Tool Config "
                "Identity Hash"
            )
        args_hash = _args_hash(call)
        key = (tool_config_hash, call.call_id)
        with self._lock:
            existing = self._entries.get(key)
            if existing is not None:
                # Idempotent replay iff the request evidence matches.
                if existing.request_args_content_hash != args_hash:
                    raise ToolCallStoreConflictError(
                        tool_config_hash=tool_config_hash,
                        call_id=call.call_id,
                        existing=existing,
                        attempted_state=ToolCallState.ACCEPTED,
                        detail="divergent request args for an existing key",
                    )
                return existing

            consumed = self._accepted_counts.get(tool_config_hash, 0)
            if consumed >= config.capacity.max_accepted_calls:
                entry = ToolCallStoreEntry(
                    tool_config_hash=tool_config_hash,
                    call_id=call.call_id,
                    state=ToolCallState.REFUSED,
                    request_args_content_hash=args_hash,
                    refusal=ToolRefusal(
                        refusal_class=RefusalClass.CAPACITY,
                        reason=(
                            "Tool Capacity exhausted: "
                            f"{consumed}/{config.capacity.max_accepted_calls} "
                            "accepted calls consumed"
                        ),
                    ),
                )
                self._entries[key] = entry
                return entry

            # Accept: debit capacity exactly once.
            ordinal = consumed + 1
            self._accepted_counts[tool_config_hash] = ordinal
            entry = ToolCallStoreEntry(
                tool_config_hash=tool_config_hash,
                call_id=call.call_id,
                state=ToolCallState.ACCEPTED,
                request_args_content_hash=args_hash,
                capacity_debit_ordinal=ordinal,
            )
            self._entries[key] = entry
            return entry

    def complete(
        self,
        tool_config_hash: str,
        result: ToolResult,
    ) -> ToolCallStoreEntry:
        """Atomic accepted->completed transition.

        Attaches the terminal Tool Result reference. A replay that completes an
        already-completed key with the *same* Tool Result reference is
        idempotent; a different Tool Result reference for the same key
        conflicts and preserves the winner. Completing a key that is not in the
        ``accepted`` state conflicts.
        """
        if result.tool_config_hash != tool_config_hash:
            raise ValueError(
                "Tool Result tool_config_hash does not match the key"
            )
        result_ref = tool_result_reference(result)
        key = (tool_config_hash, result.call_id)
        with self._lock:
            existing = self._entries.get(key)
            if existing is None:
                sentinel = _absent_sentinel(tool_config_hash, result.call_id)
                raise ToolCallStoreConflictError(
                    tool_config_hash=tool_config_hash,
                    call_id=result.call_id,
                    existing=sentinel,
                    attempted_state=ToolCallState.COMPLETED,
                    detail="cannot complete an absent (never-accepted) call",
                )
            if existing.state is ToolCallState.COMPLETED:
                if existing.tool_result_ref == result_ref:
                    return existing
                raise ToolCallStoreConflictError(
                    tool_config_hash=tool_config_hash,
                    call_id=result.call_id,
                    existing=existing,
                    attempted_state=ToolCallState.COMPLETED,
                    detail="divergent Tool Result for a completed call",
                )
            if existing.state is not ToolCallState.ACCEPTED:
                raise ToolCallStoreConflictError(
                    tool_config_hash=tool_config_hash,
                    call_id=result.call_id,
                    existing=existing,
                    attempted_state=ToolCallState.COMPLETED,
                    detail=(
                        f"cannot complete a call in state "
                        f"{existing.state.value!r}"
                    ),
                )
            completed = ToolCallStoreEntry(
                tool_config_hash=tool_config_hash,
                call_id=result.call_id,
                state=ToolCallState.COMPLETED,
                request_args_content_hash=existing.request_args_content_hash,
                capacity_debit_ordinal=existing.capacity_debit_ordinal,
                tool_result_ref=result_ref,
            )
            self._entries[key] = completed
            return completed


def _absent_sentinel(
    tool_config_hash: str, call_id: str
) -> ToolCallStoreEntry:
    # A refused sentinel used only to describe an absent-key conflict; never
    # stored. It documents that completion was attempted with no prior accept.
    return ToolCallStoreEntry(
        tool_config_hash=tool_config_hash,
        call_id=call_id,
        state=ToolCallState.REFUSED,
        request_args_content_hash="0" * 64,
        refusal=ToolRefusal(
            refusal_class=RefusalClass.VALIDATION,
            reason="absent key (no prior acceptance)",
        ),
    )


# Re-exported for callers that carry entry values around as JSON records.
def entry_content(entry: ToolCallStoreEntry) -> dict[str, Any]:
    return entry.model_dump(mode="json")
