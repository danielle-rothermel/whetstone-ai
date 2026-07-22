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
authoritative state lives in the injected :class:`~dr_store.ObjectStore` — the
per-key decision is durable through ``store.bind``/``store.resolve`` and each
entry body is content-addressed through ``store.put``/``store.get`` — so a
*fresh* store instance over the same durable backend reconstructs acceptance,
refusal, completion, and consumed capacity without any in-process memory. The
in-process lock only serializes concurrent transitions *within* one process;
cross-process exactly-once is carried by the atomic ``bind``. There is no
overwrite API.

Capacity is debited exactly once, durably, by claiming an atomic *capacity
slot* binding per accepted ordinal: ``bind(slot_key(tch, n), claim)`` is a
compare-and-set that only one call_id can win for ordinal ``n``. Consumed
capacity is therefore the number of bound slots, read back from the store — not
an in-process counter — so it survives a process restart.
"""

from __future__ import annotations

import threading
from enum import StrEnum
from typing import Any

from dr_store import (
    BindingConflictError,
    BindStatus,
    ObjectReference,
    ObjectStore,
)
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

# Durable record + binding-key schemas for the authoritative Tool Call Store.
TOOL_CALL_ENTRY_SCHEMA = "whetstone.tool_call_store_entry"
TOOL_CAPACITY_CLAIM_SCHEMA = "whetstone.tool_capacity_claim"

__all__ = [
    "TOOL_CALL_ENTRY_SCHEMA",
    "TOOL_CAPACITY_CLAIM_SCHEMA",
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


def _entry_key(tool_config_hash: str, call_id: str) -> str:
    """Opaque durable binding key for one ``(tool_config_hash, call_id)``."""
    return f"whetstone.tool_call_store_entry:{tool_config_hash}#{call_id}"


def _completed_key(tool_config_hash: str, call_id: str) -> str:
    """Opaque durable binding key for a key's *completed* decision.

    Distinct from :func:`_entry_key` because the binding table is append-only:
    the accepted->completed transition is recorded under this superseding key
    rather than rebinding the accepted decision.
    """
    return f"whetstone.tool_call_completed:{tool_config_hash}#{call_id}"


def _slot_key(tool_config_hash: str, ordinal: int) -> str:
    """Opaque durable binding key for one capacity slot ``ordinal``."""
    return f"whetstone.tool_capacity_slot:{tool_config_hash}#{ordinal}"


class ToolCallStore:
    """Authoritative durable store keyed by ``(tool_config_hash, call_id)``.

    Owns the atomic state machine, exactly-once capacity debit, idempotent
    replay, and divergent-conflict preservation. All authoritative state is
    held in the injected :class:`~dr_store.ObjectStore`: each entry body is
    content-addressed via ``put``/``get`` and each key's current decision is a
    durable ``bind`` under :func:`_entry_key`. Capacity is consumed by winning
    an atomic capacity-slot ``bind`` (:func:`_slot_key`), so consumed capacity
    is derived from the store, never from process memory — a fresh store
    instance over the same durable backend reconstructs the entire machine.

    Thread-safe within a process: a single lock serializes the compare-and-set
    transitions so a concurrent two-writer race resolves to exactly one winner
    with capacity debited once; cross-process exactly-once is carried by the
    atomic ``bind`` in the backend.
    """

    def __init__(self, store: ObjectStore) -> None:
        self._lock = threading.Lock()
        self._store = store

    def _put_entry(self, entry: ToolCallStoreEntry) -> ObjectReference:
        ref, _status = self._store.put(
            TOOL_CALL_ENTRY_SCHEMA, entry.model_dump(mode="json")
        )
        return ref

    def _bind_entry(
        self, tool_config_hash: str, call_id: str, entry: ToolCallStoreEntry
    ) -> ToolCallStoreEntry:
        """Persist an entry body and atomically bind it as the key's decision.

        Returns the decision the key actually holds after the bind: this call's
        entry when it wins (``BOUND``/``IDEMPOTENT``), or the durable winner
        loaded from the store when another writer bound first (the caller then
        reconciles idempotent-replay vs. conflict against the winner).
        """
        ref = self._put_entry(entry)
        key = _entry_key(tool_config_hash, call_id)
        try:
            self._store.bind(key, ref)
        except BindingConflictError:
            # A different decision already won this key; return the winner.
            winner = self._load_entry(tool_config_hash, call_id)
            assert winner is not None
            return winner
        return entry

    def _load_entry(
        self, tool_config_hash: str, call_id: str
    ) -> ToolCallStoreEntry | None:
        # A completed decision supersedes the accepted decision on read.
        completed = self._store.resolve(
            _completed_key(tool_config_hash, call_id)
        )
        reference = completed or self._store.resolve(
            _entry_key(tool_config_hash, call_id)
        )
        if reference is None:
            return None
        content = self._store.get(reference)
        return ToolCallStoreEntry.model_validate(content)

    def get(
        self, tool_config_hash: str, call_id: str
    ) -> ToolCallStoreEntry | None:
        """Return the current entry for a key, or None if absent.

        Read exclusively from the store's durable binding + content map, so a
        fresh store instance observes the same decision.
        """
        with self._lock:
            return self._load_entry(tool_config_hash, call_id)

    def accepted_count(self, tool_config_hash: str) -> int:
        """Current consumed capacity (accepted calls) for a Tool Config.

        Derived from the number of bound capacity slots in the store, so it is
        restored after a process restart rather than recomputed from memory.
        """
        with self._lock:
            return self._consumed_capacity(tool_config_hash)

    def _consumed_capacity(self, tool_config_hash: str) -> int:
        consumed = 0
        while self._store.resolve(_slot_key(tool_config_hash, consumed + 1)):
            consumed += 1
        return consumed

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
        idempotently; the same key with divergent args conflicts. Every
        decision is bound durably, so a restart resolves the same outcome and
        never re-debits capacity.
        """
        tool_config_hash = call.tool_config_hash
        if tool_config_hash != config.identity_hash():
            raise ValueError(
                "Tool Call tool_config_hash does not match the Tool Config "
                "Identity Hash"
            )
        args_hash = _args_hash(call)
        with self._lock:
            existing = self._load_entry(tool_config_hash, call.call_id)
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

            max_calls = config.capacity.max_accepted_calls
            ordinal = self._claim_capacity_slot(
                tool_config_hash, call.call_id, max_calls
            )
            if ordinal is None:
                entry = ToolCallStoreEntry(
                    tool_config_hash=tool_config_hash,
                    call_id=call.call_id,
                    state=ToolCallState.REFUSED,
                    request_args_content_hash=args_hash,
                    refusal=ToolRefusal(
                        refusal_class=RefusalClass.CAPACITY,
                        reason=(
                            "Tool Capacity exhausted: "
                            f"{max_calls}/{max_calls} accepted calls consumed"
                        ),
                    ),
                )
            else:
                entry = ToolCallStoreEntry(
                    tool_config_hash=tool_config_hash,
                    call_id=call.call_id,
                    state=ToolCallState.ACCEPTED,
                    request_args_content_hash=args_hash,
                    capacity_debit_ordinal=ordinal,
                )
            decided = self._bind_entry(
                tool_config_hash, call.call_id, entry
            )
            if decided != entry:
                # Another writer bound this key first (cross-process race).
                if decided.request_args_content_hash != args_hash:
                    raise ToolCallStoreConflictError(
                        tool_config_hash=tool_config_hash,
                        call_id=call.call_id,
                        existing=decided,
                        attempted_state=ToolCallState.ACCEPTED,
                        detail="divergent request args for an existing key",
                    )
            return decided

    def refuse(
        self,
        call: ToolCall,
        config: ToolConfig,
        *,
        refusal: ToolRefusal,
    ) -> ToolCallStoreEntry:
        """Durably record a non-capacity refusal (validation/budget/auth).

        A capacity refusal is produced by :meth:`accept_or_refuse` as part of
        the acceptance decision; this method records the *other* refusal
        classes (a validation-rejected argument, a budget guard, an
        authorization failure) as a durable refused Store Entry so the refusal
        is inspectable evidence that never masquerades as a measurement and
        debits **no** capacity. A capacity-class refusal is not accepted here —
        capacity is the store's own accounting, decided only by
        :meth:`accept_or_refuse`.

        Idempotent: replaying the same key with the same refusal returns the
        existing entry; an accepted/completed key, or a divergent refusal for
        an existing key, conflicts and preserves the winner.
        """
        if refusal.refusal_class is RefusalClass.CAPACITY:
            raise ValueError(
                "a capacity refusal is decided only by accept_or_refuse; "
                "refuse() records validation/budget/authorization refusals"
            )
        tool_config_hash = call.tool_config_hash
        if tool_config_hash != config.identity_hash():
            raise ValueError(
                "Tool Call tool_config_hash does not match the Tool Config "
                "Identity Hash"
            )
        args_hash = _args_hash(call)
        with self._lock:
            existing = self._load_entry(tool_config_hash, call.call_id)
            if existing is not None:
                if (
                    existing.state is ToolCallState.REFUSED
                    and existing.refusal == refusal
                    and existing.request_args_content_hash == args_hash
                ):
                    return existing
                raise ToolCallStoreConflictError(
                    tool_config_hash=tool_config_hash,
                    call_id=call.call_id,
                    existing=existing,
                    attempted_state=ToolCallState.REFUSED,
                    detail="divergent transition for an existing key",
                )
            entry = ToolCallStoreEntry(
                tool_config_hash=tool_config_hash,
                call_id=call.call_id,
                state=ToolCallState.REFUSED,
                request_args_content_hash=args_hash,
                refusal=refusal,
            )
            return self._bind_entry(tool_config_hash, call.call_id, entry)

    def _claim_capacity_slot(
        self, tool_config_hash: str, call_id: str, max_calls: int
    ) -> int | None:
        """Atomically claim the next free capacity ordinal, or None if full.

        Each ordinal 1..max_calls is a durable slot bound to the winning
        call_id's claim record. Claiming is a compare-and-set: only the first
        call_id to ``bind`` a slot wins it. A crash after the slot bind but
        before the entry bind is safe — the same call_id re-claims its own slot
        idempotently (the claim names the call_id), so no capacity is lost or
        double-debited.
        """
        claim: dict[str, Any] = {
            "tool_config_hash": tool_config_hash,
            "call_id": call_id,
        }
        claim_ref = ObjectReference.for_record(
            TOOL_CAPACITY_CLAIM_SCHEMA, claim
        )
        ordinal = 1
        while ordinal <= max_calls:
            key = _slot_key(tool_config_hash, ordinal)
            bound = self._store.resolve(key)
            if bound is None:
                self._store.put(TOOL_CAPACITY_CLAIM_SCHEMA, claim)
                status = self._store.bind(key, claim_ref)
                if status is BindStatus.BOUND:
                    return ordinal
                # Lost the race for this ordinal; fall through to inspect it.
                bound = self._store.resolve(key)
            assert bound is not None
            won = self._store.get(bound)
            if (
                isinstance(won, dict)
                and won.get("call_id") == call_id
                and won.get("tool_config_hash") == tool_config_hash
            ):
                # This call already owns this ordinal (idempotent re-claim).
                return ordinal
            ordinal += 1
        return None

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
        ``accepted`` state conflicts. The completed decision is re-bound under
        the same key, so a restart observes the completion durably.
        """
        if result.tool_config_hash != tool_config_hash:
            raise ValueError(
                "Tool Result tool_config_hash does not match the key"
            )
        result_ref = tool_result_reference(result)
        with self._lock:
            existing = self._load_entry(tool_config_hash, result.call_id)
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
            # Re-bind is not possible under the same key (append-only binding),
            # so completion is recorded under a distinct completed-decision key
            # that supersedes the accepted decision on read.
            return self._complete_bind(
                tool_config_hash, result.call_id, completed, result_ref
            )

    def _complete_bind(
        self,
        tool_config_hash: str,
        call_id: str,
        completed: ToolCallStoreEntry,
        result_ref: TypedRef,
    ) -> ToolCallStoreEntry:
        ref = self._put_entry(completed)
        key = _completed_key(tool_config_hash, call_id)
        try:
            self._store.bind(key, ref)
        except BindingConflictError:
            # A concurrent completer bound a divergent Tool Result first; the
            # durable winner is preserved and this loser conflicts.
            winner = self._load_entry(tool_config_hash, call_id)
            assert winner is not None
            if winner.tool_result_ref == result_ref:
                return winner
            raise ToolCallStoreConflictError(
                tool_config_hash=tool_config_hash,
                call_id=call_id,
                existing=winner,
                attempted_state=ToolCallState.COMPLETED,
                detail="divergent Tool Result for a completed call",
            ) from None
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
