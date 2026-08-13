from __future__ import annotations

from threading import RLock

from whetstone.optim.tools.admission import (
    ToolCallStoreEntry,
    _accepted_with_ordinal,
    _backend_scope_id,
    _complete_transition,
    _entry_key,
    _EntryKey,
    _replay_or_conflict,
    _scope_key,
    _ScopeKey,
)
from whetstone.optim.tools.contracts import (
    ToolCapacityScope,
)


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
