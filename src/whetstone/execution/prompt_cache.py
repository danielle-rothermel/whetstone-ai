"""Run-level prompt result cache (task 31).

Optimizer rounds re-evaluate heavily repeated prompts: proposed candidates are
often byte-identical paraphrases re-driven on the SAME tasks across rounds and
cells. This module is the opt-in, run-scoped store that keys on the fully
resolved provider call and returns the stored provider Result instead of
re-driving the transport.

The key (see :func:`prompt_cache_key`) is a content hash over EXACTLY the
identity-bearing surface at the call seam -- the Provider Call Request identity
(which folds the model id, the lane/protocol, and every output-affecting
sampling knob: temperature, top_p, token limit, reasoning effort, and any
provider body extension, PLUS the full resolved transcript as sent) -- together
with the REPEAT INDEX. The repeat index is mandatory: repeats exist to measure
provider nondeterminism, so a cache that ignored the repeat ordinal would
collapse the r-dimension variance to zero and silently corrupt every
repeat-based measurement. With it in the key, the SAME (prompt, settings,
repeat ordinal) reuses across cells/rounds -- the intended saving -- while the
within-cell repeat structure stays honest.

Storage follows the append-only, concurrent-writer-safe discipline of
:mod:`whetstone.execution.partials` and :mod:`whetstone.runner.events`: the
store lives under the ledger/run root at ``<root>/prompt_cache/`` as
content-addressed sharded JSON files (``<root>/prompt_cache/<aa>/<key>.json``),
so a keyed lookup is one ``Path.exists`` + read with no index to keep coherent
across concurrent writers. Writes are atomic (temp file + ``os.replace``) so a
crash mid-write never leaves a half-written entry a reader can see; a
first-writer-wins policy means a concurrent duplicate write is a harmless
no-op. No eviction: the store is run-scoped and deleted with the run root.

Honesty/provenance: a Result served from the cache is returned inside a
:class:`CallExecution` whose ``cache_hit`` is ``True`` and whose ``provenance``
names the ORIGINAL entry (its source cell/attempt, its logical call id, and the
original wall-clock the entry was first stored). Callers use that marker to
record the served row honestly -- latency ``None`` (never a fabricated zero)
and spend ``0`` with a distinct cached marker -- rather than re-attributing the
original call's cost/latency to the reuse.

A corrupt or unreadable entry is a LOUD MISS, never a crash and never silent
bad data: the read raises a typed :class:`PromptCacheError`, the wrapper logs
it, recomputes via the transport, and overwrites the bad entry.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dr_providers import ProviderCallRequest
from dr_serialize import build_identity_document, identity_document_hash
from pydantic import ValidationError

from whetstone.provider.attempt import ProviderCallResult
from whetstone.provider.driver import (
    Clock,
    Sleep,
    TransportCall,
    run_provider_call,
)
from whetstone.provider.policy import ProviderExecutionPolicy

__all__ = [
    "PROMPT_CACHE_ENTRY_SCHEMA",
    "PROMPT_CACHE_KEY_SCHEMA",
    "CacheProvenance",
    "CallExecution",
    "PartialCacheMarks",
    "PromptCacheError",
    "PromptResultCache",
    "execute_call",
    "partial_cache_marks",
    "prompt_cache_key",
]

#: Identity-document schema for the cache KEY payload. Version-stamped so a
#: future output-affecting knob added to the key surface is a DISTINCT key
#: space (a clean miss against old entries) rather than a silent collision.
PROMPT_CACHE_KEY_SCHEMA = "whetstone.execution.prompt_cache_key"
PROMPT_CACHE_KEY_SCHEMA_VERSION = 1

#: Versioned schema stamp written on every stored entry (the events/partials
#: precedent). A reader keys off it rather than sniffing fields; a stamp
#: mismatch is treated as a corrupt/foreign entry -> loud miss.
PROMPT_CACHE_ENTRY_SCHEMA = "whetstone.execution.prompt_cache_entry/v1"


class PromptCacheError(RuntimeError):
    """A cache entry was corrupt, unreadable, or schema-foreign.

    Raised by the store on a read it cannot trust. The :func:`execute_call`
    wrapper catches it, logs LOUDLY, and treats it as a miss (recompute +
    overwrite) -- so a bad entry degrades to a recompute, never a crash and
    never silently served bad data.
    """


def prompt_cache_key(
    *,
    request: ProviderCallRequest,
    repeat_index: int,
) -> str:
    """The content-hash cache key for one resolved call at one repeat ordinal.

    The key folds EXACTLY the identity-bearing surface at the call seam plus
    the mandatory repeat ordinal:

    * ``request.identity_payload()`` -- the Provider Call Config Identity Hash
      (which itself folds the model route/lane and EVERY output-affecting
      generation control: temperature, top_p, token limit, reasoning effort,
      and provider body extensions) together with the fully resolved
      Transcript (every message/text as sent). This is precisely "everything
      identity-bearing at call level".
    * ``repeat_index`` -- the repeat ordinal, so the r-dimension nondeterminism
      structure is preserved (see the module docstring).

    The Provider Execution Policy (transport timeout / retry / backoff) is
    deliberately NOT in the key: it is transport-only, excluded from Config
    identity by construction, and does not affect the generated output -- so
    two calls that differ only in transport policy MUST share a cache entry.

    Returns the full 64-char lowercase SHA-256 of the canonical Identity JSON.
    """
    if repeat_index < 0:
        raise ValueError("repeat_index must be a non-negative integer")
    document = build_identity_document(
        schema=PROMPT_CACHE_KEY_SCHEMA,
        schema_version=PROMPT_CACHE_KEY_SCHEMA_VERSION,
        payload={
            "request_identity": request.identity_payload(),
            "repeat_index": repeat_index,
        },
    )
    return identity_document_hash(document)


@dataclass(frozen=True, slots=True)
class CacheProvenance:
    """Provenance of a cache-served Result: a ref to the ORIGINAL entry.

    Emitted ONLY on a hit. ``source_phase`` + ``source_unit`` name the cell/
    attempt (phase + candidate/probe id) that first stored the entry;
    ``source_logical_call_id`` is that original call's logical id;
    ``stored_at`` is the ISO-8601 UTC wall-clock the entry was written. A row
    from cache carries these so a reader can trace the reuse back to the call
    that actually paid for it.
    """

    key: str
    source_phase: str
    source_unit: str
    source_logical_call_id: str
    stored_at: str


@dataclass(frozen=True, slots=True)
class PartialCacheMarks:
    """The cache-honesty columns a :class:`PartialCallRecord` records.

    A miss carries ``cache_hit=False`` and all-``None`` provenance; a hit
    carries ``cache_hit=True`` plus the original entry's source refs. Typed so
    a caller passes each column by name (no untyped ``**dict`` splat).
    """

    cache_hit: bool = False
    cache_source_phase: str | None = None
    cache_source_unit: str | None = None
    cache_source_call_id: str | None = None
    cache_source_at: str | None = None


def partial_cache_marks(
    cache_hit: bool, provenance: CacheProvenance | None
) -> PartialCacheMarks:
    """Build the typed :class:`PartialCacheMarks` for one row.

    A non-hit (or a hit with no provenance) is the non-cached default; a hit
    with provenance references the ORIGINAL entry's source cell/attempt +
    logical call id + original store timestamp.
    """
    if not cache_hit or provenance is None:
        return PartialCacheMarks()
    return PartialCacheMarks(
        cache_hit=True,
        cache_source_phase=provenance.source_phase,
        cache_source_unit=provenance.source_unit,
        cache_source_call_id=provenance.source_logical_call_id,
        cache_source_at=provenance.stored_at,
    )


@dataclass(frozen=True, slots=True)
class CallExecution:
    """One executed (or reused) provider call: Result + cache provenance.

    ``result`` is the terminal :class:`ProviderCallResult` -- byte-identical
    whether freshly driven (a miss) or reconstructed from the store (a hit), so
    all downstream scoring/telemetry is unchanged. ``cache_hit`` is ``True``
    only when the Result was served from the store; ``provenance`` is then the
    ref to the original entry, else ``None``. A caller marks a served row
    honestly off ``cache_hit`` (latency ``None``, spend ``0`` + cached marker).
    """

    result: ProviderCallResult
    cache_hit: bool = False
    provenance: CacheProvenance | None = None

    def cache_marks(self) -> PartialCacheMarks:
        """The typed cache-honesty marks for a :class:`PartialCallRecord`.

        On a MISS every mark is the non-cached default (``cache_hit=False``,
        all provenance ``None``). On a HIT it carries ``cache_hit=True`` plus
        the ref to the ORIGINAL entry. Callers that record latency must ALSO
        force ``latency_s=None`` on a hit (there was no wire call this time --
        never a fabricated 0); these marks own only the cache columns.
        """
        return partial_cache_marks(self.cache_hit, self.provenance)


def _default_log(message: str) -> None:
    sys.stderr.write(message)


@dataclass(frozen=True, slots=True)
class _Entry:
    result: ProviderCallResult
    provenance: CacheProvenance


@dataclass(slots=True)
class PromptResultCache:
    """Run-scoped, concurrent-writer-safe prompt result store.

    Rooted at ``<run_root>/prompt_cache/``. Entries are content-addressed
    sharded JSON files (``<shard>/<key>.json``, shard = the key's first two hex
    chars), so a lookup is a single ``exists`` + read with no index to keep
    coherent. Writes are atomic (temp file in the same shard dir +
    ``os.replace``) and first-writer-wins, so concurrent workers never corrupt
    an entry and a duplicate concurrent store is a harmless no-op.

    The in-process ``hits`` / ``misses`` / ``stores`` counters are the
    cache-hit telemetry a caller emits into the heartbeat/events seam. They are
    guarded by a process-local lock (the fan-out pool runs workers in threads).
    """

    root: Path
    log: Callable[[str], None] = _default_log
    hits: int = 0
    misses: int = 0
    stores: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def store_dir(self) -> Path:
        return self.root / "prompt_cache"

    def _path_for(self, key: str) -> Path:
        return self.store_dir / key[:2] / f"{key}.json"

    def get_result(
        self, key: str
    ) -> tuple[ProviderCallResult, CacheProvenance] | None:
        """The stored Result + provenance for ``key``, or ``None`` on a miss.

        Raises :class:`PromptCacheError` on a present-but-corrupt/unreadable/
        schema-foreign entry so the wrapper can degrade it to a LOUD miss
        rather than serve bad data.
        """
        entry = self._read_entry(key)
        if entry is None:
            return None
        return entry.result, entry.provenance

    def put(
        self,
        key: str,
        *,
        result: ProviderCallResult,
        phase: str,
        unit: str,
        logical_call_id: str,
    ) -> CacheProvenance:
        """Store ``result`` under ``key`` (first-writer-wins), return its ref.

        Writes atomically (temp file + ``os.replace``) so a crash mid-write is
        never observable, and skips the write if a VALID entry already exists
        (a concurrent duplicate store is a harmless no-op that keeps the FIRST
        writer's provenance). Returns the provenance ref for the winning entry.
        """
        provenance = CacheProvenance(
            key=key,
            source_phase=phase,
            source_unit=unit,
            source_logical_call_id=logical_call_id,
            stored_at=datetime.now(UTC).isoformat(),
        )
        path = self._path_for(key)
        if path.exists():
            # First-writer-wins: preserve the original provenance already on
            # disk. A present-but-corrupt entry (read raises) is overwritten.
            try:
                existing = self._read_entry(key)
            except PromptCacheError:
                existing = None
            if existing is not None:
                return existing.provenance
        body = {
            "schema": PROMPT_CACHE_ENTRY_SCHEMA,
            "key": key,
            "provenance": {
                "key": key,
                "source_phase": phase,
                "source_unit": unit,
                "source_logical_call_id": logical_call_id,
                "stored_at": provenance.stored_at,
            },
            "result": result.to_stable_dict(),
        }
        self._atomic_write(path, body)
        with self._lock:
            self.stores += 1
        return provenance

    def note_hit(self) -> None:
        with self._lock:
            self.hits += 1

    def note_miss(self) -> None:
        with self._lock:
            self.misses += 1

    def counters(self) -> dict[str, int]:
        """A snapshot of the (hits, misses, stores) counters."""
        with self._lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "stores": self.stores,
            }

    def _read_entry(self, key: str) -> _Entry | None:
        path = self._path_for(key)
        if not path.exists():
            return None
        try:
            raw = path.read_text()
        except OSError as exc:  # pragma: no cover - fs-level failure
            raise PromptCacheError(
                f"prompt-cache entry unreadable at {path}: {exc}"
            ) from exc
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            raise PromptCacheError(
                f"prompt-cache entry corrupt (bad JSON) at {path}: {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise PromptCacheError(
                f"prompt-cache entry corrupt (not an object) at {path}"
            )
        if data.get("schema") != PROMPT_CACHE_ENTRY_SCHEMA:
            raise PromptCacheError(
                "prompt-cache entry has a foreign/absent schema stamp "
                f"({data.get('schema')!r}) at {path}"
            )
        provenance = _provenance_from(data.get("provenance"), path)
        result = _result_from(data.get("result"), path)
        return _Entry(result=result, provenance=provenance)

    def _atomic_write(self, path: Path, body: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # A unique temp name in the SAME dir so os.replace is an atomic rename
        # on one filesystem; a reader never sees a partial file.
        tmp = path.parent / (
            f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        tmp.write_text(json.dumps(body, sort_keys=True))
        os.replace(tmp, path)


def _provenance_from(value: object, path: Path) -> CacheProvenance:
    if not isinstance(value, dict):
        raise PromptCacheError(
            f"prompt-cache entry corrupt (missing provenance) at {path}"
        )
    fields: dict[str, object] = {str(k): v for k, v in value.items()}
    try:
        return CacheProvenance(
            key=str(fields["key"]),
            source_phase=str(fields["source_phase"]),
            source_unit=str(fields["source_unit"]),
            source_logical_call_id=str(fields["source_logical_call_id"]),
            stored_at=str(fields["stored_at"]),
        )
    except KeyError as exc:
        raise PromptCacheError(
            f"prompt-cache entry corrupt (provenance field {exc}) at {path}"
        ) from exc


def _result_from(value: object, path: Path) -> ProviderCallResult:
    if not isinstance(value, dict):
        raise PromptCacheError(
            f"prompt-cache entry corrupt (missing result body) at {path}"
        )
    try:
        return ProviderCallResult.model_validate(value)
    except ValidationError as exc:
        raise PromptCacheError(
            "prompt-cache entry corrupt (result no longer validates) "
            f"at {path}: {exc}"
        ) from exc


def execute_call(
    *,
    request: ProviderCallRequest,
    policy: ProviderExecutionPolicy,
    transport: TransportCall,
    logical_call_id: str,
    repeat_index: int,
    cache: PromptResultCache | None,
    phase: str,
    unit: str,
    clock: Clock | None = None,
    sleep: Sleep | None = None,
) -> CallExecution:
    """Execute one task-role provider call, consulting the cache when present.

    This is the single seam every task-role driver routes its
    :func:`run_provider_call` through. When ``cache`` is ``None`` (the opt-out
    default) it is a thin pass-through: it drives the transport and returns
    ``CallExecution(result, cache_hit=False)`` -- byte-identical to calling
    ``run_provider_call`` directly, so no store is created and behavior is
    unchanged.

    When ``cache`` is present:

    * On a HIT (a trusted stored entry for this ``(request, repeat_index)``
      key) it returns the stored Result WITHOUT touching the transport, marked
      ``cache_hit=True`` with the original entry's :class:`CacheProvenance`.
    * On a MISS it drives the transport, stores the Result (SUCCESS OR typed
      failure -- a deterministic failure is worth reusing too), and returns it
      marked ``cache_hit=False``.
    * On a CORRUPT entry it logs LOUDLY and treats the read as a miss
      (recompute + overwrite the bad entry) -- never a crash, never bad data.

    ``phase``/``unit`` are the provenance the entry records as its source (the
    cell/attempt + candidate/probe id); ``repeat_index`` is the mandatory
    repeat ordinal that keys the entry.
    """
    if cache is None:
        result = run_provider_call(
            request=request,
            policy=policy,
            transport=transport,
            logical_call_id=logical_call_id,
            clock=clock,
            sleep=sleep,
        )
        return CallExecution(result=result, cache_hit=False)

    key = prompt_cache_key(request=request, repeat_index=repeat_index)
    try:
        found = cache.get_result(key)
    except PromptCacheError as exc:
        cache.log(
            "PROMPT-CACHE CORRUPT ENTRY -> LOUD MISS: "
            f"key={key} logical_call_id={logical_call_id!r}: {exc}. "
            "Recomputing and overwriting the bad entry.\n"
        )
        found = None
    if found is not None:
        result, provenance = found
        cache.note_hit()
        return CallExecution(
            result=result, cache_hit=True, provenance=provenance
        )

    cache.note_miss()
    result = run_provider_call(
        request=request,
        policy=policy,
        transport=transport,
        logical_call_id=logical_call_id,
        clock=clock,
        sleep=sleep,
    )
    cache.put(
        key,
        result=result,
        phase=phase,
        unit=unit,
        logical_call_id=logical_call_id,
    )
    return CallExecution(result=result, cache_hit=False)
