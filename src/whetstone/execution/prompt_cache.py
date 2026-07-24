"""Run-scoped prompt-result caching with original-call provenance."""

from __future__ import annotations

import fcntl
import json
import os
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Literal, Self
from uuid import uuid4

from dr_providers import ProviderCallRequest
from dr_serialize import build_identity_document, identity_document_hash
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictStr,
    ValidationError,
    model_validator,
)

from whetstone.execution.call_support import CallTelemetry, call_telemetry
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

PROMPT_CACHE_KEY_SCHEMA = "whetstone.execution.prompt_cache_key"
PROMPT_CACHE_KEY_SCHEMA_VERSION = 1
PROMPT_CACHE_ENTRY_SCHEMA = "whetstone.execution.prompt_cache_entry/v1"


class PromptCacheError(RuntimeError):
    """A cache entry was unreadable or failed current-schema validation."""


def prompt_cache_key(
    *,
    request: ProviderCallRequest,
    repeat_index: int,
) -> str:
    """Hash a fully resolved call identity and its repeat ordinal."""
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


class CacheProvenance(BaseModel):
    """Persistent reference to the call that originally populated an entry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: StrictStr
    source_phase: StrictStr
    source_unit: StrictStr
    source_logical_call_id: StrictStr
    stored_at: StrictStr


@dataclass(frozen=True, slots=True)
class PartialCacheMarks:
    """Cache provenance columns for one partial call record."""

    cache_hit: bool = False
    cache_source_phase: str | None = None
    cache_source_unit: str | None = None
    cache_source_call_id: str | None = None
    cache_source_at: str | None = None


def partial_cache_marks(
    cache_hit: bool,
    provenance: CacheProvenance | None,
) -> PartialCacheMarks:
    """Return honest partial-row marks for a cache miss or hit."""
    if cache_hit and provenance is None:
        raise ValueError("a cache hit requires original-entry provenance")
    if not cache_hit:
        return PartialCacheMarks()
    assert provenance is not None
    return PartialCacheMarks(
        cache_hit=True,
        cache_source_phase=provenance.source_phase,
        cache_source_unit=provenance.source_unit,
        cache_source_call_id=provenance.source_logical_call_id,
        cache_source_at=provenance.stored_at,
    )


@dataclass(frozen=True, slots=True)
class CallExecution:
    """One freshly executed or cache-served provider result."""

    result: ProviderCallResult
    cache_hit: bool = False
    provenance: CacheProvenance | None = None

    def __post_init__(self) -> None:
        if self.cache_hit != (self.provenance is not None):
            raise ValueError(
                "cache_hit and original-entry provenance must be paired"
            )

    def cache_marks(self) -> PartialCacheMarks:
        return partial_cache_marks(self.cache_hit, self.provenance)

    def telemetry(self) -> CallTelemetry:
        """Return telemetry without attributing original latency to a hit."""
        telemetry = call_telemetry(self.result)
        if not self.cache_hit:
            return telemetry
        return replace(telemetry, latency_s=None)


class _StoredEntry(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        populate_by_name=True,
    )

    schema_name: Literal["whetstone.execution.prompt_cache_entry/v1"] = Field(
        default=PROMPT_CACHE_ENTRY_SCHEMA,
        alias="schema",
    )
    key: StrictStr
    provenance: CacheProvenance
    result: ProviderCallResult

    @model_validator(mode="after")
    def _validate_key(self) -> Self:
        if self.provenance.key != self.key:
            raise ValueError("entry and provenance keys must match")
        return self


class _KeyFileLock:
    """An advisory cross-process lock scoped to one cache key."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle = None

    def __enter__(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+b")
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        assert self._handle is not None
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()
        self._handle = None


def _default_log(message: str) -> None:
    sys.stderr.write(message)


@dataclass(slots=True)
class PromptResultCache:
    """Content-addressed prompt results safe for threads and peer processes."""

    root: Path
    log: Callable[[str], None] = _default_log
    hits: int = 0
    misses: int = 0
    stores: int = 0
    _counter_lock: threading.Lock = field(
        default_factory=threading.Lock,
        repr=False,
    )

    @property
    def store_dir(self) -> Path:
        return self.root / "prompt_cache"

    def _path_for(self, key: str) -> Path:
        return self.store_dir / key[:2] / f"{key}.json"

    def _lock_path_for(self, key: str) -> Path:
        return self.store_dir / key[:2] / f".{key}.lock"

    def get_result(
        self,
        key: str,
    ) -> tuple[ProviderCallResult, CacheProvenance] | None:
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
        """Store once per key and preserve the winning writer's provenance."""
        proposed = CacheProvenance(
            key=key,
            source_phase=phase,
            source_unit=unit,
            source_logical_call_id=logical_call_id,
            stored_at=datetime.now(UTC).isoformat(),
        )
        path = self._path_for(key)
        with _KeyFileLock(self._lock_path_for(key)):
            try:
                existing = self._read_entry(key)
            except PromptCacheError:
                existing = None
            if existing is not None:
                return existing.provenance
            entry = _StoredEntry(
                key=key,
                provenance=proposed,
                result=result,
            )
            self._atomic_write(
                path,
                entry.model_dump(mode="json", by_alias=True),
            )
            with self._counter_lock:
                self.stores += 1
        return proposed

    def note_hit(self) -> None:
        with self._counter_lock:
            self.hits += 1

    def note_miss(self) -> None:
        with self._counter_lock:
            self.misses += 1

    def counters(self) -> dict[str, int]:
        with self._counter_lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "stores": self.stores,
            }

    def _read_entry(self, key: str) -> _StoredEntry | None:
        path = self._path_for(key)
        if not path.exists():
            return None
        try:
            raw = path.read_text()
        except OSError as exc:  # pragma: no cover - filesystem failure
            raise PromptCacheError(
                f"prompt-cache entry unreadable at {path}: {exc}"
            ) from exc
        try:
            entry = _StoredEntry.model_validate_json(raw)
        except ValidationError as exc:
            raise PromptCacheError(
                f"prompt-cache entry invalid at {path}: {exc}"
            ) from exc
        if entry.key != key:
            raise PromptCacheError(
                f"prompt-cache entry key mismatch at {path}"
            )
        return entry

    def _atomic_write(self, path: Path, body: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.parent / (
            f".{path.name}.{os.getpid()}.{threading.get_ident()}."
            f"{uuid4().hex}.tmp"
        )
        try:
            with temporary.open("x") as handle:
                json.dump(
                    body,
                    handle,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


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
    """Execute a provider call, optionally serving a trusted cached result."""
    if cache is None:
        return CallExecution(
            result=run_provider_call(
                request=request,
                policy=policy,
                transport=transport,
                logical_call_id=logical_call_id,
                clock=clock,
                sleep=sleep,
            )
        )

    key = prompt_cache_key(request=request, repeat_index=repeat_index)
    try:
        found = cache.get_result(key)
    except PromptCacheError as exc:
        cache.log(
            "PROMPT-CACHE CORRUPT ENTRY -> LOUD MISS: "
            f"key={key} logical_call_id={logical_call_id!r}: {exc}. "
            "Recomputing and overwriting the invalid entry.\n"
        )
        found = None
    if found is not None:
        result, provenance = found
        cache.note_hit()
        return CallExecution(
            result=result,
            cache_hit=True,
            provenance=provenance,
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
    return CallExecution(result=result)
