from __future__ import annotations

import json
import os
import re
import stat
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, NewType, Self
from uuid import uuid4

from dr_providers import ProviderCallRequest
from dr_serialize import (
    StrictJsonDecodeError,
    build_identity_document,
    decode_strict_json_bytes,
    identity_document_hash,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    ValidationError,
    model_validator,
)

from whetstone.execution._file_lock import (
    FileLock,
    ensure_private_directory,
    fsync_file,
    fsync_parent_directory,
    open_private_regular_file,
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
    "PROMPT_CACHE_ACCOUNTING_SCHEMA",
    "PROMPT_CACHE_ENTRY_SCHEMA",
    "PROMPT_CACHE_KEY_SCHEMA",
    "PROMPT_CACHE_STATS_SCHEMA",
    "CacheProvenance",
    "CallExecution",
    "PartialCacheMarks",
    "PromptCacheError",
    "PromptCacheKey",
    "PromptResultCache",
    "execute_call",
    "partial_cache_marks",
    "prompt_cache_key",
]

PROMPT_CACHE_KEY_SCHEMA = "whetstone.execution.prompt_cache_key"
PROMPT_CACHE_KEY_SCHEMA_VERSION = 2
PROMPT_CACHE_ENTRY_SCHEMA = "whetstone.execution.prompt_cache_entry/v3"
PROMPT_CACHE_STATS_SCHEMA = "whetstone.execution.prompt_cache_stats/v1"
PROMPT_CACHE_ACCOUNTING_SCHEMA = (
    "whetstone.execution.prompt_cache_accounting/v1"
)
_CACHE_KEY_PATTERN = re.compile(r"[0-9a-f]{64}")
_PUBLICATION_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
PromptCacheKey = NewType("PromptCacheKey", str)


class PromptCacheError(RuntimeError):
    """A cache entry was unreadable or failed current-schema validation."""


def _validate_cache_key(key: str) -> PromptCacheKey:
    if not isinstance(key, str):
        raise TypeError("prompt-cache key must be a string")
    if _CACHE_KEY_PATTERN.fullmatch(key) is None:
        raise ValueError(
            "prompt-cache key must be exactly 64 lowercase hexadecimal "
            "characters"
        )
    return PromptCacheKey(key)


def _validate_publication_id(publication_id: str) -> None:
    if _PUBLICATION_ID_PATTERN.fullmatch(publication_id) is None:
        raise ValueError(
            "publication_id must be exactly 32 lowercase hexadecimal "
            "characters"
        )


def prompt_cache_key(
    request: ProviderCallRequest,
    policy: ProviderExecutionPolicy,
    repeat_index: int,
    drive_ordinal: int,
) -> PromptCacheKey:
    """Hash every semantic identity component of one physical call drive."""
    return _prompt_cache_key_from_components(
        request_identity=request.identity_payload(),
        execution_policy_hash=policy.identity_hash,
        repeat_index=repeat_index,
        drive_ordinal=drive_ordinal,
    )


def _prompt_cache_key_from_components(
    *,
    request_identity: dict[str, Any],
    execution_policy_hash: str,
    repeat_index: int,
    drive_ordinal: int,
) -> PromptCacheKey:
    if isinstance(repeat_index, bool) or not isinstance(repeat_index, int):
        raise TypeError("repeat_index must be an integer")
    if repeat_index < 0:
        raise ValueError("repeat_index must be non-negative")
    if isinstance(drive_ordinal, bool) or not isinstance(drive_ordinal, int):
        raise TypeError("drive_ordinal must be an integer")
    if drive_ordinal < 0:
        raise ValueError("drive_ordinal must be non-negative")
    document = build_identity_document(
        schema=PROMPT_CACHE_KEY_SCHEMA,
        schema_version=PROMPT_CACHE_KEY_SCHEMA_VERSION,
        payload={
            # Persisted identity keys are a pinned wire contract. Do not
            # derive or enumerate them from model field names.
            "request_identity": request_identity,
            "execution_policy_hash": execution_policy_hash,
            "repeat_index": repeat_index,
            "drive_ordinal": drive_ordinal,
        },
    )
    return _validate_cache_key(identity_document_hash(document))


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

    schema_name: Literal["whetstone.execution.prompt_cache_entry/v3"] = Field(
        default=PROMPT_CACHE_ENTRY_SCHEMA,
        alias="schema",
    )
    key: StrictStr
    request_identity: dict[str, Any]
    execution_policy_hash: StrictStr
    repeat_index: StrictInt
    drive_ordinal: StrictInt
    result_policy_hash: StrictStr
    publication_id: StrictStr
    provenance: CacheProvenance
    result: ProviderCallResult

    @model_validator(mode="after")
    def _validate_key(self) -> Self:
        _validate_cache_key(self.key)
        _validate_publication_id(self.publication_id)
        if self.repeat_index < 0:
            raise ValueError("repeat_index must be non-negative")
        if self.drive_ordinal < 0:
            raise ValueError("drive_ordinal must be non-negative")
        if self.provenance.key != self.key:
            raise ValueError("entry and provenance keys must match")
        if self.result.request_identity != self.request_identity:
            raise ValueError("entry and result request identities must match")
        if self.result.execution_policy_hash != self.result_policy_hash:
            raise ValueError(
                "entry result policy hash must match the stored result"
            )
        if self.result_policy_hash != self.execution_policy_hash:
            raise ValueError(
                "entry and result execution policy hashes must match"
            )
        expected_key = _prompt_cache_key_from_components(
            request_identity=self.request_identity,
            execution_policy_hash=self.execution_policy_hash,
            repeat_index=self.repeat_index,
            drive_ordinal=self.drive_ordinal,
        )
        if self.key != expected_key:
            raise ValueError(
                "entry key does not match its identity components"
            )
        return self


class _StoredStats(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        populate_by_name=True,
    )

    schema_name: Literal["whetstone.execution.prompt_cache_stats/v1"] = Field(
        default=PROMPT_CACHE_STATS_SCHEMA,
        alias="schema",
    )
    hits: StrictInt = 0
    misses: StrictInt = 0
    stores: StrictInt = 0
    inflight_publication_ids: tuple[StrictStr, ...] = ()

    @model_validator(mode="after")
    def _validate_counts(self) -> Self:
        if self.hits < 0 or self.misses < 0 or self.stores < 0:
            raise ValueError("prompt-cache counters must be non-negative")
        if len(set(self.inflight_publication_ids)) != len(
            self.inflight_publication_ids
        ):
            raise ValueError("inflight publication IDs must be unique")
        for publication_id in self.inflight_publication_ids:
            _validate_publication_id(publication_id)
        return self


class _AccountingJournal(BaseModel):
    """Recovery record bridging entry publication and stats persistence."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        populate_by_name=True,
    )

    schema_name: Literal["whetstone.execution.prompt_cache_accounting/v1"] = (
        Field(default=PROMPT_CACHE_ACCOUNTING_SCHEMA, alias="schema")
    )
    key: StrictStr
    publication_id: StrictStr
    misses: Literal[0, 1]
    stores: Literal[1] = 1

    @model_validator(mode="after")
    def _validate_identity(self) -> Self:
        _validate_cache_key(self.key)
        _validate_publication_id(self.publication_id)
        return self


def _default_log(message: str) -> None:
    sys.stderr.write(message)


@dataclass(slots=True)
class PromptResultCache:
    """Content-addressed prompt results safe for threads and peer processes."""

    root: Path
    log: Callable[[str], None] = _default_log

    @property
    def store_dir(self) -> Path:
        return self.root / "prompt_cache"

    @property
    def _stats_path(self) -> Path:
        return self.store_dir / "stats.json"

    @property
    def _stats_lock_path(self) -> Path:
        return self.store_dir / ".stats.lock"

    def _path_for(self, key: str) -> Path:
        validated, shard = self._private_shard_for(key)
        return shard / f"{validated}.json"

    def _lock_path_for(self, key: str) -> Path:
        validated, shard = self._private_shard_for(key)
        return shard / f".{validated}.lock"

    def _pending_accounting_path_for(self, key: str) -> Path:
        validated, shard = self._private_shard_for(key)
        return shard / f".{validated}.accounting.pending.json"

    def _applied_accounting_path_for(self, key: str) -> Path:
        validated, shard = self._private_shard_for(key)
        return shard / f".{validated}.accounting.applied.json"

    def _private_shard_for(
        self,
        key: str,
    ) -> tuple[PromptCacheKey, Path]:
        # Key validation must precede every key-derived filesystem path.
        validated = _validate_cache_key(key)
        ensure_private_directory(self.store_dir)
        shard = self.store_dir / validated[:2]
        ensure_private_directory(shard)
        return validated, shard

    def get_result(
        self,
        key: str,
    ) -> tuple[ProviderCallResult, CacheProvenance] | None:
        validated = _validate_cache_key(key)
        with FileLock(self._lock_path_for(validated)):
            self._reconcile_key_accounting_locked(validated)
            entry = self._read_entry(validated)
        if entry is None:
            return None
        return entry.result, entry.provenance

    def put(
        self,
        key: str,
        *,
        request_identity: dict[str, Any],
        execution_policy_hash: str,
        repeat_index: int,
        drive_ordinal: int,
        result: ProviderCallResult,
        phase: str,
        unit: str,
        logical_call_id: str,
    ) -> CacheProvenance:
        """Store once per key and preserve the winning writer's provenance."""
        validated = _validate_cache_key(key)
        expected_key = _prompt_cache_key_from_components(
            request_identity=request_identity,
            execution_policy_hash=execution_policy_hash,
            repeat_index=repeat_index,
            drive_ordinal=drive_ordinal,
        )
        if validated != expected_key:
            raise ValueError(
                "cache key does not match its identity components"
            )
        proposed = CacheProvenance(
            key=validated,
            source_phase=phase,
            source_unit=unit,
            source_logical_call_id=logical_call_id,
            stored_at=datetime.now(UTC).isoformat(),
        )
        with FileLock(self._lock_path_for(validated)):
            self._reconcile_key_accounting_locked(validated)
            try:
                existing = self._read_entry(validated)
            except PromptCacheError as exc:
                self._log_corrupt_entry(
                    key=validated,
                    logical_call_id=logical_call_id,
                    error=exc,
                )
                self._quarantine_corrupt_entry_locked(validated)
                existing = None
            if existing is not None:
                return existing.provenance
            self._publish_entry(
                key=validated,
                request_identity=request_identity,
                execution_policy_hash=execution_policy_hash,
                repeat_index=repeat_index,
                drive_ordinal=drive_ordinal,
                provenance=proposed,
                result=result,
                misses=0,
            )
        return proposed

    def note_hit(self) -> None:
        self._update_counters(hits=1)

    def note_miss(self) -> None:
        self._update_counters(misses=1)

    def counters(self) -> dict[str, int]:
        self._reconcile_all_accounting()
        with FileLock(self._stats_lock_path, shared=True):
            stats = self._read_stats()
        return {
            "hits": stats.hits,
            "misses": stats.misses,
            "stores": stats.stores,
        }

    def _update_counters(
        self,
        *,
        hits: int = 0,
        misses: int = 0,
    ) -> None:
        with FileLock(self._stats_lock_path):
            current = self._read_stats()
            updated = current.model_copy(
                update={
                    "hits": current.hits + hits,
                    "misses": current.misses + misses,
                }
            )
            self._write_stats(updated)

    def _read_stats(self) -> _StoredStats:
        path = self._stats_path
        try:
            raw = self._read_private_bytes(path)
        except FileNotFoundError:
            return _StoredStats()
        except OSError as exc:  # pragma: no cover - filesystem failure
            raise PromptCacheError(
                f"prompt-cache stats unreadable at {path}: {exc}"
            ) from exc
        try:
            decode_strict_json_bytes(
                raw,
                max_bytes=len(raw),
                max_depth=len(raw),
            )
            return _StoredStats.model_validate_json(raw)
        except (StrictJsonDecodeError, ValidationError) as exc:
            raise PromptCacheError(
                f"prompt-cache stats invalid at {path}: {exc}"
            ) from exc

    def _write_stats(self, stats: _StoredStats) -> None:
        self._atomic_write(
            self._stats_path,
            stats.model_dump(mode="json", by_alias=True),
        )

    def _publish_entry(
        self,
        *,
        key: PromptCacheKey,
        request_identity: dict[str, Any],
        execution_policy_hash: str,
        repeat_index: int,
        drive_ordinal: int,
        provenance: CacheProvenance,
        result: ProviderCallResult,
        misses: Literal[0, 1],
    ) -> None:
        """Publish one entry with restart-recoverable accounting.

        The files are individually atomic, not a cross-file transaction.
        ``misses=1`` means an execute_call-owned provider result was durably
        published; direct ``put`` publication records a store but no miss.
        """
        publication_id = uuid4().hex
        journal = _AccountingJournal(
            key=key,
            publication_id=publication_id,
            misses=misses,
        )
        self._atomic_write(
            self._pending_accounting_path_for(key),
            journal.model_dump(mode="json", by_alias=True),
        )
        self._store_entry(
            key=key,
            request_identity=request_identity,
            execution_policy_hash=execution_policy_hash,
            repeat_index=repeat_index,
            drive_ordinal=drive_ordinal,
            provenance=provenance,
            result=result,
            publication_id=publication_id,
        )
        self._finalize_accounting_locked(key, journal)

    def _store_entry(
        self,
        *,
        key: PromptCacheKey,
        request_identity: dict[str, Any],
        execution_policy_hash: str,
        repeat_index: int,
        drive_ordinal: int,
        provenance: CacheProvenance,
        result: ProviderCallResult,
        publication_id: str,
    ) -> None:
        entry = _StoredEntry(
            key=key,
            request_identity=request_identity,
            execution_policy_hash=execution_policy_hash,
            repeat_index=repeat_index,
            drive_ordinal=drive_ordinal,
            result_policy_hash=result.execution_policy_hash,
            publication_id=publication_id,
            provenance=provenance,
            result=result,
        )
        self._atomic_write(
            self._path_for(key),
            entry.model_dump(mode="json", by_alias=True),
        )

    def _reconcile_all_accounting(self) -> None:
        ensure_private_directory(self.store_dir)
        keys: set[PromptCacheKey] = set()
        for shard in self.store_dir.iterdir():
            shard_status = shard.stat(follow_symlinks=False)
            if stat.S_ISLNK(shard_status.st_mode):
                raise PromptCacheError(
                    f"prompt-cache shard must not be a symlink: {shard}"
                )
            if not stat.S_ISDIR(shard_status.st_mode):
                continue
            ensure_private_directory(shard)
            for path in shard.iterdir():
                key = self._accounting_key_from_path(path)
                if key is not None:
                    if shard.name != key[:2]:
                        raise PromptCacheError(
                            "prompt-cache accounting journal is in the "
                            f"wrong shard: {path}"
                        )
                    keys.add(key)
        for key in sorted(keys):
            with FileLock(self._lock_path_for(key)):
                self._reconcile_key_accounting_locked(key)

    def _accounting_key_from_path(
        self,
        path: Path,
    ) -> PromptCacheKey | None:
        suffixes = (
            ".accounting.pending.json",
            ".accounting.applied.json",
        )
        for suffix in suffixes:
            if path.name.startswith(".") and path.name.endswith(suffix):
                raw_key = path.name[1 : -len(suffix)]
                try:
                    return _validate_cache_key(raw_key)
                except (TypeError, ValueError) as exc:
                    raise PromptCacheError(
                        "prompt-cache accounting journal has invalid name: "
                        f"{path}"
                    ) from exc
        return None

    def _reconcile_key_accounting_locked(
        self,
        key: PromptCacheKey,
    ) -> None:
        pending_path = self._pending_accounting_path_for(key)
        applied_path = self._applied_accounting_path_for(key)
        pending_exists = self._path_exists(pending_path)
        applied_exists = self._path_exists(applied_path)
        if pending_exists and applied_exists:
            raise PromptCacheError(
                "prompt-cache accounting has both pending and applied "
                f"journals for key {key}"
            )
        if applied_exists:
            journal = self._read_accounting_journal(applied_path, key)
            self._validate_accounted_entry(key, journal)
            self._cleanup_applied_accounting_locked(journal)
            return
        if not pending_exists:
            return

        journal = self._read_accounting_journal(pending_path, key)
        entry = self._read_entry(key)
        if entry is None:
            pending_path.unlink()
            fsync_parent_directory(pending_path)
            return
        if entry.publication_id != journal.publication_id:
            raise PromptCacheError(
                "prompt-cache pending accounting publication does not "
                f"match the entry for key {key}"
            )
        self._finalize_accounting_locked(key, journal)

    def _validate_accounted_entry(
        self,
        key: PromptCacheKey,
        journal: _AccountingJournal,
    ) -> None:
        entry = self._read_entry(key)
        if entry is None:
            raise PromptCacheError(
                "prompt-cache applied accounting has no published entry "
                f"for key {key}"
            )
        if entry.publication_id != journal.publication_id:
            raise PromptCacheError(
                "prompt-cache applied accounting publication does not "
                f"match the entry for key {key}"
            )

    def _read_accounting_journal(
        self,
        path: Path,
        key: PromptCacheKey,
    ) -> _AccountingJournal:
        try:
            raw = self._read_private_bytes(path)
        except OSError as exc:  # pragma: no cover - filesystem failure
            raise PromptCacheError(
                f"prompt-cache accounting journal unreadable at {path}: {exc}"
            ) from exc
        try:
            decode_strict_json_bytes(
                raw,
                max_bytes=len(raw),
                max_depth=len(raw),
            )
            journal = _AccountingJournal.model_validate_json(raw)
        except (StrictJsonDecodeError, ValidationError) as exc:
            raise PromptCacheError(
                f"prompt-cache accounting journal invalid at {path}: {exc}"
            ) from exc
        if journal.key != key:
            raise PromptCacheError(
                f"prompt-cache accounting journal key mismatch at {path}"
            )
        return journal

    def _finalize_accounting_locked(
        self,
        key: PromptCacheKey,
        journal: _AccountingJournal,
    ) -> None:
        pending_path = self._pending_accounting_path_for(key)
        applied_path = self._applied_accounting_path_for(key)
        with FileLock(self._stats_lock_path):
            current = self._read_stats()
            inflight = set(current.inflight_publication_ids)
            if journal.publication_id not in inflight:
                inflight.add(journal.publication_id)
                self._write_stats(
                    current.model_copy(
                        update={
                            "misses": current.misses + journal.misses,
                            "stores": current.stores + journal.stores,
                            "inflight_publication_ids": tuple(
                                sorted(inflight)
                            ),
                        }
                    )
                )
        os.replace(pending_path, applied_path)
        fsync_parent_directory(applied_path)
        self._cleanup_applied_accounting_locked(journal)

    def _cleanup_applied_accounting_locked(
        self,
        journal: _AccountingJournal,
    ) -> None:
        applied_path = self._applied_accounting_path_for(journal.key)
        with FileLock(self._stats_lock_path):
            current = self._read_stats()
            if journal.publication_id in current.inflight_publication_ids:
                self._write_stats(
                    current.model_copy(
                        update={
                            "inflight_publication_ids": tuple(
                                publication_id
                                for publication_id in (
                                    current.inflight_publication_ids
                                )
                                if publication_id != journal.publication_id
                            )
                        }
                    )
                )
        applied_path.unlink()
        fsync_parent_directory(applied_path)

    def _log_corrupt_entry(
        self,
        *,
        key: str,
        logical_call_id: str,
        error: PromptCacheError,
    ) -> None:
        self.log(
            "PROMPT-CACHE CORRUPT ENTRY -> LOUD MISS: "
            f"key={key} logical_call_id={logical_call_id!r}: {error}. "
            "Quarantining the invalid entry and recomputing.\n"
        )

    def _read_entry(self, key: str) -> _StoredEntry | None:
        validated = _validate_cache_key(key)
        path = self._path_for(validated)
        try:
            raw = self._read_private_bytes(path)
        except FileNotFoundError:
            return None
        except OSError as exc:  # pragma: no cover - filesystem failure
            raise PromptCacheError(
                f"prompt-cache entry unreadable at {path}: {exc}"
            ) from exc
        try:
            decode_strict_json_bytes(
                raw,
                max_bytes=len(raw),
                max_depth=len(raw),
            )
            entry = _StoredEntry.model_validate_json(raw)
        except (StrictJsonDecodeError, ValidationError) as exc:
            raise PromptCacheError(
                f"prompt-cache entry invalid at {path}: {exc}"
            ) from exc
        if entry.key != validated:
            raise PromptCacheError(
                f"prompt-cache entry key mismatch at {path}"
            )
        return entry

    @staticmethod
    def _read_private_bytes(path: Path) -> bytes:
        fd = open_private_regular_file(path, os.O_RDONLY)
        try:
            with os.fdopen(fd, "rb") as handle:
                fd = -1
                return handle.read()
        finally:
            if fd >= 0:
                os.close(fd)

    @staticmethod
    def _path_exists(path: Path) -> bool:
        try:
            path.stat(follow_symlinks=False)
        except FileNotFoundError:
            return False
        return True

    def _quarantine_corrupt_entry_locked(
        self,
        key: PromptCacheKey,
    ) -> None:
        path = self._path_for(key)
        if not self._path_exists(path):
            return
        quarantine = path.with_name(f".{path.name}.corrupt.{uuid4().hex}")
        os.replace(path, quarantine)
        fsync_parent_directory(quarantine)

    def _atomic_write(self, path: Path, body: dict[str, object]) -> None:
        ensure_private_directory(path.parent)
        temporary = path.parent / (
            f".{path.name}.{os.getpid()}.{threading.get_ident()}."
            f"{uuid4().hex}.tmp"
        )
        try:
            fd = open_private_regular_file(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            )
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(
                    body,
                    handle,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                handle.flush()
                fsync_file(handle.fileno())
            os.replace(temporary, path)
            fsync_parent_directory(path)
        finally:
            temporary.unlink(missing_ok=True)


def execute_call(
    *,
    request: ProviderCallRequest,
    policy: ProviderExecutionPolicy,
    transport: TransportCall,
    logical_call_id: str,
    repeat_index: int,
    drive_ordinal: int,
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

    request_identity = request.identity_payload()
    execution_policy_hash = policy.identity_hash
    key = prompt_cache_key(
        request,
        policy,
        repeat_index,
        drive_ordinal,
    )
    with FileLock(cache._lock_path_for(key)):
        cache._reconcile_key_accounting_locked(key)
        try:
            entry = cache._read_entry(key)
        except PromptCacheError as exc:
            cache._log_corrupt_entry(
                key=key,
                logical_call_id=logical_call_id,
                error=exc,
            )
            cache._quarantine_corrupt_entry_locked(key)
            entry = None
        if entry is not None:
            cache.note_hit()
            return CallExecution(
                result=entry.result,
                cache_hit=True,
                provenance=entry.provenance,
            )

        result = run_provider_call(
            request=request,
            policy=policy,
            transport=transport,
            logical_call_id=logical_call_id,
            clock=clock,
            sleep=sleep,
        )
        provenance = CacheProvenance(
            key=key,
            source_phase=phase,
            source_unit=unit,
            source_logical_call_id=logical_call_id,
            stored_at=datetime.now(UTC).isoformat(),
        )
        cache._publish_entry(
            key=key,
            request_identity=request_identity,
            execution_policy_hash=execution_policy_hash,
            repeat_index=repeat_index,
            drive_ordinal=drive_ordinal,
            provenance=provenance,
            result=result,
            misses=1,
        )
        return CallExecution(result=result)
