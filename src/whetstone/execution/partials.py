from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import secrets
import stat
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, Self, cast

from dr_serialize import StrictJsonDecodeError, decode_strict_json_bytes
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from whetstone.execution._file_lock import (
    FileLock,
    PrivateDirectory,
    fsync_file,
)

__all__ = [
    "PARTIAL_FRAME_SCHEMA",
    "PARTIAL_SCHEMA",
    "PARTIAL_STORAGE_THREAT_MODEL",
    "PartialCallRecord",
    "PartialLog",
    "partial_key",
]

PARTIAL_SCHEMA = "whetstone.execution.partial_call/v3"
PARTIAL_FRAME_SCHEMA = "whetstone.execution.partial_frame/v3"
PARTIAL_STORAGE_THREAT_MODEL = (
    "PartialLog provides crash durability, atomic publication, structural "
    "and content-integrity checks, and lock-based serialization for "
    "cooperating writers. Checksums detect accidental, malformed, or torn "
    "corruption; they do not provide authenticity or tamper resistance "
    "against a same-UID actor that can rewrite managed files, whether "
    "concurrently or between operations."
)

_CANONICAL_REQUEST_HASH = re.compile(r"[0-9a-f]{64}")
_ENTRY_NAME = re.compile(r"[0-9a-f]{64}\.json")
_TEMPORARY_NAME = re.compile(r"\.[0-9a-f]{64}\.[0-9a-f]{32}\.tmp")

_PERSISTED_FIELDS = frozenset(
    {
        "schema",
        "phase",
        "task_id",
        "unit",
        "sample_index",
        "request_hash",
        "redrive_pending",
        "split_role",
        "score",
        "failed",
        "failure_code",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "reasoning_tokens",
        "latency_s",
        "output_text",
        "raw_response",
        "observation_payload",
        "finish_reason",
        "provider_error",
        "at",
        "cache_hit",
        "cache_source_phase",
        "cache_source_unit",
        "cache_source_call_id",
        "cache_source_at",
    }
)
_FRAME_FIELDS = frozenset({"schema", "checksum", "record"})


class PartialCallRecord(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        populate_by_name=True,
    )

    phase: StrictStr
    task_id: StrictStr
    unit: StrictStr
    sample_index: StrictInt
    request_hash: StrictStr
    redrive_pending: StrictBool
    score: float | None = None
    failed: StrictBool = False
    failure_code: StrictStr = ""
    prompt_tokens: StrictInt | None = None
    completion_tokens: StrictInt | None = None
    total_tokens: StrictInt | None = None
    reasoning_tokens: StrictInt | None = None
    latency_s: float | None = None
    output_text: StrictStr | None = None
    raw_response: StrictStr = ""
    observation_payload: JsonValue | None = None
    finish_reason: StrictStr | None = None
    provider_error: dict[str, object] | None = None
    split_role: StrictStr | None = None
    at: StrictStr | None = None
    schema_name: Literal["whetstone.execution.partial_call/v3"] = Field(
        default=PARTIAL_SCHEMA,
        alias="schema",
    )
    cache_hit: StrictBool = False
    cache_source_phase: StrictStr | None = None
    cache_source_unit: StrictStr | None = None
    cache_source_call_id: StrictStr | None = None
    cache_source_at: StrictStr | None = None

    @field_validator("at")
    @classmethod
    def _validate_at(cls, value: str | None) -> str | None:
        if value is not None:
            _validate_timestamp(value)
        return value

    @field_validator("request_hash")
    @classmethod
    def _validate_request_hash(cls, value: str) -> str:
        if _CANONICAL_REQUEST_HASH.fullmatch(value) is None:
            raise ValueError(
                "request_hash must be a canonical 64-character "
                "lowercase hexadecimal digest"
            )
        return value

    @field_validator("observation_payload", mode="before")
    @classmethod
    def _validate_observation_payload(cls, value: object) -> object:
        _validate_finite_json(value, path="observation_payload")
        return value

    @model_validator(mode="after")
    def _validate_current_record(self) -> Self:
        _reject_non_finite(self.score, path="score")
        _reject_non_finite(self.latency_s, path="latency_s")
        _reject_non_finite(self.provider_error, path="provider_error")
        if not self.phase or not self.task_id or not self.unit:
            raise ValueError("partial identity fields must be non-empty")
        if self.sample_index < 0:
            raise ValueError("sample_index must be non-negative")
        sources = (
            self.cache_source_phase,
            self.cache_source_unit,
            self.cache_source_call_id,
            self.cache_source_at,
        )
        if self.cache_hit and (
            any(source is None for source in sources)
            or self.latency_s is not None
        ):
            raise ValueError(
                "a cache hit requires complete provenance and null latency"
            )
        if not self.cache_hit and any(
            source is not None for source in sources
        ):
            raise ValueError("cache provenance is only valid for a cache hit")
        return self

    def key(self) -> tuple[str, str, str, int, str]:
        return partial_key(
            self.phase,
            self.task_id,
            self.unit,
            self.sample_index,
            self.request_hash,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema_name,
            "phase": self.phase,
            "task_id": self.task_id,
            "unit": self.unit,
            "sample_index": self.sample_index,
            "request_hash": self.request_hash,
            "redrive_pending": self.redrive_pending,
            "split_role": self.split_role,
            "score": self.score,
            "failed": self.failed,
            "failure_code": self.failure_code,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "latency_s": self.latency_s,
            "output_text": self.output_text,
            "raw_response": self.raw_response,
            "observation_payload": self.observation_payload,
            "finish_reason": self.finish_reason,
            "provider_error": self.provider_error,
            "at": self.at,
            "cache_hit": self.cache_hit,
            "cache_source_phase": self.cache_source_phase,
            "cache_source_unit": self.cache_source_unit,
            "cache_source_call_id": self.cache_source_call_id,
            "cache_source_at": self.cache_source_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Self:
        fields = frozenset(data)
        if fields != _PERSISTED_FIELDS:
            missing = sorted(_PERSISTED_FIELDS - fields)
            unexpected = sorted(fields - _PERSISTED_FIELDS)
            raise ValueError(
                "partial row does not match the current schema: "
                f"missing={missing}, unexpected={unexpected}"
            )
        if data["schema"] != PARTIAL_SCHEMA:
            raise ValueError(
                "partial row schema must be "
                f"{PARTIAL_SCHEMA!r}, got {data['schema']!r}"
            )
        return cls.model_validate(data)


def partial_key(
    phase: str,
    task_id: str,
    unit: str,
    sample_index: int,
    request_hash: str,
) -> tuple[str, str, str, int, str]:
    return (phase, task_id, unit, sample_index, request_hash)


def _validate_timestamp(value: str) -> None:
    if not value:
        raise ValueError("partial row at must be a non-empty timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            "partial row at must be an ISO 8601 timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("partial row at must include a UTC offset")
    if parsed.utcoffset() != timedelta(0):
        raise ValueError("partial row at must use the UTC offset +00:00")
    if value != parsed.astimezone(UTC).isoformat():
        raise ValueError(
            "partial row at must use canonical UTC ISO 8601 formatting"
        )


def _reject_non_finite(value: object, *, path: str) -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite numbers")
        return
    if isinstance(value, dict):
        for key, nested in value.items():
            _reject_non_finite(nested, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_non_finite(nested, path=f"{path}[{index}]")


def _validate_finite_json(value: object, *, path: str) -> None:
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite numbers")
        return
    if type(value) is list:
        for index, nested in enumerate(value):
            _validate_finite_json(nested, path=f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, nested in value.items():
            if type(key) is not str:
                raise ValueError(f"{path} object keys must be strings")
            _validate_finite_json(nested, path=f"{path}.{key}")
        return
    raise ValueError(f"{path} must contain only JSON values")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _encode_frame(record: PartialCallRecord) -> bytes:
    record_body = record.as_dict()
    checksum = hashlib.sha256(_canonical_json_bytes(record_body)).hexdigest()

    frame = {
        "schema": PARTIAL_FRAME_SCHEMA,
        "checksum": checksum,
        "record": record_body,
    }
    return _canonical_json_bytes(frame) + b"\n"


def _decode_frame(
    raw: bytes,
    *,
    path: Path,
    line_number: int,
) -> PartialCallRecord:
    location = f"{path}:{line_number}"
    try:
        decoded = decode_strict_json_bytes(
            raw,
            max_bytes=len(raw),
            max_depth=len(raw),
        )
    except StrictJsonDecodeError as exc:
        raise ValueError(f"invalid partial JSON at {location}") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"partial frame must be an object at {location}")
    decoded = cast(dict[str, object], decoded)
    fields = frozenset(decoded)
    if fields != _FRAME_FIELDS:
        missing = sorted(_FRAME_FIELDS - fields)
        unexpected = sorted(fields - _FRAME_FIELDS)
        raise ValueError(
            "partial frame does not match the current schema at "
            f"{location}: missing={missing}, unexpected={unexpected}"
        )
    if decoded["schema"] != PARTIAL_FRAME_SCHEMA:
        raise ValueError(
            "partial frame schema must be "
            f"{PARTIAL_FRAME_SCHEMA!r} at {location}, "
            f"got {decoded['schema']!r}"
        )
    checksum = decoded["checksum"]
    record_body = decoded["record"]
    if not isinstance(checksum, str) or not isinstance(record_body, dict):
        raise ValueError(
            f"partial frame checksum and record are invalid at {location}"
        )
    record_body = cast(dict[str, object], record_body)
    expected = hashlib.sha256(_canonical_json_bytes(record_body)).hexdigest()
    if not hmac.compare_digest(checksum, expected):
        raise ValueError(f"partial frame checksum mismatch at {location}")
    return PartialCallRecord.from_dict(record_body)


def _same_file_snapshot(
    left: os.stat_result,
    right: os.stat_result,
) -> bool:
    return (
        left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and left.st_mode == right.st_mode
        and left.st_uid == right.st_uid
        and left.st_gid == right.st_gid
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
        and left.st_ctime_ns == right.st_ctime_ns
    )


def _entry_name(record: PartialCallRecord) -> str:
    phase, task_id, unit, sample_index, request_hash = record.key()
    identity = {
        "phase": phase,
        "task_id": task_id,
        "unit": unit,
        "sample_index": sample_index,
        "request_hash": request_hash,
    }
    return (
        f"{hashlib.sha256(_canonical_json_bytes(identity)).hexdigest()}.json"
    )


def _read_all(fd: int) -> bytes:
    chunks: list[bytes] = []
    offset = 0
    while True:
        chunk = os.pread(fd, 1024 * 1024, offset)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
        offset += len(chunk)


def _read_entry(
    fd: int,
    *,
    directory: PrivateDirectory,
    name: str,
) -> tuple[PartialCallRecord, bytes, os.stat_result]:
    path = directory.path / name
    before = os.fstat(fd)
    raw = _read_all(fd)
    after = os.fstat(fd)
    if not _same_file_snapshot(before, after):
        raise OSError(f"partial record changed while reading: {path}")
    visible = directory.stat(name)
    if not _same_file_snapshot(after, visible):
        raise OSError(f"partial record path changed while reading: {path}")
    if not raw.endswith(b"\n"):
        raise ValueError(f"partial record is not complete: {path}")
    record = _decode_frame(raw, path=path, line_number=1)
    if path.name != _entry_name(record):
        raise ValueError(
            f"partial record filename does not match its key: {path}"
        )
    return record, raw, after


@dataclass(frozen=True, slots=True)
class _PriorEntry:
    record: PartialCallRecord
    body: bytes
    snapshot: os.stat_result


def _validate_existing_key(
    directory: PrivateDirectory,
    name: str,
    expected: PartialCallRecord,
) -> _PriorEntry | None:
    try:
        fd = directory.open_regular(name, os.O_RDONLY)
    except FileNotFoundError:
        return None
    try:
        record, body, snapshot = _read_entry(
            fd,
            directory=directory,
            name=name,
        )
    finally:
        os.close(fd)
    if record.key() != expected.key():
        raise ValueError(
            "partial record key does not match target: "
            f"{directory.path / name}"
        )
    return _PriorEntry(record=record, body=body, snapshot=snapshot)


def _create_temporary(
    directory: PrivateDirectory,
    target_name: str,
) -> tuple[str, int]:
    target_stem = Path(target_name).stem
    for _ in range(4):
        token = secrets.token_hex(16)
        temporary = f".{target_stem}.{token}.tmp"
        try:
            fd = directory.open_regular(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            )
        except FileExistsError:
            continue
        return temporary, fd
    raise FileExistsError("could not allocate a unique partial temporary")


def _validate_cleanup_candidate(
    directory: PrivateDirectory,
    name: str,
    *,
    require_private_mode: bool,
) -> os.stat_result:
    path = directory.path / name
    before = directory.stat(name)
    if (
        stat.S_IFMT(before.st_mode) != stat.S_IFREG
        or before.st_uid != os.geteuid()
        or before.st_nlink != 1
        or (require_private_mode and stat.S_IMODE(before.st_mode) != 0o600)
    ):
        raise OSError(f"unsafe managed partial entry: {path}")
    return before


def _verified_unlink(
    directory: PrivateDirectory,
    name: str,
    *,
    require_private_mode: bool,
) -> None:
    path = directory.path / name
    before = _validate_cleanup_candidate(
        directory,
        name,
        require_private_mode=require_private_mode,
    )
    fd = directory.open_regular(name, os.O_RDONLY)
    try:
        opened = os.fstat(fd)
    finally:
        os.close(fd)
    if opened.st_dev != before.st_dev or opened.st_ino != before.st_ino:
        raise OSError(f"managed partial entry changed before cleanup: {path}")
    current = directory.stat(name)
    if not _same_file_snapshot(opened, current):
        raise OSError(f"managed partial entry changed before cleanup: {path}")
    directory.unlink(name)


def _validate_published_entry(
    directory: PrivateDirectory,
    name: str,
    *,
    record: PartialCallRecord,
    body: bytes,
    published_status: os.stat_result,
) -> None:
    path = directory.path / name
    fd = directory.open_regular(name, os.O_RDONLY)
    try:
        visible_record, visible_body, visible_status = _read_entry(
            fd,
            directory=directory,
            name=name,
        )
    finally:
        os.close(fd)
    if (
        visible_record.key() != record.key()
        or not hmac.compare_digest(visible_body, body)
        or visible_status.st_dev != published_status.st_dev
        or visible_status.st_ino != published_status.st_ino
    ):
        raise OSError(f"published partial record changed: {path}")
    path_status = directory.stat(name)
    if not _same_file_snapshot(visible_status, path_status):
        raise OSError(f"published partial record path changed: {path}")
    try:
        diagnostic_status = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise OSError(
            f"published partial record is not visible by path: {path}"
        ) from exc
    if not _same_file_snapshot(visible_status, diagnostic_status):
        raise OSError(f"published partial record path changed: {path}")


def _entry_exists(directory: PrivateDirectory, name: str) -> bool:
    try:
        directory.stat(name)
    except FileNotFoundError:
        return False
    return True


def _restore_prior(
    directory: PrivateDirectory,
    name: str,
    prior: _PriorEntry | None,
) -> None:
    if prior is None:
        _verified_unlink(
            directory,
            name,
            require_private_mode=False,
        )
        directory.fsync()
        return
    temporary, fd = _create_temporary(directory, name)
    restored_status: os.stat_result | None = None
    try:
        try:
            _write_all(fd, prior.body)
            fsync_file(fd)
            restored_status = os.fstat(fd)
        finally:
            os.close(fd)
        directory.replace(temporary, name)
        directory.fsync()
        assert restored_status is not None
        _validate_published_entry(
            directory,
            name,
            record=prior.record,
            body=prior.body,
            published_status=restored_status,
        )
    finally:
        if _entry_exists(directory, temporary):
            _verified_unlink(
                directory,
                temporary,
                require_private_mode=True,
            )


def _publish_entry(
    directory: PrivateDirectory,
    name: str,
    record: PartialCallRecord,
    body: bytes,
    prior: _PriorEntry | None,
) -> None:
    path = directory.path / name
    temporary, fd = _create_temporary(directory, name)
    published_status: os.stat_result | None = None
    replaced = False
    try:
        try:
            _write_all(fd, body)
            fsync_file(fd)
            published_status = os.fstat(fd)
        finally:
            os.close(fd)
        directory.replace(temporary, name)
        replaced = True
        assert published_status is not None
        try:
            directory.fsync()
            _validate_published_entry(
                directory,
                name,
                record=record,
                body=body,
                published_status=published_status,
            )
        except BaseException:
            try:
                _restore_prior(directory, name, prior)
            except BaseException as rollback_error:
                raise OSError(
                    f"partial publication rollback failed: {path}"
                ) from rollback_error
            raise

    finally:
        if not replaced and _entry_exists(directory, temporary):
            _verified_unlink(
                directory,
                temporary,
                require_private_mode=True,
            )


def _write_all(fd: int, body: bytes) -> None:
    remaining = memoryview(body)
    while remaining:
        try:
            written = os.write(fd, remaining)
        except InterruptedError:
            continue
        if written == 0:
            raise OSError("partial frame write made no progress")
        remaining = remaining[written:]


def _open_record_directory(
    parent: PrivateDirectory,
    name: str,
    *,
    create: bool,
) -> PrivateDirectory:
    try:
        return parent.open_child(name, create=create)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise ValueError(
            "partial storage must be a current per-key record directory"
        ) from exc


@dataclass(slots=True)
class PartialLog:
    path: Path

    @property
    def _lock_path(self) -> Path:
        return self.path.with_name(f".{self.path.name}.lock")

    def _entry_path(self, record: PartialCallRecord) -> Path:
        return self.path / _entry_name(record)

    def append(self, record: PartialCallRecord) -> None:
        validated = PartialCallRecord.from_dict(record.as_dict())
        stamped = validated
        if validated.at is None:
            stamped = validated.model_copy(
                update={"at": datetime.now(UTC).isoformat()}
            )
        assert stamped.at is not None
        _validate_timestamp(stamped.at)
        body = _encode_frame(stamped)
        with FileLock(self._lock_path) as lock:
            with _open_record_directory(
                lock.directory,
                self.path.name,
                create=True,
            ) as directory:
                entry_name = _entry_name(stamped)
                prior = _validate_existing_key(
                    directory,
                    entry_name,
                    stamped,
                )
                _publish_entry(
                    directory,
                    entry_name,
                    stamped,
                    body,
                    prior,
                )

    def load(self) -> list[PartialCallRecord]:
        with FileLock(self._lock_path, shared=True) as lock:
            try:
                directory = _open_record_directory(
                    lock.directory,
                    self.path.name,
                    create=False,
                )
            except FileNotFoundError:
                return []
            with directory:
                records: list[PartialCallRecord] = []
                for name in sorted(directory.list_names()):
                    if _TEMPORARY_NAME.fullmatch(name) is not None:
                        continue
                    if _ENTRY_NAME.fullmatch(name) is None:
                        raise ValueError(
                            "unexpected partial storage entry: "
                            f"{self.path / name}"
                        )
                    fd = directory.open_regular(name, os.O_RDONLY)
                    try:
                        record, _, _ = _read_entry(
                            fd,
                            directory=directory,
                            name=name,
                        )
                    finally:
                        os.close(fd)
                    records.append(record)
            records.sort(
                key=lambda record: (record.at or "", _entry_name(record))
            )
            return records

    def recorded_keys(self) -> set[tuple[str, str, str, int, str]]:
        return {record.key() for record in self.load()}

    def delete(self) -> None:
        with FileLock(self._lock_path) as lock:
            try:
                directory = _open_record_directory(
                    lock.directory,
                    self.path.name,
                    create=False,
                )
            except FileNotFoundError:
                return
            with directory:
                names = sorted(directory.list_names())
                for name in names:
                    if (
                        _ENTRY_NAME.fullmatch(name) is None
                        and _TEMPORARY_NAME.fullmatch(name) is None
                    ):
                        raise ValueError(
                            "unexpected partial storage entry: "
                            f"{self.path / name}"
                        )
                for name in names:
                    _validate_cleanup_candidate(
                        directory,
                        name,
                        require_private_mode=(
                            _TEMPORARY_NAME.fullmatch(name) is not None
                        ),
                    )
                for name in names:
                    _verified_unlink(
                        directory,
                        name,
                        require_private_mode=(
                            _TEMPORARY_NAME.fullmatch(name) is not None
                        ),
                    )
                directory.fsync()
