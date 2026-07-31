"""Current-schema incremental persistence for completed provider calls."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from whetstone.execution._file_lock import (
    FileLock,
    fsync_file,
    fsync_parent_directory,
    open_private_regular_file,
)

__all__ = [
    "PARTIAL_FRAME_SCHEMA",
    "PARTIAL_SCHEMA",
    "PartialCallRecord",
    "PartialLog",
    "partial_key",
]

PARTIAL_SCHEMA = "whetstone.execution.partial_call/v2"
PARTIAL_FRAME_SCHEMA = "whetstone.execution.partial_frame/v2"

_PERSISTED_FIELDS = frozenset(
    {
        "schema",
        "phase",
        "instance_id",
        "unit",
        "candidate_id",
        "repeat_id",
        "repeat",
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
    """One current-schema completed-call persistence record."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        populate_by_name=True,
    )

    phase: StrictStr
    instance_id: StrictStr
    unit: StrictStr
    repeat_id: StrictInt
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
    finish_reason: StrictStr | None = None
    provider_error: dict[str, object] | None = None
    split_role: StrictStr | None = None
    at: StrictStr | None = None
    schema_name: Literal["whetstone.execution.partial_call/v2"] = Field(
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

    @model_validator(mode="after")
    def _validate_current_record(self) -> Self:
        _reject_non_finite(self.score, path="score")
        _reject_non_finite(self.latency_s, path="latency_s")
        _reject_non_finite(self.provider_error, path="provider_error")
        if not self.phase or not self.instance_id or not self.unit:
            raise ValueError("partial identity fields must be non-empty")
        if self.repeat_id < 0:
            raise ValueError("repeat_id must be non-negative")
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

    def key(self) -> tuple[str, str, str, int]:
        return partial_key(
            self.phase,
            self.instance_id,
            self.unit,
            self.repeat_id,
        )

    def as_dict(self) -> dict[str, object]:
        """Return the canonical current persisted row."""
        return {
            "schema": self.schema_name,
            "phase": self.phase,
            "instance_id": self.instance_id,
            "unit": self.unit,
            "candidate_id": self.unit,
            "repeat_id": self.repeat_id,
            "repeat": self.repeat_id,
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
        """Validate and load one complete current-schema row."""
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
        if data["candidate_id"] != data["unit"]:
            raise ValueError("candidate_id must equal unit")
        if data["repeat"] != data["repeat_id"]:
            raise ValueError("repeat must equal repeat_id")
        record_data = {
            key: value
            for key, value in data.items()
            if key not in {"candidate_id", "repeat"}
        }
        return cls.model_validate(record_data)


def partial_key(
    phase: str,
    instance_id: str,
    unit: str,
    repeat_id: int,
) -> tuple[str, str, str, int]:
    """Return the stable identity of one persisted call observation."""
    return (phase, instance_id, unit, repeat_id)


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
    # Frame field names and schema are a pinned persisted-format contract.
    frame = {
        "schema": PARTIAL_FRAME_SCHEMA,
        "checksum": checksum,
        "record": record_body,
    }
    return _canonical_json_bytes(frame) + b"\n"


class _DuplicateKeyError(ValueError):
    pass


class _NonFiniteJSONNumberError(ValueError):
    pass


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    decoded: dict[str, object] = {}
    for key, value in pairs:
        if key in decoded:
            raise _DuplicateKeyError(f"duplicate JSON key {key!r}")
        decoded[key] = value
    return decoded


def _reject_json_constant(value: str) -> None:
    raise _NonFiniteJSONNumberError(
        f"non-finite JSON number is not allowed: {value}"
    )


def _decode_frame(
    raw: bytes,
    *,
    path: Path,
    line_number: int,
) -> PartialCallRecord:
    location = f"{path}:{line_number}"
    try:
        decoded = json.loads(
            raw,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (
        json.JSONDecodeError,
        UnicodeDecodeError,
        _DuplicateKeyError,
        _NonFiniteJSONNumberError,
    ) as exc:
        raise ValueError(f"invalid partial JSON at {location}") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"partial frame must be an object at {location}")
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
    expected = hashlib.sha256(_canonical_json_bytes(record_body)).hexdigest()
    if not hmac.compare_digest(checksum, expected):
        raise ValueError(f"partial frame checksum mismatch at {location}")
    return PartialCallRecord.from_dict(record_body)


def _truncate_torn_tail(fd: int) -> None:
    size = os.fstat(fd).st_size
    if size == 0 or os.pread(fd, 1, size - 1) == b"\n":
        return
    offset = size
    while offset:
        start = max(0, offset - 64 * 1024)
        chunk = os.pread(fd, offset - start, start)
        newline = chunk.rfind(b"\n")
        if newline >= 0:
            os.ftruncate(fd, start + newline + 1)
            return
        offset = start
    os.ftruncate(fd, 0)


def _write_all(fd: int, body: bytes) -> None:
    remaining = memoryview(body)
    while remaining:
        try:
            written = os.write(fd, remaining)
        except InterruptedError:
            continue
        if written == 0:  # pragma: no cover - kernel contract failure
            raise OSError("partial frame write made no progress")
        remaining = remaining[written:]


@dataclass(slots=True)
class PartialLog:
    """A durable checksummed log shared by threads and peer processes."""

    path: Path

    @property
    def _lock_path(self) -> Path:
        return self.path.with_name(f".{self.path.name}.lock")

    def append(self, record: PartialCallRecord) -> None:
        """Durably append one complete current-schema frame."""
        stamped = record
        if record.at is None:
            stamped = record.model_copy(
                update={"at": datetime.now(UTC).isoformat()}
            )
        assert stamped.at is not None
        _validate_timestamp(stamped.at)
        body = _encode_frame(stamped)
        with FileLock(self._lock_path):
            try:
                fd = open_private_regular_file(
                    self.path,
                    os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_EXCL,
                )
                created = True
            except FileExistsError:
                fd = open_private_regular_file(
                    self.path,
                    os.O_RDWR | os.O_APPEND,
                )
                created = False
            try:
                _truncate_torn_tail(fd)
                _write_all(fd, body)
                fsync_file(fd)
            finally:
                os.close(fd)
            if created:
                fsync_parent_directory(self.path)

    def load(self) -> list[PartialCallRecord]:
        """Load valid frames, ignoring only a final unterminated fragment."""
        with FileLock(self._lock_path, shared=True):
            try:
                fd = open_private_regular_file(self.path, os.O_RDONLY)
            except FileNotFoundError:
                return []
            by_key: dict[tuple[str, str, str, int], PartialCallRecord] = {}
            with os.fdopen(fd, "rb") as handle:
                for line_number, raw in enumerate(handle, start=1):
                    if not raw.endswith(b"\n"):
                        break
                    record = _decode_frame(
                        raw,
                        path=self.path,
                        line_number=line_number,
                    )
                    by_key[record.key()] = record
            return list(by_key.values())

    def recorded_keys(self) -> set[tuple[str, str, str, int]]:
        return {record.key() for record in self.load()}

    def delete(self) -> None:
        with FileLock(self._lock_path):
            try:
                fd = open_private_regular_file(self.path, os.O_RDONLY)
            except FileNotFoundError:
                return
            else:
                os.close(fd)
            try:
                self.path.unlink()
            except FileNotFoundError:
                return
            fsync_parent_directory(self.path)
