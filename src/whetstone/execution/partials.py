"""Current-schema incremental persistence for completed provider calls."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
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
    model_validator,
)

__all__ = [
    "PARTIAL_SCHEMA",
    "PartialCallRecord",
    "PartialLog",
    "partial_key",
]

PARTIAL_SCHEMA = "whetstone.execution.partial_call/v1"

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
    schema_name: Literal["whetstone.execution.partial_call/v1"] = Field(
        default=PARTIAL_SCHEMA,
        alias="schema",
    )
    cache_hit: StrictBool = False
    cache_source_phase: StrictStr | None = None
    cache_source_unit: StrictStr | None = None
    cache_source_call_id: StrictStr | None = None
    cache_source_at: StrictStr | None = None

    @model_validator(mode="after")
    def _validate_current_record(self) -> Self:
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
        if not isinstance(data["at"], str) or not data["at"]:
            raise ValueError("partial row at must be a non-empty timestamp")
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


@dataclass(slots=True)
class PartialLog:
    """A thread-safe append-only JSONL log of current-schema records."""

    path: Path
    _lock: threading.Lock = field(
        default_factory=threading.Lock,
        repr=False,
        compare=False,
    )

    def append(self, record: PartialCallRecord) -> None:
        """Append and flush one complete current-schema row."""
        stamped = record
        if record.at is None:
            stamped = record.model_copy(
                update={"at": datetime.now(UTC).isoformat()}
            )
        body = json.dumps(
            stamped.as_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.path.open("a") as handle:
            handle.write(body + "\n")
            handle.flush()

    def load(self) -> list[PartialCallRecord]:
        """Load current-schema rows, with the latest row winning per key."""
        if not self.path.exists():
            return []
        by_key: dict[tuple[str, str, str, int], PartialCallRecord] = {}
        for line_number, raw in enumerate(
            self.path.read_text().splitlines(),
            start=1,
        ):
            line = raw.strip()
            if not line:
                continue
            try:
                decoded = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid partial JSON at {self.path}:{line_number}"
                ) from exc
            if not isinstance(decoded, dict):
                raise ValueError(
                    "partial row must be an object at "
                    f"{self.path}:{line_number}"
                )
            record = PartialCallRecord.from_dict(decoded)
            by_key[record.key()] = record
        return list(by_key.values())

    def recorded_keys(self) -> set[tuple[str, str, str, int]]:
        return {record.key() for record in self.load()}

    def delete(self) -> None:
        self.path.unlink(missing_ok=True)
