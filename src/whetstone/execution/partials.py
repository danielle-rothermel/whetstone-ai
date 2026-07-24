"""Incremental per-call persistence: the ``.partial.jsonl`` append/resume log.

Round-2 pilots that crashed mid-run lost every completed call. This module is
the append-only per-call log both the pilot and cell phases write as each
provider call completes, so a crash/interrupt leaves a durable record of the
work already done:

* the **pilot** report/resume path reads the partial to emit a partial report
  (and exit non-zero with a summary) instead of losing the run;
* the **cell** resume path reads the partial to SKIP already-recorded
  ``(instance, candidate, repeat)`` observations, so a resumed cell never
  re-drives a call whose result is already on disk.

The log is keyed by ``(phase, instance, unit, repeat)`` where ``unit`` is the
probe name (pilot) or candidate id (cell). A record carries just enough to
either reconstruct the observation (the 0/1 score or a failed flag) or resume
past it. It is deliberately independent of the richer pilot ``PilotCallRecord``
so both phases share ONE resumable on-disk shape.

The path is ``<artifact>.partial.jsonl`` (e.g. ``<root>/pilots/c11.partial
.jsonl`` or ``<root>/partials/<cell_id>.partial.jsonl``); a clean run may
delete it once the final artifact is written.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

__all__ = [
    "PARTIAL_SCHEMA",
    "PartialCallRecord",
    "PartialLog",
    "partial_key",
]

#: Versioned schema stamp on every partial row written GOING FORWARD (the
#: ``power_analysis/v1`` / ``events/v1`` precedent). A reader keys off it
#: rather than sniffing which fields happen to be present. Old rows (no stamp)
#: read back as ``schema=None`` -- coverage-honest, not an error.
PARTIAL_SCHEMA = "whetstone.execution.partial_call/v1"


@dataclass(frozen=True, slots=True)
class PartialCallRecord:
    """One completed provider call, appended as it finishes.

    The RESUME key is ``(phase, instance_id, unit, repeat_id)`` -- ``phase`` is
    ``"pilot"`` or ``"cell"`` and ``unit`` is the probe name (pilot) or the
    candidate id (cell). These key fields are UNCHANGED (resume must stay
    byte-identical across vintages). ``score`` is the 0/1 oracle score for a
    succeeded call (``None`` when it failed or produced no score); ``failed`` +
    ``failure_code`` carry the typed failure.

    Task 26 writes each row in the FINALIZED-row superset going forward: the
    finalized-row field NAMES are emitted as first-class fields
    (``candidate_id`` = ``unit``, ``repeat`` = ``repeat_id``, plus
    ``split_role``) so a consumer never has to remap the partial schema onto
    the ``rollout_outputs`` shape; the model ``output_text`` is persisted (it
    was 100% empty before); ``finish_reason`` / ``provider_error`` land the
    same per-call provenance the finalized rows carry; ``at`` is the ISO-8601
    UTC wall-clock the row was recorded; and ``schema`` version-stamps the row.
    Every added field is null/absent when unknown (never a populated-but-empty
    value). ``raw_response`` stays the ed1 resume-payload slot.
    """

    phase: str
    instance_id: str
    unit: str
    repeat_id: int
    score: float | None = None
    failed: bool = False
    failure_code: str = ""
    #: Optional measured usage (pilot token sanity); absent on the cell path.
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    #: Task-20 telemetry (present only from that commit forward; a pre-tel
    #: record leaves these ``None`` -- coverage-honest, never 0-conflated).
    reasoning_tokens: int | None = None
    latency_s: float | None = None
    #: The FULL model output text of the driven call (task 26: persisted going
    #: forward -- previously always empty on the cell path). ``None`` when the
    #: call produced no text (a failure) or was restored (not re-driven).
    output_text: str | None = None
    #: Resume-payload slot. On the ed1 enc-dec path this carries a compact JSON
    #: blob of the dual extras (compression + encoder/decoder text) the reducer
    #: needs to reconstruct a row WITHOUT re-driving; the QA/d1 paths leave it
    #: empty and persist the model text on ``output_text`` instead. Retained as
    #: a stored field (distinct from ``output_text``) so ed1 resume is
    #: byte-identical.
    raw_response: str = ""
    #: Task-26 per-call provenance (``None`` when unknown, never 0-conflated).
    finish_reason: str | None = None
    provider_error: dict[str, object] | None = None
    #: Finalized-row split role (``official`` / ``official_naive`` /
    #: ``official_ceiling`` / internal), so a consumer reads it directly rather
    #: than inferring an arm from the ``unit`` suffix. ``None`` when the writer
    #: did not supply one (old rows).
    split_role: str | None = None
    #: ISO-8601 UTC wall-clock the row was recorded (task 26). ``None`` on old
    #: rows (never captured) -- distinct from a populated-but-empty string.
    at: str | None = None
    #: Versioned schema stamp (:data:`PARTIAL_SCHEMA`); ``None`` on old rows.
    schema: str | None = None
    #: Task-31 prompt-cache honesty. ``cache_hit`` is True when this row's
    #: Result was SERVED from the run-level prompt cache (not re-driven), so a
    #: reader never mistakes reuse for a fresh paid call. A cached row keeps
    #: ``latency_s=None`` (never a fabricated 0 -- there was no wire call this
    #: time) and records spend as 0 via this marker (distinct from a genuinely
    #: free call, which has ``cache_hit=False``). ``cache_source_*`` reference
    #: the ORIGINAL entry (the cell/attempt + logical call id that first paid
    #: for it) and ``cache_source_at`` its original store timestamp -- ``None``
    #: on every non-cached row.
    cache_hit: bool = False
    cache_source_phase: str | None = None
    cache_source_unit: str | None = None
    cache_source_call_id: str | None = None
    cache_source_at: str | None = None

    def key(self) -> tuple[str, str, str, int]:
        return partial_key(
            self.phase, self.instance_id, self.unit, self.repeat_id
        )

    def as_dict(self) -> dict[str, object]:
        # The finalized-row superset. ``candidate_id`` / ``repeat`` mirror the
        # resume-key ``unit`` / ``repeat_id`` under the finalized field names
        # so a consumer never remaps the partial schema. ``raw_response`` stays
        # the verbatim resume-payload slot (the ed1 dual blob); ``output_text``
        # is the model text a QA/d1 row now persists.
        return {
            "schema": self.schema,
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
    def from_dict(cls, data: dict[str, object]) -> PartialCallRecord:
        # ``output_text`` reads as ``None`` when absent (the honest "not
        # recorded" state), never a populated-but-empty value. ``raw_response``
        # stays verbatim (ed1 resume decodes its JSON blob from it).
        out = data.get("output_text")
        return cls(
            phase=str(data["phase"]),
            instance_id=str(data["instance_id"]),
            unit=str(data["unit"]),
            repeat_id=_as_int(data["repeat_id"]),
            score=_opt_float(data.get("score")),
            failed=bool(data.get("failed", False)),
            failure_code=str(data.get("failure_code", "")),
            prompt_tokens=_opt_int(data.get("prompt_tokens")),
            completion_tokens=_opt_int(data.get("completion_tokens")),
            total_tokens=_opt_int(data.get("total_tokens")),
            reasoning_tokens=_opt_int(data.get("reasoning_tokens")),
            latency_s=_opt_float(data.get("latency_s")),
            output_text=None if out is None else str(out),
            raw_response=str(data.get("raw_response", "")),
            finish_reason=_opt_str(data.get("finish_reason")),
            provider_error=_opt_mapping(data.get("provider_error")),
            split_role=_opt_str(data.get("split_role")),
            at=_opt_str(data.get("at")),
            schema=_opt_str(data.get("schema")),
            cache_hit=bool(data.get("cache_hit", False)),
            cache_source_phase=_opt_str(data.get("cache_source_phase")),
            cache_source_unit=_opt_str(data.get("cache_source_unit")),
            cache_source_call_id=_opt_str(data.get("cache_source_call_id")),
            cache_source_at=_opt_str(data.get("cache_source_at")),
        )


def _as_int(value: object) -> int:
    assert isinstance(value, int | float | str)
    return int(value)


def _opt_int(value: object) -> int | None:
    return None if value is None else _as_int(value)


def _opt_float(value: object) -> float | None:
    if value is None:
        return None
    assert isinstance(value, int | float | str)
    return float(value)


def _opt_str(value: object) -> str | None:
    return None if value is None else str(value)


def _opt_mapping(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    assert isinstance(value, dict)
    return {str(k): v for k, v in value.items()}


def partial_key(
    phase: str, instance_id: str, unit: str, repeat_id: int
) -> tuple[str, str, str, int]:
    """The resume key for one per-call observation."""
    return (phase, instance_id, unit, repeat_id)


@dataclass(slots=True)
class PartialLog:
    """An append-only ``.partial.jsonl`` of per-call records.

    Constructed over a target ``path``; :meth:`append` writes one record as its
    call completes (flushed so a crash keeps it), and :meth:`load` reads the
    existing records back for the resume/partial-report path. Last-write-wins
    by key, so a re-appended observation supersedes an earlier one.

    :meth:`append` is thread-safe: the fan-out pool's workers each append their
    completed call concurrently, so a crash mid-drive still leaves every
    already-finished call durably on disk.
    """

    path: Path
    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    def append(self, record: PartialCallRecord) -> None:
        """Append one per-call record, flushing so a crash keeps it.

        Stamps the wall-clock ``at`` (ISO-8601 UTC) + the versioned ``schema``
        on the row at write time when the caller left them unset, so every
        going-forward row is timestamped and version-stamped without each call
        site repeating it. Thread-safe: serialized under a lock so concurrent
        workers' appends never interleave a half-written line.
        """
        if record.at is None or record.schema is None:
            record = replace(
                record,
                at=record.at or datetime.now(UTC).isoformat(),
                schema=record.schema or PARTIAL_SCHEMA,
            )
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as handle:
                handle.write(
                    json.dumps(record.as_dict(), sort_keys=True) + "\n"
                )
                handle.flush()

    def load(self) -> list[PartialCallRecord]:
        """Read the existing records (last-write-wins by key, in order)."""
        if not self.path.exists():
            return []
        by_key: dict[tuple[str, str, str, int], PartialCallRecord] = {}
        for raw in self.path.read_text().splitlines():
            line = raw.strip()
            if not line:
                continue
            record = PartialCallRecord.from_dict(json.loads(line))
            by_key[record.key()] = record
        return list(by_key.values())

    def recorded_keys(self) -> set[tuple[str, str, str, int]]:
        """Keys of observations already on disk (resume skips these)."""
        return {r.key() for r in self.load()}

    def delete(self) -> None:
        """Remove the partial log (a clean run does this post-finalize)."""
        self.path.unlink(missing_ok=True)
