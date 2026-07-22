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
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "PartialCallRecord",
    "PartialLog",
    "partial_key",
]


@dataclass(frozen=True, slots=True)
class PartialCallRecord:
    """One completed provider call, appended as it finishes.

    ``phase`` is ``"pilot"`` or ``"cell"``; ``unit`` is the probe name (pilot)
    or the candidate id (cell). ``score`` is the 0/1 oracle score for a
    succeeded call (``None`` when it failed or produced no score). ``failed``
    and ``failure_code`` carry the typed failure for a failed call.
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
    raw_response: str = ""

    def key(self) -> tuple[str, str, str, int]:
        return partial_key(
            self.phase, self.instance_id, self.unit, self.repeat_id
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "instance_id": self.instance_id,
            "unit": self.unit,
            "repeat_id": self.repeat_id,
            "score": self.score,
            "failed": self.failed,
            "failure_code": self.failure_code,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "raw_response": self.raw_response,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> PartialCallRecord:
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
            raw_response=str(data.get("raw_response", "")),
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

        Thread-safe: serialized under a lock so concurrent workers' appends
        never interleave a half-written line.
        """
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
