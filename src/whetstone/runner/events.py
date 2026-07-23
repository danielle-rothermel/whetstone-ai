"""Push-based run telemetry: the append-only ``logs/events.jsonl`` stream.

A live validation run drives many cells (often concurrently, one shared key),
and the only way to know what a run is doing has been to poll many logs. This
module is the ONE structured event stream a dumb watcher can wake on: the
runner appends typed JSONL events to ``<root>/logs/events.jsonl`` (run-root
level, append-only, one stream keyed by cell/screen id) the moment something
meaningful happens, and mirrors each event with a LOUD plain-text stderr
marker so existing ``grep``-based watchers keep working and gain the new
signatures.

The event types (all pushed the instant the runner observes them, not polled):

* :data:`RATE_LIMIT_PRESSURE` -- heartbeat-window counters of 429s / rate-limit
  retries / concurrency-halved incidents, emitted only when nonzero.
* :data:`ATTEMPT_SKIPPED` -- the runner SKIPPED an already-completed attempt
  (correct for wave relaunches, but previously SILENT while it echoed stale
  stats -- the c18 collision defect class). The skip behavior is unchanged; the
  event + loud warning is what is new.
* :data:`CELL_FINALIZED` -- a cell reached a terminal statistical status
  (status, delta + CI when applicable, REALIZED spend, duration, attempt id).
* :data:`CELL_FAILED` -- a cell failed with a TYPED failure reason (never a
  bare string dump of an exception).
* :data:`ARM_INCOMPLETE` -- the incomplete-official-arm transient-failure class
  (currently only visible deep in the logs).
* :data:`LATENCY_SNAPSHOT` -- per-model rolling median call latency at each
  heartbeat window.
* :data:`TRACEBACK` -- an unhandled exception surfaced as an event BEFORE the
  process dies.

Truthfulness rules (matching the existing telemetry coverage-honesty rule):

* **null-not-zero for unknown values** -- an unknown median latency is
  ``None``, never ``0.0``; an unknown spend is ``None``.
* **estimates are labeled** -- heartbeat spend estimates over-attribute under
  concurrency (many cells share one credits meter), so such fields are named
  ``spend_estimate_usd`` and never presented as realized spend. A finalized
  cell's REALIZED spend (the credits-delta paired to its own snapshots) is the
  distinct ``realized_spend_usd`` field.

Nothing here makes a live paid call: events describe work the runner already
did. Emission is best-effort and MUST never affect a cell's result -- a failed
write is swallowed loudly (a broken event stream cannot break a run).
"""

from __future__ import annotations

import contextlib
import json
import sys
import threading
import traceback as traceback_mod
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr

__all__ = [
    "ARM_INCOMPLETE",
    "ATTEMPT_SKIPPED",
    "CELL_FAILED",
    "CELL_FINALIZED",
    "EVENTS_SCHEMA",
    "EVENT_MARKERS",
    "LATENCY_SNAPSHOT",
    "RATE_LIMIT_PRESSURE",
    "TRACEBACK",
    "EventStream",
    "EventUnit",
    "RunEvent",
    "arm_incomplete_event",
    "attempt_skipped_event",
    "cell_failed_event",
    "cell_finalized_event",
    "emit_traceback_on_unhandled",
    "is_rate_limit_code",
    "latency_snapshot_event",
    "rate_limit_pressure_event",
    "traceback_event",
]

#: The event-schema tag stamped on every row (precedent:
#: ``whetstone.runner.power_analysis/v1``). A structured reader keys off it so
#: the envelope can evolve without silently breaking downstream joins.
EVENTS_SCHEMA = "whetstone.runner.events/v1"

#: The typed event-name constants (the ``event`` field's closed value set).
RATE_LIMIT_PRESSURE = "rate_limit_pressure"
ATTEMPT_SKIPPED = "attempt_skipped"
CELL_FINALIZED = "cell_finalized"
CELL_FAILED = "cell_failed"
ARM_INCOMPLETE = "arm_incomplete"
LATENCY_SNAPSHOT = "latency_snapshot"
TRACEBACK = "traceback"

#: The LOUD plain-text stderr marker mirroring each event. The set is chosen so
#: existing grep watchers matching
#: ``429|rate.limit|halved|RATE-LIMIT|Traceback``
#: keep working and gain the new signatures. Each marker is a stable, unique,
#: greppable token (upper-case, hyphenated) a watcher can ``grep -m1`` on.
EVENT_MARKERS: dict[str, str] = {
    RATE_LIMIT_PRESSURE: "RATE-LIMIT PRESSURE",
    ATTEMPT_SKIPPED: "ATTEMPT-SKIPPED",
    CELL_FINALIZED: "CELL-FINALIZED",
    CELL_FAILED: "CELL-FAILED",
    ARM_INCOMPLETE: "ARM-INCOMPLETE",
    LATENCY_SNAPSHOT: "LATENCY-SNAPSHOT",
    TRACEBACK: "TRACEBACK",
}


def _utc_now() -> str:
    """An ISO-8601 UTC timestamp (the default event clock)."""
    return datetime.now(UTC).isoformat()


def is_rate_limit_code(code: str) -> bool:
    """Whether a recorded failure ``code`` is a rate-limit (429) signal.

    Matches the transport codes the runner records for a rate-limited call
    (``http_status_429``) and the semantic-class value (``rate-limit``), so a
    watcher's ``429|rate.limit`` grep and this counter agree. ``""`` (a
    success) is not a rate limit.
    """
    lowered = code.lower()
    return (
        "429" in lowered
        or "rate_limit" in lowered
        or "rate-limit" in lowered
    )


class EventUnit(BaseModel):
    """The STRUCTURED identity of the unit an event is about.

    The composite ``cell_id`` (``opt:env:aN``) / ``screen_id`` display strings
    are recorded ALONGSIDE the components -- but a downstream reader (the
    viewer) joins on the SEPARATE fields (``env`` / ``optimizer`` / ``attempt``
    / ``lane`` / ``model``) so it never has to parse the composite id, and a
    future id format change cannot break its joins. Every component is ``None``
    when it does not apply (a screen-level event has no ``attempt``; a
    pre-model event has no ``model``) -- null-not-zero, never a placeholder.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    cell_id: StrictStr | None = None
    screen_id: StrictStr | None = None
    env: StrictStr | None = None
    optimizer: StrictStr | None = None
    attempt: StrictInt | None = None
    lane: StrictStr | None = None
    model: StrictStr | None = None

    @classmethod
    def for_cell(
        cls,
        *,
        cell_id: str,
        env: str,
        optimizer: str,
        attempt: int,
        lane: str,
        model: str | None,
    ) -> EventUnit:
        """The identity of a cell-level event (all components populated)."""
        return cls(
            cell_id=cell_id,
            env=env,
            optimizer=optimizer,
            attempt=attempt,
            lane=lane,
            model=model,
        )

    @property
    def display_id(self) -> str:
        return self.cell_id or self.screen_id or "-"


class RunEvent(BaseModel):
    """One typed telemetry event -- one line of ``logs/events.jsonl``.

    ``schema`` tags the envelope version (:data:`EVENTS_SCHEMA`); ``event`` is
    one of the module's event-name constants; ``at`` is a real ISO-8601 UTC
    timestamp (never an empty string -- a hard precedent to avoid, see module
    docstring). ``unit`` carries the STRUCTURED identity (env/optimizer/
    attempt/lane/model as separate fields, plus the composite display id) so a
    reader joins on components, not a parsed id string. ``marker`` is the loud
    plain-text token mirrored to stderr. ``fields`` holds the event-type
    payload (already truthfulness-checked by the builder: null-not-zero,
    estimates labeled). ``extra="forbid"`` keeps the envelope typed; the open
    ``fields`` dict is where per-type data lives.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_: StrictStr = EVENTS_SCHEMA
    event: StrictStr
    at: StrictStr
    marker: StrictStr
    unit: EventUnit
    fields: dict[str, Any] = {}

    @property
    def cell_id(self) -> str | None:
        return self.unit.cell_id

    @property
    def screen_id(self) -> str | None:
        return self.unit.screen_id

    def model_dump_json_dict(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")
        # Serialize the reserved-word attribute as ``schema`` on the wire.
        data["schema"] = data.pop("schema_")
        return data

    def to_line(self) -> str:
        return json.dumps(self.model_dump_json_dict(), sort_keys=True)

    def marker_line(self) -> str:
        """The loud one-line stderr form: ``<MARKER> <id> key=val ...``."""
        parts = [self.marker, self.unit.display_id]
        for key in sorted(self.fields):
            parts.append(f"{key}={self.fields[key]}")
        return " ".join(parts)

    @classmethod
    def from_line(cls, line: str) -> RunEvent:
        data = json.loads(line)
        if "schema" in data:
            data["schema_"] = data.pop("schema")
        return cls.model_validate(data)


def _event(
    event: str,
    *,
    at: str,
    unit: EventUnit,
    fields: dict[str, Any],
) -> RunEvent:
    return RunEvent(
        event=event,
        at=at,
        marker=EVENT_MARKERS[event],
        unit=unit,
        fields=fields,
    )


def rate_limit_pressure_event(
    *,
    unit: EventUnit,
    rate_limit_rows: int,
    concurrency_halved: bool,
    guard_timeouts: int,
    window_label: str,
    at: str | None = None,
) -> RunEvent:
    """A heartbeat-window rate-limit-pressure event (the task-24 core).

    ``rate_limit_rows`` counts observation rows in this window that hit a
    429/rate-limit typed failure (a call that retried past a 429 still counts
    -- it observed pressure on the shared key). ``concurrency_halved`` /
    ``guard_timeouts`` mirror the fan-out backpressure signals. The caller
    emits this ONLY when at least one counter is nonzero (never a 0-0-0 beat).
    """
    return _event(
        RATE_LIMIT_PRESSURE,
        at=at or _utc_now(),
        unit=unit,
        fields={
            "rate_limit_rows": rate_limit_rows,
            "concurrency_halved": concurrency_halved,
            "guard_timeouts": guard_timeouts,
            "window": window_label,
        },
    )


def attempt_skipped_event(
    *,
    unit: EventUnit,
    prior_status: str,
    at: str | None = None,
) -> RunEvent:
    """An event for a SKIPPED already-completed attempt (was SILENT before).

    The skip behavior is unchanged (correct for wave relaunches); this makes
    the skip LOUD so a watcher never mistakes the runner's echo of the prior
    line's stale stats for fresh work (the c18 collision defect class).
    ``prior_status`` is the completed record's status the runner is echoing.
    """
    return _event(
        ATTEMPT_SKIPPED,
        at=at or _utc_now(),
        unit=unit,
        fields={"prior_status": prior_status},
    )


def cell_finalized_event(
    *,
    unit: EventUnit,
    status: str,
    delta: float | None,
    delta_ci95: tuple[float, float] | None,
    realized_spend_usd: float | None,
    duration_s: float | None,
    at: str | None = None,
) -> RunEvent:
    """A cell reached a terminal statistical status.

    ``delta`` / ``delta_ci95`` are ``None`` when the cell has no delta (e.g. an
    eval-only row or a cut-short cell) -- null-not-zero. ``realized_spend_usd``
    is the cell's OWN credits-delta (the ``spend_for_cell`` pairing), never a
    heartbeat estimate; ``None`` when spend is unknown (non-openrouter lane or
    an unbounded gap).
    """
    ci = list(delta_ci95) if delta_ci95 is not None else None
    return _event(
        CELL_FINALIZED,
        at=at or _utc_now(),
        unit=unit,
        fields={
            "status": status,
            "delta": delta,
            "delta_ci95": ci,
            "realized_spend_usd": realized_spend_usd,
            "duration_s": duration_s,
        },
    )


def cell_failed_event(
    *,
    unit: EventUnit,
    reason_class: str,
    detail: str,
    at: str | None = None,
) -> RunEvent:
    """A cell failed with a TYPED failure reason (never a bare string dump).

    ``reason_class`` is the failure's TYPE name (e.g.
    ``"CellBaselineFailure"``); ``detail`` is a bounded human message. A
    watcher keys off ``reason_class``, not the free-form detail.
    """
    return _event(
        CELL_FAILED,
        at=at or _utc_now(),
        unit=unit,
        fields={"reason_class": reason_class, "detail": detail},
    )


def arm_incomplete_event(
    *,
    unit: EventUnit,
    detail: str,
    at: str | None = None,
) -> RunEvent:
    """An incomplete-official-arm finalization (the transient-failure class).

    ``detail`` names the failed arm(s) + row accounting exactly as the ledger
    note does; the cell is NOT a certified result (a re-run supersedes it).
    """
    return _event(
        ARM_INCOMPLETE,
        at=at or _utc_now(),
        unit=unit,
        fields={"detail": detail},
    )


def latency_snapshot_event(
    *,
    unit: EventUnit,
    median_latency_s: float | None,
    coverage: int,
    window_label: str,
    at: str | None = None,
) -> RunEvent:
    """A per-model rolling-median call-latency snapshot at a heartbeat window.

    The model is on ``unit.model`` (a structured field, not parsed from an id).
    ``median_latency_s`` is ``None`` when NO call in the window reported a
    latency (null-not-zero, never a fake 0); ``coverage`` is the number of rows
    the median was computed over so a partial-coverage window is never mistaken
    for a full one.
    """
    return _event(
        LATENCY_SNAPSHOT,
        at=at or _utc_now(),
        unit=unit,
        fields={
            "median_latency_s": median_latency_s,
            "coverage": coverage,
            "window": window_label,
        },
    )


def traceback_event(
    *,
    unit: EventUnit,
    exc_type: str,
    message: str,
    traceback_text: str,
    at: str | None = None,
) -> RunEvent:
    """An unhandled exception surfaced as an event BEFORE the process dies.

    ``exc_type`` is the exception's type name; ``traceback_text`` is the full
    formatted traceback (so the ``Traceback`` grep signature is preserved in
    the marker line and the structured field). Emitted from the unhandled
    boundary, then the exception re-raises so the process still fails loudly.
    """
    return _event(
        TRACEBACK,
        at=at or _utc_now(),
        unit=unit,
        fields={
            "exc_type": exc_type,
            "message": message,
            "traceback": traceback_text,
        },
    )


class EventStream:
    """The append-only, concurrent-writer-safe ``logs/events.jsonl`` writer.

    Constructed over the RUN ROOT; the stream lives at ``<root>/logs/events
    .jsonl`` (one shared stream for the whole run, keyed by the cell/screen id
    fields). :meth:`emit` appends one event's JSON line (flushed so a crash
    keeps it) AND mirrors its loud marker to stderr, so a ``grep``-based
    watcher on the plain log and a structured JSONL reader both see it.

    Appends are serialized under a process-local lock so concurrent threads in
    ONE process never interleave a half-written line. Across PROCESSES, each
    ``emit`` opens/writes/closes a single ``O_APPEND`` line, which POSIX writes
    atomically for a line this small -- the same single-line-append discipline
    the partial log relies on.

    Emission is best-effort: a write failure is swallowed (loudly, to stderr)
    so a broken event stream can NEVER break a cell's actual work.
    """

    def __init__(
        self,
        root: Path,
        *,
        marker_sink: Callable[[str], None] | None = None,
    ) -> None:
        self._root = root
        self._marker_sink = marker_sink or (
            lambda line: sys.stderr.write(line + "\n")
        )
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._root / "logs" / "events.jsonl"

    def emit(self, event: RunEvent) -> None:
        """Append ``event`` (flushed) and mirror its loud marker to stderr.

        Never raises: a JSONL write failure is reported to the marker sink and
        swallowed, so a broken event stream cannot break a run.
        """
        # The loud marker always fires (it is the grep-able signal); it is the
        # cheapest and most important half, so do it first and independently.
        try:
            self._marker_sink(event.marker_line())
        except Exception:  # pragma: no cover - marker sink is best-effort
            pass
        try:
            line = event.to_line()
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a") as handle:
                    handle.write(line + "\n")
                    handle.flush()
        except Exception as exc:  # pragma: no cover - stream is best-effort
            try:
                self._marker_sink(
                    f"EVENT-STREAM-WRITE-FAILED {event.event}: {exc}"
                )
            except Exception:
                pass

    def load(self) -> list[RunEvent]:
        """Read the existing events back (for tests / a structured reader)."""
        if not self.path.exists():
            return []
        events: list[RunEvent] = []
        for raw in self.path.read_text().splitlines():
            line = raw.strip()
            if not line:
                continue
            events.append(RunEvent.from_line(line))
        return events


@contextlib.contextmanager
def emit_traceback_on_unhandled(
    events: EventStream | None,
    *,
    unit: EventUnit,
    reraise: type[BaseException] | tuple[type[BaseException], ...] = (),
) -> Iterator[None]:
    """Surface an unhandled exception as a ``traceback`` event, then re-raise.

    Wraps the process-death boundary (the CLI's live run): if the body raises,
    a :func:`traceback_event` is pushed (the full formatted traceback in the
    ``traceback`` field AND the loud ``TRACEBACK`` marker, preserving the
    ``Traceback`` grep signature) BEFORE the exception propagates and the
    process dies. ``reraise`` names exception types this boundary lets pass
    WITHOUT emitting -- the runner's own typed, already-handled failures
    (``CellBaselineFailure`` / ``ReserveError``), which the CLI reports itself
    and are not "unhandled".

    A ``None`` ``events`` makes this a transparent pass-through. Emission is
    best-effort (:meth:`EventStream.emit` never raises), so it can never mask
    or replace the original exception.
    """
    try:
        yield
    except BaseException as exc:
        if events is not None and not (reraise and isinstance(exc, reraise)):
            with contextlib.suppress(Exception):
                events.emit(
                    traceback_event(
                        unit=unit,
                        exc_type=type(exc).__name__,
                        message=str(exc),
                        traceback_text="".join(
                            traceback_mod.format_exception(exc)
                        ),
                    )
                )
        raise
