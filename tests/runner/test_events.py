from __future__ import annotations

import pytest

from whetstone.runner.events import (
    CELL_FAILED,
    CELL_FINALIZED,
    EVENT_MARKERS,
    EVENTS_SCHEMA,
    EventStream,
    EventUnit,
    RunEvent,
    arm_incomplete_event,
    cell_failed_event,
    cell_finalized_event,
    emit_traceback_on_unhandled,
    is_rate_limit_code,
    latency_snapshot_event,
    rate_limit_pressure_event,
)


def _unit() -> EventUnit:
    return EventUnit.for_cell(
        cell_id="copro:c18:a0",
        env="c18",
        optimizer="copro",
        attempt=0,
        lane="test",
        model="openai/test",
    )


def test_event_stream_appends_validated_jsonl(tmp_path) -> None:
    stream = EventStream(tmp_path)
    event = cell_finalized_event(
        unit=_unit(),
        status="no-improvement",
        delta=0.0,
        delta_ci95=(-0.1, 0.1),
        realized_spend_usd=0.0,
        duration_s=1.25,
        at="2026-07-24T00:00:00+00:00",
    )

    stream.emit(event)

    loaded = RunEvent.from_line(stream.path.read_text().strip())
    assert loaded == event
    assert loaded.event == CELL_FINALIZED


def test_failed_event_preserves_typed_reason() -> None:
    event = cell_failed_event(
        unit=_unit(),
        reason_class="CellBaselineFailure",
        detail="baseline incomplete",
        at="2026-07-24T00:00:00+00:00",
    )

    assert event.event == CELL_FAILED
    assert event.fields["reason_class"] == "CellBaselineFailure"
    assert event.fields["detail"] == "baseline incomplete"


def test_wire_line_carries_the_schema_key_not_the_attribute() -> None:
    event = arm_incomplete_event(
        unit=_unit(), detail="baseline arm short 3 rows", at="2026-07-24T00:00"
    )

    payload = event.model_dump_json_dict()

    assert payload["schema"] == EVENTS_SCHEMA
    assert "schema_" not in payload
    assert RunEvent.from_line(event.to_line()) == event


def test_unknown_latency_is_null_not_zero() -> None:
    event = latency_snapshot_event(
        unit=_unit(),
        median_latency_s=None,
        coverage=0,
        window_label="w1",
        at="2026-07-24T00:00",
    )

    assert event.fields["median_latency_s"] is None
    assert event.fields["coverage"] == 0


def test_marker_line_is_greppable_and_sorted() -> None:
    event = rate_limit_pressure_event(
        unit=_unit(),
        rate_limit_rows=4,
        concurrency_halved=True,
        guard_timeouts=0,
        window_label="w1",
        at="2026-07-24T00:00",
    )

    line = event.marker_line()

    assert line.startswith("RATE-LIMIT PRESSURE copro:c18:a0 ")
    assert "rate_limit_rows=4" in line
    assert line.index("concurrency_halved") < line.index("rate_limit_rows")


def test_every_event_name_has_a_unique_marker() -> None:
    assert len(set(EVENT_MARKERS.values())) == len(EVENT_MARKERS)


@pytest.mark.parametrize(
    "code",
    ["http_status_429", "RATE_LIMIT", "rate-limit", "provider_429_backoff"],
)
def test_rate_limit_codes_match_the_watcher_grep(code: str) -> None:
    assert is_rate_limit_code(code)


@pytest.mark.parametrize("code", ["", "http_status_500", "timeout"])
def test_non_rate_limit_codes_do_not_match(code: str) -> None:
    assert not is_rate_limit_code(code)


def test_emit_is_best_effort_and_never_breaks_the_caller(tmp_path) -> None:
    # A stream rooted at a *file* cannot create its logs directory, so emit
    # must swallow the failure while the loud marker still fires.
    blocked = tmp_path / "not-a-dir"
    blocked.write_text("")
    markers: list[str] = []
    stream = EventStream(blocked, marker_sink=markers.append)

    stream.emit(
        cell_failed_event(
            unit=_unit(),
            reason_class="Boom",
            detail="d",
            at="2026-07-24T00:00",
        )
    )

    assert markers[0].startswith("CELL-FAILED")
    assert any("EVENT-STREAM-WRITE-FAILED" in line for line in markers)
    assert stream.load() == []


def test_traceback_boundary_emits_then_reraises(tmp_path) -> None:
    stream = EventStream(tmp_path)

    with pytest.raises(ValueError, match="boom"):
        with emit_traceback_on_unhandled(stream, unit=_unit()):
            raise ValueError("boom")

    (event,) = stream.load()
    assert event.fields["exc_type"] == "ValueError"
    assert "ValueError: boom" in event.fields["traceback"]


def test_traceback_boundary_skips_declared_handled_failures(tmp_path) -> None:
    stream = EventStream(tmp_path)

    with pytest.raises(KeyError):
        with emit_traceback_on_unhandled(
            stream, unit=_unit(), reraise=KeyError
        ):
            raise KeyError("handled")

    assert stream.load() == []
