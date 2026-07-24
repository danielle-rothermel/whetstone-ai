from __future__ import annotations

from whetstone.runner.events import (
    CELL_FAILED,
    CELL_FINALIZED,
    EventStream,
    EventUnit,
    RunEvent,
    cell_failed_event,
    cell_finalized_event,
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
