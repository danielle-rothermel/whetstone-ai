"""Truncated Codex output is never presented as a contiguous stream.

Under a finite ``payload_output`` budget dr-exec retains a head and a tail
and drops the middle. Concatenating them yields bytes that were never
adjacent in the real stream: the last retained head line and the first
retained tail line join into one line that Codex never emitted. Parsed as
JSONL that either fails on a synthetic malformed boundary or -- worse --
decodes into an event no process ever produced.

So the join carries an explicit marker and the artifact says the stream
was truncated and by how much.
"""

from __future__ import annotations

import pytest

from whetstone.optim.codex.adapter import OpaqueStepError
from whetstone.optim.codex.runner import (
    CODEX_ELIDED_MARKER_PREFIX,
    _parse_jsonl_events,
    _retained_bytes,
)


class _FakeStream:
    """Stands in for dr-exec's ``RetainedPayloadStream``."""

    def __init__(self, *, head: bytes, tail: bytes, dropped_bytes: int) -> None:
        self.head = head
        self.tail = tail
        self.dropped_bytes = dropped_bytes


def test_an_untruncated_stream_is_returned_byte_for_byte() -> None:
    """No truncation, no marker: the retained bytes are the real stream."""
    stream = _FakeStream(head=b'{"a": 1}\n{"b": 2}\n', tail=b"", dropped_bytes=0)

    retained = _retained_bytes(stream)

    assert retained.data == b'{"a": 1}\n{"b": 2}\n'
    assert retained.truncated is False
    assert retained.dropped_bytes == 0


def test_a_truncated_stream_marks_where_the_middle_was_elided() -> None:
    """The head and tail are separated by an explicit marker line."""
    stream = _FakeStream(
        head=b'{"a": 1}\n{"partial": ',
        tail=b'ue}\n{"b": 2}\n',
        dropped_bytes=4096,
    )

    retained = _retained_bytes(stream)

    assert retained.truncated is True
    assert retained.dropped_bytes == 4096
    lines = retained.data.splitlines()
    marker = [
        line for line in lines if line.startswith(CODEX_ELIDED_MARKER_PREFIX)
    ]
    assert len(marker) == 1
    assert b"4096" in marker[0]
    # The two real fragments never merge into one synthetic line.
    assert b'{"partial": ue}' not in retained.data


def test_the_marker_terminates_a_head_that_ends_mid_line() -> None:
    """A head cut mid-line must not run into the marker itself."""
    stream = _FakeStream(
        head=b'{"a": 1}\n{"cut": ', tail=b'{"b": 2}\n', dropped_bytes=12
    )

    retained = _retained_bytes(stream)

    lines = retained.data.splitlines()
    assert b'{"cut": ' in lines
    assert any(line.startswith(CODEX_ELIDED_MARKER_PREFIX) for line in lines)
    assert b'{"b": 2}' in lines


def test_a_truncated_transcript_parses_to_its_complete_records() -> None:
    """The marker and the cut boundary lines never fail the parse.

    An over-budget stdout cut mid-line is the expected end of a chatty
    agent, not a contract violation. The run's final artifact is still
    valid, so the transcript yields exactly the records that survived
    retention intact, and says how many it lost.
    """
    stream = _FakeStream(
        head=b'{"a": 1}\n{"cut": ',
        tail=b'ue}\n{"b": 2}\n',
        dropped_bytes=4096,
    )
    retained = _retained_bytes(stream)

    parsed = _parse_jsonl_events(retained.data, truncated=True)

    assert parsed.events == ({"a": 1}, {"b": 2})
    # The head's cut tail and the tail's cut head: two damaged lines.
    assert parsed.dropped_partial_lines == 2


def test_an_untruncated_stream_still_rejects_a_malformed_event() -> None:
    """Tolerance is for budget damage only, never for what Codex wrote."""
    with pytest.raises(OpaqueStepError, match="malformed"):
        _parse_jsonl_events(b'{"a": 1}\nnot json\n', truncated=False)


def test_a_truncated_stream_rejects_damage_away_from_the_stitch() -> None:
    """Only the two lines the marker sits between may be dropped."""
    # The malformed line is the *first* of the head, two lines clear of
    # the marker, so retention cannot explain it.
    stitched = (
        b'not json\n{"a": 1}\n'
        + CODEX_ELIDED_MARKER_PREFIX
        + b"9 bytes elided ...]\n"
        + b'{"b": 2}\n'
    )

    with pytest.raises(OpaqueStepError, match="malformed"):
        _parse_jsonl_events(stitched, truncated=True)


def test_a_marker_shaped_line_does_not_capture_the_stitch_search() -> None:
    """Only the real marker locates the stitch, so real damage stays forgiven.

    The elision marker is whetstone's own synthetic line, and the parser
    has to tell it from Codex's output by something Codex cannot
    plausibly emit. A human-readable ``[... `` prefix is not that: a
    line opening with a bracketed aside is ordinary agent prose.

    Matching loosely costs more than one line. ``_is_stitch_boundary``
    stops at the *first* marker-shaped line, so a decoy captures the
    search and points it away from the real stitch -- and then the two
    lines the retention window genuinely cut are no longer recognized
    as budget damage. A truncated run whose final artifact is perfectly
    valid fails on exactly the damage the parser exists to tolerate.

    Here the decoy sits in the head, ahead of the real marker. The
    decoy is not JSON, so it is a real contract violation and the parse
    must fail -- but it must fail on the *decoy*, not on the innocent
    stitch line the decoy misdirected the search away from.
    """
    decoy = b"[... the model narrating its reasoning ...]"
    stream = _FakeStream(
        head=b'{"a": 1}\n' + decoy + b'\n{"cut": ',
        tail=b'ue}\n{"b": 2}\n',
        dropped_bytes=4096,
    )
    retained = _retained_bytes(stream)

    with pytest.raises(OpaqueStepError, match="event 2 is malformed"):
        _parse_jsonl_events(retained.data, truncated=True)


def test_only_the_exact_marker_line_is_skipped() -> None:
    """The marker is matched whole, on its sentinel, never on a prefix."""
    stitched = (
        b'{"a": 1}\n'
        + CODEX_ELIDED_MARKER_PREFIX
        + b"4096 bytes elided ...]\n"
        + b'{"b": 2}\n'
    )

    parsed = _parse_jsonl_events(stitched, truncated=True)

    assert parsed.events == ({"a": 1}, {"b": 2})
    assert parsed.dropped_partial_lines == 0


def test_a_complete_malformed_record_beside_the_marker_still_fails() -> None:
    """Adjacency to the stitch does not excuse a whole malformed record.

    Retention can end exactly on a record boundary, and then the line
    before the marker is a complete line Codex really emitted. If that
    line is malformed it is genuine process output that violates the
    contract, and forgiving it on position alone deletes it silently --
    the persisted ``jsonl_events`` then differ from the retained stream
    with nothing but a ``dropped_partial_lines`` bump to show for it.

    ``not json`` is balanced: it neither opens a record it fails to
    close nor closes one it never opened, so it is not a partial record
    in either direction.
    """
    stitched = (
        b'{"a": 1}\nnot json\n'
        + CODEX_ELIDED_MARKER_PREFIX
        + b"4096 bytes elided ...]\n"
        + b'{"b": 2}\n'
    )

    with pytest.raises(OpaqueStepError, match="event 2 is malformed"):
        _parse_jsonl_events(stitched, truncated=True)


def test_a_complete_malformed_record_after_the_marker_still_fails() -> None:
    """The same rule on the tail side, where retention began cleanly.

    A tail whose first retained byte is a record boundary hands the
    parser a whole line. A malformed whole line there is Codex's output,
    not the budget's damage.
    """
    stitched = (
        b'{"a": 1}\n'
        + CODEX_ELIDED_MARKER_PREFIX
        + b"4096 bytes elided ...]\n"
        + b'{"b": 2} trailing garbage\n'
    )

    with pytest.raises(OpaqueStepError, match="event 3 is malformed"):
        _parse_jsonl_events(stitched, truncated=True)


def test_a_genuinely_cut_boundary_line_is_still_forgiven() -> None:
    """The tolerance the parser exists for must survive the tightening.

    The head fragment opens an object it never closes and the tail
    fragment closes one it never opened, so both are demonstrably
    partial records rather than complete lines.
    """
    stitched = (
        b'{"a": 1}\n{"cut": "val\n'
        + CODEX_ELIDED_MARKER_PREFIX
        + b"4096 bytes elided ...]\n"
        + b'ue"}\n{"b": 2}\n'
    )

    parsed = _parse_jsonl_events(stitched, truncated=True)

    assert parsed.events == ({"a": 1}, {"b": 2})
    assert parsed.dropped_partial_lines == 2


def test_a_complete_malformed_tail_beside_the_marker_still_fails() -> None:
    """Closing without opening is not proof the budget cut the record.

    ``not json}`` does not begin on ``{`` and does end on ``}``, so the
    tail line's own shape is indistinguishable from a real fragment --
    yet here the retained head ended on a whole record, so nothing in
    the stream shows a record spanning the elision. Forgiving this line
    would delete genuine malformed process output and report it as
    retention damage instead of rejecting the run.
    """
    stitched = (
        b'{"a": 1}\n'
        + CODEX_ELIDED_MARKER_PREFIX
        + b"4096 bytes elided ...]\n"
        + b"not json}\n"
        + b'{"b": 2}\n'
    )

    with pytest.raises(OpaqueStepError, match="event 3 is malformed"):
        _parse_jsonl_events(stitched, truncated=True)


def test_a_cut_head_still_corroborates_a_cut_tail() -> None:
    """The tolerance survives: a head cut mid-record witnesses the tail.

    The head's last line opens a record it never closes, so the stream
    itself shows the budget falling inside a record. The tail line that
    closes that record is then demonstrably its back half, and both
    count as retention damage rather than contract violations.
    """
    stitched = (
        b'{"a": 1}\n{"cut": "val\n'
        + CODEX_ELIDED_MARKER_PREFIX
        + b"4096 bytes elided ...]\n"
        + b"not json}\n"
        + b'{"b": 2}\n'
    )

    parsed = _parse_jsonl_events(stitched, truncated=True)

    assert parsed.events == ({"a": 1}, {"b": 2})
    assert parsed.dropped_partial_lines == 2
