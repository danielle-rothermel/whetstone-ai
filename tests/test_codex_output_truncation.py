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

from whetstone.optim.codex.runner import (
    CODEX_ELIDED_MARKER_PREFIX,
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
