from __future__ import annotations

from typing import Any

import pytest

from dr_dspy.eval_failures import RecordingFailureError
from dr_dspy.lm.logging import _LoggingMixin


class _StubLoggingMixin(_LoggingMixin):
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def _log(self, event_type: str, **kwargs: Any) -> None:
        self.events.append((event_type, kwargs))


def test_log_response_propagates_recording_failure() -> None:
    mixin = _StubLoggingMixin()
    with pytest.raises(RecordingFailureError):
        mixin._log_response("req", {"bad": object()}, 0.1)
    assert mixin.events == []


def test_run_logged_forward_propagates_response_recording_failure() -> None:
    mixin = _StubLoggingMixin()

    def forward() -> dict[str, object]:
        return {"bad": object()}

    with pytest.raises(RecordingFailureError):
        mixin._run_logged_forward(
            forward,
            messages=[{"role": "user", "content": "hi"}],
            kwargs={},
        )
    assert [event for event, _ in mixin.events if event == "lm.request"]
    assert not any(event == "lm.response" for event, _ in mixin.events)
