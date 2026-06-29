from __future__ import annotations

from typing import Any

import pytest

from dr_dspy.eval_failures import (
    FailureClass,
    RecordingFailureError,
    ensure_recordable,
    failure_metadata_from_exception,
    should_retry_step,
    summarize_exception,
)
from dr_dspy.serialization import JsonEncodeError, MaxDepthExceededError


def test_ensure_recordable_wraps_encode_error() -> None:
    with pytest.raises(RecordingFailureError) as exc_info:
        ensure_recordable({"bad": object()})
    assert exc_info.value.underlying is not None
    assert isinstance(exc_info.value.underlying, JsonEncodeError)


def test_ensure_recordable_wraps_depth_error() -> None:
    nested: list[Any] = []
    current: list[Any] = nested
    for _ in range(101):
        inner: list[Any] = []
        current.append(inner)
        current = inner
    with pytest.raises(RecordingFailureError) as exc_info:
        ensure_recordable(nested)
    assert isinstance(exc_info.value.underlying, MaxDepthExceededError)


def test_summarize_recording_failure_is_permanent() -> None:
    error = RecordingFailureError(
        "not JSON-serializable",
        underlying=JsonEncodeError(
            path=("bad",),
            type_name="object",
            detail="...",
            underlying=TypeError("object"),
            value_preview="...",
        ),
    )
    summary = summarize_exception(error)
    assert summary.failure_class is FailureClass.PERMANENT
    assert should_retry_step(error) is False
    assert summary.failure_metadata["type_name"] == "object"
    assert "RecordingFailureError" in summary.failure_exception_type


def test_failure_metadata_from_wrapped_error() -> None:
    underlying = JsonEncodeError(
        path=("x",),
        type_name="object",
        detail="detail",
        underlying=TypeError("object"),
        value_preview="preview",
    )
    error = RecordingFailureError("wrapped", underlying=underlying)
    metadata = failure_metadata_from_exception(error)
    assert metadata["path"] == ["x"]
    assert metadata["type_name"] == "object"
