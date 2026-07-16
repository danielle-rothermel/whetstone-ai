from __future__ import annotations

from typing import Any

import pytest
from dr_serialize import (
    JsonEncodeError,
    MaxDepthExceededError,
    ModelDumpError,
    PayloadTooLargeError,
)

from whetstone.db.io import experiment_row
from whetstone.eval_failures import (
    FailureClass,
    RecordingFailureError,
    ensure_recordable,
    failure_metadata_dict_from_exception,
    failure_metadata_from_exception,
    recordable_text,
    should_retry_step,
    summarize_exception,
)
from whetstone.eval_failures.exceptions import TransientFailureError
from whetstone.records import ExperimentRecord


def test_recordable_text_passthrough_str() -> None:
    assert recordable_text("hello") == "hello"


def test_recordable_text_canonicalizes_dict() -> None:
    assert recordable_text({"code": "x"}) == '{"code":"x"}'


def test_recordable_text_propagates_encode_error() -> None:
    with pytest.raises(JsonEncodeError):
        recordable_text({"bad": object()})


def test_ensure_recordable_wraps_encode_error() -> None:
    with pytest.raises(RecordingFailureError) as exc_info:
        ensure_recordable({"bad": object()})
    assert exc_info.value.underlying is not None
    assert isinstance(exc_info.value.underlying, JsonEncodeError)


def test_jsonb_row_rejects_unserializable_record_value() -> None:
    unserializable = object()
    record = ExperimentRecord(
        experiment_name="payload-boundary",
        config_metadata={"value": unserializable},
    )

    assert record.config_metadata["value"] is unserializable
    with pytest.raises(RecordingFailureError):
        experiment_row(record)


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
    metadata = failure_metadata_dict_from_exception(error)
    assert metadata["path"] == ["x"]
    assert metadata["type_name"] == "object"


def test_failure_metadata_from_exception_builds_payload() -> None:
    error = TransientFailureError("provider timeout")
    payload = failure_metadata_from_exception(error)
    assert payload.failure_class is FailureClass.TRANSIENT
    assert payload.error_type.endswith("TransientFailureError")
    assert payload.message == "provider timeout"


def test_ensure_recordable_wraps_payload_too_large_error() -> None:
    with pytest.raises(RecordingFailureError) as exc_info:
        ensure_recordable({"data": "x" * 500}, max_bytes=50)
    assert isinstance(exc_info.value.underlying, PayloadTooLargeError)


def test_ensure_recordable_wraps_model_dump_error() -> None:
    from tests.serialization_support import BadModel

    with pytest.raises(RecordingFailureError) as exc_info:
        ensure_recordable(BadModel(x=object()))
    assert isinstance(exc_info.value.underlying, ModelDumpError)
