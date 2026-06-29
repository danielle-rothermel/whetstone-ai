from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import dspy
from dr_dspy.dspy_runner import run_predictor
from dr_dspy.eval_failures import (
    EmptyGenerationError,
    FailureClass,
    PredictionParseError,
    failure_metadata_from_exception,
    require_generation_text,
    should_retry_step,
    summarize_exception,
)
from dr_dspy.lm_utils import LmEventBuffer


@pytest.mark.parametrize("text", [None, "", "   "])
def test_require_generation_text_rejects_empty(text: str | None) -> None:
    with pytest.raises(EmptyGenerationError) as exc_info:
        require_generation_text(text, output_field="code")
    assert exc_info.value.metadata["output_field"] == "code"


def test_require_generation_text_returns_non_empty_text() -> None:
    assert require_generation_text("def f(): pass", output_field="code") == (
        "def f(): pass"
    )


def test_summarize_empty_generation_failure_is_permanent() -> None:
    error = EmptyGenerationError(
        "empty generation for output field 'code'",
        metadata={"output_field": "code"},
    )
    summary = summarize_exception(error)
    assert summary.failure_class is FailureClass.PERMANENT
    assert should_retry_step(error) is False
    assert summary.failure_metadata["output_field"] == "code"
    assert "EmptyGenerationError" in summary.failure_exception_type


def test_summarize_prediction_parse_failure_preserves_underlying() -> None:
    error = PredictionParseError(
        "predictor failed for output field 'code'",
        underlying=ValueError("invalid output"),
        metadata={
            "output_field": "code",
            "lm_response_preview": "not valid python",
        },
    )
    summary = summarize_exception(error)
    assert summary.failure_class is FailureClass.PERMANENT
    assert should_retry_step(error) is False
    assert summary.underlying_exception_type.endswith("ValueError")
    preview = summary.failure_metadata["lm_response_preview"]
    assert preview == "not valid python"


def test_failure_metadata_from_eval_failure_error() -> None:
    error = PredictionParseError(
        "parse failed",
        underlying=ValueError("bad"),
        metadata={"output_field": "description"},
    )
    metadata = failure_metadata_from_exception(error)
    assert metadata == {"output_field": "description"}


class _StubPrediction:
    def __init__(self, **fields: str) -> None:
        for name, value in fields.items():
            setattr(self, name, value)


@contextmanager
def _stub_predict(
    *,
    side_effect: BaseException | None = None,
    prediction: Any = None,
) -> Iterator[MagicMock]:
    predict_call = MagicMock()
    if side_effect is not None:
        predict_call.side_effect = side_effect
    else:
        predict_call.return_value = prediction
    context_manager = MagicMock()
    context_manager.__enter__.return_value = None
    context_manager.__exit__.return_value = False
    with patch("dr_dspy.dspy_runner.dspy.Predict", return_value=predict_call):
        context_patch = patch(
            "dr_dspy.dspy_runner.dspy.context",
            return_value=context_manager,
        )
        with context_patch:
            yield predict_call


def test_run_predictor_raises_prediction_parse_error() -> None:
    event_buffer = LmEventBuffer()
    event_buffer.put_event(
        "lm.response",
        payload={"response": {"choices": [{"message": {"content": "raw"}}]}},
    )
    with _stub_predict(side_effect=ValueError("parse failed")):
        with pytest.raises(PredictionParseError) as exc_info:
            run_predictor(
                signature=MagicMock(spec=dspy.Signature),
                input_kwargs={"prompt": "hi"},
                output_field="code",
                lm=MagicMock(spec=dspy.BaseLM),
                event_buffer=event_buffer,
            )
    assert exc_info.value.metadata["output_field"] == "code"
    assert exc_info.value.metadata["lm_response_preview"] == "raw"
    assert isinstance(exc_info.value.underlying, ValueError)


def test_run_predictor_raises_empty_generation_error() -> None:
    event_buffer = LmEventBuffer()
    with _stub_predict(prediction=_StubPrediction(code="   ")):
        with pytest.raises(EmptyGenerationError):
            run_predictor(
                signature=MagicMock(spec=dspy.Signature),
                input_kwargs={"prompt": "hi"},
                output_field="code",
                lm=MagicMock(spec=dspy.BaseLM),
                event_buffer=event_buffer,
            )
