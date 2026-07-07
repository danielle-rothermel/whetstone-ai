from __future__ import annotations

import pytest

from whetstone.eval_failures import (
    EmptyGenerationError,
    FailureClass,
    PredictionParseError,
    failure_metadata_dict_from_exception,
    require_generation_text,
    should_retry_step,
    summarize_exception,
    validate_direct_generation,
    validate_encdec_generation,
)


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
    metadata = failure_metadata_dict_from_exception(error)
    assert metadata == {"output_field": "description"}


def test_validate_encdec_generation_rejects_empty_description() -> None:
    with pytest.raises(EmptyGenerationError) as exc_info:
        validate_encdec_generation(description="", code="def f(): pass")
    assert exc_info.value.metadata["output_field"] == "description"


def test_validate_encdec_generation_rejects_empty_code() -> None:
    with pytest.raises(EmptyGenerationError) as exc_info:
        validate_encdec_generation(description="describe task", code="  ")
    assert exc_info.value.metadata["output_field"] == "code"


def test_validate_encdec_generation_accepts_non_empty_outputs() -> None:
    validate_encdec_generation(
        description="describe task",
        code="def f(): pass",
    )


def test_validate_direct_generation_rejects_empty_code() -> None:
    with pytest.raises(EmptyGenerationError) as exc_info:
        validate_direct_generation(code="")
    assert exc_info.value.metadata["output_field"] == "code"


def test_validate_direct_generation_accepts_non_empty_code() -> None:
    validate_direct_generation(code="def f(): pass")
