from __future__ import annotations

import pytest

from whetstone.provider.failures import (
    EmptyProviderGenerationError,
    FailureClass,
    PermanentFailureError,
    failure_metadata_dict_from_exception,
    should_retry_step,
    summarize_exception,
)
from whetstone.provider.language_model import require_provider_generation_text


@pytest.mark.parametrize("text", [None, "", "   "])
def test_require_provider_generation_text_rejects_empty(
    text: str | None,
) -> None:
    with pytest.raises(EmptyProviderGenerationError) as exc_info:
        require_provider_generation_text(text, output_field="code")
    assert exc_info.value.metadata["output_field"] == "code"


def test_require_provider_generation_text_returns_non_empty_text() -> None:
    assert require_provider_generation_text(
        "def f(): pass", output_field="code"
    ) == ("def f(): pass")


def test_summarize_empty_generation_failure_is_permanent() -> None:
    error = EmptyProviderGenerationError(
        "empty generation for output field 'code'",
        metadata={"output_field": "code"},
    )
    summary = summarize_exception(error)
    assert summary.failure_class is FailureClass.PERMANENT
    assert should_retry_step(error) is False
    assert summary.failure_metadata["output_field"] == "code"
    assert "EmptyProviderGenerationError" in summary.failure_exception_type


def test_summarize_permanent_failure_preserves_underlying() -> None:
    error = PermanentFailureError(
        "parse failed for output field 'code'",
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
    error = PermanentFailureError(
        "parse failed",
        underlying=ValueError("bad"),
        metadata={"output_field": "description"},
    )
    metadata = failure_metadata_dict_from_exception(error)
    assert metadata == {"output_field": "description"}
