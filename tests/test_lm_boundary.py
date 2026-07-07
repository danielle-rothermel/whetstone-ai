"""Contract tests for the thin lm boundary adapter.

Wire mechanics (payload building, parsing, transport, classification)
are tested in dr-providers; these cover whetstone's adapter surface:
node parameters → LlmRequest, LlmResponse → ProviderResult, and kernel
failure translation.
"""

from __future__ import annotations

import pytest
from dr_providers.kernel import (
    CostInfo,
    FailureClass,
    LlmResponse,
    LlmWarning,
    MessageRole,
    PromptMessage,
    RateLimitedProviderError,
    build_payload,
    failure_record,
    openai_responses_config,
    openrouter_chat_config,
    raise_failure,
)

from whetstone.eval_failures import (
    EmptyGenerationError,
    PermanentFailureError,
    RateLimitedFailureError,
)
from whetstone.lm.boundary import (
    PlainPromptAdapter,
    llm_request_for_node,
    provider_result_from_response,
    translate_provider_failure,
)


class TestPlainPromptAdapter:
    def test_messages_with_system(self) -> None:
        adapter = PlainPromptAdapter()
        messages = adapter.messages(
            user_content="write add", system_content="be brief"
        )
        assert [m.role for m in messages] == [
            MessageRole.SYSTEM,
            MessageRole.USER,
        ]

    def test_messages_without_system(self) -> None:
        messages = PlainPromptAdapter().messages(user_content="write add")
        assert len(messages) == 1
        assert messages[0].role is MessageRole.USER


class TestLlmRequestForNode:
    def test_maps_parameters_and_merges_extra_kwargs(self) -> None:
        request = llm_request_for_node(
            config=openrouter_chat_config(model="m"),
            messages=(
                PromptMessage(role=MessageRole.USER, content="hi"),
            ),
            parameters={
                "temperature": 0.2,
                "token_limit": 10,
                "reasoning": {"effort": "low"},
                "extra_body": {"a": 1},
                "extra_kwargs": {"b": 2},
            },
            idempotency_key="attempt-1",
        )
        assert request.temperature == 0.2
        assert request.token_limit == 10
        assert request.reasoning == {"effort": "low"}
        assert request.extra_body == {"a": 1, "b": 2}
        assert request.idempotency_key == "attempt-1"
        payload = build_payload(request)
        assert payload["max_completion_tokens"] == 10
        assert payload["a"] == 1
        assert payload["b"] == 2

    def test_absent_parameters_stay_unset(self) -> None:
        request = llm_request_for_node(
            config=openai_responses_config(model="m"),
            messages=(
                PromptMessage(role=MessageRole.USER, content="hi"),
            ),
            parameters={},
        )
        assert request.temperature is None
        assert request.token_limit is None
        assert request.reasoning == {}
        payload = build_payload(request)
        assert "temperature" not in payload
        assert "max_output_tokens" not in payload


class TestProviderResultFromResponse:
    def test_maps_parts_to_record_fields(self) -> None:
        response = LlmResponse(
            text="hello",
            cost=CostInfo(total_cost=0.02),
            finish_reason="stop",
            response_id="resp-1",
            model="m-actual",
            provider_metadata={
                "id": "resp-1",
                "usage": {"total_tokens": 3},
            },
        )
        result = provider_result_from_response(response)
        assert result.text == "hello"
        assert result.usage_metadata == {"total_tokens": 3}
        assert result.provider_cost == 0.02
        assert result.response_id == "resp-1"
        assert result.model == "m-actual"
        assert result.finish_reason == "stop"
        assert result.response_metadata["id"] == "resp-1"

    def test_conformance_warnings_ride_in_metadata(self) -> None:
        response = LlmResponse(
            text="hello",
            warnings=(
                LlmWarning(code="model_substitution", message="swapped"),
            ),
        )
        result = provider_result_from_response(response)
        recorded = result.response_metadata["conformance_warnings"]
        assert recorded[0]["code"] == "model_substitution"

    def test_blank_text_raises_empty_generation(self) -> None:
        response = LlmResponse(text="   ")
        with pytest.raises(EmptyGenerationError):
            provider_result_from_response(response, output_field="code")


class TestTranslateProviderFailure:
    def test_rate_limited_maps_to_rate_limited_eval_failure(self) -> None:
        failure = failure_record(
            failure_class=FailureClass.RATE_LIMITED,
            code="http_status_429",
            message="slow down",
        )
        carrier = raise_failure(failure)
        assert isinstance(carrier, RateLimitedProviderError)
        translated = translate_provider_failure(carrier)
        assert isinstance(translated, RateLimitedFailureError)
        assert translated.underlying is carrier
        assert (
            translated.metadata["provider_failure"]["code"]
            == "http_status_429"
        )

    def test_permanent_maps_to_permanent_eval_failure(self) -> None:
        carrier = raise_failure(
            failure_record(
                failure_class=FailureClass.PERMANENT,
                message="bad request",
            )
        )
        translated = translate_provider_failure(carrier)
        assert isinstance(translated, PermanentFailureError)
