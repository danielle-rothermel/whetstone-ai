"""Contract tests for the thin lm boundary adapter.

Wire mechanics (payload building, parsing, transport, classification)
are tested in dr-providers; these cover whetstone's adapter surface:
caller parameters -> Provider Call Request, Provider Transport Response ->
ProviderResult, and kernel failure translation.
"""

from __future__ import annotations

import pytest
from dr_providers import (
    CostInfo,
    FailureClass,
    MessageRole,
    PromptMessage,
    ProviderTransportResponse,
    ProviderTransportWarning,
    RateLimitedProviderError,
    ReasoningEffort,
    ResponsesDiagnostics,
    TokenUsage,
    build_payload,
    failure_record,
    openai_responses_config,
    openrouter_chat_config,
    raise_failure,
)

from whetstone.provider.failures import (
    EmptyGenerationError,
    PermanentFailureError,
    RateLimitedFailureError,
)
from whetstone.provider.language_model import (
    PlainPromptAdapter,
    StructuredPromptAdapter,
    provider_call_request_from_parameters,
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


class TestStructuredPromptAdapter:
    def test_mixed_messages_preserve_exact_wire_and_identity_order(
        self,
    ) -> None:
        messages = StructuredPromptAdapter().messages_from_records(
            (
                {"role": "system", "content": "follow the format"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe this"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://example.test/image.png",
                                "detail": "high",
                            },
                        },
                    ],
                },
                {"role": "assistant", "content": "description"},
            )
        )

        expected = [
            {"role": "system", "content": "follow the format"},
            {
                "role": "user",
                "content": (
                    {"type": "text", "text": "describe this"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://example.test/image.png",
                            "detail": "high",
                        },
                    },
                ),
            },
            {"role": "assistant", "content": "description"},
        ]
        assert [message.provider_dict() for message in messages] == expected
        assert [message.identity_payload() for message in messages] == expected

    @pytest.mark.parametrize(
        "record",
        [
            {"role": "user"},
            {"content": "hello"},
            {"role": "user", "content": "hello", "name": "caller"},
        ],
    )
    def test_rejects_missing_or_extra_record_keys(
        self, record: dict[str, object]
    ) -> None:
        with pytest.raises(
            ValueError,
            match="structured prompt messages require role and content",
        ):
            StructuredPromptAdapter().messages_from_records((record,))

    def test_rejects_invalid_role(self) -> None:
        with pytest.raises(
            ValueError, match="'invalid' is not a valid MessageRole"
        ):
            StructuredPromptAdapter().messages_from_records(
                ({"role": "invalid", "content": "hello"},)
            )

    def test_rejects_empty_records(self) -> None:
        with pytest.raises(
            ValueError, match="structured prompt messages cannot be empty"
        ):
            StructuredPromptAdapter().messages_from_records(())

    def test_rejects_empty_content_parts(self) -> None:
        with pytest.raises(
            ValueError,
            match="structured prompt content must be text or content parts",
        ):
            StructuredPromptAdapter().messages_from_records(
                ({"role": "user", "content": []},)
            )

    def test_rejects_non_object_content_parts(self) -> None:
        with pytest.raises(
            ValueError,
            match="structured prompt content parts must be objects",
        ):
            StructuredPromptAdapter().messages_from_records(
                (
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "hello"}, "bad"],
                    },
                )
            )

    def test_nested_source_mutation_cannot_change_wire_or_identity(
        self,
    ) -> None:
        nested_image = {
            "url": "https://example.test/original.png",
            "metadata": {"labels": ["original"]},
        }
        records = (
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": nested_image},
                ],
            },
        )
        message = StructuredPromptAdapter().messages_from_records(records)[0]
        provider_payload = message.provider_dict()
        identity_payload = message.identity_payload()

        nested_image["url"] = "https://example.test/mutated.png"
        nested_image["metadata"]["labels"].append("mutated")

        expected = {
            "role": "user",
            "content": (
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.test/original.png",
                        "metadata": {"labels": ["original"]},
                    },
                },
            ),
        }
        assert provider_payload == expected
        assert identity_payload == expected
        assert message.provider_dict() == expected
        assert message.identity_payload() == expected


class TestProviderCallRequestFromParameters:
    def test_maps_parameters_into_config_controls(self) -> None:
        request = provider_call_request_from_parameters(
            config=openrouter_chat_config(model="m"),
            messages=(PromptMessage(role=MessageRole.USER, content="hi"),),
            parameters={
                "temperature": 0.2,
                "token_limit": 10,
                "reasoning": "low",
                "extra_body": {"a": 1},
            },
        )
        controls = request.config.controls
        assert controls.temperature == 0.2
        assert controls.token_limit == 10
        assert controls.reasoning is ReasoningEffort.LOW
        assert request.config.extensions.extra_body == {"a": 1}
        payload = build_payload(request)
        assert payload["max_completion_tokens"] == 10
        assert payload["a"] == 1

    def test_rejects_unrecognized_reasoning_effort(self) -> None:
        with pytest.raises(ValueError, match="invalid reasoning effort"):
            provider_call_request_from_parameters(
                config=openrouter_chat_config(model="m"),
                messages=(PromptMessage(role=MessageRole.USER, content="hi"),),
                parameters={"reasoning": {"effort": "low"}},
            )

    def test_absent_parameters_stay_unset(self) -> None:
        request = provider_call_request_from_parameters(
            config=openai_responses_config(model="m"),
            messages=(PromptMessage(role=MessageRole.USER, content="hi"),),
            parameters={},
        )
        controls = request.config.controls
        assert controls.temperature is None
        assert controls.token_limit is None
        assert controls.reasoning is None
        payload = build_payload(request)
        assert "temperature" not in payload
        assert "max_output_tokens" not in payload


class TestProviderResultFromResponse:
    def test_maps_parts_to_record_fields(self) -> None:
        response = ProviderTransportResponse(
            text="hello",
            cost=CostInfo(total_cost=0.02),
            finish_reason="stop",
            response_id="resp-1",
            model="m-actual",
            usage=TokenUsage(total_tokens=3),
            raw_body={"id": "resp-1"},
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
        response = ProviderTransportResponse(
            text="hello",
            warnings=(
                ProviderTransportWarning(
                    code="model_substitution", message="swapped"
                ),
            ),
        )
        result = provider_result_from_response(response)
        recorded = result.response_metadata["conformance_warnings"]
        assert recorded[0]["code"] == "model_substitution"

    def test_response_diagnostics_ride_in_metadata(self) -> None:
        response = ProviderTransportResponse(
            text="hello",
            diagnostics=ResponsesDiagnostics(
                response_status="completed",
                output_item_types={"message": 1},
                content_part_types={"output_text": 1},
                output_text_len=5,
                response_id_hash="abc123",
            ),
        )

        result = provider_result_from_response(response)

        assert result.response_metadata["diagnostics"] == {
            "response_status": "completed",
            "incomplete_reason": None,
            "output_item_types": {"message": 1},
            "content_part_types": {"output_text": 1},
            "output_text_len": 5,
            "refusal_len": None,
            "response_id_hash": "abc123",
        }

    def test_blank_text_raises_empty_generation(self) -> None:
        response = ProviderTransportResponse(text="   ")
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
