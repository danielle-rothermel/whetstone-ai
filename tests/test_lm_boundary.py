from __future__ import annotations

from typing import Any

import pytest

import dspy
from dr_dspy.eval_failures import (
    EmptyGenerationError,
    FailureClass,
    ProviderResponseParseError,
    RateLimitedFailureError,
    summarize_exception,
)
from dr_dspy.lm.boundary import (
    EndpointKind,
    PlainPromptAdapter,
    ProviderConfig,
    ProviderKind,
    ReasoningRequestShape,
    TokenLimitParameter,
    build_chat_completions_request,
    build_responses_request,
    call_provider_request,
    openai_chat_config,
    openai_responses_config,
    openrouter_chat_config,
    parse_provider_response,
)
from dr_dspy.lm.openrouter import LoggingOpenRouterLM
from dr_dspy.lm.utils import LmEventBuffer


def test_plain_prompt_adapter_builds_exact_user_message() -> None:
    adapter = PlainPromptAdapter(output_field="code")

    messages = adapter.messages(user_content="solve this")

    assert [message.provider_dict() for message in messages] == [
        {"role": "user", "content": "solve this"}
    ]


def test_plain_prompt_adapter_builds_exact_system_and_user_messages() -> None:
    adapter = PlainPromptAdapter(output_field="code")

    messages = adapter.messages(
        system_content="You write Python.",
        user_content="solve this",
    )

    assert [message.provider_dict() for message in messages] == [
        {"role": "system", "content": "You write Python."},
        {"role": "user", "content": "solve this"},
    ]
    assert "[[ ##" not in repr(messages)


def test_plain_prompt_adapter_returns_configured_output_field() -> None:
    adapter = PlainPromptAdapter(output_field="code")
    result = parse_provider_response(
        {
            "id": "cmpl-1",
            "model": "model/test",
            "choices": [
                {
                    "message": {"content": "def f(): pass"},
                    "finish_reason": "stop",
                }
            ],
        },
        config=openrouter_chat_config(model="model/test"),
    )

    assert adapter.output_from_result(result) == {"code": "def f(): pass"}


def test_openrouter_request_places_reasoning_in_extra_body() -> None:
    request = build_chat_completions_request(
        config=openrouter_chat_config(model="model/test"),
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.2,
        token_limit=12,
        reasoning={"effort": "low"},
        extra_body={"provider": {"order": ["OpenAI"]}},
        extra_kwargs={"seed": 7},
    )

    assert request.kwargs == {
        "model": "model/test",
        "messages": [{"role": "user", "content": "hello"}],
        "seed": 7,
        "temperature": 0.2,
        "max_completion_tokens": 12,
        "extra_body": {
            "provider": {"order": ["OpenAI"]},
            "reasoning": {"effort": "low"},
        },
    }


def test_chat_request_suppresses_unsupported_temperature() -> None:
    config = ProviderConfig(
        provider_kind=ProviderKind.OPENAI,
        endpoint_kind=EndpointKind.CHAT_COMPLETIONS,
        model="o-test",
        api_key_env="OPENAI_API_KEY",
        temperature_supported=False,
        reasoning_shape=ReasoningRequestShape.TOP_LEVEL,
        token_limit_parameter=TokenLimitParameter.MAX_COMPLETION_TOKENS,
    )

    request = build_chat_completions_request(
        config=config,
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.7,
        token_limit=10,
        reasoning={"effort": "low"},
    )

    assert "temperature" not in request.kwargs
    assert request.kwargs["reasoning"] == {"effort": "low"}
    assert request.kwargs["max_completion_tokens"] == 10


def test_openai_chat_request_uses_boundary_config() -> None:
    request = build_chat_completions_request(
        config=openai_chat_config(model="gpt-test"),
        messages=[{"role": "user", "content": "hello"}],
        token_limit=10,
    )

    assert request.provider_kind is ProviderKind.OPENAI
    assert request.endpoint_kind is EndpointKind.CHAT_COMPLETIONS
    assert request.kwargs["model"] == "gpt-test"
    assert request.kwargs["max_completion_tokens"] == 10


def test_openai_responses_request_maps_system_to_instructions() -> None:
    request = build_responses_request(
        config=openai_responses_config(model="gpt-test"),
        messages=[
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "hello"},
        ],
        temperature=0.3,
        token_limit=10,
        reasoning={"effort": "low"},
    )

    assert request.kwargs == {
        "model": "gpt-test",
        "instructions": "Be brief.",
        "input": [{"role": "user", "content": "hello"}],
        "temperature": 0.3,
        "max_output_tokens": 10,
        "reasoning": {"effort": "low"},
    }


def test_parse_chat_completion_extracts_text_usage_and_cost() -> None:
    result = parse_provider_response(
        {
            "id": "cmpl-1",
            "model": "provider-model",
            "choices": [
                {
                    "message": {"content": [{"text": "ok"}]},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "total_tokens": 3,
                "cost": 0.01,
            },
        },
        config=openrouter_chat_config(model="model/test"),
    )

    assert result.text == "ok"
    assert result.usage_metadata["total_tokens"] == 3
    assert result.provider_cost == 0.01
    assert result.response_id == "cmpl-1"
    assert result.model == "provider-model"
    assert result.finish_reason == "stop"


def test_parse_chat_completion_allows_absent_cost() -> None:
    result = parse_provider_response(
        {
            "choices": [
                {"message": {"content": "ok"}, "finish_reason": "stop"}
            ],
            "usage": {"total_tokens": 3},
        },
        config=openrouter_chat_config(model="model/test"),
    )

    assert result.provider_cost is None
    assert result.usage_metadata == {"total_tokens": 3}


def test_parse_empty_chat_completion_text_raises_empty_generation() -> None:
    with pytest.raises(EmptyGenerationError):
        parse_provider_response(
            {"choices": [{"message": {"content": ""}}]},
            config=openrouter_chat_config(model="model/test"),
            output_field="code",
        )


def test_parse_malformed_chat_completion_raises_parse_failure() -> None:
    with pytest.raises(ProviderResponseParseError):
        parse_provider_response(
            {"model": "model/test"},
            config=openrouter_chat_config(model="model/test"),
        )


def test_parse_responses_response_extracts_output_text() -> None:
    result = parse_provider_response(
        {
            "id": "resp-1",
            "model": "gpt-test",
            "status": "completed",
            "output_text": "ok",
            "usage": {"input_tokens": 1, "output_tokens": 2},
        },
        config=openai_responses_config(model="gpt-test"),
    )

    assert result.text == "ok"
    assert result.finish_reason == "completed"
    assert result.usage_metadata == {"input_tokens": 1, "output_tokens": 2}


class _FakeCompletions:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> dict[str, Any]:
        self.kwargs = kwargs
        return {"choices": [{"message": {"content": "ok"}}]}


class _FakeResponses:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> dict[str, Any]:
        self.kwargs = kwargs
        return {"output_text": "ok"}


class _FakeClient:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()
        self.responses = _FakeResponses()
        self.chat: Any = type(
            "FakeChat",
            (),
            {"completions": self.completions},
        )()


def test_call_provider_request_uses_chat_client() -> None:
    client = _FakeClient()
    request = build_chat_completions_request(
        config=openrouter_chat_config(model="model/test"),
        messages=[{"role": "user", "content": "hello"}],
    )

    response = call_provider_request(client, request)

    assert response == {"choices": [{"message": {"content": "ok"}}]}
    assert client.completions.kwargs == request.kwargs


def test_call_provider_request_uses_responses_client() -> None:
    client = _FakeClient()
    request = build_responses_request(
        config=openai_responses_config(model="gpt-test"),
        messages=[{"role": "user", "content": "hello"}],
    )

    response = call_provider_request(client, request)

    assert response == {"output_text": "ok"}
    assert client.responses.kwargs == request.kwargs


def test_provider_error_translation_preserves_rate_limit_cause() -> None:
    import httpx
    import openai

    request = build_chat_completions_request(
        config=openrouter_chat_config(model="model/test"),
        messages=[{"role": "user", "content": "hello"}],
    )

    class RateLimitedCompletions:
        def create(self, **kwargs: Any) -> None:
            response = httpx.Response(
                429,
                request=httpx.Request("POST", "https://example.test"),
            )
            raise openai.RateLimitError(
                "limited",
                response=response,
                body=None,
            )

    client = _FakeClient()
    client.chat = type(
        "FakeChat",
        (),
        {"completions": RateLimitedCompletions()},
    )()

    with pytest.raises(RateLimitedFailureError) as exc_info:
        call_provider_request(client, request)

    assert isinstance(exc_info.value.underlying, openai.RateLimitError)
    summary = summarize_exception(exc_info.value)
    assert summary.failure_class is FailureClass.RATE_LIMITED


def test_logging_openrouter_lm_uses_provider_boundary() -> None:
    client = _FakeClient()
    events = LmEventBuffer()
    lm = LoggingOpenRouterLM(
        "model/test",
        log=events.put_event,
        client=client,
        cache=False,
        max_completion_tokens=12,
        reasoning={"enabled": False},
    )
    request = dspy.LMRequest.from_call(
        model="model/test",
        prompt="hello",
        max_completion_tokens=12,
        cache=False,
    )

    response = lm.forward(request)

    assert lm.forward_contract == "typed_lm"
    assert isinstance(response, dspy.LMResponse)
    assert response.text == "ok"
    assert client.completions.kwargs == {
        "model": "model/test",
        "messages": [{"role": "user", "content": "hello"}],
        "max_completion_tokens": 12,
        "extra_body": {"reasoning": {"enabled": False}},
    }
    assert events.latest_response_text() == "ok"
