"""Typed prompt and provider boundary helpers for LM calls."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr

from dr_dspy.eval_failures.exceptions import (
    EvalFailureError,
    ProviderResponseParseError,
    failure_exception_type_for_class,
)
from dr_dspy.eval_failures.generation import require_generation_text
from dr_dspy.eval_failures.policy import classify_exception
from dr_dspy.lm.utils import content_to_text, provider_cost_from_response

OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OUTPUT_FIELD_TEXT = "text"

CHAT_COMPLETIONS_METHOD = "chat.completions.create"
RESPONSES_METHOD = "responses.create"

__all__ = [
    "OPENAI_API_KEY_ENV",
    "OPENROUTER_API_KEY_ENV",
    "OPENROUTER_BASE_URL",
    "EndpointKind",
    "MessageRole",
    "PlainPromptAdapter",
    "PromptMessage",
    "ProviderConfig",
    "ProviderKind",
    "ProviderRequest",
    "ProviderResult",
    "ReasoningRequestShape",
    "TokenLimitParameter",
    "build_chat_completions_request",
    "build_responses_request",
    "call_provider_request",
    "message_dicts",
    "openai_chat_config",
    "openai_responses_config",
    "openrouter_chat_config",
    "parse_chat_completion_response",
    "parse_provider_response",
    "parse_responses_response",
    "translate_provider_error",
]


class ProviderKind(StrEnum):
    OPENROUTER = "openrouter"
    OPENAI = "openai"


class EndpointKind(StrEnum):
    CHAT_COMPLETIONS = "chat_completions"
    RESPONSES = "responses"


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class TokenLimitParameter(StrEnum):
    MAX_TOKENS = "max_tokens"
    MAX_COMPLETION_TOKENS = "max_completion_tokens"
    MAX_OUTPUT_TOKENS = "max_output_tokens"


class ReasoningRequestShape(StrEnum):
    NONE = "none"
    EXTRA_BODY = "extra_body"
    TOP_LEVEL = "top_level"


class PromptMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: MessageRole
    content: StrictStr

    def provider_dict(self) -> dict[str, str]:
        return {"role": self.role.value, "content": self.content}


class ProviderConfig(BaseModel):
    """Runtime provider call configuration.

    Graph workflow persistence stores a narrower provider ref (kind, endpoint,
    model, throttle_key, parameters). Reconstruct full runtime config from
    pinned spec fields via template helpers such as ``openrouter_chat_config``.
    """

    model_config = ConfigDict(extra="forbid")

    provider_kind: ProviderKind
    endpoint_kind: EndpointKind
    model: StrictStr
    api_key_env: StrictStr
    base_url: StrictStr | None = None
    temperature_supported: StrictBool = True
    reasoning_supported: StrictBool = True
    reasoning_shape: ReasoningRequestShape = ReasoningRequestShape.NONE
    token_limit_parameter: TokenLimitParameter
    extra_body: dict[str, Any] = Field(default_factory=dict)
    throttle_key: StrictStr | None = None

    @property
    def throttle_identity(self) -> str:
        if self.throttle_key:
            return self.throttle_key
        return ":".join(
            (self.provider_kind.value, self.endpoint_kind.value, self.model)
        )


class ProviderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider_kind: ProviderKind
    endpoint_kind: EndpointKind
    method: StrictStr
    kwargs: dict[str, Any]
    throttle_key: StrictStr


class ProviderResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: StrictStr
    response_metadata: dict[str, Any] = Field(default_factory=dict)
    usage_metadata: dict[str, Any] = Field(default_factory=dict)
    provider_cost: float | None = None
    response_id: StrictStr | None = None
    model: StrictStr | None = None
    finish_reason: StrictStr | None = None


class PlainPromptAdapter(BaseModel):
    """Minimal prompt adapter with no hidden DSPy formatting."""

    model_config = ConfigDict(extra="forbid")

    output_field: StrictStr = OUTPUT_FIELD_TEXT

    def messages(
        self,
        *,
        user_content: str,
        system_content: str | None = None,
    ) -> tuple[PromptMessage, ...]:
        messages: list[PromptMessage] = []
        if system_content is not None:
            messages.append(
                PromptMessage(
                    role=MessageRole.SYSTEM,
                    content=system_content,
                )
            )
        messages.append(
            PromptMessage(role=MessageRole.USER, content=user_content)
        )
        return tuple(messages)

    def output_from_result(self, result: ProviderResult) -> dict[str, str]:
        return {self.output_field: result.text}


def openrouter_chat_config(
    *,
    model: str,
    base_url: str = OPENROUTER_BASE_URL,
) -> ProviderConfig:
    return ProviderConfig(
        provider_kind=ProviderKind.OPENROUTER,
        endpoint_kind=EndpointKind.CHAT_COMPLETIONS,
        model=model,
        api_key_env=OPENROUTER_API_KEY_ENV,
        base_url=base_url,
        reasoning_shape=ReasoningRequestShape.EXTRA_BODY,
        token_limit_parameter=TokenLimitParameter.MAX_COMPLETION_TOKENS,
    )


def openai_chat_config(*, model: str) -> ProviderConfig:
    return ProviderConfig(
        provider_kind=ProviderKind.OPENAI,
        endpoint_kind=EndpointKind.CHAT_COMPLETIONS,
        model=model,
        api_key_env=OPENAI_API_KEY_ENV,
        reasoning_shape=ReasoningRequestShape.TOP_LEVEL,
        token_limit_parameter=TokenLimitParameter.MAX_COMPLETION_TOKENS,
    )


def openai_responses_config(*, model: str) -> ProviderConfig:
    return ProviderConfig(
        provider_kind=ProviderKind.OPENAI,
        endpoint_kind=EndpointKind.RESPONSES,
        model=model,
        api_key_env=OPENAI_API_KEY_ENV,
        reasoning_shape=ReasoningRequestShape.TOP_LEVEL,
        token_limit_parameter=TokenLimitParameter.MAX_OUTPUT_TOKENS,
    )


def message_dicts(
    messages: Sequence[PromptMessage | Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, PromptMessage):
            result.append(message.provider_dict())
        else:
            result.append(dict(message))
    return result


def build_chat_completions_request(
    *,
    config: ProviderConfig,
    messages: Sequence[PromptMessage | Mapping[str, Any]],
    temperature: float | None = None,
    token_limit: int | None = None,
    reasoning: Mapping[str, Any] | None = None,
    extra_body: Mapping[str, Any] | None = None,
    extra_kwargs: Mapping[str, Any] | None = None,
) -> ProviderRequest:
    if config.endpoint_kind is not EndpointKind.CHAT_COMPLETIONS:
        raise ValueError("chat completions request requires chat config")

    kwargs = _without_none(
        {
            "model": config.model,
            "messages": message_dicts(messages),
            **dict(extra_kwargs or {}),
        }
    )
    if temperature is not None and config.temperature_supported:
        kwargs["temperature"] = temperature
    if token_limit is not None:
        kwargs[config.token_limit_parameter.value] = token_limit

    _add_reasoning_and_extra_body(
        kwargs,
        config=config,
        reasoning=reasoning,
        extra_body=extra_body,
    )
    return ProviderRequest(
        provider_kind=config.provider_kind,
        endpoint_kind=config.endpoint_kind,
        method=CHAT_COMPLETIONS_METHOD,
        kwargs=kwargs,
        throttle_key=config.throttle_identity,
    )


def build_responses_request(
    *,
    config: ProviderConfig,
    messages: Sequence[PromptMessage | Mapping[str, Any]],
    temperature: float | None = None,
    token_limit: int | None = None,
    reasoning: Mapping[str, Any] | None = None,
    extra_body: Mapping[str, Any] | None = None,
    extra_kwargs: Mapping[str, Any] | None = None,
) -> ProviderRequest:
    if config.endpoint_kind is not EndpointKind.RESPONSES:
        raise ValueError("responses request requires responses config")

    instructions, input_messages = _responses_input_messages(messages)
    kwargs = _without_none(
        {
            "model": config.model,
            "instructions": instructions,
            "input": input_messages,
            **dict(extra_kwargs or {}),
        }
    )
    if temperature is not None and config.temperature_supported:
        kwargs["temperature"] = temperature
    if token_limit is not None:
        kwargs[config.token_limit_parameter.value] = token_limit

    _add_reasoning_and_extra_body(
        kwargs,
        config=config,
        reasoning=reasoning,
        extra_body=extra_body,
    )
    return ProviderRequest(
        provider_kind=config.provider_kind,
        endpoint_kind=config.endpoint_kind,
        method=RESPONSES_METHOD,
        kwargs=kwargs,
        throttle_key=config.throttle_identity,
    )


def parse_provider_response(
    response: Any,
    *,
    config: ProviderConfig,
    output_field: str = OUTPUT_FIELD_TEXT,
) -> ProviderResult:
    if config.endpoint_kind is EndpointKind.CHAT_COMPLETIONS:
        return parse_chat_completion_response(
            response,
            config=config,
            output_field=output_field,
        )
    if config.endpoint_kind is EndpointKind.RESPONSES:
        return parse_responses_response(
            response,
            config=config,
            output_field=output_field,
        )
    raise ProviderResponseParseError(
        f"unsupported endpoint kind {config.endpoint_kind.value!r}",
        metadata={"endpoint_kind": config.endpoint_kind.value},
    )


def parse_chat_completion_response(
    response: Any,
    *,
    config: ProviderConfig,
    output_field: str = OUTPUT_FIELD_TEXT,
) -> ProviderResult:
    metadata = _response_metadata(response)
    choices = _get_value(response, "choices")
    if not isinstance(choices, Sequence) or isinstance(choices, str | bytes):
        raise _parse_error(
            "provider response missing choices",
            config=config,
            response=response,
        )
    if not choices:
        raise _parse_error(
            "provider response has empty choices",
            config=config,
            response=response,
        )
    choice = choices[0]
    message = _get_value(choice, "message")
    text: str | None = None
    if message is not None:
        text = content_to_text(_get_value(message, "content"))
    if text is None:
        value = _get_value(choice, "text")
        if isinstance(value, str):
            text = value
    return ProviderResult(
        text=require_generation_text(text, output_field=output_field),
        response_metadata=metadata,
        usage_metadata=_usage_metadata(metadata),
        provider_cost=provider_cost_from_response(metadata),
        response_id=_optional_str(_get_value(response, "id")),
        model=_optional_str(_get_value(response, "model")) or config.model,
        finish_reason=_optional_str(_get_value(choice, "finish_reason")),
    )


def parse_responses_response(
    response: Any,
    *,
    config: ProviderConfig,
    output_field: str = OUTPUT_FIELD_TEXT,
) -> ProviderResult:
    metadata = _response_metadata(response)
    text = _optional_str(_get_value(response, "output_text"))
    if text is None:
        text = _text_from_responses_output(_get_value(response, "output"))
    return ProviderResult(
        text=require_generation_text(text, output_field=output_field),
        response_metadata=metadata,
        usage_metadata=_usage_metadata(metadata),
        provider_cost=provider_cost_from_response(metadata),
        response_id=_optional_str(_get_value(response, "id")),
        model=_optional_str(_get_value(response, "model")) or config.model,
        finish_reason=_finish_reason_from_responses_response(response),
    )


def call_provider_request(client: Any, request: ProviderRequest) -> Any:
    try:
        if request.endpoint_kind is EndpointKind.CHAT_COMPLETIONS:
            return client.chat.completions.create(**request.kwargs)
        if request.endpoint_kind is EndpointKind.RESPONSES:
            return client.responses.create(**request.kwargs)
    except Exception as exc:
        translated = translate_provider_error(exc, request=request)
        if translated is exc:
            raise
        raise translated from exc
    raise ValueError(
        f"unsupported endpoint kind {request.endpoint_kind.value}"
    )


def translate_provider_error(
    error: Exception,
    *,
    request: ProviderRequest,
) -> EvalFailureError:
    if isinstance(error, EvalFailureError):
        return error
    failure_class = classify_exception(error)
    exception_type = failure_exception_type_for_class(failure_class)
    return exception_type(
        str(error),
        underlying=error,
        metadata={
            "provider_kind": request.provider_kind.value,
            "endpoint_kind": request.endpoint_kind.value,
            "method": request.method,
        },
    )


def _add_reasoning_and_extra_body(
    kwargs: dict[str, Any],
    *,
    config: ProviderConfig,
    reasoning: Mapping[str, Any] | None,
    extra_body: Mapping[str, Any] | None,
) -> None:
    merged_extra_body = {**config.extra_body, **dict(extra_body or {})}
    reasoning_payload = dict(reasoning or {})
    if (
        config.reasoning_supported
        and reasoning_payload
        and config.reasoning_shape is ReasoningRequestShape.TOP_LEVEL
    ):
        kwargs["reasoning"] = reasoning_payload
    if (
        config.reasoning_supported
        and reasoning_payload
        and config.reasoning_shape is ReasoningRequestShape.EXTRA_BODY
    ):
        merged_extra_body["reasoning"] = reasoning_payload
    if merged_extra_body:
        kwargs["extra_body"] = merged_extra_body


def _responses_input_messages(
    messages: Sequence[PromptMessage | Mapping[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    dicts = message_dicts(messages)
    if dicts and dicts[0].get("role") == MessageRole.SYSTEM.value:
        system_content = dicts[0].get("content")
        instructions = (
            system_content if isinstance(system_content, str) else None
        )
        return instructions, dicts[1:]
    return None, dicts


def _without_none(data: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if value is not None}


def _response_metadata(response: Any) -> dict[str, Any]:
    from dr_dspy.serialization import to_metadata_dict

    return to_metadata_dict(response)


def _usage_metadata(response_metadata: Mapping[str, Any]) -> dict[str, Any]:
    usage = response_metadata.get("usage")
    return dict(usage) if isinstance(usage, Mapping) else {}


def _get_value(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


RESPONSES_INCOMPLETE_REASON_LENGTH = "max_output_tokens"
RESPONSES_INCOMPLETE_REASON_CONTENT_FILTER = "content_filter"
RESPONSES_STATUS_COMPLETED = "completed"
CHAT_FINISH_REASON_LENGTH = "length"
CHAT_FINISH_REASON_CONTENT_FILTER = "content_filter"
CHAT_FINISH_REASON_STOP = "stop"


def _finish_reason_from_responses_response(response: Any) -> str | None:
    incomplete_details = _get_value(response, "incomplete_details")
    if isinstance(incomplete_details, Mapping):
        reason = _optional_str(_get_value(incomplete_details, "reason"))
        if reason == RESPONSES_INCOMPLETE_REASON_LENGTH:
            return CHAT_FINISH_REASON_LENGTH
        if reason == RESPONSES_INCOMPLETE_REASON_CONTENT_FILTER:
            return CHAT_FINISH_REASON_CONTENT_FILTER
    status = _optional_str(_get_value(response, "status"))
    if status == RESPONSES_STATUS_COMPLETED:
        return CHAT_FINISH_REASON_STOP
    return None


def _text_from_responses_output(output: Any) -> str | None:
    if not isinstance(output, Sequence) or isinstance(output, str | bytes):
        return None
    parts: list[str] = []
    for item in output:
        content = _get_value(item, "content")
        if not isinstance(content, Sequence) or isinstance(
            content, str | bytes
        ):
            continue
        for part in content:
            if _get_value(part, "type") != "output_text":
                continue
            text = _get_value(part, "text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts) or None


def _parse_error(
    message: str,
    *,
    config: ProviderConfig,
    response: Any,
) -> ProviderResponseParseError:
    return ProviderResponseParseError(
        message,
        metadata={
            "provider_kind": config.provider_kind.value,
            "endpoint_kind": config.endpoint_kind.value,
            "response_preview": repr(response)[:512],
        },
    )
