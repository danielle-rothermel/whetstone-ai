"""OpenRouter-backed DSPy LM wrappers."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from openai import OpenAI

import dspy
from dr_dspy.lm.boundary import (
    OPENROUTER_API_KEY_ENV,
    OPENROUTER_BASE_URL,
    ProviderConfig,
    ProviderRequest,
    ProviderResult,
    build_chat_completions_request,
    call_provider_request,
    openrouter_chat_config,
    parse_provider_response,
)
from dr_dspy.lm.logging import PutEventFn, _LoggingMixin
from dspy.clients.openai_format import to_openai_chat_request
from dspy.core.types import LMOutput, LMResponse, LMTextPart

DSPY_ONLY_KWARGS = frozenset({"cache", "rollout_id"})
TOKEN_LIMIT_KEYS = (
    "max_completion_tokens",
    "max_tokens",
    "max_output_tokens",
)

__all__ = [
    "OPENROUTER_API_KEY_ENV",
    "OPENROUTER_BASE_URL",
    "LoggingOpenRouterLM",
]


class _OpenRouterLM(dspy.BaseLM):
    """DSPy typed LM that calls OpenRouter chat completions directly."""

    forward_contract = "typed_lm"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str = OPENROUTER_BASE_URL,
        reasoning: Mapping[str, Any] | None = None,
        client: OpenAI | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.api_key = api_key
        self.base_url = base_url
        self.reasoning = dict(reasoning or {})
        self._client = client

    @property
    def supported_params(self) -> set[str]:
        return {
            "max_completion_tokens",
            "max_tokens",
            "reasoning",
            "response_format",
            "seed",
            "tool_choice",
            "tools",
        }

    @property
    def supports_reasoning(self) -> bool:
        return True

    @property
    def supports_response_schema(self) -> bool:
        return False

    @property
    def supports_function_calling(self) -> bool:
        return True

    def _get_client(self) -> OpenAI:
        if self._client is not None:
            return self._client

        api_key = self.api_key or os.getenv(OPENROUTER_API_KEY_ENV)
        if not api_key:
            raise dspy.LMNotConfiguredError(
                f"{OPENROUTER_API_KEY_ENV} is not set",
                model=self.model,
                provider="openrouter",
            )

        self._client = OpenAI(api_key=api_key, base_url=self.base_url)
        return self._client

    def _provider_config(self) -> ProviderConfig:
        return openrouter_chat_config(
            model=self.model,
            base_url=self.base_url,
        )

    def _request_kwargs(
        self,
        request: dspy.LMRequest,
    ) -> dict[str, Any]:
        return self._provider_request(request).kwargs

    def _provider_request(
        self,
        request: dspy.LMRequest,
    ) -> ProviderRequest:
        config = self._provider_config()
        request_kwargs = {
            key: value
            for key, value in to_openai_chat_request(request).items()
            if key not in DSPY_ONLY_KWARGS and value is not None
        }
        request_kwargs.pop("model", None)
        messages = request_kwargs.pop("messages", [])
        temperature = request_kwargs.pop("temperature", None)
        extra_body = request_kwargs.pop("extra_body", {}) or {}
        token_limit = _pop_token_limit(request_kwargs, request)
        return build_chat_completions_request(
            config=config,
            messages=messages,
            temperature=temperature,
            token_limit=token_limit,
            reasoning=self.reasoning,
            extra_body=extra_body,
            extra_kwargs=request_kwargs,
        )

    def _create_completion(self, request: ProviderRequest) -> Any:
        return call_provider_request(self._get_client(), request)

    def _provider_completion(self, request: dspy.LMRequest) -> Any:
        return self._create_completion(self._provider_request(request))

    def _forward_request(
        self,
        prompt: str | dspy.LMRequest | None,
        messages: list[dict[str, Any]] | None,
        kwargs: dict[str, Any],
    ) -> dspy.LMRequest:
        if (
            isinstance(prompt, dspy.LMRequest)
            and messages is None
            and not kwargs
        ):
            return prompt
        raise TypeError(
            f"{type(self).__name__}.forward() requires a dspy.LMRequest; "
            "call the LM object instead of calling forward() directly."
        )

    def forward(
        self,
        prompt: str | dspy.LMRequest | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dspy.LMResponse:
        request = self._forward_request(prompt, messages, kwargs)
        completion = self._provider_completion(request)
        result = parse_provider_response(
            completion,
            config=self._provider_config(),
        )
        return _provider_result_to_lm_response(result, completion=completion)

    async def aforward(
        self,
        prompt: str | dspy.LMRequest | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dspy.LMResponse:
        return self.forward(prompt=prompt, messages=messages, **kwargs)


class LoggingOpenRouterLM(_LoggingMixin, _OpenRouterLM):
    """OpenRouter typed LM with lm.request/response/error logging."""

    forward_contract = "typed_lm"

    def __init__(self, model: str, *, log: PutEventFn, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        self._log = log

    def forward(
        self,
        prompt: str | dspy.LMRequest | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dspy.LMResponse:
        request = self._forward_request(prompt, messages, kwargs)
        provider_request = self._provider_request(request)
        completion = self._run_logged_forward(
            lambda: self._create_completion(provider_request),
            messages=provider_request.kwargs.get("messages"),
            kwargs={
                key: value
                for key, value in provider_request.kwargs.items()
                if key != "messages"
            },
        )
        result = parse_provider_response(
            completion,
            config=self._provider_config(),
        )
        return _provider_result_to_lm_response(result, completion=completion)

    async def aforward(
        self,
        prompt: str | dspy.LMRequest | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dspy.LMResponse:
        return self.forward(prompt=prompt, messages=messages, **kwargs)


def _pop_token_limit(
    request_kwargs: dict[str, Any],
    request: dspy.LMRequest,
) -> int | None:
    for key in TOKEN_LIMIT_KEYS:
        value = request_kwargs.pop(key, None)
        if type(value) is int:
            return value
    return request.config.max_tokens


def _provider_result_to_lm_response(
    result: ProviderResult,
    *,
    completion: Any,
) -> LMResponse:
    output = LMOutput(
        parts=[LMTextPart(text=result.text)],
        finish_reason=result.finish_reason,
        truncated=result.finish_reason == "length",
    )
    return LMResponse(
        model=result.model,
        outputs=[output],
        usage=result.usage_metadata or None,
        cost=result.provider_cost,
        response_id=result.response_id,
        provider_response=completion,
        metadata={"response_metadata": result.response_metadata},
    )
