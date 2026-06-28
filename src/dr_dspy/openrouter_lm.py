"""OpenRouter-backed DSPy LM wrappers."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from openai import OpenAI

import dspy
from dr_dspy.lm_logging import PutEventFn, _LoggingMixin
from dspy.clients.openai_format import (
    completion_to_lm_response,
    to_openai_chat_request,
)

OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DSPY_ONLY_KWARGS = frozenset({"cache", "rollout_id"})

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

    def _request_kwargs(
        self,
        request: dspy.LMRequest,
    ) -> dict[str, Any]:
        request_kwargs = {
            key: value
            for key, value in to_openai_chat_request(request).items()
            if key not in DSPY_ONLY_KWARGS and value is not None
        }
        if self.reasoning:
            extra_body = dict(request_kwargs.pop("extra_body", {}) or {})
            extra_body["reasoning"] = dict(self.reasoning)
            request_kwargs["extra_body"] = extra_body
        return request_kwargs

    def _create_completion(self, request_kwargs: dict[str, Any]) -> Any:
        return self._get_client().chat.completions.create(**request_kwargs)

    def _provider_completion(self, request: dspy.LMRequest) -> Any:
        return self._create_completion(self._request_kwargs(request))

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
        return completion_to_lm_response(
            self._provider_completion(request), request
        )

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
        request_kwargs = self._request_kwargs(request)
        completion = self._run_logged_forward(
            lambda: self._create_completion(request_kwargs),
            messages=request_kwargs.get("messages"),
            kwargs={
                key: value
                for key, value in request_kwargs.items()
                if key != "messages"
            },
        )
        return completion_to_lm_response(completion, request)

    async def aforward(
        self,
        prompt: str | dspy.LMRequest | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dspy.LMResponse:
        return self.forward(prompt=prompt, messages=messages, **kwargs)
