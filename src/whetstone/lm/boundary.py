"""Thin adapter over the dr-providers kernel.

The wire mechanics (config records, payload building, transport,
parsing, failure classification) live in dr-providers. This module keeps
only whetstone's domain shapes: ``ProviderResult`` (the provider outcome
surface), the ``LlmRequest`` construction from node parameters, the
``LlmResponse`` → ``ProviderResult`` conversion, and translation of
kernel failure carriers into whetstone eval-failure exceptions.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dr_providers import (
    LlmRequest,
    LlmResponse,
    MessageRole,
    PromptMessage,
    ProviderConfig,
    ProviderFailureError,
    ReasoningEffort,
)
from pydantic import BaseModel, ConfigDict, Field, StrictStr

from whetstone.eval_failures.exceptions import (
    EvalFailureError,
    failure_exception_type_for_class,
)
from whetstone.eval_failures.generation import require_generation_text

OUTPUT_FIELD_TEXT = "text"

TEMPERATURE_PARAMETER = "temperature"
TOKEN_LIMIT_PARAMETER = "token_limit"
REASONING_PARAMETER = "reasoning"
EXTRA_BODY_PARAMETER = "extra_body"
EXTRA_KWARGS_PARAMETER = "extra_kwargs"

__all__ = [
    "OUTPUT_FIELD_TEXT",
    "PlainPromptAdapter",
    "ProviderResult",
    "llm_request_for_node",
    "provider_result_from_response",
    "reasoning_effort_from_parameter",
    "translate_provider_failure",
]


class ProviderResult(BaseModel):
    """Structured provider outcome surfaced to callers."""

    model_config = ConfigDict(extra="forbid")

    text: StrictStr
    response_metadata: dict[str, Any] = Field(default_factory=dict)
    usage_metadata: dict[str, Any] = Field(default_factory=dict)
    provider_cost: float | None = None
    response_id: StrictStr | None = None
    model: StrictStr | None = None
    finish_reason: StrictStr | None = None


class PlainPromptAdapter(BaseModel):
    """Minimal prompt adapter with no hidden framework formatting."""

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


def reasoning_effort_from_parameter(value: Any) -> ReasoningEffort | None:
    """Coerce a node's ``reasoning`` parameter into a typed effort level.

    dr-providers 0.2 made ``LlmRequest.reasoning`` a ``ReasoningEffort``
    enum (was a free-form dict). Whetstone's node parameter surface still
    supplies the effort either as the legacy ``{"effort": "low"}`` mapping
    or as a bare effort string; both map to the enum here. Absent or empty
    values mean "no reasoning override" (``None``); an unrecognized effort
    fails loudly, consistent with the kernel's no-silent-defaults stance.
    """
    if value is None:
        return None
    if isinstance(value, ReasoningEffort):
        return value
    if isinstance(value, Mapping):
        if not value:
            return None
        effort = value.get("effort")
    else:
        effort = value
    if effort is None or effort == "":
        return None
    try:
        return ReasoningEffort(effort)
    except ValueError as exc:
        valid = ", ".join(level.value for level in ReasoningEffort)
        raise ValueError(
            f"invalid reasoning effort {effort!r}; "
            f"expected one of: {valid}"
        ) from exc


def llm_request_for_node(
    *,
    config: ProviderConfig,
    messages: tuple[PromptMessage, ...],
    parameters: Mapping[str, Any],
    idempotency_key: str | None = None,
) -> LlmRequest:
    """Build the kernel request from a node's merged parameters.

    Legacy ``extra_kwargs`` merge into ``extra_body``: the kernel's
    raw-httpx payload is the request body, so both land inline exactly
    as the SDK-era indirection did on the wire.
    """
    extra_body = {
        **dict(parameters.get(EXTRA_BODY_PARAMETER) or {}),
        **dict(parameters.get(EXTRA_KWARGS_PARAMETER) or {}),
    }
    return LlmRequest(
        provider_config=config,
        messages=messages,
        temperature=parameters.get(TEMPERATURE_PARAMETER),
        token_limit=parameters.get(TOKEN_LIMIT_PARAMETER),
        reasoning=reasoning_effort_from_parameter(
            parameters.get(REASONING_PARAMETER)
        ),
        extra_body=extra_body,
        idempotency_key=idempotency_key,
    )


def provider_result_from_response(
    response: LlmResponse,
    *,
    output_field: str = OUTPUT_FIELD_TEXT,
) -> ProviderResult:
    metadata = dict(response.provider_metadata)
    if response.diagnostics is not None:
        metadata["diagnostics"] = response.diagnostics.model_dump(mode="json")
    if response.warnings:
        metadata["conformance_warnings"] = [
            warning.model_dump(mode="json")
            for warning in response.warnings
        ]
    usage = metadata.get("usage")
    return ProviderResult(
        text=require_generation_text(
            response.text, output_field=output_field
        ),
        response_metadata=metadata,
        usage_metadata=dict(usage) if isinstance(usage, Mapping) else {},
        provider_cost=(
            response.cost.total_cost if response.cost is not None else None
        ),
        response_id=response.response_id,
        model=response.model,
        finish_reason=response.finish_reason,
    )


def translate_provider_failure(
    error: ProviderFailureError,
) -> EvalFailureError:
    """Wrap a kernel failure carrier in whetstone's eval-failure family.

    The failure class carries over 1:1; the kernel failure record rides
    in the exception metadata so persisted failure rows keep the full
    provider diagnostics.
    """
    exception_type = failure_exception_type_for_class(
        error.failure.failure_class
    )
    return exception_type(
        error.failure.message,
        underlying=error,
        metadata={
            "provider_failure": error.failure.model_dump(mode="json"),
        },
    )
