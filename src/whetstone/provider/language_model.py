"""Thin adapter over the dr-providers kernel.

The wire mechanics (config records, payload building, transport,
parsing, failure classification) live in dr-providers. This module keeps
only whetstone's domain shapes: ``ProviderResult`` (the provider outcome
surface), the ``ProviderCallRequest`` construction from caller parameters,
the ``ProviderTransportResponse`` -> ``ProviderResult`` conversion, and
translation of kernel failure carriers into whetstone eval-failure
exceptions.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Literal

from dr_providers import (
    MessageRole,
    PromptMessage,
    ProviderBodyExtensions,
    ProviderCallConfig,
    ProviderCallRequest,
    ProviderFailureError,
    ProviderTransportResponse,
    ReasoningEffort,
    Transcript,
)
from pydantic import BaseModel, ConfigDict, Field, StrictStr

from whetstone.provider.failures.exceptions import (
    EmptyGenerationError,
    EvalFailureError,
    failure_exception_type_for_class,
)

OUTPUT_FIELD_TEXT = "text"

TEMPERATURE_PARAMETER = "temperature"
TOKEN_LIMIT_PARAMETER = "token_limit"
REASONING_PARAMETER = "reasoning"
EXTRA_BODY_PARAMETER = "extra_body"

__all__ = [
    "OUTPUT_FIELD_TEXT",
    "PlainPromptAdapter",
    "ProviderResult",
    "StructuredPromptAdapter",
    "provider_call_config_with_parameters",
    "provider_call_request_from_parameters",
    "provider_result_from_response",
    "reasoning_effort_from_parameter",
    "require_generation_text",
    "translate_provider_failure",
]


def require_generation_text(text: str | None, *, output_field: str) -> str:
    """Require a nonblank generation before it becomes a result field."""
    if text is None or not text.strip():
        raise EmptyGenerationError(
            f"empty generation for output field {output_field!r}",
            metadata={"output_field": output_field},
        )
    return text


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


class _TransientStructuredPromptMessage(PromptMessage):
    """Ephemeral kernel bridge; durable state uses strict JSON envelopes."""

    content: StrictStr | tuple[dict[str, Any], ...]

    def provider_dict(self) -> dict[str, Any]:
        return {"role": self.role.value, "content": deepcopy(self.content)}

    def identity_payload(self) -> dict[str, Any]:
        return self.provider_dict()


class StructuredPromptAdapter(PlainPromptAdapter):
    """Identity-bound text or ordered structured-content adapter."""

    content_mode: Literal["text_and_structured_parts/v1"] = (
        "text_and_structured_parts/v1"
    )

    def messages_from_records(
        self,
        records: tuple[dict[str, Any], ...],
    ) -> tuple[PromptMessage, ...]:
        messages: list[PromptMessage] = []
        for record in records:
            if set(record) != {"role", "content"}:
                raise ValueError(
                    "structured prompt messages require role and content"
                )
            role = MessageRole(record["role"])
            raw_content = record["content"]
            if isinstance(raw_content, str):
                content: str | tuple[dict[str, Any], ...] = raw_content
            elif isinstance(raw_content, (list, tuple)) and raw_content:
                if any(not isinstance(part, dict) for part in raw_content):
                    raise ValueError(
                        "structured prompt content parts must be objects"
                    )
                content = tuple(deepcopy(part) for part in raw_content)
            else:
                raise ValueError(
                    "structured prompt content must be text or content parts"
                )
            messages.append(
                _TransientStructuredPromptMessage(role=role, content=content)
            )
        if not messages:
            raise ValueError("structured prompt messages cannot be empty")
        return tuple(messages)


def reasoning_effort_from_parameter(value: Any) -> ReasoningEffort | None:
    """Coerce a ``reasoning`` parameter into a typed effort level.

    Accepts a ``ReasoningEffort`` or a bare effort string. Absent or
    empty values mean "no reasoning override" (``None``); an
    unrecognized effort fails loudly, consistent with the kernel's
    no-silent-defaults stance.
    """
    if value is None or value == "":
        return None
    if isinstance(value, ReasoningEffort):
        return value
    try:
        return ReasoningEffort(value)
    except ValueError as exc:
        valid = ", ".join(level.value for level in ReasoningEffort)
        raise ValueError(
            f"invalid reasoning effort {value!r}; expected one of: {valid}"
        ) from exc


def provider_call_config_with_parameters(
    config: ProviderCallConfig,
    parameters: Mapping[str, Any],
) -> ProviderCallConfig:
    """Fold caller generation parameters into the Config's controls.

    Under the released dr-providers contract, output-affecting controls
    (temperature/token-limit/reasoning) and body extensions are part of the
    Provider Call Config identity, not loose request fields. This projects a
    caller's merged parameters onto a base Config, re-materializing it
    through its owning Definition so the assignment stays validated. Only set
    parameters override the base control; absent ones leave it unchanged.
    """
    controls = config.controls
    updates: dict[str, Any] = {}
    if TEMPERATURE_PARAMETER in parameters:
        updates["temperature"] = parameters.get(TEMPERATURE_PARAMETER)
    if TOKEN_LIMIT_PARAMETER in parameters:
        updates["token_limit"] = parameters.get(TOKEN_LIMIT_PARAMETER)
    if REASONING_PARAMETER in parameters:
        updates["reasoning"] = reasoning_effort_from_parameter(
            parameters.get(REASONING_PARAMETER)
        )
    new_controls = controls.model_copy(update=updates) if updates else controls
    extra_body = dict(parameters.get(EXTRA_BODY_PARAMETER) or {})
    if extra_body:
        extensions = ProviderBodyExtensions(extra_body=extra_body)
        # Extension body keys are Definition-declared Variables; declare any
        # caller-supplied keys so the re-materialized Config stays validated.
        definition = config.definition.model_copy(
            update={
                "extension_keys": (
                    config.definition.extension_keys | frozenset(extra_body)
                )
            }
        )
    else:
        extensions = config.extensions
        definition = config.definition
    return definition.materialize(
        controls=new_controls,
        extensions=extensions,
    )


def provider_call_request_from_parameters(
    *,
    config: ProviderCallConfig,
    messages: tuple[PromptMessage, ...],
    parameters: Mapping[str, Any],
) -> ProviderCallRequest:
    """Build the native Provider Call Request from caller parameters.

    The request references exactly one Provider Call Config (with the
    caller's generation parameters folded into its controls) plus one
    Transcript. It carries no copied controls and no transport policy.
    """
    return ProviderCallRequest(
        config=provider_call_config_with_parameters(config, parameters),
        transcript=Transcript(messages=messages),
    )


def provider_result_from_response(
    response: ProviderTransportResponse,
    *,
    output_field: str = OUTPUT_FIELD_TEXT,
) -> ProviderResult:
    """Project a native Provider Transport Response onto the whetstone
    ``ProviderResult`` surface.

    The transport response's least-processed ``raw_body`` becomes the base
    of ``response_metadata``; conformance warnings and Responses diagnostics
    ride alongside it. Usage and cost are read from the native typed fields.
    """
    metadata: dict[str, Any] = dict(response.raw_body)
    if response.diagnostics is not None:
        metadata["diagnostics"] = response.diagnostics.model_dump(mode="json")
    if response.warnings:
        metadata["conformance_warnings"] = [
            warning.model_dump(mode="json") for warning in response.warnings
        ]
    usage_metadata: dict[str, Any] = (
        response.usage.model_dump(mode="json", exclude_none=True)
        if response.usage is not None
        else {}
    )
    return ProviderResult(
        text=require_generation_text(response.text, output_field=output_field),
        response_metadata=metadata,
        usage_metadata=usage_metadata,
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
