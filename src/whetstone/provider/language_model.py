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
    RequestControl,
    Transcript,
)
from pydantic import BaseModel, ConfigDict, Field, StrictStr

from whetstone.execution.call_metadata import METADATA_FAILURE_CODE_KEY
from whetstone.provider.classification import SemanticFailureClass
from whetstone.provider.failures.exceptions import (
    EmptyProviderGenerationError,
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
    "require_provider_generation_text",
    "translate_provider_failure",
]


def require_provider_generation_text(
    text: str | None, *, output_field: str
) -> str:
    if text is None or not text.strip():
        raise EmptyProviderGenerationError(
            f"empty generation for output field {output_field!r}",
            metadata={
                "output_field": output_field,
                METADATA_FAILURE_CODE_KEY: (
                    SemanticFailureClass.BLANK_PROVIDER_GENERATION.value
                ),
            },
        )
    return text


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
    content: StrictStr | tuple[dict[str, Any], ...]

    def provider_dict(self) -> dict[str, Any]:
        return {"role": self.role.value, "content": deepcopy(self.content)}

    def identity_payload(self) -> dict[str, Any]:
        return self.provider_dict()


class StructuredPromptAdapter(PlainPromptAdapter):
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
    *,
    seed: int | None = None,
) -> ProviderCallConfig:
    """Apply eval parameters and the eval-derived seed.

    ``seed`` is threaded separately from ``parameters`` so it stays sourced
    from eval derivation. A definition that cannot transport
    ``RequestControl.SEED`` refuses here rather than running unseeded:
    the eval contract requires the seed on the wire, and a silent omission
    would be invisible to identity-based evidence.
    """
    controls = config.controls
    updates: dict[str, Any] = {}
    if seed is not None:
        if not config.definition.constraints.supports(RequestControl.SEED):
            raise ValueError(
                "definition "
                f"{config.definition.definition_id!r} does not advertise "
                "RequestControl.SEED and cannot transport the eval-derived "
                "rng_seed; use a seed-advertising definition for seeded "
                "eval traffic"
            )
        updates["seed"] = seed
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
    seed: int | None = None,
) -> ProviderCallRequest:
    return ProviderCallRequest(
        config=provider_call_config_with_parameters(
            config,
            parameters,
            seed=seed,
        ),
        transcript=Transcript(messages=messages),
    )


def provider_result_from_response(
    response: ProviderTransportResponse,
    *,
    output_field: str = OUTPUT_FIELD_TEXT,
) -> ProviderResult:
    metadata: dict[str, Any] = dict(response.response_body)
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
        text=require_provider_generation_text(
            response.text, output_field=output_field
        ),
        response_metadata=metadata,
        usage_metadata=usage_metadata,
        provider_cost=(
            response.cost.total_cost if response.cost is not None else None
        ),
        response_id=response.response_id,
        model=response.model,
        finish_reason=response.stop_reason,
    )


def translate_provider_failure(
    error: ProviderFailureError,
) -> EvalFailureError:
    exception_type = failure_exception_type_for_class(
        error.failure.recoverability
    )
    return exception_type(
        error.failure.message,
        underlying=error,
        metadata={
            "provider_failure": error.failure.model_dump(mode="json"),
        },
    )
