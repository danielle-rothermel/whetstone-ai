from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from dr_providers import (
    PromptMessage,
    ProviderCallConfig,
    ProviderCallRequest,
)

from whetstone.evaluation.drivers.graph_execution import (
    cache_marks_metadata,
    telemetry_metadata,
)
from whetstone.execution.prompt_cache import (
    CallExecution,
    PromptResultCache,
    execute_call,
)
from whetstone.provider.driver import Clock, Sleep, TransportCall
from whetstone.provider.language_model import (
    OUTPUT_FIELD_TEXT,
    PlainPromptAdapter,
    StructuredPromptAdapter,
    provider_call_request_from_parameters,
    require_provider_generation_text,
)
from whetstone.provider.policy import ProviderExecutionPolicy

__all__ = [
    "LlmCallContext",
    "build_provider_request",
    "call_execution_metadata",
    "execute_llm_call",
    "provider_result_text",
]


@dataclass(frozen=True, slots=True)
class LlmCallContext:
    execution_policy: ProviderExecutionPolicy
    transport: TransportCall
    prompt_adapter: PlainPromptAdapter | StructuredPromptAdapter
    clock: Clock | None = None
    sleep: Sleep | None = None
    prompt_cache: PromptResultCache | None = None


def build_provider_request(
    *,
    provider_config: ProviderCallConfig,
    prompt: str | None = None,
    messages: tuple[PromptMessage, ...] | None = None,
    parameters: Mapping[str, object] | None = None,
    prompt_adapter: PlainPromptAdapter | StructuredPromptAdapter,
) -> ProviderCallRequest:
    if messages is None:
        if prompt is None:
            raise ValueError("build_provider_request requires prompt or messages")
        messages = prompt_adapter.messages(user_content=prompt)
    elif prompt is not None:
        raise ValueError("prompt and messages are mutually exclusive")
    resolved_parameters: dict[str, Any] = (
        {} if parameters is None else dict(parameters)
    )
    return provider_call_request_from_parameters(
        config=provider_config,
        messages=messages,
        parameters=resolved_parameters,
    )


def execute_llm_call(
    *,
    context: LlmCallContext,
    request: ProviderCallRequest,
    logical_call_id: str,
    sample_index: int = 0,
    drive_ordinal: int = 0,
    phase: str = "",
    unit: str = "",
) -> CallExecution:
    return execute_call(
        request=request,
        policy=context.execution_policy,
        transport=context.transport,
        logical_call_id=logical_call_id,
        sample_index=sample_index,
        drive_ordinal=drive_ordinal,
        cache=context.prompt_cache,
        phase=phase,
        unit=unit,
        clock=context.clock,
        sleep=context.sleep,
    )


def call_execution_metadata(execution: CallExecution) -> dict[str, object]:
    metadata = dict(telemetry_metadata(execution.telemetry()))
    metadata.update(cache_marks_metadata(execution.cache_marks()))
    return metadata


def provider_result_text(
    result: Any,
    *,
    output_field: str = OUTPUT_FIELD_TEXT,
) -> str:
    if result.provider_generation is None:
        failure = result.semantic_failure
        assert failure is not None
        raise ValueError(
            f"provider call failed with {failure.failure_class.value}: "
            f"{failure.message}"
        )
    return require_provider_generation_text(
        result.provider_generation.text,
        output_field=output_field,
    )
