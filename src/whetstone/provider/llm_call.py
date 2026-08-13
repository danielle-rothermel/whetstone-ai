from __future__ import annotations

import hashlib
import warnings
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
    "PendingProviderSeedSupportWarning",
    "build_provider_request",
    "call_execution_metadata",
    "derive_rng_seed",
    "execute_llm_call",
    "provider_result_text",
]


class PendingProviderSeedSupportWarning(UserWarning):
    """Eval rng_seed is required but not yet wired to dr-providers."""


_PENDING_PROVIDER_SEED_WARNING_EMITTED = False


def derive_rng_seed(*parts: str | int) -> int:
    """Derive a stable non-negative eval rng_seed from ordered identity parts."""
    encoded = "|".join(str(part) for part in parts).encode()
    digest = hashlib.sha256(encoded).digest()
    # OpenAI-style provider seeds are ints; stay within signed 31-bit range.
    return int.from_bytes(digest[:8], "big") % (2**31)


def _validate_eval_rng_seed(rng_seed: int) -> None:
    if isinstance(rng_seed, bool) or not isinstance(rng_seed, int):
        raise TypeError("rng_seed must be an integer")
    if rng_seed < 0:
        raise ValueError("rng_seed must be non-negative")


def _reject_conflicting_seed_parameters(parameters: Mapping[str, object]) -> None:
    if "seed" in parameters:
        raise ValueError(
            "parameters must not include seed; pass rng_seed to "
            "build_provider_request instead"
        )
    extra_body = parameters.get("extra_body")
    if isinstance(extra_body, Mapping) and "seed" in extra_body:
        raise ValueError(
            "parameters.extra_body must not include seed; pass rng_seed to "
            "build_provider_request instead"
        )


def _warn_pending_provider_seed_support() -> None:
    global _PENDING_PROVIDER_SEED_WARNING_EMITTED
    if _PENDING_PROVIDER_SEED_WARNING_EMITTED:
        return
    warnings.warn(
        "build_provider_request requires rng_seed, but dr-providers does not "
        "support provider seed yet; rng_seed is discarded until dr-providers "
        "is updated.",
        PendingProviderSeedSupportWarning,
        stacklevel=3,
    )
    _PENDING_PROVIDER_SEED_WARNING_EMITTED = True


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
    rng_seed: int,
    prompt: str | None = None,
    messages: tuple[PromptMessage, ...] | None = None,
    parameters: Mapping[str, object] | None = None,
    prompt_adapter: PlainPromptAdapter | StructuredPromptAdapter,
) -> ProviderCallRequest:
    _validate_eval_rng_seed(rng_seed)
    _warn_pending_provider_seed_support()
    if messages is None:
        if prompt is None:
            raise ValueError("build_provider_request requires prompt or messages")
        messages = prompt_adapter.messages(user_content=prompt)
    elif prompt is not None:
        raise ValueError("prompt and messages are mutually exclusive")
    resolved_parameters: dict[str, Any] = (
        {} if parameters is None else dict(parameters)
    )
    _reject_conflicting_seed_parameters(resolved_parameters)
    _ = rng_seed
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
