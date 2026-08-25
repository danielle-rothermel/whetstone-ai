from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from dr_providers import (
    PromptMessage,
    ProviderCallConfig,
    ProviderCallRequest,
)

from whetstone.execution.call_metadata import (
    METADATA_FAILURE_CODE_KEY,
    cache_marks_metadata,
    telemetry_metadata,
)
from whetstone.execution.call_support import failure_code_of
from whetstone.execution.partials import PartialCallRecord, PartialLog
from whetstone.execution.prompt_cache import (
    CallExecution,
    PromptResultCache,
    execute_call,
)
from whetstone.provider.driver import Clock, Sleep, TransportCall
from whetstone.provider.classification import SemanticFailureClass
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
    "ProviderSemanticFailureError",
    "build_provider_request",
    "call_execution_metadata",
    "derive_rng_seed",
    "resolve_eval_rng_seed",
    "execute_llm_call",
    "provider_result_text",
]


class ProviderSemanticFailureError(ValueError):
    """A provider call produced no usable generation.

    Carries the ``failure_code`` metadata the row attribution table keys on,
    so the accepting/rejecting decision stays owned by the classification
    layer rather than being re-derived from the message text.
    """

    def __init__(
        self, message: str, *, metadata: Mapping[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.metadata: dict[str, Any] = dict(metadata or {})


def derive_rng_seed(*parts: str | int) -> int:
    """Derive a stable non-negative eval rng_seed from ordered identity parts."""
    encoded = "|".join(str(part) for part in parts).encode()
    digest = hashlib.sha256(encoded).digest()
    # OpenAI-style provider seeds are ints; stay within signed 31-bit range.
    return int.from_bytes(digest[:8], "big") % (2**31)


def resolve_eval_rng_seed(
    *,
    candidate_id: str,
    task_id: str,
    task_hash: str,
    seed_index: int,
    seed_plan: Any,
) -> int:
    """Prefer explicit plan provenance; fall back to deterministic derivation."""
    keyed = f"{task_hash}#{seed_index}"
    rng_map = dict(getattr(seed_plan, "rng_seeds", ()))
    if keyed in rng_map:
        return int(rng_map[keyed])
    return derive_rng_seed(candidate_id, task_id, seed_index)


def _validate_eval_rng_seed(rng_seed: int) -> None:
    if isinstance(rng_seed, bool) or not isinstance(rng_seed, int):
        raise TypeError("rng_seed must be an integer")
    if rng_seed < 0:
        raise ValueError("rng_seed must be non-negative")


def _reject_conflicting_seed_parameters(parameters: Mapping[str, object]) -> None:
    """Keep the provider seed single-sourced from eval seed derivation.

    dr-providers reserves the ``seed`` wire key against ``extra_body`` itself,
    so this guard only covers whetstone's own parameters surface, where a
    caller-supplied seed would otherwise compete with the derived rng_seed.
    """
    if "seed" in parameters:
        raise ValueError(
            "parameters must not include seed; pass rng_seed to "
            "build_provider_request instead"
        )


@dataclass(frozen=True, slots=True)
class LlmCallContext:
    execution_policy: ProviderExecutionPolicy
    transport: TransportCall
    prompt_adapter: PlainPromptAdapter | StructuredPromptAdapter
    clock: Clock | None = None
    sleep: Sleep | None = None
    prompt_cache: PromptResultCache | None = None
    partial_log: PartialLog | None = None


def _append_partial_call_record(
    *,
    partial_log: PartialLog,
    execution: CallExecution,
    request: ProviderCallRequest,
    task_id: str,
    seed_index: int,
    phase: str,
    unit: str,
    split_role: str | None = None,
) -> None:
    telemetry = execution.telemetry()
    marks = execution.cache_marks()
    result = execution.result
    output_text: str | None = None
    if result.succeeded and result.provider_generation is not None:
        output_text = require_provider_generation_text(
            result.provider_generation.text,
            output_field=OUTPUT_FIELD_TEXT,
        )
    failed = not result.succeeded
    partial_log.append(
        PartialCallRecord(
            phase=phase,
            task_id=task_id,
            unit=unit,
            seed_index=seed_index,
            request_hash=request.identity_hash,
            redrive_pending=False,
            split_role=split_role or phase,
            failed=failed,
            failure_code=failure_code_of(result) if failed else "",
            prompt_tokens=telemetry.prompt_tokens,
            completion_tokens=telemetry.completion_tokens,
            total_tokens=telemetry.total_tokens,
            reasoning_tokens=telemetry.reasoning_tokens,
            latency_s=telemetry.latency_s,
            output_text=output_text,
            finish_reason=telemetry.finish_reason,
            provider_error=telemetry.provider_error,
            cache_hit=marks.cache_hit,
            cache_source_phase=marks.cache_source_phase,
            cache_source_unit=marks.cache_source_unit,
            cache_source_call_id=marks.cache_source_call_id,
            cache_source_at=marks.cache_source_at,
        )
    )


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
    return provider_call_request_from_parameters(
        config=provider_config,
        messages=messages,
        parameters=resolved_parameters,
        seed=rng_seed,
    )


def execute_llm_call(
    *,
    context: LlmCallContext,
    request: ProviderCallRequest,
    logical_call_id: str,
    task_id: str = "",
    seed_index: int = 0,
    drive_ordinal: int = 0,
    phase: str = "",
    unit: str = "",
    split_role: str | None = None,
    request_identity_sink: list[str] | None = None,
) -> CallExecution:
    execution = execute_call(
        request=request,
        policy=context.execution_policy,
        transport=context.transport,
        logical_call_id=logical_call_id,
        seed_index=seed_index,
        drive_ordinal=drive_ordinal,
        cache=context.prompt_cache,
        phase=phase,
        unit=unit,
        clock=context.clock,
        sleep=context.sleep,
    )
    if request_identity_sink is not None:
        request_identity_sink.append(request.identity_hash)
    if context.partial_log is not None and task_id:
        _append_partial_call_record(
            partial_log=context.partial_log,
            execution=execution,
            request=request,
            task_id=task_id,
            seed_index=seed_index,
            phase=phase,
            unit=unit,
            split_role=split_role,
        )
    return execution


def call_execution_metadata(execution: CallExecution) -> dict[str, object]:
    metadata = dict(telemetry_metadata(execution.telemetry()))
    metadata.update(cache_marks_metadata(execution.cache_marks()))
    return metadata


def provider_result_text(
    result: Any,
    *,
    output_field: str = OUTPUT_FIELD_TEXT,
) -> tuple[str, str]:
    """Return this call's generation text and the failure code marking it.

    An empty generation is an *observed model output*, not a failure to
    observe one: the model was asked, it answered, and the answer was
    nothing. That is a result -- just one that cannot possibly pass. So a
    blank generation is returned as the empty string and routed through the
    eval procedure like any other output, where the family scorer fails it
    on its own terms. The returned failure code is informational: it records
    *that* the generation was blank so the row stays explainable, without
    determining the row's state.

    Every other semantic failure class means no generation exists to score.
    Those still raise carrying their failure class, which is the value the
    row attribution table keys on to separate an infrastructure failure from
    a contract-rejected ``invalid`` row.
    """
    if result.provider_generation is None:
        failure = result.semantic_failure
        assert failure is not None
        if failure.failure_class is SemanticFailureClass.BLANK_PROVIDER_GENERATION:
            # The provider answered; the answer was empty. Hand the empty
            # text to the eval procedure rather than aborting the row.
            return "", failure.failure_class.value
        raise ProviderSemanticFailureError(
            f"provider call failed with {failure.failure_class.value}: "
            f"{failure.message}",
            metadata={METADATA_FAILURE_CODE_KEY: failure.failure_class.value},
        )
    return (
        require_provider_generation_text(
            result.provider_generation.text,
            output_field=output_field,
        ),
        "",
    )
