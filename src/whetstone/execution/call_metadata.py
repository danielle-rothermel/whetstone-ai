"""Canonical wire keys and codecs for graph-node call metadata.

These keys are a persisted format: they are written into graph node
metadata and read back by the eval drivers. They are owned here, in the
execution layer, because the values they encode (`CallTelemetry`,
`PartialCacheMarks`) are execution-layer types shared by the provider,
experiment, and eval layers. Keeping the codecs here keeps the provider
layer free of any import into the eval driver package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from whetstone.execution.call_support import CallTelemetry
from whetstone.execution.prompt_cache import PartialCacheMarks

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "METADATA_CACHE_HIT_KEY",
    "METADATA_CACHE_PROVENANCE_KEY",
    "METADATA_CACHE_SOURCE_AT_KEY",
    "METADATA_CACHE_SOURCE_CALL_ID_KEY",
    "METADATA_CACHE_SOURCE_PHASE_KEY",
    "METADATA_CACHE_SOURCE_UNIT_KEY",
    "METADATA_FAILURE_CODE_KEY",
    "METADATA_PROMPT_KEY",
    "METADATA_REDRIVABLE_KEY",
    "METADATA_SUBMISSION_RESULT_KEY",
    "cache_marks_from_metadata",
    "cache_marks_metadata",
    "telemetry_from_metadata",
    "telemetry_metadata",
]


METADATA_PROMPT_KEY = "prompt"
METADATA_FAILURE_CODE_KEY = "failure_code"
METADATA_REDRIVABLE_KEY = "redrivable"
METADATA_CACHE_HIT_KEY = "cache_hit"
METADATA_CACHE_PROVENANCE_KEY = "cache_provenance"
METADATA_CACHE_SOURCE_PHASE_KEY = "cache_source_phase"
METADATA_CACHE_SOURCE_UNIT_KEY = "cache_source_unit"
METADATA_CACHE_SOURCE_CALL_ID_KEY = "cache_source_call_id"
METADATA_CACHE_SOURCE_AT_KEY = "cache_source_at"
METADATA_SUBMISSION_RESULT_KEY = "submission_result"

_TELEMETRY_KEYS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "reasoning_tokens",
    "latency_s",
    "finish_reason",
    "provider_error",
    "provider_cost",
)


def telemetry_metadata(telemetry: CallTelemetry) -> dict[str, Any]:
    return {
        "prompt_tokens": telemetry.prompt_tokens,
        "completion_tokens": telemetry.completion_tokens,
        "total_tokens": telemetry.total_tokens,
        "reasoning_tokens": telemetry.reasoning_tokens,
        "latency_s": telemetry.latency_s,
        "finish_reason": telemetry.finish_reason,
        "provider_error": telemetry.provider_error,
        "provider_cost": telemetry.provider_cost,
    }


def telemetry_from_metadata(metadata: Mapping[str, Any]) -> CallTelemetry:
    values = {key: metadata.get(key) for key in _TELEMETRY_KEYS}
    latency = values["latency_s"]
    return CallTelemetry(
        prompt_tokens=values["prompt_tokens"],
        completion_tokens=values["completion_tokens"],
        total_tokens=values["total_tokens"],
        reasoning_tokens=values["reasoning_tokens"],
        latency_s=None if latency is None else float(latency),
        finish_reason=values["finish_reason"],
        provider_error=values["provider_error"],
        provider_cost=(
            None
            if values["provider_cost"] is None
            else float(values["provider_cost"])
        ),
    )


def cache_marks_metadata(marks: PartialCacheMarks) -> dict[str, Any]:
    return {
        METADATA_CACHE_HIT_KEY: marks.cache_hit,
        METADATA_CACHE_SOURCE_PHASE_KEY: marks.cache_source_phase,
        METADATA_CACHE_SOURCE_UNIT_KEY: marks.cache_source_unit,
        METADATA_CACHE_SOURCE_CALL_ID_KEY: marks.cache_source_call_id,
        METADATA_CACHE_SOURCE_AT_KEY: marks.cache_source_at,
    }


def cache_marks_from_metadata(
    metadata: Mapping[str, Any],
) -> PartialCacheMarks:
    return PartialCacheMarks(
        cache_hit=metadata.get(METADATA_CACHE_HIT_KEY) is True,
        cache_source_phase=metadata.get(METADATA_CACHE_SOURCE_PHASE_KEY),
        cache_source_unit=metadata.get(METADATA_CACHE_SOURCE_UNIT_KEY),
        cache_source_call_id=metadata.get(METADATA_CACHE_SOURCE_CALL_ID_KEY),
        cache_source_at=metadata.get(METADATA_CACHE_SOURCE_AT_KEY),
    )
