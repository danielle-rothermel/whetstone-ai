from __future__ import annotations

from typing import Any

from dr_dspy.serialization import SerializationError, to_jsonable

# Domain-tier payload caps (much smaller than Postgres JSONB limits).
TASK_INPUTS_MAX_BYTES = 256 * 1024
NODE_OUTPUT_MAX_BYTES = 1024 * 1024
PROVIDER_TELEMETRY_MAX_BYTES = 1024 * 1024
GRAPH_SNAPSHOT_MAX_BYTES = 256 * 1024
BATCH_SUBMIT_SPEC_MAX_BYTES = 128 * 1024
PER_TEST_RESULTS_MAX_COUNT = 512
PER_TEST_RESULTS_MAX_BYTES = 4 * 1024 * 1024


def validate_payload_size(
    value: Any,
    *,
    max_bytes: int,
    label: str,
) -> None:
    try:
        to_jsonable(value, max_bytes=max_bytes)
    except SerializationError as exc:
        raise ValueError(f"{label}: {exc}") from exc
