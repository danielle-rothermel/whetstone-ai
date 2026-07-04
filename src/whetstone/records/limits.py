from __future__ import annotations

from typing import Any

from whetstone.serialization import (
    PAYLOAD_MAX_BYTES,
    SerializationError,
    to_jsonable,
)

# Tier-1 catastrophe guards — aligned with serialization ceiling.
DOMAIN_PAYLOAD_MAX_BYTES = PAYLOAD_MAX_BYTES  # ~768 MiB

TASK_INPUTS_MAX_BYTES = DOMAIN_PAYLOAD_MAX_BYTES
NODE_OUTPUT_MAX_BYTES = DOMAIN_PAYLOAD_MAX_BYTES
PROVIDER_TELEMETRY_MAX_BYTES = DOMAIN_PAYLOAD_MAX_BYTES
GRAPH_SNAPSHOT_MAX_BYTES = DOMAIN_PAYLOAD_MAX_BYTES
BATCH_SUBMIT_SPEC_MAX_BYTES = DOMAIN_PAYLOAD_MAX_BYTES
PER_TEST_RESULTS_MAX_BYTES = DOMAIN_PAYLOAD_MAX_BYTES
METRICS_MAX_BYTES = DOMAIN_PAYLOAD_MAX_BYTES
METRICS_STAGES_MAX_COUNT = 10_000


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
