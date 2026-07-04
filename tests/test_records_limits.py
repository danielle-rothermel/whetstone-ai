from __future__ import annotations

from whetstone.records import limits
from whetstone.serialization import PAYLOAD_MAX_BYTES

_BYTE_CAP_NAMES = (
    "TASK_INPUTS_MAX_BYTES",
    "NODE_OUTPUT_MAX_BYTES",
    "PROVIDER_TELEMETRY_MAX_BYTES",
    "GRAPH_SNAPSHOT_MAX_BYTES",
    "BATCH_SUBMIT_SPEC_MAX_BYTES",
    "PER_TEST_RESULTS_MAX_BYTES",
    "METRICS_MAX_BYTES",
)


def test_domain_byte_caps_match_payload_max_bytes() -> None:
    assert limits.DOMAIN_PAYLOAD_MAX_BYTES == PAYLOAD_MAX_BYTES
    for name in _BYTE_CAP_NAMES:
        assert getattr(limits, name) == PAYLOAD_MAX_BYTES, name


def test_metrics_stages_max_count_is_extraordinarily_high() -> None:
    assert limits.METRICS_STAGES_MAX_COUNT >= 1000
