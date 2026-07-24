"""The Rollout Work Request and its opaque typed-reference transport.

Proves the Work Request carries exactly its declared fields (and NO
Materialization Record reference), and that its typed Object Reference encodes
to / decodes from the opaque string dr-platform transports byte-for-byte
without parsing.
"""

from __future__ import annotations

import pytest
from dr_store import ObjectReference

from whetstone.orchestration import (
    ROLLOUT_WORK_REQUEST_SCHEMA,
    decode_object_reference,
    encode_object_reference,
    encode_work_request_ref,
    work_request_reference,
)
from whetstone.orchestration.work_request import RolloutWorkRequest

from .support import execution_key, full_hash, work_request


def test_work_request_carries_its_declared_fields() -> None:
    request = work_request()
    assert request.graph_config_ref
    assert request.evaluation_context_ref
    assert request.task_inputs == {"prompt": "task.prompt"}
    assert request.repeat_data.repeat_index == 0
    assert (
        request.expected_schema_identities.rollout_result_schema
        == "whetstone.rollout_result"
    )


def test_work_request_has_no_materialization_record_field() -> None:
    """No field could carry a Materialization Record reference."""
    fields = set(RolloutWorkRequest.model_fields)
    for forbidden in fields:
        assert "materialization" not in forbidden.lower()
        assert "record_ref" not in forbidden.lower()


def test_work_request_rejects_repeat_id_mismatch() -> None:
    key = execution_key(repeat_id="r0")
    with pytest.raises(ValueError, match="repeat_id"):
        RolloutWorkRequest(
            rollout_execution_key=key,
            graph_config_ref="g",
            evaluation_context_ref="e",
            repeat_data={"repeat_id": "r9", "repeat_index": 0},
            expected_schema_identities={
                "rollout_result_schema": ROLLOUT_WORK_REQUEST_SCHEMA
            },
        )


def test_work_request_is_content_addressed() -> None:
    request = work_request()
    reference = work_request_reference(request)
    assert reference.schema == ROLLOUT_WORK_REQUEST_SCHEMA
    assert len(reference.content_hash) == 64
    # Equal content -> equal reference.
    assert work_request_reference(work_request()) == reference


def test_object_reference_round_trips_through_opaque_string() -> None:
    reference = ObjectReference(
        schema="whetstone.rollout_result",
        content_hash=full_hash("c"),
    )
    encoded = encode_object_reference(reference)
    assert isinstance(encoded, str) and encoded
    assert decode_object_reference(encoded) == reference


def test_opaque_string_survives_a_schema_with_delimiters() -> None:
    """A schema containing '/' or '?' cannot corrupt the opaque shape."""
    reference = ObjectReference(
        schema="whetstone/rollout?weird=1",
        content_hash=full_hash("d"),
    )
    encoded = encode_object_reference(reference)
    assert decode_object_reference(encoded) == reference


def test_work_request_input_ref_encodes_its_own_reference() -> None:
    request = work_request()
    encoded = encode_work_request_ref(request)
    decoded = decode_object_reference(encoded)
    assert decoded == work_request_reference(request)
    assert decoded.schema == ROLLOUT_WORK_REQUEST_SCHEMA


def test_decode_rejects_a_non_whetstone_string() -> None:
    with pytest.raises(ValueError, match="Whetstone object reference"):
        decode_object_reference("https://example.test/not-a-ref")


def test_decode_rejects_a_missing_or_bad_content_hash() -> None:
    with pytest.raises(ValueError, match="content_hash"):
        decode_object_reference("objref://whetstone.rollout_result/?x=1")
    with pytest.raises(ValueError, match="content_hash"):
        decode_object_reference(
            "objref://whetstone.rollout_result/?content_hash=short"
        )
