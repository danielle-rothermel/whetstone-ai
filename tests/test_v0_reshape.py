from __future__ import annotations

from dr_dspy.migration.v0_reshape import (
    reshape_v0_direct_row,
    reshape_v0_encdec_row,
)
from dr_dspy.records import GenerationRunStatus, NodeAttemptStatus
from tests.integration.v0_sample_loader import load_v0_sample


def test_v0_direct_success_maps_to_success_status() -> None:
    result = reshape_v0_direct_row(load_v0_sample("direct_success.json"))

    assert result.generation_run is not None
    assert result.generation_run.status is GenerationRunStatus.SUCCESS
    assert len(result.node_attempts) == 1
    assert result.node_attempts[0].status is NodeAttemptStatus.SUCCESS
    assert result.source_metadata["v0_prediction_id"] == "v0-direct-success-1"
    v0_prediction_id = result.source_metadata["v0_prediction_id"]
    assert result.spec.prediction_id != v0_prediction_id


def test_v0_direct_error_maps_to_error_status() -> None:
    result = reshape_v0_direct_row(
        load_v0_sample("direct_generation_error.json")
    )

    assert result.generation_run is not None
    assert result.generation_run.status is GenerationRunStatus.ERROR
    assert result.node_attempts[0].status is NodeAttemptStatus.ERROR


def test_v0_encdec_encoder_failure_maps_to_error_status() -> None:
    result = reshape_v0_encdec_row(
        load_v0_sample("encdec_encoder_failure.json")
    )

    assert result.generation_run is not None
    assert result.generation_run.status is GenerationRunStatus.ERROR
    assert len(result.node_attempts) == 1
    assert result.node_attempts[0].node_id == "encoder"


def test_v0_encdec_success_produces_two_node_attempts() -> None:
    result = reshape_v0_encdec_row(load_v0_sample("encdec_success.json"))

    assert result.generation_run is not None
    assert result.generation_run.status is GenerationRunStatus.SUCCESS
    assert {attempt.node_id for attempt in result.node_attempts} == {
        "encoder",
        "decoder",
    }


def test_v0_pending_row_has_spec_only() -> None:
    row = load_v0_sample("direct_success.json")
    row = {**row, "generation_status": "pending"}
    result = reshape_v0_direct_row(row)

    assert result.spec.prediction_id
    assert result.generation_run is None
    assert result.node_attempts == ()
