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


def test_v0_encdec_extraction_edge_preserves_extraction_metadata() -> None:
    result = reshape_v0_encdec_row(
        load_v0_sample("encdec_extraction_edge.json")
    )

    assert result.generation_run is not None
    assert result.generation_run.summary is not None
    metadata = result.generation_run.summary.metadata
    assert metadata.get("extraction_error") is None
    assert result.generation_run.summary.terminal_output == (
        "def power(a, b):\n    return a ** b"
    )


def test_v0_encdec_decoder_failure_maps_to_partial_status() -> None:
    result = reshape_v0_encdec_row(
        load_v0_sample("encdec_decoder_failure.json")
    )

    assert result.generation_run is not None
    assert result.generation_run.status is GenerationRunStatus.PARTIAL
    assert result.generation_run.terminal_node_id == "decoder"
    assert len(result.node_attempts) == 2
    attempts_by_node = {
        attempt.node_id: attempt for attempt in result.node_attempts
    }
    assert attempts_by_node["encoder"].status is NodeAttemptStatus.SUCCESS
    assert attempts_by_node["decoder"].status is NodeAttemptStatus.ERROR
    assert result.generation_run.summary is not None
    assert result.generation_run.summary.execution_order == (
        "encoder",
        "decoder",
    )
    assert result.generation_run.summary.terminal_error is None


def test_v0_encdec_pending_row_has_spec_only() -> None:
    row = load_v0_sample("encdec_success.json")
    row = {**row, "generation_status": "pending"}
    result = reshape_v0_encdec_row(row)

    assert result.spec.prediction_id
    assert result.generation_run is None
    assert result.node_attempts == ()


def test_v0_failure_from_invalid_failure_class_defaults_permanent() -> None:
    row = load_v0_sample("encdec_encoder_failure.json")
    row = {**row, "generation_failure_class": "not-a-real-class"}
    result = reshape_v0_encdec_row(row)

    assert result.generation_run is not None
    failure = result.node_attempts[0].failure
    assert failure is not None
    assert failure.failure_class is not None
    assert failure.failure_class.value == "permanent"


def test_v0_generation_timestamps_clamp_completed_before_started() -> None:
    row = load_v0_sample("direct_success.json")
    row = {
        **row,
        "generated_at": "2026-06-29T12:00:10+00:00",
        "scored_at": "2026-06-29T12:00:05+00:00",
    }
    result = reshape_v0_direct_row(row)

    assert result.generation_run is not None
    started_at = result.generation_run.started_at
    assert started_at == result.generation_run.completed_at


def test_v0_pending_row_has_spec_only() -> None:
    row = load_v0_sample("direct_success.json")
    row = {**row, "generation_status": "pending"}
    result = reshape_v0_direct_row(row)

    assert result.spec.prediction_id
    assert result.generation_run is None
    assert result.node_attempts == ()
