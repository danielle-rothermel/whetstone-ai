from typing import Any, cast

import pytest
from dr_platform import ProjectionColumnType

import whetstone.publication as publication
from whetstone.publication import (
    ANALYSIS_MEMBERS,
    DETAIL_MEMBERS,
    analysis_projection_specs,
    detail_projection_specs,
    validate_projection_specs,
)


def test_export_requires_reconciliation_dependencies(tmp_path) -> None:
    call = cast("Any", publication.export_whetstone)
    with pytest.raises(TypeError, match="reconciliation"):
        call(
            cast("Any", object()),
            destination_path=tmp_path / "publication.duckdb",
        )


def test_export_uses_one_reconciliation_dependency_and_whetstone_schema(
    monkeypatch, tmp_path
) -> None:
    reconciliation = cast("Any", object())
    calls: list[dict[str, Any]] = []

    def fake_export(*args: Any, **kwargs: Any) -> str:
        calls.append(kwargs)
        return str(len(calls))

    monkeypatch.setattr(publication, "export", fake_export)
    result = publication.export_whetstone(
        cast("Any", object()),
        reconciliation=reconciliation,
        destination_path=tmp_path / "publication.duckdb",
    )

    assert result == ("1", "2")
    assert len(calls) == 2
    assert all(call["reconciliation"] is reconciliation for call in calls)
    assert all(call["schema"] == publication.PLATFORM_SCHEMA for call in calls)


def test_analysis_inventory_is_exact_and_referentially_closed() -> None:
    specs = analysis_projection_specs()
    assert tuple(spec.member for spec in specs) == ANALYSIS_MEMBERS
    validate_projection_specs(specs)
    assert "node_attempts" not in ANALYSIS_MEMBERS
    by_member = {spec.member: spec for spec in specs}
    assert {
        "experiment_id",
        "display_name",
        "experiment_kind",
        "updated_at",
    }.issubset(by_member["experiments"].columns)
    assert {
        "task_id",
        "model",
        "result_state",
        "provider_cost",
        "compression_ratio",
        "score",
        "failure_class",
        "bundle_id",
        "snapshot_seq",
    }.issubset(by_member["predictions"].columns)
    assert all(spec.full_rebuild_builder is not None for spec in specs)
    assert all(
        tuple(column.name for column in spec.column_schema) == spec.columns
        for spec in specs
    )
    assert by_member["predictions"].column_schema[4].type is (
        ProjectionColumnType.INTEGER
    )
    assert by_member["predictions"].column_schema[11].type is (
        ProjectionColumnType.NUMERIC
    )
    assert by_member["predictions"].column_schema[16].type is (
        ProjectionColumnType.TIMESTAMP
    )
    assert by_member["predictions"].column_schema[18].type is (
        ProjectionColumnType.JSON
    )


def test_detail_inventory_is_root_cascaded() -> None:
    specs = detail_projection_specs()
    assert tuple(spec.member for spec in specs) == DETAIL_MEMBERS
    validate_projection_specs(specs)
    assert "detail_platform_attempts" in DETAIL_MEMBERS
    by_member = {spec.member: spec for spec in specs}
    assert {
        "input_text",
        "output_text",
        "prompt_text",
        "code_text",
        "metrics_json",
        "request_json",
        "response_json",
        "validation_json",
    }.issubset(by_member["detail_prediction_payloads"].columns)
    assert by_member["detail_node_attempts"].unique_key == ("node_attempt_id",)
    assert by_member["detail_platform_attempts"].unique_key == (
        "platform_item_id",
        "platform_attempt",
    )
    assert all(spec.full_rebuild_builder is not None for spec in specs)
    assert all(
        tuple(column.name for column in spec.column_schema) == spec.columns
        for spec in specs
    )
