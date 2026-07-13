from whetstone.publication import (
    ANALYSIS_MEMBERS,
    DETAIL_MEMBERS,
    analysis_projection_specs,
    detail_projection_specs,
    validate_projection_specs,
)


def test_analysis_inventory_is_exact_and_referentially_closed() -> None:
    specs = analysis_projection_specs()
    assert tuple(spec.member for spec in specs) == ANALYSIS_MEMBERS
    validate_projection_specs(specs)
    assert "node_attempts" not in ANALYSIS_MEMBERS
    by_member = {spec.member: spec for spec in specs}
    assert {
        "experiment_id", "display_name", "experiment_kind", "updated_at"
    }.issubset(by_member["experiments"].columns)
    assert {
        "task_id", "model", "result_state", "provider_cost",
        "compression_ratio", "score", "failure_class", "bundle_id",
        "snapshot_seq",
    }.issubset(by_member["predictions"].columns)
    assert all(spec.full_rebuild_builder is not None for spec in specs)


def test_detail_inventory_is_root_cascaded() -> None:
    specs = detail_projection_specs()
    assert tuple(spec.member for spec in specs) == DETAIL_MEMBERS
    validate_projection_specs(specs)
    assert "detail_platform_attempts" in DETAIL_MEMBERS
    by_member = {spec.member: spec for spec in specs}
    assert {
        "input_text", "output_text", "prompt_text", "code_text",
        "metrics_json", "request_json", "response_json", "validation_json",
    }.issubset(by_member["detail_prediction_payloads"].columns)
    assert by_member["detail_node_attempts"].unique_key == ("node_attempt_id",)
    assert by_member["detail_platform_attempts"].unique_key == (
        "platform_item_id", "platform_attempt"
    )
    assert all(spec.full_rebuild_builder is not None for spec in specs)
