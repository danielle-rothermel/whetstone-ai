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


def test_detail_inventory_is_root_cascaded() -> None:
    specs = detail_projection_specs()
    assert tuple(spec.member for spec in specs) == DETAIL_MEMBERS
    validate_projection_specs(specs)
    assert "detail_platform_attempts" in DETAIL_MEMBERS
