"""Viewer projection tests: strict schema, exact bytes, generation rows."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from whetstone.core.identity import TypedRef
from whetstone.optimization.contracts import StepStatus
from whetstone.runner.viewer_projection import (
    VIEWER_GENERATION_ROW_SCHEMA,
    VIEWER_PROJECTION_SCHEMA,
    ViewerStepSummary,
)


def test_step_summary_rejects_negative_counts() -> None:
    """Counts are counts; a negative one is a defect, not a datum."""
    with pytest.raises(ValidationError, match="cannot be negative"):
        ViewerStepSummary(
            step_index=0,
            step_result_ref=TypedRef(
                schema_name="whetstone.optimization.step_result",
                content_hash="a" * 64,
            ),
            status=StepStatus.COMPLETE,
            proposed_count=-1,
            accepted_count=0,
            resolved_intent_count=0,
            tool_evidence_count=0,
        )


def test_viewer_schema_literals_are_pinned() -> None:
    """Golden literals: renaming one orphans every published cell."""
    assert VIEWER_PROJECTION_SCHEMA == "whetstone.runner.viewer_projection/v1"
    assert (
        VIEWER_GENERATION_ROW_SCHEMA
        == "whetstone.runner.viewer_generation_row/v1"
    )
