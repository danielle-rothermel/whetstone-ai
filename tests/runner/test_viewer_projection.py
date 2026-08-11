"""Viewer projection tests: strict schema, exact bytes, generation rows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tests.runner.support import cell_config
from whetstone.core.identity import TypedRef
from whetstone.optimization.contracts import StepStatus
from whetstone.runner.cell import run_cell
from whetstone.runner.viewer_projection import (
    VIEWER_GENERATION_ROW_SCHEMA,
    VIEWER_PROJECTION_SCHEMA,
    ViewerCellProjection,
    ViewerStepSummary,
    build_viewer_cell_projection,
)


def _published(tmp_path: Path):
    outcome = run_cell(cell_config(tmp_path))
    publication = outcome.record.artifacts.viewer_publication
    assert publication is not None
    return outcome, publication, tmp_path / "ledger"


def test_projection_bytes_match_their_recorded_hashes(tmp_path: Path) -> None:
    """The ledger records the hash of exactly the bytes it committed."""
    _outcome, publication, root = _published(tmp_path)

    for reference in (publication.projection, publication.generation_outputs):
        body = (root / reference.relative_path).read_bytes()
        assert hashlib.sha256(body).hexdigest() == reference.sha256


def test_projection_reports_every_official_arm(tmp_path: Path) -> None:
    """Three official arms produce three summaries and generation rows."""
    outcome, publication, root = _published(tmp_path)
    projection = json.loads(
        (root / publication.projection.relative_path).read_text()
    )
    lines = [
        json.loads(line)
        for line in (root / publication.generation_outputs.relative_path)
        .read_text()
        .splitlines()
        if line
    ]

    assert projection["schema"] == VIEWER_PROJECTION_SCHEMA
    assert projection["cell_id"] == outcome.record.cell_id
    assert len(projection["evidence_summaries"]) == 3
    assert projection["generation_row_count"] == len(lines)
    assert {
        summary["purpose"] for summary in projection["evidence_summaries"]
    } == {
        "official_baseline",
        "official_ceiling",
        "official_best",
    }
    for row in lines:
        assert row["schema"] == VIEWER_GENERATION_ROW_SCHEMA
        assert row["cell_id"] == outcome.record.cell_id
        assert row["task_hash"]


def test_projection_composes_every_ordered_step(tmp_path: Path) -> None:
    """Steps are contiguous and the terminal status is the final step's."""
    outcome, publication, root = _published(tmp_path)
    projection = json.loads(
        (root / publication.projection.relative_path).read_text()
    )

    indices = [step["step_index"] for step in projection["steps"]]
    assert indices == list(range(len(indices)))
    assert projection["terminal_status"] == projection["steps"][-1]["status"]
    assert projection["optimization_result_ref"] == json.loads(
        outcome.record.artifacts.optimization_result_ref.model_dump_json()
    )


def test_projection_serialization_is_canonical_and_stable(
    tmp_path: Path,
) -> None:
    """The same projection always serializes to the same bytes."""
    outcome, publication, root = _published(tmp_path)
    body = (root / publication.projection.relative_path).read_bytes()
    reparsed = ViewerCellProjection.model_validate_json(body)

    assert reparsed.to_bytes() == body
    assert body.endswith(b"\n")
    assert b", " not in body
    assert reparsed.cell_id == outcome.record.cell_id


def test_projection_refuses_a_misaligned_cell_identity(
    tmp_path: Path,
) -> None:
    """Identity fields are one fact, so a mismatched cell_id is refused."""
    _outcome, publication, root = _published(tmp_path)
    payload = json.loads(
        (root / publication.projection.relative_path).read_text()
    )
    payload["attempt"] = payload["attempt"] + 1

    with pytest.raises(ValidationError, match="identity fields do not align"):
        ViewerCellProjection.model_validate(payload)


def test_projection_refuses_a_terminal_status_that_contradicts_its_steps(
    tmp_path: Path,
) -> None:
    """A terminal status must be read off the steps, never asserted."""
    _outcome, publication, root = _published(tmp_path)
    payload = json.loads(
        (root / publication.projection.relative_path).read_text()
    )
    payload["terminal_status"] = StepStatus.FAILED.value

    with pytest.raises(ValidationError, match="terminal_status must match"):
        ViewerCellProjection.model_validate(payload)


def test_projection_refuses_a_result_from_another_cell(
    tmp_path: Path,
) -> None:
    """A projection may only describe the run its cell actually drove."""
    config = cell_config(tmp_path)
    outcome = run_cell(config)
    assert outcome.result is not None
    result_ref = outcome.record.artifacts.optimization_result_ref
    assert result_ref is not None

    with pytest.raises(ValueError, match="does not belong to this cell"):
        build_viewer_cell_projection(
            cell_id="identity:c18:a7",
            optimizer="identity",
            env="c18",
            attempt=7,
            result=outcome.result,
            result_ref=result_ref,
            store=config.store,
            official_evaluations=(),
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
