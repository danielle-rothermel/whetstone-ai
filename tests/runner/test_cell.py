"""Cell lifecycle tests over real evaluation evidence.

Every cell here runs the real engine against real in-process rows, so the
scores, per-task vectors, and viewer rows these assertions read are the same
artifacts production produces.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.runner.support import ENV_NAME, cell_config
from whetstone.runner.cell import run_cell
from whetstone.runner.ledger import COMPLETED_CELL_STATUSES


def test_cell_records_official_arms_and_publishes_viewer(
    tmp_path: Path,
) -> None:
    """A finished cell reports all three arms and commits its viewer files."""
    outcome = run_cell(cell_config(tmp_path))
    record = outcome.record

    assert record.cell_id == f"identity:{ENV_NAME}:a0"
    assert record.baseline_official is not None
    assert record.ceiling_official is not None
    assert record.best_official is not None
    # The identity adapter returns its input candidate, so best equals baseline
    # exactly; the cell reports a real measured zero rather than a missing arm.
    assert record.delta == pytest.approx(0.0)
    assert record.naive_ci95 is not None
    assert record.headroom_ci95 is not None
    assert record.status in COMPLETED_CELL_STATUSES

    publication = record.artifacts.viewer_publication
    assert publication is not None
    root = tmp_path / "ledger"
    projection = json.loads(
        (root / publication.projection.relative_path).read_text()
    )
    assert projection["cell_id"] == record.cell_id
    assert projection["generation_row_count"] == 3
    assert len(projection["evidence_summaries"]) == 3
