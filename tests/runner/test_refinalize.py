"""Refinalize recomputes a cell status from persisted evidence only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from whetstone.runner.ledger import (
    COMPLETED_CELL_STATUSES,
    CellArtifacts,
    CellModels,
    CellRecord,
    Ledger,
    ViewerCellPublicationRef,
    ViewerPublishedFileRef,
)
from whetstone.runner.refinalize import (
    REFINALIZED_NOTE,
    recompute_status,
    refinalize_cell,
)

_HASH = "a" * 64
_OTHER_HASH = "b" * 64


def _publication() -> ViewerCellPublicationRef:
    return ViewerCellPublicationRef(
        projection=ViewerPublishedFileRef(
            relative_path="viewer_cells/copro__c18__a0/projection.json",
            sha256=_HASH,
        ),
        generation_outputs=ViewerPublishedFileRef(
            relative_path=(
                "viewer_cells/copro__c18__a0/generation_outputs.jsonl"
            ),
            sha256=_OTHER_HASH,
        ),
    )


def _cell(
    *,
    status: str,
    baseline_official: float | None = 0.5,
    best_official: float | None = 0.6,
    delta: float | None = 0.1,
    delta_ci95: tuple[float, float] | None = None,
    headroom_delta: float | None = None,
    headroom_ci95: tuple[float, float] | None = None,
    escalation_note: str = "",
) -> CellRecord:
    return CellRecord(
        cell_id="copro:c18:a0",
        optimizer="copro",
        env="c18",
        attempt=0,
        canonical=True,
        models=CellModels(task="t", proposer="p"),
        baseline_official=baseline_official,
        ceiling_official=0.9,
        best_official=best_official,
        delta=delta,
        delta_ci95=delta_ci95,
        headroom_delta=headroom_delta,
        headroom_ci95=headroom_ci95,
        internal_evals_count=4,
        optimizer_steps=2,
        spend_usd=0.25,
        wall_s=12.5,
        lane="openrouter",
        status=status,
        escalation_note=escalation_note,
        artifacts=(
            CellArtifacts(viewer_publication=_publication())
            if status in COMPLETED_CELL_STATUSES
            else CellArtifacts()
        ),
    )


# --------------------------------------------------------------------------
# recompute_status: the pure evidence-only rule
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    (
        "cell_kwargs",
        "expected_status",
        "reason_fragment",
    ),
    [
        ({}, "improved", "unchanged"),
        (
            {
                "status": "halted",
                "best_official": None,
                "delta": None,
            },
            "halted",
            "work was cut short",
        ),
        (
            {
                "status": "halted",
                "delta": 0.1,
                "delta_ci95": (0.05, 0.2),
            },
            "improved",
            "every phase completed",
        ),
        (
            {
                "status": "halted",
                "delta": 0.1,
                "delta_ci95": (-0.05, 0.2),
            },
            "inconclusive",
            None,
        ),
        (
            {
                "status": "halted",
                "delta": 0.1,
                "delta_ci95": None,
            },
            "inconclusive",
            None,
        ),
        (
            {
                "status": "halted",
                "delta": 0.0,
                "delta_ci95": (-0.3, -0.1),
            },
            "no-improvement",
            None,
        ),
        (
            {
                "status": "halted",
                "delta": -0.2,
                "delta_ci95": (-0.3, -0.1),
            },
            "no-improvement",
            None,
        ),
        (
            {
                "status": "halted",
                "delta": -0.1,
                "delta_ci95": (-0.3, -0.05),
            },
            "no-improvement",
            None,
        ),
        (
            {
                "status": "halted",
                "delta": 0.1,
                "delta_ci95": (-0.3, -0.05),
            },
            "inconclusive",
            None,
        ),
        (
            {
                "status": "improved",
                "baseline_official": None,
            },
            "incomplete-arm",
            "incomplete official arm",
        ),
        (
            {
                "status": "inconclusive",
                "baseline_official": None,
            },
            "incomplete-arm",
            "incomplete official arm",
        ),
        (
            {
                "status": "no-improvement",
                "baseline_official": None,
            },
            "incomplete-arm",
            "incomplete official arm",
        ),
        (
            {
                "status": "no-improvement",
                "best_official": None,
            },
            "incomplete-arm",
            "best=None",
        ),
        (
            {
                "status": "no-improvement",
                "baseline_official": None,
                "best_official": None,
            },
            "incomplete-arm",
            "naive, best=None",
        ),
        (
            {
                "status": "plumbing-retry",
                "baseline_official": None,
            },
            "plumbing-retry",
            None,
        ),
    ],
)
def test_recompute_status_cases(
    cell_kwargs: dict[str, Any],
    expected_status: str,
    reason_fragment: str | None,
) -> None:
    status = cell_kwargs.pop("status", "improved")
    record = _cell(status=str(status), **cell_kwargs)
    corrected, reason = recompute_status(record)

    assert corrected == expected_status
    if reason_fragment is not None:
        assert reason_fragment in reason


def test_refinalize_matches_the_live_cell_status_rule() -> None:
    """Refinalize duplicates the live rule, so the two may never disagree."""
    from whetstone.evaluation.analysis.statistics import BootstrapCI
    from whetstone.optimization.contracts import StepStatus
    from whetstone.runner.cell import _status_for
    from whetstone.runner.refinalize import _status_from

    cases: list[tuple[float, tuple[float, float]]] = [
        (0.1, (0.05, 0.2)),
        (0.1, (-0.05, 0.2)),
        (0.1, (-0.3, -0.05)),
        (-0.1, (-0.3, -0.05)),
        (0.1, (0.0, 0.2)),
    ]
    for delta, pair in cases:
        low, high = pair
        live = _status_for(
            terminal_status=StepStatus.COMPLETE,
            best_score=1.0,
            ceiling_expected=False,
            ceiling_score=None,
            delta=delta,
            delta_ci=BootstrapCI(
                point=(low + high) / 2,
                low=low,
                high=high,
                level=0.95,
                resamples=1000,
            ),
        )
        assert _status_from(delta, pair) == live, (delta, pair)


# --------------------------------------------------------------------------
# refinalize_cell: appending the corrected line
# --------------------------------------------------------------------------


def test_an_unchanged_cell_appends_nothing(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "run")
    ledger.append_cell(_cell(status="improved"))

    outcome = refinalize_cell(ledger, optimizer="copro", env="c18", attempt=0)

    assert not outcome.changed
    assert outcome.corrected is None
    assert len(Ledger(tmp_path / "run").cells()) == 1


def test_a_correction_appends_and_preserves_the_original(
    tmp_path: Path,
) -> None:
    ledger = Ledger(tmp_path / "run")
    original = _cell(status="halted", delta=0.1, delta_ci95=(0.05, 0.2))
    ledger.append_cell(original)

    outcome = refinalize_cell(ledger, optimizer="copro", env="c18", attempt=0)

    assert outcome.changed
    assert outcome.corrected is not None
    assert outcome.corrected.status == "improved"
    assert outcome.corrected.escalation_note.startswith(REFINALIZED_NOTE)
    lines = Ledger(tmp_path / "run").cells()
    assert len(lines) == 2
    assert lines[0] == original
    assert lines[1].status == "improved"

    note_ledger = Ledger(tmp_path / "note-run")
    note_ledger.append_cell(
        _cell(
            status="halted",
            delta=0.1,
            delta_ci95=(0.05, 0.2),
            escalation_note="repeats doubled",
        )
    )
    noted = refinalize_cell(
        note_ledger, optimizer="copro", env="c18", attempt=0
    )
    assert noted.corrected is not None
    assert "original note: repeats doubled" in noted.corrected.escalation_note


def test_an_incomplete_arm_correction_strips_the_headroom_verdict(
    tmp_path: Path,
) -> None:
    ledger = Ledger(tmp_path / "run")
    ledger.append_cell(
        _cell(
            status="no-improvement",
            baseline_official=None,
            headroom_delta=0.4,
            headroom_ci95=(0.2, 0.6),
        )
    )

    outcome = refinalize_cell(ledger, optimizer="copro", env="c18", attempt=0)

    assert outcome.corrected is not None
    assert outcome.corrected.status == "incomplete-arm"
    assert outcome.corrected.headroom_delta is None
    assert outcome.corrected.headroom_ci95 is None
    assert not outcome.corrected.is_completed()
    assert outcome.corrected.artifacts.viewer_publication is None


def test_the_corrected_line_survives_a_full_reload(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "run")
    ledger.append_cell(_cell(status="no-improvement", best_official=None))

    refinalize_cell(ledger, optimizer="copro", env="c18", attempt=0)

    assert len(Ledger(tmp_path / "run").cells()) == 2


def test_an_absent_cell_is_refused(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "run")

    with pytest.raises(ValueError, match="no cell line for"):
        refinalize_cell(ledger, optimizer="copro", env="c18", attempt=0)


def test_refinalize_reads_the_latest_line_for_the_attempt(
    tmp_path: Path,
) -> None:
    ledger = Ledger(tmp_path / "run")
    ledger.append_cell(_cell(status="halted", delta=0.1))
    ledger.append_cell(_cell(status="improved", delta=0.1))

    outcome = refinalize_cell(ledger, optimizer="copro", env="c18", attempt=0)

    assert not outcome.changed
    assert outcome.original.status == "improved"
