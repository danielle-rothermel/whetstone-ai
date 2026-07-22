"""Refinalize: recompute a cell's status from persisted evidence.

FIX 4: a cell wrongly stamped ``halted`` that actually COMPLETED every planned
phase is corrected to its true statistical status; a corrected line is appended
(original preserved) with the ``refinalized`` provenance note. A genuinely
cut-short cell (no ``best_official``) and any non-halted cell are unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from whetstone.runner.ledger import (
    CellArtifacts,
    CellModels,
    CellRecord,
    Ledger,
)
from whetstone.runner.refinalize import (
    REFINALIZED_NOTE,
    recompute_status,
    refinalize_cell,
)


def _record(
    *,
    status: str,
    best_official: float | None,
    delta: float | None,
    delta_ci95: tuple[float, float] | None,
    optimizer: str = "eval",
    env: str = "c11",
    note: str = "",
    baseline_official: float | None = 0.667,
    headroom_delta: float | None = None,
    headroom_ci95: tuple[float, float] | None = None,
    no_demonstrable_headroom: bool | None = None,
) -> CellRecord:
    return CellRecord(
        cell_id=f"{optimizer}:{env}:a0",
        optimizer=optimizer, env=env, attempt=0, canonical=True,
        models=CellModels(task="openai/gpt-5-nano", proposer="p"),
        baseline_official=baseline_official, ceiling_official=0.981,
        best_official=best_official, delta=delta, ci95=delta_ci95,
        delta_ci95=delta_ci95,
        headroom_delta=headroom_delta, headroom_ci95=headroom_ci95,
        no_demonstrable_headroom=no_demonstrable_headroom,
        internal_evals_count=1, optimizer_steps=0, spend_usd=1.5,
        wall_s=3870.0, lane="openrouter", status=status,
        escalation_note=note, artifacts=CellArtifacts(),
    )


def test_halted_but_completed_recomputes_to_no_improvement() -> None:
    # The c11 shape: halted at wall-deadline AFTER completing every phase
    # (best_official present), delta 0 -> true status is no-improvement.
    rec = _record(
        status="halted", best_official=0.667, delta=0.0,
        delta_ci95=(0.0, 0.0),
        note="halted: whole-cell wall deadline 3600s reached (elapsed 3870s)",
    )
    new_status, reason = recompute_status(rec)
    assert new_status == "no-improvement"
    assert "every phase completed" in reason


def test_halted_but_completed_positive_delta_ci_excludes_zero() -> None:
    rec = _record(
        status="halted", best_official=0.9, delta=0.2, delta_ci95=(0.1, 0.3)
    )
    assert recompute_status(rec)[0] == "improved"


def test_halted_but_completed_positive_delta_ci_spans_zero() -> None:
    rec = _record(
        status="halted", best_official=0.9, delta=0.2,
        delta_ci95=(-0.05, 0.4),
    )
    assert recompute_status(rec)[0] == "inconclusive"


def test_genuinely_cut_short_halted_is_unchanged() -> None:
    # No best_official: optimize/best never ran -> work WAS cut short -> stays
    # halted (refinalize must not fabricate a completion).
    rec = _record(
        status="halted", best_official=None, delta=None, delta_ci95=None
    )
    assert recompute_status(rec)[0] == "halted"


def test_non_halted_is_unchanged() -> None:
    rec = _record(
        status="improved", best_official=0.9, delta=0.2, delta_ci95=(0.1, 0.3)
    )
    assert recompute_status(rec)[0] == "improved"


def test_refinalize_appends_corrected_line_preserving_original(
    tmp_path: Path,
) -> None:
    ledger = Ledger(root=tmp_path)
    rec = _record(
        status="halted", best_official=0.667, delta=0.0,
        delta_ci95=(0.0, 0.0),
        note="halted: whole-cell wall deadline 3600s reached (elapsed 3870s)",
    )
    ledger.append_cell(rec)

    outcome = refinalize_cell(ledger, optimizer="eval", env="c11", attempt=0)
    assert outcome.changed
    assert outcome.corrected is not None
    assert outcome.corrected.status == "no-improvement"
    assert REFINALIZED_NOTE in outcome.corrected.escalation_note
    # Original halt note is preserved in the corrected line's provenance.
    assert "wall deadline" in outcome.corrected.escalation_note

    # The on-disk ledger has BOTH lines; the original halted line is preserved.
    fresh = Ledger(root=tmp_path)
    lines = fresh.load()
    assert len(lines) == 2
    assert lines[0].status == "halted"
    assert lines[1].status == "no-improvement"
    # The corrected line supersedes for the resumability key.
    latest = fresh.latest_for("eval", "c11")
    assert latest is not None
    assert latest.status == "no-improvement"


def test_refinalize_no_change_appends_nothing(tmp_path: Path) -> None:
    ledger = Ledger(root=tmp_path)
    rec = _record(
        status="improved", best_official=0.9, delta=0.2, delta_ci95=(0.1, 0.3)
    )
    ledger.append_cell(rec)
    outcome = refinalize_cell(ledger, optimizer="eval", env="c11", attempt=0)
    assert not outcome.changed
    assert Ledger(root=tmp_path).load() == [rec]


def test_incomplete_naive_arm_certified_line_corrected_to_incomplete_arm() -> (
    None
):
    # The exact c18:a1 defect: naive arm never resolved (baseline None)
    # yet the line was stamped a terminal 'no-improvement' AND carried a
    # headroom / no_demonstrable_headroom verdict. That is a certified-looking
    # result off a partial vector -> corrected to 'incomplete-arm'.
    rec = _record(
        status="no-improvement", best_official=1.0, delta=None,
        delta_ci95=None, baseline_official=None,
        headroom_delta=0.0167, headroom_ci95=(-0.0042, 0.0375),
        no_demonstrable_headroom=True,
    )
    new_status, reason = recompute_status(rec)
    assert new_status == "incomplete-arm"
    assert "INCOMPLETE official arm" in reason
    assert "naive" in reason


def test_refinalize_incomplete_arm_strips_headroom_verdict(
    tmp_path: Path,
) -> None:
    ledger = Ledger(root=tmp_path)
    rec = _record(
        status="no-improvement", best_official=1.0, delta=None,
        delta_ci95=None, baseline_official=None,
        headroom_delta=0.0167, headroom_ci95=(-0.0042, 0.0375),
        no_demonstrable_headroom=True,
    )
    ledger.append_cell(rec)
    outcome = refinalize_cell(ledger, optimizer="eval", env="c11", attempt=0)
    assert outcome.changed
    assert outcome.corrected is not None
    assert outcome.corrected.status == "incomplete-arm"
    # The certified-looking headroom verdict is STRIPPED on the corrected line.
    assert outcome.corrected.headroom_delta is None
    assert outcome.corrected.headroom_ci95 is None
    assert outcome.corrected.no_demonstrable_headroom is None
    assert REFINALIZED_NOTE in outcome.corrected.escalation_note
    # Both lines on disk; the corrected line supersedes for resumability.
    fresh = Ledger(root=tmp_path)
    assert len(fresh.load()) == 2
    latest = fresh.latest_for("eval", "c11")
    assert latest is not None
    assert latest.status == "incomplete-arm"


def test_incomplete_best_arm_certified_line_corrected() -> None:
    # Symmetric: best_official=None while stamped a statistical status.
    rec = _record(
        status="improved", best_official=None, delta=None,
        delta_ci95=None, baseline_official=0.5,
    )
    new_status, reason = recompute_status(rec)
    assert new_status == "incomplete-arm"
    assert "best" in reason


def test_refinalize_missing_cell_raises(tmp_path: Path) -> None:
    ledger = Ledger(root=tmp_path)
    with pytest.raises(ValueError, match="no cell line"):
        refinalize_cell(ledger, optimizer="eval", env="c99", attempt=0)
