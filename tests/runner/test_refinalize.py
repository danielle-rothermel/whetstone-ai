"""Refinalize recomputes a cell status from persisted evidence only."""

from __future__ import annotations

from pathlib import Path

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


def test_a_non_halted_complete_cell_is_unchanged() -> None:
    status, reason = recompute_status(_cell(status="improved"))

    assert status == "improved"
    assert "unchanged" in reason


def test_halted_with_no_best_arm_stays_halted() -> None:
    status, reason = recompute_status(
        _cell(status="halted", best_official=None, delta=None)
    )

    assert status == "halted"
    assert "work was cut short" in reason


def test_halted_after_every_phase_completed_is_corrected() -> None:
    status, reason = recompute_status(
        _cell(status="halted", delta=0.1, delta_ci95=(0.05, 0.2))
    )

    assert status == "improved"
    assert "every phase completed" in reason


def test_a_positive_delta_spanning_zero_is_inconclusive() -> None:
    status, _ = recompute_status(
        _cell(status="halted", delta=0.1, delta_ci95=(-0.05, 0.2))
    )

    assert status == "inconclusive"


def test_a_positive_delta_with_no_interval_is_inconclusive() -> None:
    status, _ = recompute_status(
        _cell(status="halted", delta=0.1, delta_ci95=None)
    )

    assert status == "inconclusive"


@pytest.mark.parametrize("delta", [0.0, -0.2])
def test_a_non_positive_delta_is_no_improvement(delta: float) -> None:
    status, _ = recompute_status(
        _cell(status="halted", delta=delta, delta_ci95=(-0.3, -0.1))
    )

    assert status == "no-improvement"


def test_a_negative_interval_excluding_zero_still_needs_a_positive_delta() -> (
    None
):
    # A CI strictly below 0 excludes 0, but the delta gates first, so this can
    # never read as "improved".
    status, _ = recompute_status(
        _cell(status="halted", delta=-0.1, delta_ci95=(-0.3, -0.05))
    )

    assert status == "no-improvement"


def test_a_positive_delta_under_a_negative_interval_is_inconclusive() -> None:
    """An internally inconsistent record is never certified as improved.

    ``delta > 0`` with an interval lying wholly below 0 is skewed or
    inconsistent evidence. Certifying it would also let a refinalized line
    disagree with the live rule, which requires the interval's low bound to
    clear 0.
    """
    status, _ = recompute_status(
        _cell(status="halted", delta=0.1, delta_ci95=(-0.3, -0.05))
    )

    assert status == "inconclusive"


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


@pytest.mark.parametrize(
    "status", ["improved", "inconclusive", "no-improvement"]
)
def test_a_statistical_verdict_on_a_partial_arm_is_incomplete(
    status: str,
) -> None:
    corrected, reason = recompute_status(
        _cell(status=status, baseline_official=None)
    )

    assert corrected == "incomplete-arm"
    assert "incomplete official arm" in reason
    assert "naive=None" in reason


def test_a_missing_best_arm_is_named_in_the_reason() -> None:
    _, reason = recompute_status(
        _cell(status="no-improvement", best_official=None)
    )

    assert "best=None" in reason


def test_both_missing_arms_are_named() -> None:
    _, reason = recompute_status(
        _cell(
            status="no-improvement",
            baseline_official=None,
            best_official=None,
        )
    )

    assert "naive, best=None" in reason


def test_a_non_statistical_status_on_a_partial_arm_is_left_alone() -> None:
    status, _ = recompute_status(
        _cell(status="plumbing-retry", baseline_official=None)
    )

    assert status == "plumbing-retry"


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
    lines = Ledger(tmp_path / "run").cells()
    assert len(lines) == 2
    assert lines[0] == original
    assert lines[1].status == "improved"


def test_the_corrected_line_carries_the_provenance_note(
    tmp_path: Path,
) -> None:
    ledger = Ledger(tmp_path / "run")
    ledger.append_cell(
        _cell(status="halted", delta=0.1, delta_ci95=(0.05, 0.2))
    )

    outcome = refinalize_cell(ledger, optimizer="copro", env="c18", attempt=0)

    assert outcome.corrected is not None
    assert outcome.corrected.escalation_note.startswith(REFINALIZED_NOTE)


def test_an_existing_note_is_preserved_in_the_correction(
    tmp_path: Path,
) -> None:
    ledger = Ledger(tmp_path / "run")
    ledger.append_cell(
        _cell(
            status="halted",
            delta=0.1,
            delta_ci95=(0.05, 0.2),
            escalation_note="repeats doubled",
        )
    )

    outcome = refinalize_cell(ledger, optimizer="copro", env="c18", attempt=0)

    assert outcome.corrected is not None
    assert "original note: repeats doubled" in (
        outcome.corrected.escalation_note
    )


def test_the_corrected_line_supersedes_for_resumability(
    tmp_path: Path,
) -> None:
    ledger = Ledger(tmp_path / "run")
    ledger.append_cell(_cell(status="halted", delta=0.1))

    refinalize_cell(ledger, optimizer="copro", env="c18", attempt=0)

    reread = Ledger(tmp_path / "run")
    latest = reread.for_attempt("copro", "c18", 0)
    assert latest is not None
    assert latest.status == "inconclusive"
    assert reread.completed_keys() == {("copro", "c18", 0)}


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
    # The certified-looking headroom verdict came off a partial vector, so the
    # superseding line records none.
    assert outcome.corrected.headroom_delta is None
    assert outcome.corrected.headroom_ci95 is None


def test_an_incomplete_arm_correction_drops_the_viewer_publication(
    tmp_path: Path,
) -> None:
    ledger = Ledger(tmp_path / "run")
    ledger.append_cell(_cell(status="no-improvement", best_official=None))

    outcome = refinalize_cell(ledger, optimizer="copro", env="c18", attempt=0)

    assert outcome.corrected is not None
    assert not outcome.corrected.is_completed()
    assert outcome.corrected.artifacts.viewer_publication is None


def test_the_corrected_line_survives_a_full_reload(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "run")
    ledger.append_cell(_cell(status="no-improvement", best_official=None))

    refinalize_cell(ledger, optimizer="copro", env="c18", attempt=0)

    # Reloading validates every line, so an unrepresentable correction would
    # fail here rather than silently persist.
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
