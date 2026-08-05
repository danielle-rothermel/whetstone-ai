"""Cell lifecycle tests over real evaluation evidence.

Every cell here runs the real engine against real in-process rows, so the
scores, per-task vectors, and viewer rows these assertions read are the same
artifacts production produces.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.optimization.support import candidate, make_store
from tests.runner.support import (
    BASELINE_TEMPLATE,
    ENV_NAME,
    cell_config,
    official_binding,
    official_engine,
)
from whetstone.runner.budget import BudgetGuard, CreditsSnapshot, StopLossError
from whetstone.runner.cell import (
    CELL_RUN_CONTROL_SCHEMA,
    OFFICIAL_ARM_BINDING_SCHEMA,
    CellError,
    bind_cell_launch,
    prepare_cell_launch,
    run_cell,
)
from whetstone.runner.events import EventStream
from whetstone.runner.ledger import COMPLETED_CELL_STATUSES, Ledger


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
    assert projection["rollout_row_count"] == 3
    assert len(projection["evidence_summaries"]) == 3


def test_completed_cell_is_skipped_without_rerunning(tmp_path: Path) -> None:
    """A completed attempt returns its ledger line instead of paying again."""
    first = run_cell(cell_config(tmp_path))
    assert not first.skipped

    stream = EventStream(tmp_path / "run")
    second_config = cell_config(tmp_path, event_stream=stream)
    second = run_cell(second_config)

    assert second.skipped
    assert second.result is None
    assert second.record == first.record
    events = [
        json.loads(line)
        for line in stream.path.read_text().splitlines()
        if line
    ]
    assert [event["event"] for event in events] == ["attempt_skipped"]


def test_official_arms_bind_before_they_are_paid_for(tmp_path: Path) -> None:
    """Binding happens at launch, so the arms exist before any evaluation."""
    config = cell_config(tmp_path)
    bind_cell_launch(config)

    for arm in ("baseline", "ceiling"):
        bound = config.store.resolve(
            f"{OFFICIAL_ARM_BINDING_SCHEMA}:{config.cell_id}#{arm}"
        )
        assert bound is not None
    assert (
        config.store.resolve(f"{CELL_RUN_CONTROL_SCHEMA}:{config.cell_id}")
        is not None
    )


def test_rebinding_a_changed_official_arm_is_refused(tmp_path: Path) -> None:
    """A cell id may never be reused for a different candidate."""
    store = make_store(tmp_path)
    ledger = Ledger(tmp_path / "ledger")
    first = cell_config(tmp_path, store=store, ledger=ledger)
    bind_cell_launch(first)

    engine = official_engine(store)
    changed = cell_config(
        tmp_path,
        store=store,
        ledger=ledger,
        official_engine=engine,
        official_evaluation_binding=official_binding(engine),
        baseline=candidate("baseline", text=BASELINE_TEMPLATE + " now"),
    )
    with pytest.raises(CellError, match="already bound"):
        bind_cell_launch(changed)


def test_cell_identity_must_match_its_run_control(tmp_path: Path) -> None:
    """The cell id and the optimization run id are one identity, not two."""
    with pytest.raises(CellError, match="exact cell identity"):
        cell_config(tmp_path, env="other-env")


def test_spend_checkpoints_precede_every_paid_boundary(
    tmp_path: Path,
) -> None:
    """Each paid arm is preceded by its own durable checkpoint record."""
    snapshots = iter(
        [
            CreditsSnapshot(total_credits=100.0, total_usage=0.0),
            CreditsSnapshot(total_credits=100.0, total_usage=1.0),
            CreditsSnapshot(total_credits=100.0, total_usage=2.0),
            CreditsSnapshot(total_credits=100.0, total_usage=3.0),
            CreditsSnapshot(total_credits=100.0, total_usage=4.0),
            CreditsSnapshot(total_credits=100.0, total_usage=5.0),
        ]
    )
    outcome = run_cell(
        cell_config(
            tmp_path,
            credits_fetcher=lambda: next(snapshots),
            budget_guard=BudgetGuard(
                reserve_usd=0.0, expected_cell_usd=1000.0
            ),
        )
    )
    ledger = Ledger(tmp_path / "ledger")
    phases = [record.phase for record in ledger.spend_records()]

    assert phases[0] == "before"
    assert phases[-1] == "after"
    for arm in ("baseline", "ceiling", "best"):
        assert f"checkpoint:official:{arm}" in phases
    assert "checkpoint:optimization" in phases
    # Every checkpoint precedes the arm it guards, so a crash mid-cell leaves
    # evidence of what had already been spent.
    assert phases.index("checkpoint:official:baseline") < phases.index(
        "checkpoint:official:best"
    )
    assert outcome.record.spend_usd == pytest.approx(5.0)


def test_stop_loss_halts_before_the_next_paid_arm(tmp_path: Path) -> None:
    """A checkpoint past the stop loss refuses the boundary it guards."""
    snapshots = iter(
        [
            CreditsSnapshot(total_credits=100.0, total_usage=0.0),
            CreditsSnapshot(total_credits=100.0, total_usage=0.0),
            CreditsSnapshot(total_credits=100.0, total_usage=99.0),
        ]
    )
    config = cell_config(
        tmp_path,
        credits_fetcher=lambda: next(snapshots),
        budget_guard=BudgetGuard(reserve_usd=0.0, expected_cell_usd=1.0),
    )
    with pytest.raises(StopLossError):
        run_cell(config)

    # The cell never reached a terminal line, so nothing claims it completed.
    assert Ledger(tmp_path / "ledger").load() == []


def test_preflight_rejects_a_candidate_the_env_cannot_render(
    tmp_path: Path,
) -> None:
    """Preflight runs before any spend, so a bad template never costs money."""
    from whetstone.envs.rollout_definition import PromptInputError

    config = cell_config(
        tmp_path, baseline=candidate("baseline", text="{nonexistent}")
    )
    with pytest.raises(PromptInputError):
        prepare_cell_launch(config)


def test_cell_without_a_ceiling_reports_no_headroom(tmp_path: Path) -> None:
    """A ceiling-free cell is complete, with headroom genuinely absent."""
    outcome = run_cell(cell_config(tmp_path, ceiling=False))
    record = outcome.record

    assert record.ceiling_official is None
    assert record.ceiling_ci95 is None
    assert record.headroom_delta is None
    assert record.headroom_ci95 is None
    assert record.status in COMPLETED_CELL_STATUSES
    assert not (tmp_path / "ledger" / "official_anchors").exists()


def test_official_anchor_projects_both_reference_arms(tmp_path: Path) -> None:
    """A cell with a ceiling writes the viewer-facing official anchor."""
    outcome = run_cell(cell_config(tmp_path))
    anchor_path = Ledger(tmp_path / "ledger").official_anchor_path(
        outcome.record.cell_id
    )
    anchor = json.loads(anchor_path.read_text())

    assert anchor["cell_id"] == outcome.record.cell_id
    assert anchor["baseline_official"] == outcome.record.baseline_official
    assert anchor["ceiling_official"] == outcome.record.ceiling_official
    assert len(anchor["official_instance_ids"]) == len(
        anchor["baseline_per_task"]
    )


def test_a_completed_line_always_cites_committed_viewer_files(
    tmp_path: Path,
) -> None:
    """The ledger's own contract: no terminal line without published files."""
    outcome = run_cell(cell_config(tmp_path))
    record = outcome.record
    assert record.is_completed()
    publication = record.artifacts.viewer_publication
    assert publication is not None

    root = tmp_path / "ledger"
    for reference in (publication.projection, publication.rollout_outputs):
        assert (root / reference.relative_path).is_file()
