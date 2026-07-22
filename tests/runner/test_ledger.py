"""Ledger schema validation + resumability tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from whetstone.runner.ledger import (
    CellArtifacts,
    CellModels,
    CellRecord,
    Ledger,
    SpendRecord,
)


def _record(**overrides: object) -> CellRecord:
    base = CellRecord(
        cell_id="copro:c11:a0",
        optimizer="copro",
        env="c11",
        attempt=0,
        canonical=True,
        models=CellModels(
            task="openai/gpt-5-nano", proposer="openai/gpt-5.4-nano"
        ),
        baseline_official=0.2,
        ceiling_official=0.9,
        best_official=0.5,
        delta=0.3,
        ci95=(0.1, 0.5),
        internal_evals_count=9,
        optimizer_steps=8,
        spend_usd=1.2,
        wall_s=42.0,
        lane="openrouter",
        window_notes="",
        status="improved",
        artifacts=CellArtifacts(),
    )
    if not overrides:
        return base
    # Re-validate so an override (e.g. an invalid status) is enforced.
    payload = base.model_dump(mode="json")
    payload.update(overrides)
    return CellRecord.model_validate(payload)


def test_cell_record_exact_schema_fields() -> None:
    record = _record()
    dumped = record.model_dump(mode="json")
    assert set(dumped) == {
        "cell_id", "optimizer", "env", "attempt", "canonical", "models",
        "baseline_official", "ceiling_official", "best_official", "delta",
        "ci95", "internal_evals_count", "optimizer_steps", "spend_usd",
        "wall_s", "lane", "window_notes", "status", "artifacts",
    }
    assert set(dumped["models"]) == {"task", "proposer"}
    assert set(dumped["artifacts"]) == {
        "optimization_result_ref",
        "official_record_before",
        "official_record_after",
    }


def test_invalid_status_rejected() -> None:
    with pytest.raises(ValidationError):
        _record(status="totally-invalid")


def test_all_plan_statuses_accepted() -> None:
    for status in ("improved", "no-improvement", "plumbing-retry", "halted"):
        assert _record(status=status).status == status


def test_null_delta_and_ci_allowed() -> None:
    record = _record(delta=None, ci95=None, baseline_official=None,
                     best_official=None)
    assert record.delta is None
    assert record.ci95 is None


def test_round_trip_through_jsonl_line() -> None:
    record = _record()
    line = record.to_line()
    restored = CellRecord.model_validate_json(line)
    assert restored == record


def test_ledger_append_and_completed_keys(tmp_path: Path) -> None:
    ledger = Ledger(root=tmp_path)
    ledger.append_cell(_record(status="improved"))
    ledger.append_cell(
        _record(optimizer="miprov2", cell_id="miprov2:c11:a0",
                status="no-improvement")
    )
    completed = ledger.completed_keys()
    assert ("copro", "c11", 0) in completed
    assert ("miprov2", "c11", 0) in completed
    assert ledger.is_completed("copro", "c11", 0)
    assert not ledger.is_completed("gepa", "c11", 0)


def test_plumbing_retry_is_not_completed(tmp_path: Path) -> None:
    ledger = Ledger(root=tmp_path)
    ledger.append_cell(_record(status="plumbing-retry"))
    # A plumbing-retry cell does NOT count as completed (must be re-run).
    assert not ledger.is_completed("copro", "c11", 0)


def test_ceiling_cache_lookup(tmp_path: Path) -> None:
    ledger = Ledger(root=tmp_path)
    ledger.append_cell(_record(ceiling_official=0.87))
    assert ledger.ceiling_for("c11") == 0.87
    assert ledger.ceiling_for("c22") is None


def test_reload_from_disk(tmp_path: Path) -> None:
    ledger = Ledger(root=tmp_path)
    ledger.append_cell(_record())
    fresh = Ledger(root=tmp_path)
    loaded = fresh.load()
    assert len(loaded) == 1
    assert loaded[0].cell_id == "copro:c11:a0"


def test_spend_record_round_trip(tmp_path: Path) -> None:
    ledger = Ledger(root=tmp_path)
    ledger.append_spend(
        SpendRecord(
            cell_id="copro:c11:a0", phase="before", lane="openrouter",
            total_credits=710.0, total_usage=616.97, remaining_usd=93.03,
        )
    )
    records = ledger.spend_records()
    assert len(records) == 1
    assert records[0].phase == "before"
    assert records[0].remaining_usd == 93.03


def test_total_spend_sums_cells(tmp_path: Path) -> None:
    ledger = Ledger(root=tmp_path)
    ledger.append_cell(_record(spend_usd=1.5))
    ledger.append_cell(
        _record(cell_id="miprov2:c11:a0", optimizer="miprov2", spend_usd=0.5)
    )
    assert ledger.total_spend_usd() == pytest.approx(2.0)
