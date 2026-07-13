from __future__ import annotations

from pathlib import Path

import pytest

from whetstone.platform.live_sweep import GENERATION_CEILING_USD, SweepLedger


def _cell(cell_id: str) -> dict[str, str]:
    return {"cell_id": cell_id}


def test_ledger_reservation_is_idempotent_and_excludes_remaining(
    tmp_path: Path,
) -> None:
    ledger = SweepLedger(
        tmp_path / "live-sweep.sqlite3", manifest_hash="manifest-a"
    )
    try:
        cells = [_cell("a"), _cell("b")]
        estimates = {"a": 0.10, "b": 0.20}
        assert ledger.reserve(cells[:1], estimates) == cells[:1]
        assert ledger.reserve(cells[:1], estimates) == []
        assert ledger.selected_remaining(cells) == cells[1:]
    finally:
        ledger.close()


def test_ledger_fails_closed_for_unknown_cost_and_ceiling(
    tmp_path: Path,
) -> None:
    ledger = SweepLedger(
        tmp_path / "live-sweep.sqlite3", manifest_hash="manifest-a"
    )
    try:
        with pytest.raises(ValueError, match="unknown"):
            ledger.reserve([_cell("a")], {})
        with pytest.raises(ValueError, match="ceiling"):
            ledger.reserve([_cell("a")], {"a": GENERATION_CEILING_USD + 0.01})
    finally:
        ledger.close()


def test_ledger_requires_an_absolute_external_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute"):
        SweepLedger(Path("live-sweep.sqlite3"), manifest_hash="manifest-a")
