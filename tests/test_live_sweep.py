from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from dr_platform.status import AttemptExecutionState
from sqlalchemy.engine import Engine

from whetstone.platform import live_sweep
from whetstone.platform.live_sweep import (
    GENERATION_CEILING_USD,
    CellReconciliation,
    SweepLedger,
    reconcile_ledger,
)
from whetstone.records import GenerationRunStatus


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


def test_reconciliation_uses_persisted_platform_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = SweepLedger(
        tmp_path / "ledger.sqlite3", manifest_hash="manifest-a"
    )
    try:
        ledger.reserve([_cell("a")], {"a": 0.10})
        ledger.submitted(
            [_cell("a")],
            operation_key="operation-a",
            prediction_ids={"a": "prediction-a"},
        )
        row = ledger.rows()[0]
        observed = SimpleNamespace(
            item_id=row["platform_item_id"],
            attempt=0,
            execution_state=AttemptExecutionState.SUCCEEDED,
        )
        monkeypatch.setattr(
            live_sweep,
            "_attempt_facts",
            lambda _engine, _operation: {(observed.item_id, 0): observed},
        )
        monkeypatch.setattr(
            live_sweep,
            "_node_cost_facts",
            lambda _engine, **_kwargs: (
                GenerationRunStatus.SUCCESS,
                0.07,
                {"total_tokens": 12},
            ),
        )
        monkeypatch.setattr(
            live_sweep,
            "_score_terminal_status",
            lambda _engine, **_kwargs: None,
        )

        facts = reconcile_ledger(ledger, engine=cast("Engine", object()))

        assert facts[0].status == "succeeded"
        assert facts[0].actual_cost == 0.07
        assert ledger.summary()["succeeded"]["actual_usd"] == 0.07
    finally:
        ledger.close()


def test_reconciliation_keeps_unknown_cost_reserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = SweepLedger(
        tmp_path / "ledger.sqlite3", manifest_hash="manifest-a"
    )
    try:
        ledger.reserve([_cell("a")], {"a": 0.10})
        ledger.submitted(
            [_cell("a")],
            operation_key="operation-a",
            prediction_ids={"a": "prediction-a"},
        )
        row = ledger.rows()[0]
        observed = SimpleNamespace(
            item_id=row["platform_item_id"],
            attempt=0,
            execution_state=AttemptExecutionState.SUCCEEDED,
        )
        monkeypatch.setattr(
            live_sweep,
            "_attempt_facts",
            lambda _engine, _operation: {(observed.item_id, 0): observed},
        )
        monkeypatch.setattr(
            live_sweep,
            "_node_cost_facts",
            lambda _engine, **_kwargs: (GenerationRunStatus.SUCCESS, None, {}),
        )
        monkeypatch.setattr(
            live_sweep,
            "_score_terminal_status",
            lambda _engine, **_kwargs: None,
        )

        reconcile_ledger(ledger, engine=cast("Engine", object()))

        summary = ledger.summary()["succeeded"]
        assert summary["actual_usd"] == 0
        assert summary["reserved_usd"] == 0.10
    finally:
        ledger.close()


def test_retry_resume_records_one_lineage_increment(tmp_path: Path) -> None:
    ledger = SweepLedger(
        tmp_path / "ledger.sqlite3", manifest_hash="manifest-a"
    )
    try:
        ledger.reserve([_cell("a")], {"a": 0.10})
        ledger.submitted(
            [_cell("a")],
            operation_key="operation-a",
            prediction_ids={"a": "prediction-a"},
        )
        failure = CellReconciliation(
            cell_id="a",
            status="typed_failure",
            platform_attempt=0,
            retry_count=0,
            actual_cost=None,
            provider_tokens={},
            error_classification="generation_error",
        )
        ledger.reconciliation([failure])
        assert ledger.claim_retry(failure)
        # A process crash after the claim is safe: replaying the deterministic
        # Platform request records the same child Attempt only once.
        ledger.retried(cell_id="a", source_attempt=0, created_attempt=1)
        ledger.retried(cell_id="a", source_attempt=0, created_attempt=1)

        row = ledger.rows()[0]
        assert row["retry_count"] == 1
        assert row["retry_of_attempt"] == 0
        assert row["platform_attempt"] == 1
        assert row["attempt_ids_json"] == "[0, 1]"
    finally:
        ledger.close()


def test_concurrent_retry_claim_replays_one_attempt_lineage(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ledger.sqlite3"
    first = SweepLedger(path, manifest_hash="manifest-a")
    second = SweepLedger(path, manifest_hash="manifest-a")
    try:
        first.reserve([_cell("a")], {"a": 0.10})
        first.submitted(
            [_cell("a")],
            operation_key="operation-a",
            prediction_ids={"a": "prediction-a"},
        )
        failure = CellReconciliation(
            cell_id="a",
            status="typed_failure",
            platform_attempt=0,
            retry_count=0,
            actual_cost=None,
            provider_tokens={},
            error_classification="generation_error",
        )
        first.reconciliation([failure])

        assert first.claim_retry(failure)
        assert second.claim_retry(failure)
        first.retried(cell_id="a", source_attempt=0, created_attempt=1)
        second.retried(cell_id="a", source_attempt=0, created_attempt=1)

        row = first.rows()[0]
        assert row["retry_count"] == 1
        assert row["attempt_ids_json"] == "[0, 1]"
    finally:
        second.close()
        first.close()
