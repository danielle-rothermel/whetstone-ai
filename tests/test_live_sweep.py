from __future__ import annotations

import json
import shutil
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import typer
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
        estimates = {"a": Decimal("0.10"), "b": Decimal("0.20")}
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
            ledger.reserve(
                [_cell("a")], {"a": GENERATION_CEILING_USD + Decimal("0.01")}
            )
    finally:
        ledger.close()


@pytest.mark.parametrize("estimate", [float("nan"), float("inf"), "0.10"])
def test_ledger_rejects_non_json_or_nonfinite_money(
    tmp_path: Path, estimate: object
) -> None:
    ledger = SweepLedger(
        tmp_path / "ledger.sqlite3", manifest_hash="manifest-a"
    )
    try:
        with pytest.raises(ValueError):
            ledger.reserve([_cell("a")], {"a": estimate})  # type: ignore[arg-type]
    finally:
        ledger.close()


def test_remaining_never_resubmits_typed_failure(tmp_path: Path) -> None:
    ledger = SweepLedger(
        tmp_path / "ledger.sqlite3", manifest_hash="manifest-a"
    )
    try:
        ledger.reserve([_cell("a")], {"a": Decimal("0.10")})
        ledger.reconciliation(
            [
                CellReconciliation(
                    "a",
                    "typed_failure",
                    0,
                    0,
                    Decimal("0.10"),
                    {},
                    "generation_error",
                )
            ]
        )
        assert ledger.selected_remaining([_cell("a"), _cell("b")]) == [
            _cell("b")
        ]
    finally:
        ledger.close()


def test_sqlite_intent_replays_after_platform_commit_before_local_ack(
    tmp_path: Path,
) -> None:
    """The durable intent is enough to repeat the same Platform submission."""
    ledger = SweepLedger(
        tmp_path / "ledger.sqlite3", manifest_hash="manifest-a"
    )
    try:
        cell = _cell("a")
        ledger.reserve([cell], {"a": Decimal("0.10")})
        ledger.submission_intent(
            [cell],
            operation_key="operation-a",
            prediction_ids={"a": "prediction-a"},
        )
        # Simulate process death here: reopening SQLite sees only the intent.
        assert ledger.reserve([cell], {"a": Decimal("0.10")}) == [cell]
        row = ledger.rows()[0]
        assert row["status"] == "submitting"
        assert row["operation_key"] == "operation-a"
        assert row["platform_item_id"] == live_sweep.item_id(
            operation_key="operation-a", item_key="prediction-a"
        )
    finally:
        ledger.close()


def test_retry_reserves_each_known_cost_attempt_and_fails_closed_unknown(
    tmp_path: Path,
) -> None:
    ledger = SweepLedger(
        tmp_path / "ledger.sqlite3", manifest_hash="manifest-a"
    )
    try:
        ledger.reserve([_cell("a")], {"a": Decimal("2.31")})
        ledger.submitted(
            [_cell("a")],
            operation_key="operation-a",
            prediction_ids={"a": "prediction-a"},
        )
        known = CellReconciliation(
            "a", "typed_failure", 0, 0, Decimal("2.31"), {}, "error"
        )
        ledger.reconciliation([known])
        assert ledger.claim_retry(known)
        assert ledger.rows()[0]["reserved_cost"] == "2.31"
        unknown = CellReconciliation(
            "a", "typed_failure", 1, 1, None, {}, "error"
        )
        assert not ledger.claim_retry(unknown)
    finally:
        ledger.close()


def test_executable_artifact_binds_generated_spec_and_rejects_tamper(
    tmp_path: Path,
) -> None:
    source = Path("/private/tmp/platform-v6-live-sweep-161")
    target = tmp_path / "campaign"
    shutil.copytree(source, target)
    _metadata, cells, _hash = live_sweep.validate_campaign(target)
    assert len(cells) == 5_904
    manifest = target / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    rows[0]["prediction_id"] = "tampered"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(typer.BadParameter):
        live_sweep.validate_campaign(target)


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
        ledger.reserve([_cell("a")], {"a": Decimal("0.10")})
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
                Decimal("0.07"),
                {"total_tokens": 12},
            ),
        )
        monkeypatch.setattr(
            live_sweep,
            "_score_terminal_status",
            lambda _engine, **_kwargs: None,
        )
        monkeypatch.setattr(
            live_sweep, "_safe_diagnostics", lambda *_args, **_kwargs: {}
        )

        facts = reconcile_ledger(ledger, engine=cast("Engine", object()))

        assert facts[0].status == "succeeded"
        assert facts[0].actual_cost == Decimal("0.07")
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
        ledger.reserve([_cell("a")], {"a": Decimal("0.10")})
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
        monkeypatch.setattr(
            live_sweep, "_safe_diagnostics", lambda *_args, **_kwargs: {}
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
        ledger.reserve([_cell("a")], {"a": Decimal("0.10")})
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
            actual_cost=Decimal("0.10"),
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
        first.reserve([_cell("a")], {"a": Decimal("0.10")})
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
            actual_cost=Decimal("0.10"),
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
