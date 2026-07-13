from __future__ import annotations

import json
import shutil
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import typer
from dr_code.humaneval.sampling import write_human_eval_snapshot_rows
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


def _portable_snapshot_campaign(
    campaign_dir: Path,
) -> dict[str, Any]:
    locator = "snapshots/humanevalplus_snapshot.json"
    snapshot_path = write_human_eval_snapshot_rows(
        [
            {
                "task_id": "HumanEval/0",
                "prompt": "def add_one(value):\n",
                "canonical_solution": "    return value + 1\n",
                "entry_point": "add_one",
                "test": (
                    "def check(candidate):\n"
                    "    inputs = [(1,)]\n"
                    "    results = [2]\n"
                    "    for inp, expected in zip(inputs, results):\n"
                    "        assertion(candidate(*inp), expected)\n"
                ),
            }
        ],
        snapshot_path=campaign_dir / locator,
        dataset_name="local/fixture",
    )
    snapshot = live_sweep.load_humaneval_snapshot(
        dataset_name="local/fixture",
        dataset_split="test",
        snapshot_path=snapshot_path,
    )
    (campaign_dir / "split-full.json").write_text(
        json.dumps(
            {
                "name": "portable-fixture",
                "dataset": {
                    "name": "local/fixture",
                    "split": "test",
                    "snapshot_path": locator,
                    "sample_seed": 0,
                    "sample_count": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    return {
        "snapshot": {
            "path": locator,
            "sha256": snapshot.identity.sha256,
            "header": snapshot.identity.header.model_dump(mode="json"),
            "task_count": 1,
        }
    }


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
    if not source.exists():
        pytest.skip("locked operator campaign is not present")
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


def test_campaign_snapshot_is_portable_and_injected_after_move(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = tmp_path / "original" / "campaign"
    original.mkdir(parents=True)
    metadata = _portable_snapshot_campaign(original)
    moved = tmp_path / "moved" / "campaign"
    moved.parent.mkdir()
    original.rename(moved)
    captured: dict[str, object] = {}

    def capture_specs(
        _path: Path, *, configs_root: Path, snapshot: object
    ) -> tuple[()]:
        captured["configs_root"] = configs_root
        captured["snapshot"] = snapshot
        return ()

    monkeypatch.setattr(
        live_sweep, "iter_experiment_specs_from_file", capture_specs
    )

    assert live_sweep._specs_for_cells(moved, metadata, []) == {}
    assert captured["configs_root"] == moved
    snapshot = captured["snapshot"]
    assert isinstance(snapshot, live_sweep.HumanEvalSnapshot)
    assert snapshot.identity.sha256 == metadata["snapshot"]["sha256"]


def test_campaign_snapshot_rejects_locator_escape_and_content_tamper(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    metadata = _portable_snapshot_campaign(campaign)
    snapshot_metadata = metadata["snapshot"]
    assert isinstance(snapshot_metadata, dict)
    snapshot_metadata["path"] = "../outside.json"
    with pytest.raises(ValueError, match="remain inside campaign"):
        live_sweep._campaign_snapshot(campaign, metadata)

    metadata = _portable_snapshot_campaign(campaign)
    snapshot_metadata = metadata["snapshot"]
    assert isinstance(snapshot_metadata, dict)
    snapshot_path = campaign / str(snapshot_metadata["path"])
    snapshot_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        live_sweep._campaign_snapshot(campaign, metadata)


def test_dry_run_output_is_json_serializable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    live_sweep._emit(
        "plan", cells=[_cell("a")], manifest_hash="manifest-a", execute=False
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["generation_ceiling_usd"] == "4.62"


def test_ledger_requires_an_absolute_external_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute"):
        SweepLedger(Path("live-sweep.sqlite3"), manifest_hash="manifest-a")


@pytest.mark.parametrize(
    ("values", "present", "nonblank", "disposition"),
    [
        ({"text": ""}, True, False, "blank_output"),
        ({"text": " \t\n "}, True, False, "blank_output"),
        ({"other": "generated"}, False, False, "missing_output"),
        ({"text": {"code": "generated"}}, True, True, "success"),
        ({"text": {}}, True, False, "blank_output"),
    ],
)
def test_safe_diagnostics_reports_output_key_and_nonblank_semantics(
    values: dict[str, object],
    present: bool,
    nonblank: bool,
    disposition: str,
) -> None:
    diagnostics = live_sweep._project_safe_diagnostics(
        node={
            "status": "success",
            "model": "gpt-5.4",
            "output": {"values": values},
            "response_metadata": {},
            "failure": None,
        },
        score=None,
    )

    assert diagnostics["expected_output_field"] == "text"
    assert diagnostics["output_field_present"] is present
    assert diagnostics["output_nonblank"] is nonblank
    assert diagnostics["adapter_disposition"] == disposition


@pytest.mark.parametrize(
    ("failure", "disposition", "code"),
    [
        (
            {
                "failure_class": "permanent",
                "error_type": (
                    "whetstone.eval_failures.exceptions."
                    "PredictionParseError"
                ),
            },
            "parse_failure",
            "prediction_parse",
        ),
        (
            {
                "failure_class": "transient",
                "error_type": (
                    "whetstone.eval_failures.exceptions."
                    "TransientFailureError"
                ),
                "metadata": {"provider_failure": {"message": "secret"}},
            },
            "provider_failure",
            "provider_failure",
        ),
    ],
)
def test_safe_diagnostics_uses_typed_failure_taxonomy(
    failure: dict[str, object], disposition: str, code: str
) -> None:
    diagnostics = live_sweep._project_safe_diagnostics(
        node={
            "status": "error",
            "model": "gpt-5.4",
            "output": None,
            "response_metadata": {},
            "failure": failure,
        },
        score=None,
    )

    assert diagnostics["adapter_disposition"] == disposition
    assert diagnostics["typed_failure_code"] == code
    assert diagnostics["typed_failure_class"] == failure["failure_class"]


def test_safe_diagnostics_allowlists_and_hashes_provider_facts() -> None:
    secret = "Bearer secret-token"
    prompt = "the private prompt"
    diagnostics = live_sweep._project_safe_diagnostics(
        node={
            "status": "success",
            "model": "gpt-5.4",
            "output": {
                "values": {"text": "generated", "prompt": prompt},
                "metadata": {
                    "response_id": "response-123",
                    "model": "gpt-5.4",
                    "finish_reason": "stop",
                    "authorization": secret,
                    "base_url": "https://api.example.test/v1",
                },
            },
            "response_metadata": {
                "headers": {"Authorization": secret},
                "url": "https://api.example.test/v1?api_key=secret",
                "prompt": prompt,
                "encoded_secret": "Bearer%20secret-token",
            },
            "failure": {
                "failure_class": "permanent",
                "error_type": (
                    "whetstone.eval_failures.exceptions."
                    "EmptyGenerationError"
                ),
                "message": secret,
                "metadata": {"prompt": prompt, "headers": secret},
            },
        },
        score={
            "parser_profile_id": "humaneval",
            "parser_version": "v1",
            "status": "success",
        },
    )

    serialized = json.dumps(diagnostics, sort_keys=True)
    assert diagnostics["response_id_hash"] == sha256(
        b"response-123"
    ).hexdigest()
    assert diagnostics["returned_model"] == "gpt-5.4"
    assert diagnostics["finish_reason"] == "stop"
    assert diagnostics["parser_profile"] == "humaneval"
    assert diagnostics["parser_version"] == "v1"
    assert diagnostics["parser_status"] == "success"
    for forbidden in (
        secret,
        prompt,
        "Bearer%20secret-token",
        "api.example.test",
        "authorization",
        "headers",
        "base_url",
        "response-123",
    ):
        assert forbidden not in serialized


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


def test_reconciliation_persists_allowlisted_diagnostics(
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
                {},
            ),
        )
        monkeypatch.setattr(
            live_sweep,
            "_score_terminal_status",
            lambda _engine, **_kwargs: None,
        )
        diagnostics = live_sweep._project_safe_diagnostics(
            node={
                "status": "success",
                "model": "gpt-5.4",
                "output": {"values": {"text": "generated"}},
                "response_metadata": {},
                "failure": None,
            },
            score={
                "parser_profile_id": "humaneval",
                "parser_version": "v1",
                "status": "success",
            },
        )
        monkeypatch.setattr(
            live_sweep,
            "_safe_diagnostics",
            lambda *_args, **_kwargs: diagnostics,
        )

        reconcile_ledger(ledger, engine=cast("Engine", object()))

        persisted = json.loads(str(ledger.rows()[0]["diagnostics_json"]))
        assert persisted == diagnostics
        assert persisted["output_nonblank"] is True
        assert persisted["parser_status"] == "success"
        assert persisted["adapter_disposition"] == "success"
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
