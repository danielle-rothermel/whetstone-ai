from __future__ import annotations

import json
import shutil
import sqlite3
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
from typer.testing import CliRunner

from whetstone.platform import live_sweep
from whetstone.platform.live_sweep import (
    CellReconciliation,
    SweepLedger,
    reconcile_ledger,
    require_terminal_lifecycle,
)
from whetstone.records import GenerationRunStatus


def _cell(cell_id: str) -> dict[str, str]:
    return {"cell_id": cell_id}


def test_bounded_page_requires_terminal_lifecycle() -> None:
    fact = CellReconciliation(
        cell_id="a",
        status="in_flight",
        platform_attempt=0,
        retry_count=0,
        actual_cost=None,
        provider_tokens={},
        error_classification=None,
    )

    with pytest.raises(RuntimeError, match="terminal lifecycle"):
        require_terminal_lifecycle([fact], cell_ids={"a"})


def test_bounded_page_accepts_terminal_lifecycle_without_actual_cost() -> None:
    fact = CellReconciliation(
        cell_id="a",
        status="typed_failure",
        platform_attempt=0,
        retry_count=0,
        actual_cost=None,
        provider_tokens={},
        error_classification="provider_rejected_before_charge",
    )

    require_terminal_lifecycle([fact], cell_ids={"a"})


def test_bounded_page_rejects_unknown_lifecycle_even_with_cost() -> None:
    fact = CellReconciliation(
        cell_id="a",
        status="unknown",
        platform_attempt=0,
        retry_count=0,
        actual_cost=Decimal("0.10"),
        provider_tokens={},
        error_classification="unrecognized_platform_lifecycle",
    )

    with pytest.raises(RuntimeError, match="terminal lifecycle"):
        require_terminal_lifecycle([fact], cell_ids={"a"})


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


def test_ledger_intent_is_idempotent_and_excludes_remaining(
    tmp_path: Path,
) -> None:
    ledger = SweepLedger(
        tmp_path / "live-sweep.sqlite3", manifest_hash="manifest-a"
    )
    try:
        cells = [_cell("a"), _cell("b")]
        assert ledger.record_intent(cells[:1]) == cells[:1]
        assert ledger.record_intent(cells[:1]) == []
        assert ledger.selected_remaining(cells) == cells[1:]
    finally:
        ledger.close()


def test_legacy_cost_columns_are_removed_without_gating_intent(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ledger.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute(
        """CREATE TABLE sweep_cells (
        manifest_hash TEXT NOT NULL, cell_id TEXT NOT NULL,
        estimated_cost TEXT NOT NULL, reserved_cost TEXT, actual_cost TEXT,
        operation_key TEXT, prediction_id TEXT, platform_item_id TEXT,
        platform_attempt INTEGER, attempt_ids_json TEXT NOT NULL DEFAULT '[]',
        status TEXT NOT NULL, retry_count INTEGER NOT NULL DEFAULT 0,
        retry_of_attempt INTEGER, error_classification TEXT,
        provider_tokens_json TEXT, score_status TEXT, diagnostics_json TEXT,
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
        PRIMARY KEY (manifest_hash, cell_id))"""
    )
    connection.execute(
        "INSERT INTO sweep_cells(manifest_hash,cell_id,estimated_cost,"
        "reserved_cost,status,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
        ("manifest-a", "old", "999999", "999999", "reserved", "t", "t"),
    )
    connection.commit()
    connection.close()

    ledger = SweepLedger(path, manifest_hash="manifest-a")
    try:
        assert ledger.record_intent([_cell("new")]) == [_cell("new")]
        columns = {
            str(row[1])
            for row in ledger.connection.execute(
                "PRAGMA table_info(sweep_cells)"
            )
        }
        assert "estimated_cost" not in columns
        assert "reserved_cost" not in columns
        assert [row["cell_id"] for row in ledger.rows()] == ["new", "old"]
        assert ledger.rows()[1]["status"] == "pending"
    finally:
        ledger.close()


def test_remaining_never_resubmits_typed_failure(tmp_path: Path) -> None:
    ledger = SweepLedger(
        tmp_path / "ledger.sqlite3", manifest_hash="manifest-a"
    )
    try:
        ledger.record_intent([_cell("a")])
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
        ledger.record_intent([cell])
        ledger.submission_intent(
            [cell],
            operation_key="operation-a",
            prediction_ids={"a": "prediction-a"},
        )
        # Simulate process death here: reopening SQLite sees only the intent.
        assert ledger.record_intent([cell]) == [cell]
        row = ledger.rows()[0]
        assert row["status"] == "submitting"
        assert row["operation_key"] == "operation-a"
        assert row["platform_item_id"] == live_sweep.item_id(
            operation_key="operation-a", item_key="prediction-a"
        )
    finally:
        ledger.close()


def test_retry_claim_does_not_depend_on_actual_cost(
    tmp_path: Path,
) -> None:
    ledger = SweepLedger(
        tmp_path / "ledger.sqlite3", manifest_hash="manifest-a"
    )
    try:
        ledger.record_intent([_cell("a")])
        ledger.submitted(
            [_cell("a")],
            operation_key="operation-a",
            prediction_ids={"a": "prediction-a"},
        )
        unknown = CellReconciliation(
            "a", "typed_failure", 0, 0, None, {}, "error"
        )
        ledger.reconciliation([unknown])
        assert ledger.claim_retry(unknown)
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
    assert payload["cell_count"] == 1
    assert "generation_ceiling_usd" not in payload


def test_submit_canary_executes_without_estimate_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    cells = [
        {
            "cell_id": f"cell-{index}",
            "task_id": "HumanEval/0",
            "repetition_seed": 0,
        }
        for index in range(12)
    ]
    submitted: list[dict[str, Any]] = []
    monkeypatch.setattr(
        live_sweep,
        "validate_campaign",
        lambda _path: ({"campaign": "test"}, cells, "manifest-a"),
    )
    monkeypatch.setattr(
        live_sweep,
        "_submit",
        lambda _path, _metadata, selected, _ledger: submitted.extend(selected),
    )

    result = CliRunner().invoke(
        live_sweep.APP,
        [
            "submit-canary",
            str(campaign),
            "--execute",
            "--ledger",
            str(tmp_path / "ledger.sqlite3"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert submitted == cells


@pytest.mark.parametrize(
    "command", ["submit-canary", "submit-remaining", "submit-retry"]
)
def test_live_sweep_cli_has_no_estimates_option(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    monkeypatch.setattr(
        live_sweep,
        "validate_campaign",
        lambda _path: ({}, [], "manifest-a"),
    )
    runner = CliRunner()

    help_result = runner.invoke(live_sweep.APP, [command, "--help"])
    rejected = runner.invoke(
        live_sweep.APP,
        [command, str(campaign), "--estimates", "legacy.json"],
    )

    assert help_result.exit_code == 0
    assert "--estimates" not in help_result.output
    assert rejected.exit_code != 0
    assert "No such option" in rejected.output


@pytest.mark.parametrize(
    ("status", "expected_exit", "expected_submissions"),
    [("succeeded", 0, 3), ("in_flight", 1, 1)],
)
def test_remaining_pages_gate_only_on_terminal_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    expected_exit: int,
    expected_submissions: int,
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    cells = [_cell(f"cell-{index}") for index in range(3)]
    submitted: list[str] = []
    monkeypatch.setattr(
        live_sweep,
        "validate_campaign",
        lambda _path: ({"campaign": "test"}, cells, "manifest-a"),
    )

    def fake_submit(
        _path: Path,
        _metadata: dict[str, Any],
        selected: list[dict[str, Any]],
        ledger: SweepLedger,
    ) -> None:
        for cell in selected:
            cell_id = str(cell["cell_id"])
            submitted.append(cell_id)
            ledger.submission_intent(
                [cell],
                operation_key=f"operation-{cell_id}",
                prediction_ids={cell_id: f"prediction-{cell_id}"},
            )
            ledger.submitted(
                [cell],
                operation_key=f"operation-{cell_id}",
                prediction_ids={cell_id: f"prediction-{cell_id}"},
            )

    def fake_reconcile(
        ledger: SweepLedger, *, engine: Engine
    ) -> list[CellReconciliation]:
        del engine
        return [
            CellReconciliation(
                cell_id=str(row["cell_id"]),
                status=status,
                platform_attempt=0,
                retry_count=0,
                actual_cost=None,
                provider_tokens={},
                error_classification=None,
            )
            for row in ledger.rows()
        ]

    monkeypatch.setattr(live_sweep, "_submit", fake_submit)
    monkeypatch.setattr(live_sweep, "reconcile_ledger", fake_reconcile)
    monkeypatch.setattr(
        live_sweep,
        "create_engine",
        lambda _url: SimpleNamespace(dispose=lambda: None),
    )

    result = CliRunner().invoke(
        live_sweep.APP,
        [
            "submit-remaining",
            str(campaign),
            "--execute",
            "--ledger",
            str(tmp_path / "ledger.sqlite3"),
            "--page-size",
            "1",
        ],
    )

    assert result.exit_code == expected_exit, result.output
    assert len(submitted) == expected_submissions


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
        ledger.record_intent([_cell("a")])
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
        assert json.loads(ledger.rows()[0]["provider_tokens_json"]) == {
            "total_tokens": 12
        }
    finally:
        ledger.close()


def test_reconciliation_persists_allowlisted_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = SweepLedger(
        tmp_path / "ledger.sqlite3", manifest_hash="manifest-a"
    )
    try:
        ledger.record_intent([_cell("a")])
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


def test_reconciliation_reports_unknown_observed_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = SweepLedger(
        tmp_path / "ledger.sqlite3", manifest_hash="manifest-a"
    )
    try:
        ledger.record_intent([_cell("a")])
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
        assert summary["unknown_cost_count"] == 1
        assert ledger.summary()["observed_cost"] == {
            "actual_usd": 0.0,
            "unknown_cost_count": 1,
        }
    finally:
        ledger.close()


def test_retry_resume_records_one_lineage_increment(tmp_path: Path) -> None:
    ledger = SweepLedger(
        tmp_path / "ledger.sqlite3", manifest_hash="manifest-a"
    )
    try:
        ledger.record_intent([_cell("a")])
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


def test_retry_claim_preserves_two_retry_limit(tmp_path: Path) -> None:
    ledger = SweepLedger(
        tmp_path / "ledger.sqlite3", manifest_hash="manifest-a"
    )
    try:
        ledger.record_intent([_cell("a")])
        exhausted = CellReconciliation(
            cell_id="a",
            status="typed_failure",
            platform_attempt=2,
            retry_count=2,
            actual_cost=None,
            provider_tokens={},
            error_classification="generation_error",
        )

        assert not ledger.claim_retry(exhausted)
    finally:
        ledger.close()


def test_concurrent_retry_claim_replays_one_attempt_lineage(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ledger.sqlite3"
    first = SweepLedger(path, manifest_hash="manifest-a")
    second = SweepLedger(path, manifest_hash="manifest-a")
    try:
        first.record_intent([_cell("a")])
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
