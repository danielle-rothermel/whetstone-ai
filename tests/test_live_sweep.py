from __future__ import annotations

import json
import shutil
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
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

from whetstone.db.io import node_attempt_row
from whetstone.platform import live_sweep
from whetstone.platform.live_sweep import (
    CellReconciliation,
    SweepLedger,
    reconcile_ledger,
    require_terminal_lifecycle,
)
from whetstone.records import GenerationRunStatus, NodeAttemptRecord


def _cell(cell_id: str) -> dict[str, str]:
    return {"cell_id": cell_id}


def _persisted_success_node(
    response_metadata: dict[str, Any],
) -> dict[str, Any]:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    record = NodeAttemptRecord.model_validate(
        {
            "node_attempt_id": "node-attempt-a",
            "generation_run_id": "run-a",
            "prediction_id": "prediction-a",
            "node_id": "generate",
            "attempt_index": 0,
            "status": "success",
            "provider_config": {
                "provider_kind": "openai",
                "endpoint_kind": "responses",
                "model": "gpt-5.4-nano",
                "throttle_key": "openai:gpt-5.4-nano",
            },
            "output": {"values": {"text": "generated"}},
            "response_metadata": {
                "response_metadata": response_metadata,
            },
            "started_at": timestamp,
            "completed_at": timestamp,
        }
    )
    return node_attempt_row(record)


def _persisted_failure_node(
    provider_failure: dict[str, Any],
) -> dict[str, Any]:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    record = NodeAttemptRecord.model_validate(
        {
            "node_attempt_id": "node-attempt-a",
            "generation_run_id": "run-a",
            "prediction_id": "prediction-a",
            "node_id": "generate",
            "attempt_index": 0,
            "status": "error",
            "failure": {
                "failure_class": "permanent",
                "error_type": "whetstone.eval_failures.PermanentFailureError",
                "underlying_exception_type": (
                    "dr_providers.ProviderFailureError"
                ),
                "message": "provider failure",
                "metadata": {"provider_failure": provider_failure},
            },
            "started_at": timestamp,
            "completed_at": timestamp,
        }
    )
    return node_attempt_row(record)


def _scoring_target(
    *, prediction_id: str = "prediction-a", generation_run_id: str = "run-a"
) -> live_sweep.ScoringTargetSpec:
    return live_sweep.ScoringTargetSpec.model_validate(
        {
            "prediction_id": prediction_id,
            "generation_run_id": generation_run_id,
            "scoring_profile_id": "humaneval",
            "scoring_profile_version": "v1",
            "parser_profile_id": "humaneval-best-effort",
            "parser_version": "v1",
            "dataset_name": "evalplus/humanevalplus",
            "dataset_split": "test",
            "dataset_snapshot": {
                "sha256": "a" * 64,
                "header": {
                    "schema_version": 1,
                    "dataset_id": "evalplus/humanevalplus",
                    "hf_revision": "frozen",
                    "overrides_digest": "b" * 64,
                },
            },
        }
    )


def _scoring_item_ids(
    operation_key: str, targets: tuple[live_sweep.ScoringTargetSpec, ...]
) -> list[str]:
    return [
        live_sweep.item_id(
            operation_key=operation_key, item_key=target.item_key
        )
        for target in targets
    ]


def _scoring_selection_digest(
    targets: tuple[live_sweep.ScoringTargetSpec, ...],
) -> str:
    return live_sweep.sha256_json_digest(
        [target.model_dump(mode="json") for target in targets]
    )


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


def test_generation_shards_are_bounded_stable_and_exhaustive() -> None:
    cells = [_cell(f"cell-{index:03d}") for index in range(217)]
    canary_ids = [str(cell["cell_id"]) for cell in cells[:12]]

    first = live_sweep._generation_shards(
        campaign="campaign", cells=cells, canary_ids=canary_ids
    )
    second = live_sweep._generation_shards(
        campaign="campaign", cells=cells, canary_ids=canary_ids
    )

    assert first == second
    assert [len(shard["cell_ids"]) for shard in first] == [12, 100, 100, 5]
    assert [cell_id for shard in first for cell_id in shard["cell_ids"]] == [
        *canary_ids,
        *(str(cell["cell_id"]) for cell in cells[12:]),
    ]
    assert len({shard["operation_key"] for shard in first}) == 4


def _write_relock_campaign(campaign: Path) -> None:
    campaign.mkdir()
    cells = [
        {
            "cell_id": f"canary-{index}",
            "task_id": "HumanEval/0",
            "repetition_seed": 0,
            "prediction_id": f"prediction-canary-{index}",
        }
        for index in range(12)
    ] + [
        {
            "cell_id": "remaining",
            "task_id": "HumanEval/1",
            "repetition_seed": 0,
            "prediction_id": "prediction-remaining",
        }
    ]
    manifest = campaign / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(cell, sort_keys=True) + "\n" for cell in cells),
        encoding="utf-8",
    )
    (campaign / "campaign-metadata.json").write_text(
        json.dumps({"campaign": "campaign"}), encoding="utf-8"
    )
    (campaign / "canary-12-cells.json").write_text(
        json.dumps({"cell_ids": [cell["cell_id"] for cell in cells[:12]]}),
        encoding="utf-8",
    )
    (campaign / "manifest-index.json").write_text(
        json.dumps(
            {"manifest_sha256": sha256(manifest.read_bytes()).hexdigest()}
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("crash_after_fsync", range(1, 7))
def test_relock_recovers_after_every_durability_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_after_fsync: int,
) -> None:
    campaign = tmp_path / "campaign"
    _write_relock_campaign(campaign)
    real_fsync_directory = live_sweep._fsync_directory
    fsyncs = 0

    def crash_after(path: Path) -> None:
        nonlocal fsyncs
        real_fsync_directory(path)
        fsyncs += 1
        if fsyncs == crash_after_fsync:
            raise RuntimeError("injected relock crash")

    monkeypatch.setattr(live_sweep, "_fsync_directory", crash_after)
    crashed = CliRunner().invoke(
        live_sweep.APP, ["relock-generation-shards", str(campaign)]
    )
    assert crashed.exit_code == 1
    monkeypatch.setattr(
        live_sweep, "_fsync_directory", real_fsync_directory
    )

    recovered = CliRunner().invoke(
        live_sweep.APP, ["relock-generation-shards", str(campaign)]
    )
    assert recovered.exit_code == 0, recovered.output
    pointer_before = (campaign / "generation-lock.json").read_bytes()
    manifest, shards, index = live_sweep._locked_generation_paths(campaign)
    pointer = json.loads(pointer_before)
    assert pointer["sha256"] == {
        manifest.name: sha256(manifest.read_bytes()).hexdigest(),
        shards.name: sha256(shards.read_bytes()).hexdigest(),
        index.name: sha256(index.read_bytes()).hexdigest(),
    }

    repeated = CliRunner().invoke(
        live_sweep.APP, ["relock-generation-shards", str(campaign)]
    )
    assert repeated.exit_code == 0, repeated.output
    assert (campaign / "generation-lock.json").read_bytes() == pointer_before


def test_relock_pointer_is_authoritative_over_legacy_files(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "campaign"
    _write_relock_campaign(campaign)
    first = CliRunner().invoke(
        live_sweep.APP, ["relock-generation-shards", str(campaign)]
    )
    assert first.exit_code == 0, first.output
    authoritative_manifest = live_sweep._locked_generation_paths(campaign)[0]
    (campaign / "manifest.jsonl").write_text("corrupt legacy file\n")

    second = CliRunner().invoke(
        live_sweep.APP, ["relock-generation-shards", str(campaign)]
    )

    assert second.exit_code == 0, second.output
    assert live_sweep._locked_generation_paths(campaign)[0] == (
        authoritative_manifest
    )


def test_scoring_intent_is_replay_safe_and_snapshot_bound(
    tmp_path: Path,
) -> None:
    ledger = SweepLedger(
        tmp_path / "ledger.sqlite3", manifest_hash="manifest-a"
    )
    intent = {
        "operation_key": "scoring-a",
        "generation_cut_digest": "cut-a",
        "selection_digest": _scoring_selection_digest(
            (_scoring_target(),)
        ),
        "snapshot_sha256": "a" * 64,
        "scoring_profile_id": "humaneval",
        "scoring_profile_version": "v1",
        "parser_profile_id": "humaneval-best-effort",
        "parser_version": "v1",
        "item_ids": _scoring_item_ids("scoring-a", (_scoring_target(),)),
        "targets": (_scoring_target(),),
    }
    try:
        ledger.scoring_intent(**intent)
        ledger.scoring_intent(**intent)
        with pytest.raises(ValueError, match="identity changed"):
            ledger.scoring_intent(
                operation_key="scoring-a",
                generation_cut_digest="cut-a",
                selection_digest=_scoring_selection_digest(
                    (_scoring_target(),)
                ),
                snapshot_sha256="b" * 64,
                scoring_profile_id="humaneval",
                scoring_profile_version="v1",
                parser_profile_id="humaneval-best-effort",
                parser_version="v1",
                item_ids=_scoring_item_ids(
                    "scoring-a", (_scoring_target(),)
                ),
                targets=(_scoring_target(),),
            )
        with pytest.raises(ValueError, match="identity changed"):
            ledger.scoring_intent(
                operation_key="scoring-b",
                generation_cut_digest="cut-a",
                selection_digest=_scoring_selection_digest(
                    (_scoring_target(),)
                ),
                snapshot_sha256="a" * 64,
                scoring_profile_id="humaneval",
                scoring_profile_version="v1",
                parser_profile_id="humaneval-best-effort",
                parser_version="v1",
                item_ids=_scoring_item_ids(
                    "scoring-b", (_scoring_target(),)
                ),
                targets=(_scoring_target(),),
            )
        count = ledger.connection.execute(
            "SELECT COUNT(*) FROM sweep_scoring_operations"
        ).fetchone()[0]
        assert count == 1
    finally:
        ledger.close()


def test_response_parse_provider_failure_is_adapter_parse_failure() -> None:
    diagnostics = live_sweep._project_safe_diagnostics(
        node={
            "status": "error",
            "model": "gpt-5.4-nano",
            "output": None,
            "response_metadata": {},
            "failure": {
                "failure_class": "permanent",
                "error_type": "whetstone.eval_failures.PermanentFailureError",
                "underlying_exception_type": (
                    "dr_providers.ProviderFailureError"
                ),
                "metadata": {
                    "provider_failure": {
                        "code": "response_parse_error",
                        "metadata": {
                            "diagnostics": {
                                "response_status": "completed"
                            }
                        },
                    }
                },
            },
        },
        score=None,
    )

    assert diagnostics["adapter_disposition"] == "parse_failure"
    assert diagnostics["typed_failure_code"] == "provider_response_parse"
    assert diagnostics["response_status"] == "completed"
    assert diagnostics["output_field_present"] is False


@pytest.mark.parametrize(
    "provider_code",
    [
        "response_refusal",
        "response_incomplete_no_text",
        "response_failed",
        "response_no_text",
    ],
)
def test_typed_response_outcomes_remain_provider_outcomes(
    provider_code: str,
) -> None:
    diagnostics = live_sweep._project_safe_diagnostics(
        node={
            "status": "error",
            "model": "gpt-5.4-nano",
            "output": None,
            "response_metadata": {},
            "failure": {
                "failure_class": "permanent",
                "error_type": (
                    "whetstone.eval_failures.PermanentFailureError"
                ),
                "underlying_exception_type": (
                    "dr_providers.ProviderFailureError"
                ),
                "metadata": {
                    "provider_failure": {
                        "code": provider_code,
                        "metadata": {
                            "diagnostics": {
                                "response_status": "incomplete",
                                "incomplete_reason": "max_output_tokens",
                                "output_item_types": {"message": 1},
                                "content_part_types": {"refusal": 1},
                                "output_text_len": 0,
                                "refusal_len": 12,
                                "response_id_hash": "0123456789abcdef",
                            }
                        },
                    }
                },
            },
        },
        score=None,
    )

    assert diagnostics["adapter_disposition"] == "provider_failure"
    assert diagnostics["typed_failure_code"] == provider_code
    assert diagnostics["response_status"] == "incomplete"
    assert diagnostics["incomplete_reason"] == "max_output_tokens"
    assert diagnostics["output_item_types"] == {"message": 1}
    assert diagnostics["content_part_types"] == {"refusal": 1}
    assert diagnostics["output_text_len"] == 0
    assert diagnostics["refusal_len"] == 12
    assert diagnostics["response_id_hash"] == "0123456789abcdef"


def test_successful_response_diagnostics_are_projected_safely() -> None:
    diagnostics = live_sweep._project_safe_diagnostics(
        node=_persisted_success_node(
            {
                "status": "PRIVATE_RAW_STATUS",
                "diagnostics": {
                    "response_status": "completed",
                    "output_item_types": {
                        "message": 1,
                        "PRIVATE_ITEM_TYPE": 2,
                    },
                    "content_part_types": {"output_text": 1},
                    "output_text_len": 9,
                    "response_id_hash": "fedcba9876543210",
                },
            }
        ),
        score=None,
    )

    assert diagnostics["adapter_disposition"] == "success"
    assert diagnostics["response_status"] == "completed"
    assert diagnostics["output_item_types"] == {"message": 1}
    assert diagnostics["content_part_types"] == {"output_text": 1}
    assert diagnostics["output_text_len"] == 9
    assert diagnostics["response_id_hash"] == "fedcba9876543210"
    assert "PRIVATE" not in json.dumps(diagnostics, sort_keys=True)


def test_legacy_flattened_response_status_uses_closed_allowlist() -> None:
    diagnostics = live_sweep._project_safe_diagnostics(
        node={
            "status": "success",
            "model": "gpt-5.4-nano",
            "output": {"values": {"text": "generated"}},
            "response_metadata": {"status": "completed"},
            "failure": None,
        },
        score=None,
    )

    assert diagnostics["response_status"] == "completed"


def test_private_raw_status_is_omitted_from_ledger_diagnostics(
    tmp_path: Path,
) -> None:
    diagnostics = live_sweep._project_safe_diagnostics(
        node=_persisted_success_node({"status": "PRIVATE_RAW_STATUS"}),
        score=None,
    )
    assert "response_status" not in diagnostics

    ledger = SweepLedger(
        tmp_path / "ledger.sqlite3", manifest_hash="manifest-a"
    )
    try:
        ledger.record_intent([_cell("a")])
        ledger.reconciliation(
            [
                CellReconciliation(
                    cell_id="a",
                    status="succeeded",
                    platform_attempt=0,
                    retry_count=0,
                    actual_cost=None,
                    provider_tokens={},
                    error_classification=None,
                    diagnostics=diagnostics,
                )
            ]
        )

        serialized = str(ledger.rows()[0]["diagnostics_json"])
        assert "PRIVATE_RAW_STATUS" not in serialized
    finally:
        ledger.close()


def test_success_and_failure_hashes_use_provider_truncated_digest(
    tmp_path: Path,
) -> None:
    response_id = "resp-correlation-1"
    truncated_digest = sha256(response_id.encode()).hexdigest()[:16]

    success = live_sweep._project_safe_diagnostics(
        node=_persisted_success_node(
            {"id": response_id, "status": "completed"}
        ),
        score=None,
    )
    failure = live_sweep._project_safe_diagnostics(
        node=_persisted_failure_node(
            {
                "code": "response_failed",
                "metadata": {
                    "diagnostics": {
                        "response_status": "failed",
                        "response_id_hash": truncated_digest,
                    }
                },
            }
        ),
        score=None,
    )

    assert success["response_id_hash"] == truncated_digest
    assert failure["response_id_hash"] == truncated_digest

    ledger = SweepLedger(
        tmp_path / "ledger.sqlite3", manifest_hash="manifest-a"
    )
    try:
        ledger.record_intent([_cell("a")])
        ledger.reconciliation(
            [
                CellReconciliation(
                    cell_id="a",
                    status="succeeded",
                    platform_attempt=0,
                    retry_count=0,
                    actual_cost=None,
                    provider_tokens={},
                    error_classification=None,
                    diagnostics=success,
                )
            ]
        )

        serialized = str(ledger.rows()[0]["diagnostics_json"])
        assert truncated_digest in serialized
        assert sha256(response_id.encode()).hexdigest() not in serialized
        assert response_id not in serialized
    finally:
        ledger.close()


def test_legacy_flat_failure_status_recovers_via_closed_allowlist() -> None:
    diagnostics = live_sweep._project_safe_diagnostics(
        node=_persisted_failure_node(
            {
                "code": "provider_http_error",
                "metadata": {"response_status": "failed"},
            }
        ),
        score=None,
    )

    assert diagnostics["response_status"] == "failed"


def test_nested_diagnostics_status_wins_over_legacy_flat_key() -> None:
    diagnostics = live_sweep._project_safe_diagnostics(
        node=_persisted_failure_node(
            {
                "code": "response_failed",
                "metadata": {
                    "response_status": "cancelled",
                    "diagnostics": {"response_status": "incomplete"},
                },
            }
        ),
        score=None,
    )

    assert diagnostics["response_status"] == "incomplete"


def test_legacy_flat_raw_status_is_omitted_from_ledger(
    tmp_path: Path,
) -> None:
    diagnostics = live_sweep._project_safe_diagnostics(
        node=_persisted_failure_node(
            {
                "code": "provider_http_error",
                "metadata": {"response_status": "PRIVATE_RAW_STATUS"},
            }
        ),
        score=None,
    )
    assert "response_status" not in diagnostics

    ledger = SweepLedger(
        tmp_path / "ledger.sqlite3", manifest_hash="manifest-a"
    )
    try:
        ledger.record_intent([_cell("a")])
        ledger.reconciliation(
            [
                CellReconciliation(
                    cell_id="a",
                    status="typed_failure",
                    platform_attempt=0,
                    retry_count=0,
                    actual_cost=None,
                    provider_tokens={},
                    error_classification="permanent",
                    diagnostics=diagnostics,
                )
            ]
        )

        serialized = str(ledger.rows()[0]["diagnostics_json"])
        assert "PRIVATE_RAW_STATUS" not in serialized
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


@pytest.mark.parametrize("status", ["in_flight", "incomplete"])
def test_scoring_cut_rejects_nonterminal_or_unreconciled_member(
    tmp_path: Path, status: str
) -> None:
    cell = {
        "cell_id": "cell-a",
        "prediction_id": "prediction-a",
        "operation_key": "generation-a",
        "platform_item_id": "item-a",
        "generation_shard_ordinal": 1,
    }
    ledger = SweepLedger(tmp_path / "ledger.sqlite3", manifest_hash="manifest")
    try:
        ledger.record_intent([cell])
        ledger.submission_intent(
            [cell],
            operation_key="generation-a",
            prediction_ids={"cell-a": "prediction-a"},
        )
        ledger.reconciliation(
            [
                CellReconciliation(
                    "cell-a", status, 0, 0, None, {}, None
                )
            ]
        )
        with pytest.raises(RuntimeError, match="fully reconciled"):
            live_sweep._complete_generation_members([cell], ledger)
    finally:
        ledger.close()


def test_scoring_cut_rejects_partial_member_inventory(tmp_path: Path) -> None:
    cells = [
        {"cell_id": "cell-a", "prediction_id": "prediction-a"},
        {"cell_id": "cell-b", "prediction_id": "prediction-b"},
    ]
    ledger = SweepLedger(tmp_path / "ledger.sqlite3", manifest_hash="manifest")
    try:
        ledger.record_intent([cells[0]])
        with pytest.raises(RuntimeError, match="every locked"):
            live_sweep._complete_generation_members(cells, ledger)
    finally:
        ledger.close()


@contextmanager
def _fake_enqueue_runtime() -> Iterator[SimpleNamespace]:
    """Unit tests never launch DBOS; the live canary proves the runtime."""
    yield SimpleNamespace(
        queue_lookup=None, enqueue_adapter=None, workflow_observer=None
    )


def test_scoring_replays_frozen_intent_once_then_duplicate_is_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    ledger_path = tmp_path / "ledger.sqlite3"
    target = _scoring_target()
    ledger = SweepLedger(ledger_path, manifest_hash="manifest-a")
    try:
        ledger.scoring_intent(
            operation_key="scoring-a",
            generation_cut_digest="cut-a",
            selection_digest=_scoring_selection_digest((target,)),
            snapshot_sha256="a" * 64,
            scoring_profile_id="humaneval",
            scoring_profile_version="v1",
            parser_profile_id="humaneval-best-effort",
            parser_version="v1",
            item_ids=_scoring_item_ids("scoring-a", (target,)),
            targets=(target,),
        )
    finally:
        ledger.close()
    monkeypatch.setattr(
        live_sweep,
        "validate_campaign",
        lambda _path: (
            {"campaign": "campaign", "snapshot": {"sha256": "a" * 64}},
            [{"cell_id": "cell-a", "prediction_id": "prediction-a"}],
            "manifest-a",
        ),
    )
    monkeypatch.setattr(
        live_sweep,
        "create_engine",
        lambda _url: SimpleNamespace(dispose=lambda: None),
    )
    monkeypatch.setattr(
        live_sweep, "resolve_application_database_url", lambda: "sqlite://"
    )
    monkeypatch.setattr(
        live_sweep, "_platform_enqueue_runtime", _fake_enqueue_runtime
    )
    submissions: list[tuple[live_sweep.ScoringTargetSpec, ...]] = []
    monkeypatch.setattr(
        live_sweep,
        "submit_scoring_targets",
        lambda _engine, **kwargs: submissions.append(tuple(kwargs["targets"])),
    )
    arguments = [
        "submit-scoring",
        str(campaign),
        "--execute",
        "--ledger",
        str(ledger_path),
    ]
    runner = CliRunner()

    replay = runner.invoke(live_sweep.APP, arguments)
    duplicate = runner.invoke(live_sweep.APP, arguments)

    assert replay.exit_code == 0, replay.output
    assert duplicate.exit_code == 0, duplicate.output
    assert submissions == [(target,)]
    assert json.loads(duplicate.output)["dispatch"] is False


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
    manifest = live_sweep._locked_generation_paths(target)[0]
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


def test_submit_remaining_fresh_ledger_never_records_canary_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    canary = [
        {
            "cell_id": f"canary-{index}",
            "generation_shard_ordinal": 1,
            "operation_key": "canary-operation",
        }
        for index in range(12)
    ]
    remaining = [
        {
            "cell_id": "remaining",
            "generation_shard_ordinal": 2,
            "operation_key": "remaining-operation",
        }
    ]
    # Deliberately put the canary after paid work: position is not the gate.
    cells = remaining + canary
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
        submitted.extend(str(cell["cell_id"]) for cell in selected)
        for cell in selected:
            cell_id = str(cell["cell_id"])
            ledger.submitted(
                [cell],
                operation_key=str(cell["operation_key"]),
                prediction_ids={cell_id: f"prediction-{cell_id}"},
            )

    monkeypatch.setattr(live_sweep, "_submit", fake_submit)
    monkeypatch.setattr(
        live_sweep,
        "reconcile_ledger",
        lambda ledger, *, engine: [
            CellReconciliation(
                str(row["cell_id"]), "succeeded", 0, 0, None, {}, None
            )
            for row in ledger.rows()
        ],
    )
    monkeypatch.setattr(
        live_sweep,
        "create_engine",
        lambda _url: SimpleNamespace(dispose=lambda: None),
    )
    monkeypatch.setattr(
        live_sweep, "resolve_application_database_url", lambda: "sqlite://"
    )

    result = CliRunner().invoke(
        live_sweep.APP,
        [
            "submit-remaining",
            str(campaign),
            "--execute",
            "--ledger",
            str(tmp_path / "ledger.sqlite3"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert submitted == ["remaining"]
    ledger = SweepLedger(
        tmp_path / "ledger.sqlite3", manifest_hash="manifest-a"
    )
    try:
        assert [str(row["cell_id"]) for row in ledger.rows()] == ["remaining"]
    finally:
        ledger.close()


def test_submit_remaining_replays_only_prejournaled_canary_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    cells = [
        {
            "cell_id": f"canary-{index}",
            "generation_shard_ordinal": 1,
            "operation_key": "canary-operation",
        }
        for index in range(12)
    ]
    prediction_ids = {
        str(cell["cell_id"]): f"prediction-{cell['cell_id']}"
        for cell in cells
    }
    ledger_path = tmp_path / "ledger.sqlite3"
    ledger = SweepLedger(ledger_path, manifest_hash="manifest-a")
    try:
        ledger.record_intent(cells)
    finally:
        ledger.close()
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
        submitted.extend(str(cell["cell_id"]) for cell in selected)
        ledger.submitted(
            selected,
            operation_key="canary-operation",
            prediction_ids=prediction_ids,
        )

    monkeypatch.setattr(live_sweep, "_submit", fake_submit)
    monkeypatch.setattr(
        live_sweep,
        "reconcile_ledger",
        lambda ledger, *, engine: [
            CellReconciliation(
                str(row["cell_id"]), "succeeded", 0, 0, None, {}, None
            )
            for row in ledger.rows()
        ],
    )
    monkeypatch.setattr(
        live_sweep,
        "create_engine",
        lambda _url: SimpleNamespace(dispose=lambda: None),
    )
    monkeypatch.setattr(
        live_sweep, "resolve_application_database_url", lambda: "sqlite://"
    )

    result = CliRunner().invoke(
        live_sweep.APP,
        [
            "submit-remaining",
            str(campaign),
            "--execute",
            "--ledger",
            str(ledger_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert submitted == [str(cell["cell_id"]) for cell in cells]


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
    cells = [
        {
            "cell_id": "canary",
            "generation_shard_ordinal": 1,
        },
        *[
            {
                "cell_id": f"cell-{index}",
                "generation_shard_ordinal": 2 + index,
            }
            for index in range(3)
        ],
    ]
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
    monkeypatch.setattr(
        live_sweep,
        "resolve_application_database_url",
        lambda: "sqlite://",
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
                    "whetstone.eval_failures.exceptions.PredictionParseError"
                ),
            },
            "parse_failure",
            "prediction_parse",
        ),
        (
            {
                "failure_class": "transient",
                "error_type": (
                    "whetstone.eval_failures.exceptions.TransientFailureError"
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
                    "whetstone.eval_failures.exceptions.EmptyGenerationError"
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
    assert (
        diagnostics["response_id_hash"]
        == sha256(b"response-123").hexdigest()[:16]
    )
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
