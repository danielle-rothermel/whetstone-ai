"""Ledger schema validation + resumability tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from whetstone.runner.ledger import (
    FULL_CONFIG_EVAL_HASH,
    CellArtifacts,
    CellModels,
    CellRecord,
    EnvOfficialCache,
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
        # Statistical-confidence upgrade fields.
        "naive_ci95", "ceiling_ci95", "delta_ci95", "headroom_delta",
        "headroom_ci95", "no_demonstrable_headroom", "official_repeats_used",
        "escalated", "escalation_note", "pooled_observation_counts",
        # Reduced-sampling overrides (--official-n / --official-repeats).
        "sampling_overrides",
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
    for status in (
        "improved", "inconclusive", "no-improvement", "plumbing-retry",
        "halted",
    ):
        assert _record(status=status).status == status


def test_inconclusive_counts_as_completed(tmp_path: Path) -> None:
    ledger = Ledger(root=tmp_path)
    ledger.append_cell(_record(status="inconclusive"))
    # An inconclusive cell is terminal (resolved), so it skips on resume.
    assert ledger.is_completed("copro", "c11", 0)


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


def test_round_trip_preserves_stats_upgrade_fields() -> None:
    record = _record(
        status="inconclusive",
        naive_ci95=(0.1, 0.4),
        ceiling_ci95=(0.7, 1.0),
        delta_ci95=(-0.05, 0.35),
        headroom_delta=0.6,
        headroom_ci95=(0.3, 0.9),
        no_demonstrable_headroom=False,
        official_repeats_used=10,
        escalated=True,
        escalation_note="escalated: doubled official repeats and pooled",
        pooled_observation_counts={"naive": 60, "best": 60},
    )
    restored = CellRecord.model_validate_json(record.to_line())
    assert restored == record
    assert restored.delta_ci95 == (-0.05, 0.35)
    assert restored.headroom_ci95 == (0.3, 0.9)
    assert restored.escalated is True
    assert restored.official_repeats_used == 10
    assert restored.pooled_observation_counts == {"naive": 60, "best": 60}


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


def _cache(**overrides: object) -> EnvOfficialCache:
    base = {
        "env": "c11",
        "naive_official": 0.2,
        "ceiling_official": 0.9,
        "naive_per_task": (0.2, 0.2),
        "ceiling_per_task": (0.9, 0.9),
        "official_repeats_used": 5,
        "task_model": "openai/gpt-5-nano",
        "eval_config_hash": "full-config-hash",
    }
    base.update(overrides)
    return EnvOfficialCache(**base)  # type: ignore[arg-type]


def test_env_cache_keyed_by_eval_config_hash(tmp_path: Path) -> None:
    # A cell whose official Eval Config identity differs (reduced sampling ->
    # a different eval_config_hash) MISSES the full-config cache entry for the
    # same (env, task_model); the matching-hash read HITS.
    ledger = Ledger(root=tmp_path)
    ledger.append_env_cache(_cache(eval_config_hash="full-config-hash"))
    hit = ledger.env_cache_for(
        "c11",
        task_model="openai/gpt-5-nano",
        eval_config_hash="full-config-hash",
    )
    assert hit is not None
    miss = ledger.env_cache_for(
        "c11",
        task_model="openai/gpt-5-nano",
        eval_config_hash="reduced-config-hash",
    )
    assert miss is None


def test_old_cache_line_defaults_to_full_config_sentinel() -> None:
    # A pre-migration cache line (no eval_config_hash field) resolves to the
    # full-config sentinel.
    line = (
        '{"env": "c11", "naive_official": 0.2, "ceiling_official": 0.9, '
        '"naive_per_task": [0.2], "ceiling_per_task": [0.9], '
        '"official_repeats_used": 5, "task_model": "openai/gpt-5-nano"}'
    )
    record = EnvOfficialCache.model_validate_json(line)
    assert record.eval_config_hash == FULL_CONFIG_EVAL_HASH


def test_old_sentinel_line_matches_only_default_config_reads(
    tmp_path: Path,
) -> None:
    # An old sentinel cache line is matchable ONLY by a full-config
    # (default_config=True) read -- a reduced-sampling read never reuses it.
    ledger = Ledger(root=tmp_path)
    ledger.append_env_cache(_cache(eval_config_hash=FULL_CONFIG_EVAL_HASH))
    # Reduced-sampling read (a concrete hash, default_config=False): MISS.
    assert (
        ledger.env_cache_for(
            "c11",
            task_model="openai/gpt-5-nano",
            eval_config_hash="reduced-config-hash",
        )
        is None
    )
    # Full-config read (default_config=True): HIT regardless of requested hash.
    assert (
        ledger.env_cache_for(
            "c11",
            task_model="openai/gpt-5-nano",
            eval_config_hash="full-config-hash",
            default_config=True,
        )
        is not None
    )


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


def _spend(ledger: Ledger, cell_id: str, phase: str, remaining: float) -> None:
    ledger.append_spend(
        SpendRecord(
            cell_id=cell_id, phase=phase, lane="openrouter",
            total_credits=710.0, total_usage=710.0 - remaining,
            remaining_usd=remaining,
        )
    )


def test_spend_for_cell_sums_clean_before_after(tmp_path: Path) -> None:
    ledger = Ledger(root=tmp_path)
    _spend(ledger, "eval:c11:a0", "before", 90.0)
    _spend(ledger, "eval:c11:a0", "after", 88.5)
    total, gaps = ledger.spend_for_cell("eval:c11:a0")
    assert total == pytest.approx(1.5)
    assert gaps == []


def test_spend_for_cell_includes_crashed_attempt(tmp_path: Path) -> None:
    # FIX 8: attempt 0 CRASHED (before at 90.0, no after) burning 1.0 before
    # dying; the NEXT snapshot (attempt 1 before at 89.0) bounds it. Attempt 1
    # then completes (before 89.0 -> after 87.0 = 2.0). The cell's total spend
    # sums BOTH attempts: 1.0 (crashed) + 2.0 (completed) = 3.0.
    ledger = Ledger(root=tmp_path)
    _spend(ledger, "eval:c11:a0", "before", 90.0)   # attempt 0 before
    _spend(ledger, "eval:c11:a1", "before", 89.0)   # attempt 0 crashed here
    _spend(ledger, "eval:c11:a1", "after", 87.0)    # attempt 1 completed
    total, gaps = ledger.spend_for_cell("eval:c11:a0")
    # Only cell_id 'eval:c11:a0' is summed here (attempt 0's crashed 1.0).
    assert total == pytest.approx(1.0)
    assert any("crashed" in g for g in gaps)
    # And the a1 attempt sums its own 2.0.
    total_a1, gaps_a1 = ledger.spend_for_cell("eval:c11:a1")
    assert total_a1 == pytest.approx(2.0)
    assert gaps_a1 == []


def test_spend_for_cell_pairs_by_cell_id_under_interleaving(
    tmp_path: Path,
) -> None:
    # The c18:a1 spend=0.0 defect: CONCURRENT cells interleave their
    # before/after snapshots into one shared spend.jsonl, so the record right
    # after a cell's `before` may be a DIFFERENT cell's `before`. Pairing must
    # be by cell_id (this cell's own next `after`), not the globally-next
    # record. Here c18:a1 before -> [c23 before, c19 before] -> c18:a1 after:
    # the true delta (0.804) must be attributed, not $0.00.
    ledger = Ledger(root=tmp_path)
    _spend(ledger, "eval:c18:a1", "before", 87.086)   # this cell's before
    _spend(ledger, "eval:c23:a0", "before", 87.086)   # a concurrent cell
    _spend(ledger, "eval:c19:a0", "before", 87.086)   # another concurrent cell
    _spend(ledger, "eval:c18:a1", "after", 86.282)    # this cell's own after
    total, gaps = ledger.spend_for_cell("eval:c18:a1")
    assert total == pytest.approx(0.804, abs=1e-3)
    assert gaps == []


def test_spend_for_cell_reports_unbounded_trailing_before(
    tmp_path: Path,
) -> None:
    # A before with nothing after it (still running / crashed last-in-file)
    # cannot be bounded -> reported as a gap, contributes 0.
    ledger = Ledger(root=tmp_path)
    _spend(ledger, "eval:c11:a0", "before", 90.0)
    total, gaps = ledger.spend_for_cell("eval:c11:a0")
    assert total == pytest.approx(0.0)
    assert gaps and "no following snapshot" in gaps[0]
