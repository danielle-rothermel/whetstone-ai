"""The append-only ``.partial.jsonl`` log: append, load, and resume keys."""

from __future__ import annotations

from pathlib import Path

from whetstone.execution.partials import (
    PartialCallRecord,
    PartialLog,
    partial_key,
)


def test_append_then_load_roundtrips(tmp_path: Path) -> None:
    log = PartialLog(path=tmp_path / "c.partial.jsonl")
    rec = PartialCallRecord(
        phase="cell", instance_id="i1", unit="cand", repeat_id=0,
        score=1.0, total_tokens=42,
    )
    log.append(rec)
    loaded = log.load()
    assert loaded == [rec]
    assert log.recorded_keys() == {partial_key("cell", "i1", "cand", 0)}


def test_last_write_wins_by_key(tmp_path: Path) -> None:
    log = PartialLog(path=tmp_path / "c.partial.jsonl")
    log.append(PartialCallRecord(
        phase="pilot", instance_id="i1", unit="naive", repeat_id=0, score=0.0
    ))
    log.append(PartialCallRecord(
        phase="pilot", instance_id="i1", unit="naive", repeat_id=0, score=1.0
    ))
    loaded = log.load()
    assert len(loaded) == 1
    assert loaded[0].score == 1.0


def test_failed_record_roundtrips_with_code(tmp_path: Path) -> None:
    log = PartialLog(path=tmp_path / "c.partial.jsonl")
    rec = PartialCallRecord(
        phase="cell", instance_id="i2", unit="cand", repeat_id=1,
        score=None, failed=True, failure_code="rate-limit",
    )
    log.append(rec)
    assert log.load() == [rec]


def test_missing_log_loads_empty(tmp_path: Path) -> None:
    log = PartialLog(path=tmp_path / "absent.partial.jsonl")
    assert log.load() == []
    assert log.recorded_keys() == set()


def test_delete_removes_the_log(tmp_path: Path) -> None:
    log = PartialLog(path=tmp_path / "c.partial.jsonl")
    log.append(PartialCallRecord(
        phase="cell", instance_id="i", unit="u", repeat_id=0, score=1.0
    ))
    assert log.path.exists()
    log.delete()
    assert not log.path.exists()
    log.delete()  # idempotent
