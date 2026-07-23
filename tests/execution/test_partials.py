"""The append-only ``.partial.jsonl`` log: append, load, and resume keys."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from whetstone.execution.partials import (
    PARTIAL_SCHEMA,
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
    # append() stamps a real ISO ``at`` + the versioned ``schema`` at write
    # time (task 26); the other fields round-trip verbatim.
    assert len(loaded) == 1
    assert loaded[0] == replace(
        rec, at=loaded[0].at, schema=loaded[0].schema
    )
    assert loaded[0].at is not None and loaded[0].schema == PARTIAL_SCHEMA
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
    loaded = log.load()
    assert loaded == [replace(rec, at=loaded[0].at, schema=loaded[0].schema)]
    assert loaded[0].failure_code == "rate-limit" and loaded[0].failed


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


def test_finalized_superset_names_persisted(tmp_path: Path) -> None:
    # Task 26 item 7: the finalized-row field NAMES ride alongside the resume
    # key, output_text is persisted, and per-call provenance lands -- so a
    # consumer never remaps the partial schema.
    log = PartialLog(path=tmp_path / "c.partial.jsonl")
    log.append(PartialCallRecord(
        phase="official", instance_id="i1", unit="cand-a", repeat_id=2,
        score=1.0, split_role="official_naive", output_text="def f(): ...",
        finish_reason="stop", provider_error=None,
    ))
    raw = (tmp_path / "c.partial.jsonl").read_text().splitlines()[0]
    line = json.loads(raw)
    assert line["candidate_id"] == "cand-a"
    assert line["repeat"] == 2
    assert line["split_role"] == "official_naive"
    assert line["output_text"] == "def f(): ..."
    assert line["finish_reason"] == "stop"
    assert line["schema"] == PARTIAL_SCHEMA
    assert line["at"] is not None


def test_provider_error_and_finish_reason_round_trip(tmp_path: Path) -> None:
    log = PartialLog(path=tmp_path / "c.partial.jsonl")
    err: dict[str, object] = {
        "failure_class": "provider-rejection", "message": "400 bad request",
    }
    rec = PartialCallRecord(
        phase="cell", instance_id="i", unit="u", repeat_id=0,
        failed=True, failure_code="permanent", provider_error=err,
        finish_reason="length",
    )
    log.append(rec)
    loaded = log.load()[0]
    assert loaded.provider_error == err
    assert loaded.finish_reason == "length"


def test_legacy_row_reads_output_text_as_null(tmp_path: Path) -> None:
    # A legacy row (empty raw_response, no output_text) reads back as
    # output_text=None -- the honest "not recorded" state, never "".
    path = tmp_path / "c.partial.jsonl"
    path.write_text(json.dumps({
        "phase": "cell", "instance_id": "i", "unit": "u", "repeat_id": 0,
        "score": 1.0, "raw_response": "",
    }) + "\n")
    loaded = PartialLog(path=path).load()[0]
    assert loaded.output_text is None
