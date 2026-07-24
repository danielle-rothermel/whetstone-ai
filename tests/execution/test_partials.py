"""Current-schema partial-call persistence and resume identities."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from whetstone.execution.partials import (
    PARTIAL_SCHEMA,
    PartialCallRecord,
    PartialLog,
    partial_key,
)


def test_append_load_and_resume_key_round_trip(tmp_path: Path) -> None:
    log = PartialLog(path=tmp_path / "calls.partial.jsonl")
    record = PartialCallRecord(
        phase="official",
        instance_id="task-1",
        unit="candidate-1",
        repeat_id=2,
        split_role="official",
        score=1.0,
        total_tokens=42,
        output_text="answer",
        finish_reason="stop",
    )
    log.append(record)

    loaded = log.load()
    assert len(loaded) == 1
    assert loaded[0] == record.model_copy(update={"at": loaded[0].at})
    assert loaded[0].schema_name == PARTIAL_SCHEMA
    assert loaded[0].at is not None
    assert log.recorded_keys() == {
        partial_key("official", "task-1", "candidate-1", 2)
    }


def test_latest_complete_row_wins_by_key(tmp_path: Path) -> None:
    log = PartialLog(path=tmp_path / "calls.partial.jsonl")
    base = PartialCallRecord(
        phase="internal",
        instance_id="task-1",
        unit="candidate-1",
        repeat_id=0,
        score=0.0,
    )
    log.append(base)
    log.append(base.model_copy(update={"score": 1.0}))
    assert [record.score for record in log.load()] == [1.0]


def test_persisted_row_contains_provenance_and_cache_fields(
    tmp_path: Path,
) -> None:
    path = tmp_path / "calls.partial.jsonl"
    log = PartialLog(path=path)
    provider_error: dict[str, object] = {
        "failure_class": "provider-rejection",
        "message": "bad request",
    }
    log.append(
        PartialCallRecord(
            phase="internal",
            instance_id="task-1",
            unit="candidate-1",
            repeat_id=0,
            failed=True,
            failure_code="provider-rejection",
            provider_error=provider_error,
            cache_hit=True,
            cache_source_phase="official",
            cache_source_unit="candidate-original",
            cache_source_call_id="original-call",
            cache_source_at="2026-07-24T12:00:00+00:00",
            latency_s=None,
        )
    )
    data = json.loads(path.read_text())
    assert data["candidate_id"] == data["unit"]
    assert data["repeat"] == data["repeat_id"]
    assert data["provider_error"] == provider_error
    assert data["cache_hit"] is True
    assert data["cache_source_call_id"] == "original-call"
    assert data["latency_s"] is None
    assert PartialLog(path=path).load()[0].provider_error == provider_error


def test_old_or_malformed_rows_fail_loudly(tmp_path: Path) -> None:
    path = tmp_path / "calls.partial.jsonl"
    path.write_text(
        json.dumps(
            {
                "phase": "internal",
                "instance_id": "task-1",
                "unit": "candidate-1",
                "repeat_id": 0,
                "score": 1.0,
                "raw_response": "",
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="current schema"):
        PartialLog(path=path).load()


def test_current_schema_rejects_inconsistent_mirror_fields(
    tmp_path: Path,
) -> None:
    path = tmp_path / "calls.partial.jsonl"
    log = PartialLog(path=path)
    log.append(
        PartialCallRecord(
            phase="internal",
            instance_id="task-1",
            unit="candidate-1",
            repeat_id=0,
        )
    )
    data = json.loads(path.read_text())
    data["candidate_id"] = "different"
    path.write_text(json.dumps(data) + "\n")
    with pytest.raises(ValueError, match="candidate_id"):
        log.load()


def test_cache_hit_requires_complete_provenance_and_null_latency() -> None:
    with pytest.raises(ValueError, match="complete provenance"):
        PartialCallRecord(
            phase="internal",
            instance_id="task-1",
            unit="candidate-1",
            repeat_id=0,
            cache_hit=True,
            latency_s=0.0,
        )


def test_missing_log_and_delete_are_idempotent(tmp_path: Path) -> None:
    log = PartialLog(path=tmp_path / "calls.partial.jsonl")
    assert log.load() == []
    assert log.recorded_keys() == set()
    log.delete()
    log.append(
        PartialCallRecord(
            phase="internal",
            instance_id="task-1",
            unit="candidate-1",
            repeat_id=0,
        )
    )
    log.delete()
    assert not log.path.exists()
