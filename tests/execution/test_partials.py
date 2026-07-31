"""Durable checksummed partial-call persistence and resume identities."""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import stat
from pathlib import Path

import pytest

from tests.execution.storage_workers import (
    append_partial_worker,
    hold_partial_lock,
    run_partial_operation,
    write_torn_partial_worker,
)
from whetstone.execution.partials import (
    PARTIAL_FRAME_SCHEMA,
    PARTIAL_SCHEMA,
    PartialCallRecord,
    PartialLog,
    partial_key,
)


def _record(
    *,
    unit: str = "candidate-1",
    repeat_id: int = 0,
    at: str | None = None,
) -> PartialCallRecord:
    return PartialCallRecord(
        phase="internal",
        instance_id="task-1",
        unit=unit,
        repeat_id=repeat_id,
        at=at,
    )


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _rewrite_frame(
    path: Path,
    *,
    record_update: dict[str, object],
    refresh_checksum: bool,
) -> None:
    frame = json.loads(path.read_text())
    frame["record"].update(record_update)
    if refresh_checksum:
        frame["checksum"] = hashlib.sha256(
            _canonical_json_bytes(frame["record"])
        ).hexdigest()
    path.write_bytes(_canonical_json_bytes(frame) + b"\n")


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
    base = _record()
    log.append(base.model_copy(update={"score": 0.0}))
    log.append(base.model_copy(update={"score": 1.0}))
    assert [record.score for record in log.load()] == [1.0]


def test_v2_frame_golden_pins_schema_fields_and_checksum(
    tmp_path: Path,
) -> None:
    path = tmp_path / "calls.partial.jsonl"
    PartialLog(path=path).append(_record(at="2026-07-31T12:00:00+00:00"))
    frame = json.loads(path.read_text())

    assert set(frame) == {"schema", "checksum", "record"}
    assert frame["schema"] == "whetstone.execution.partial_frame/v2"
    assert frame["schema"] == PARTIAL_FRAME_SCHEMA
    assert frame["record"]["schema"] == "whetstone.execution.partial_call/v2"
    assert frame["record"]["schema"] == PARTIAL_SCHEMA
    assert set(frame["record"]) == {
        "schema",
        "phase",
        "instance_id",
        "unit",
        "candidate_id",
        "repeat_id",
        "repeat",
        "split_role",
        "score",
        "failed",
        "failure_code",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "reasoning_tokens",
        "latency_s",
        "output_text",
        "raw_response",
        "finish_reason",
        "provider_error",
        "at",
        "cache_hit",
        "cache_source_phase",
        "cache_source_unit",
        "cache_source_call_id",
        "cache_source_at",
    }
    assert (
        frame["checksum"]
        == "c583d4f5f56b2bcdbb4cc6a90a033002d499ec9f45b6004f98a28f9bed43a949"
    )


def test_persisted_record_contains_provenance_and_cache_fields(
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
    frame = json.loads(path.read_text())
    data = frame["record"]
    assert data["candidate_id"] == data["unit"]
    assert data["repeat"] == data["repeat_id"]
    assert data["provider_error"] == provider_error
    assert data["cache_hit"] is True
    assert data["cache_source_call_id"] == "original-call"
    assert data["latency_s"] is None
    assert PartialLog(path=path).load()[0].provider_error == provider_error


def test_plain_nonframe_rows_fail_loudly(tmp_path: Path) -> None:
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
    with pytest.raises(ValueError, match="partial frame"):
        PartialLog(path=path).load()


def test_complete_frame_rejects_inconsistent_mirror_fields(
    tmp_path: Path,
) -> None:
    path = tmp_path / "calls.partial.jsonl"
    log = PartialLog(path=path)
    log.append(_record())
    _rewrite_frame(
        path,
        record_update={"candidate_id": "different"},
        refresh_checksum=True,
    )
    with pytest.raises(ValueError, match="candidate_id"):
        log.load()


def test_newline_terminated_checksum_corruption_is_fatal(
    tmp_path: Path,
) -> None:
    path = tmp_path / "calls.partial.jsonl"
    log = PartialLog(path=path)
    log.append(_record())
    _rewrite_frame(
        path,
        record_update={"score": 1.0},
        refresh_checksum=False,
    )
    assert path.read_bytes().endswith(b"\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        log.load()


def test_only_final_unterminated_fragment_is_ignored_and_repaired(
    tmp_path: Path,
) -> None:
    path = tmp_path / "calls.partial.jsonl"
    log = PartialLog(path=path)
    log.append(_record(unit="first", repeat_id=0))
    with path.open("ab") as handle:
        handle.write(b'{"schema":"whetstone.execution.partial_frame/v2"')
        handle.flush()
        os.fsync(handle.fileno())

    assert [record.unit for record in log.load()] == ["first"]
    log.append(_record(unit="second", repeat_id=1))
    assert [record.unit for record in log.load()] == ["first", "second"]
    assert path.read_bytes().count(b"\n") == 2


def test_killed_partial_writer_tail_is_recovered_before_next_append(
    tmp_path: Path,
) -> None:
    path = tmp_path / "calls.partial.jsonl"
    log = PartialLog(path=path)
    log.append(_record(unit="first", repeat_id=0))
    context = multiprocessing.get_context("spawn")
    started = context.Event()
    writer = context.Process(
        target=write_torn_partial_worker,
        args=(str(path), started),
    )
    writer.start()
    try:
        assert started.wait(timeout=10)
        writer.terminate()
        writer.join(timeout=10)
    finally:
        if writer.is_alive():
            writer.kill()
            writer.join(timeout=10)
    assert writer.exitcode is not None

    assert [record.unit for record in log.load()] == ["first"]
    log.append(_record(unit="second", repeat_id=1))
    restarted = PartialLog(path=path)
    assert [record.unit for record in restarted.load()] == [
        "first",
        "second",
    ]


def test_multiprocess_multi_megabyte_appends_all_validate(
    tmp_path: Path,
) -> None:
    path = tmp_path / "calls.partial.jsonl"
    context = multiprocessing.get_context("spawn")
    worker_count = 6
    barrier = context.Barrier(worker_count)
    processes = [
        context.Process(
            target=append_partial_worker,
            args=(str(path), worker_id, 1024 * 1024, barrier),
        )
        for worker_id in range(worker_count)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0

    loaded = PartialLog(path=path).load()
    assert len(loaded) == worker_count
    assert {len(record.output_text or "") for record in loaded} == {
        1024 * 1024
    }


def test_append_survives_immediate_child_hard_exit(tmp_path: Path) -> None:
    path = tmp_path / "calls.partial.jsonl"
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=append_partial_worker,
        args=(str(path), 1, 32),
        kwargs={"exit_immediately": True},
    )
    process.start()
    process.join(timeout=10)
    assert process.exitcode == 0
    assert [record.unit for record in PartialLog(path=path).load()] == [
        "candidate-1"
    ]


@pytest.mark.parametrize("operation", ["append", "load", "delete"])
def test_separate_instances_serialize_all_operations(
    tmp_path: Path,
    operation: str,
) -> None:
    path = tmp_path / "calls.partial.jsonl"
    PartialLog(path=path).append(_record())
    context = multiprocessing.get_context("spawn")
    entered = context.Event()
    release = context.Event()
    holder = context.Process(
        target=hold_partial_lock,
        args=(str(path), entered, release),
    )
    holder.start()
    assert entered.wait(timeout=10)

    output = context.Queue()
    attempted = context.Event()
    acquired = context.Event()
    operation_process = context.Process(
        target=run_partial_operation,
        args=(str(path), operation, output, attempted, acquired),
    )
    operation_process.start()
    try:
        assert attempted.wait(timeout=10)
        assert not acquired.is_set()
        release.set()
        assert acquired.wait(timeout=10)
        holder.join(timeout=10)
        operation_process.join(timeout=10)
    finally:
        release.set()
        for process in (holder, operation_process):
            if process.is_alive():
                process.kill()
                process.join(timeout=10)
    assert holder.exitcode == 0
    assert operation_process.exitcode == 0
    assert output.get(timeout=5) in {"appended", "deleted", 1}


def test_short_writes_are_completed(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "calls.partial.jsonl"
    original_write = os.write

    def short_write(fd: int, body) -> int:
        return original_write(fd, body[:7])

    monkeypatch.setattr("whetstone.execution.partials.os.write", short_write)
    PartialLog(path=path).append(_record())
    loaded = PartialLog(path=path).load()
    assert len(loaded) == 1
    assert loaded[0].unit == "candidate-1"


@pytest.mark.parametrize("value", ["", "not-a-timestamp", "2026-07-31"])
def test_invalid_at_is_rejected_before_any_bytes(
    tmp_path: Path,
    value: str,
) -> None:
    path = tmp_path / "calls.partial.jsonl"
    with pytest.raises(ValueError, match="partial row at"):
        PartialCallRecord(
            phase="internal",
            instance_id="task-1",
            unit="candidate-1",
            repeat_id=0,
            at=value,
        )
    assert not path.exists()


def test_append_revalidates_at_before_opening_log(tmp_path: Path) -> None:
    path = tmp_path / "calls.partial.jsonl"
    invalid = PartialCallRecord.model_construct(
        phase="internal",
        instance_id="task-1",
        unit="candidate-1",
        repeat_id=0,
        at="",
    )
    with pytest.raises(ValueError, match="non-empty timestamp"):
        PartialLog(path=path).append(invalid)
    assert not path.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("score", math.nan),
        ("latency_s", math.inf),
        ("provider_error", {"outer": [{"inner": -math.inf}]}),
    ],
)
def test_non_finite_numbers_are_rejected_at_record_boundary(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match="finite numbers"):
        PartialCallRecord.model_validate(
            {
                "phase": "internal",
                "instance_id": "task-1",
                "unit": "candidate-1",
                "repeat_id": 0,
                field: value,
            }
        )


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_nonstandard_json_numbers_fail_loudly_on_load(
    tmp_path: Path,
    constant: str,
) -> None:
    path = tmp_path / "calls.partial.jsonl"
    record = _record(at="2026-07-31T12:00:00+00:00").as_dict()
    record["provider_error"] = {"nested": float(constant)}
    frame = {
        "schema": PARTIAL_FRAME_SCHEMA,
        "checksum": hashlib.sha256(_canonical_json_bytes(record)).hexdigest(),
        "record": record,
    }
    path.write_bytes(_canonical_json_bytes(frame) + b"\n")

    with pytest.raises(ValueError, match="invalid partial JSON"):
        PartialLog(path=path).load()


@pytest.mark.parametrize("storage_kind", ["data", "lock"])
def test_partial_symlinks_do_not_touch_external_target(
    tmp_path: Path,
    storage_kind: str,
) -> None:
    path = tmp_path / "calls.partial.jsonl"
    log = PartialLog(path=path)
    victim = tmp_path / f"{storage_kind}-victim"
    body = f"{storage_kind}-external".encode()
    victim.write_bytes(body)
    victim.chmod(0o644)
    target = path if storage_kind == "data" else log._lock_path
    target.symlink_to(victim)

    with pytest.raises(OSError):
        log.append(_record())
    assert victim.read_bytes() == body
    assert stat.S_IMODE(victim.stat().st_mode) == 0o644


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


def test_creation_and_delete_fsync_parent_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[Path] = []
    monkeypatch.setattr(
        "whetstone.execution.partials.fsync_parent_directory",
        observed.append,
    )
    path = tmp_path / "calls.partial.jsonl"
    log = PartialLog(path=path)
    log.append(_record())
    log.append(_record(unit="second", repeat_id=1))
    log.delete()
    assert observed == [path, path]


def test_missing_log_and_delete_are_idempotent(tmp_path: Path) -> None:
    log = PartialLog(path=tmp_path / "calls.partial.jsonl")
    assert log.load() == []
    assert log.recorded_keys() == set()
    log.delete()
    log.append(_record())
    log.delete()
    assert not log.path.exists()
