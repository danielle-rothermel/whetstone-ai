from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import stat
from pathlib import Path

import pytest

import whetstone.execution.partials as partials_module
from tests.execution.storage_workers import (
    append_partial_worker,
    hold_partial_lock,
    run_partial_operation,
    write_torn_partial_worker,
)
from tests.optimization.processes import join_processes, terminate_processes
from whetstone.execution._file_lock import PrivateDirectory
from whetstone.execution.partials import (
    PARTIAL_FRAME_SCHEMA,
    PARTIAL_SCHEMA,
    PARTIAL_STORAGE_THREAT_MODEL,
    PartialCallRecord,
    PartialLog,
    partial_key,
)

REQUEST_IDENTITY_A = "a" * 64
REQUEST_IDENTITY_B = "b" * 64


def _record(
    *,
    unit: str = "candidate-1",
    repeat_id: int = 0,
    request_identity: str = REQUEST_IDENTITY_A,
    at: str | None = None,
) -> PartialCallRecord:
    return PartialCallRecord(
        phase="internal",
        instance_id="task-1",
        unit=unit,
        repeat_id=repeat_id,
        request_identity=request_identity,
        redrive_pending=False,
        at=at,
    )


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _record_files(path: Path) -> list[Path]:
    return sorted(path.glob("*.json"))


def _only_record_file(path: Path) -> Path:
    files = _record_files(path)
    assert len(files) == 1
    return files[0]


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
    log = PartialLog(path=tmp_path / "calls.partial")
    record = PartialCallRecord(
        phase="official",
        instance_id="task-1",
        unit="candidate-1",
        repeat_id=2,
        request_identity=REQUEST_IDENTITY_A,
        redrive_pending=False,
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
        partial_key(
            "official",
            "task-1",
            "candidate-1",
            2,
            REQUEST_IDENTITY_A,
        )
    }


def test_latest_complete_record_wins_for_same_exact_key(
    tmp_path: Path,
) -> None:
    log = PartialLog(path=tmp_path / "calls.partial")
    base = _record()
    log.append(base.model_copy(update={"score": 0.0}))
    log.append(base.model_copy(update={"score": 1.0}))

    assert [record.score for record in log.load()] == [1.0]
    assert len(_record_files(log.path)) == 1


def test_distinct_requests_at_same_coordinates_both_survive(
    tmp_path: Path,
) -> None:
    log = PartialLog(path=tmp_path / "calls.partial")
    log.append(_record(request_identity=REQUEST_IDENTITY_A))
    log.append(_record(request_identity=REQUEST_IDENTITY_B))

    assert {record.request_identity for record in log.load()} == {
        REQUEST_IDENTITY_A,
        REQUEST_IDENTITY_B,
    }


def test_v3_frame_and_filename_golden_pin_persisted_contract(
    tmp_path: Path,
) -> None:
    log = PartialLog(path=tmp_path / "calls.partial")
    log.append(_record(at="2026-07-31T12:00:00+00:00"))
    entry = _only_record_file(log.path)
    frame = json.loads(entry.read_text())

    assert entry.name == (
        "c982833de29102cb2f7c7e06975853e42c590c4c3b5a4cf1ab63c862430200b7.json"
    )
    assert set(frame) == {"schema", "checksum", "record"}
    assert frame["schema"] == "whetstone.execution.partial_frame/v3"
    assert frame["schema"] == PARTIAL_FRAME_SCHEMA
    assert frame["record"]["schema"] == "whetstone.execution.partial_call/v3"
    assert frame["record"]["schema"] == PARTIAL_SCHEMA
    assert set(frame["record"]) == {
        "schema",
        "phase",
        "instance_id",
        "unit",
        "repeat_id",
        "request_identity",
        "redrive_pending",
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
        "observation_payload",
        "finish_reason",
        "provider_error",
        "at",
        "cache_hit",
        "cache_source_phase",
        "cache_source_unit",
        "cache_source_call_id",
        "cache_source_at",
    }
    assert frame["checksum"] == (
        "39b6c1a7db213e8b5251a9bb63c41f222355b1f305901ef01e3c24c19b4403d5"
    )
    assert stat.S_IMODE(log.path.stat().st_mode) == 0o700
    assert stat.S_IMODE(entry.stat().st_mode) == 0o600


def test_storage_threat_model_is_explicit_and_pinned() -> None:
    assert PARTIAL_STORAGE_THREAT_MODEL == (
        "PartialLog provides crash durability, atomic publication, "
        "structural and content-integrity checks, and lock-based "
        "serialization for cooperating writers. Checksums detect "
        "accidental, malformed, or torn corruption; they do not provide "
        "authenticity or tamper resistance against a same-UID actor that "
        "can rewrite managed files, whether concurrently or between "
        "operations."
    )
    assert PartialLog.__doc__ is not None
    assert "crash durability" in PartialLog.__doc__
    assert "do not provide authenticity" in PartialLog.__doc__
    assert "concurrently or between operations" in PartialLog.__doc__
    assert PartialLog.delete.__doc__ == (
        "Clear managed entries durably while retaining the private store."
    )


def test_v2_and_monolithic_storage_fail_loudly(tmp_path: Path) -> None:
    path = tmp_path / "calls.partial"
    body = (
        b'{"schema":"whetstone.execution.partial_frame/v2",'
        b'"checksum":"unrecoverable","record":{}}\n'
    )
    path.write_bytes(body)
    log = PartialLog(path=path)

    with pytest.raises(ValueError, match="per-key record directory"):
        log.append(_record())
    with pytest.raises(ValueError, match="per-key record directory"):
        log.load()
    assert path.read_bytes() == body


def test_v2_frame_and_record_files_fail_loudly(tmp_path: Path) -> None:
    log = PartialLog(path=tmp_path / "calls.partial")
    log.append(_record())
    entry = _only_record_file(log.path)
    entry.write_bytes(
        b'{"schema":"whetstone.execution.partial_frame/v2",'
        b'"checksum":"unrecoverable","record":{}}\n'
    )
    with pytest.raises(ValueError, match="partial frame schema"):
        log.load()

    log.delete()
    log.append(_record())
    entry = _only_record_file(log.path)
    _rewrite_frame(
        entry,
        record_update={"schema": "whetstone.execution.partial_call/v2"},
        refresh_checksum=True,
    )
    with pytest.raises(ValueError, match="partial row schema"):
        log.load()


@pytest.mark.parametrize("retired_key", ["candidate_id", "repeat"])
def test_retired_wire_keys_are_rejected(
    tmp_path: Path,
    retired_key: str,
) -> None:
    log = PartialLog(path=tmp_path / "calls.partial")
    log.append(_record())
    _rewrite_frame(
        _only_record_file(log.path),
        record_update={retired_key: "retired"},
        refresh_checksum=True,
    )

    with pytest.raises(ValueError, match=retired_key):
        log.load()


def test_same_key_corruption_is_not_silently_overwritten(
    tmp_path: Path,
) -> None:
    log = PartialLog(path=tmp_path / "calls.partial")
    log.append(_record())
    entry = _only_record_file(log.path)
    _rewrite_frame(
        entry,
        record_update={"score": 1.0},
        refresh_checksum=False,
    )
    before = entry.read_bytes()

    with pytest.raises(ValueError, match="checksum mismatch"):
        log.append(_record())
    assert entry.read_bytes() == before


def test_recomputed_checksum_validates_content_but_not_authenticity(
    tmp_path: Path,
) -> None:
    log = PartialLog(path=tmp_path / "calls.partial")
    log.append(_record())
    _rewrite_frame(
        _only_record_file(log.path),
        record_update={"score": 1.0},
        refresh_checksum=True,
    )

    assert [record.score for record in log.load()] == [1.0]


def test_unrelated_append_does_not_bless_corrupt_record(
    tmp_path: Path,
) -> None:
    log = PartialLog(path=tmp_path / "calls.partial")
    log.append(_record(unit="first"))
    corrupt = _only_record_file(log.path)
    _rewrite_frame(
        corrupt,
        record_update={"score": 1.0},
        refresh_checksum=False,
    )

    log.append(_record(unit="second", repeat_id=1))
    assert len(_record_files(log.path)) == 2
    with pytest.raises(ValueError, match="checksum mismatch"):
        log.load()


def test_filename_to_key_mismatch_fails_closed(tmp_path: Path) -> None:
    log = PartialLog(path=tmp_path / "calls.partial")
    log.append(_record())
    entry = _only_record_file(log.path)
    entry.rename(log.path / f"{'0' * 64}.json")

    with pytest.raises(ValueError, match="filename does not match"):
        log.load()


def test_visible_path_swap_during_read_is_detected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = PartialLog(path=tmp_path / "calls.partial")
    log.append(_record())
    entry = _only_record_file(log.path)
    real_read_all = partials_module._read_all
    swapped = False

    def read_then_swap(fd: int) -> bytes:
        nonlocal swapped
        body = real_read_all(fd)
        if not swapped:
            swapped = True
            attacker = tmp_path / "read-attacker"
            attacker.write_bytes(b'{"replaced":true}\n')
            os.replace(attacker, entry)
        return body

    monkeypatch.setattr(partials_module, "_read_all", read_then_swap)
    with pytest.raises(OSError, match="changed while reading"):
        log.load()


def test_record_open_stays_bound_to_verified_directory_after_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = PartialLog(path=tmp_path / "calls.partial")
    log.append(_record())
    relocated = tmp_path / "relocated-partials"
    external = tmp_path / "external-partials"
    external.mkdir(mode=0o700)
    real_open = PrivateDirectory.open_regular
    substituted = False

    def substitute_before_record_open(
        directory: PrivateDirectory,
        name: str,
        flags: int,
        mode: int = 0o600,
    ) -> int:
        nonlocal substituted
        if (
            not substituted
            and directory.path == log.path
            and name.endswith(".json")
        ):
            substituted = True
            log.path.rename(relocated)
            log.path.symlink_to(external, target_is_directory=True)
        return real_open(directory, name, flags, mode)

    monkeypatch.setattr(
        PrivateDirectory,
        "open_regular",
        substitute_before_record_open,
    )

    assert [record.unit for record in log.load()] == ["candidate-1"]
    assert len(_record_files(relocated)) == 1
    assert list(external.iterdir()) == []


def test_temp_open_cannot_escape_verified_directory_after_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = PartialLog(path=tmp_path / "calls.partial")
    relocated = tmp_path / "relocated-partials"
    external = tmp_path / "external-partials"
    external.mkdir(mode=0o700)
    real_open = PrivateDirectory.open_regular
    substituted = False

    def substitute_before_temp_open(
        directory: PrivateDirectory,
        name: str,
        flags: int,
        mode: int = 0o600,
    ) -> int:
        nonlocal substituted
        if (
            not substituted
            and directory.path == log.path
            and name.endswith(".tmp")
        ):
            substituted = True
            log.path.rename(relocated)
            log.path.symlink_to(external, target_is_directory=True)
        return real_open(directory, name, flags, mode)

    monkeypatch.setattr(
        PrivateDirectory,
        "open_regular",
        substitute_before_temp_open,
    )

    with pytest.raises(OSError, match="not visible by path"):
        log.append(_record())
    assert list(relocated.iterdir()) == []
    assert list(external.iterdir()) == []


@pytest.mark.parametrize("operation", ["append", "load", "delete"])
def test_record_container_open_uses_lock_parent_after_real_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    active_parent = tmp_path / "active"
    active_parent.mkdir(mode=0o700)
    log = PartialLog(path=active_parent / "calls.partial")
    log.append(_record(unit="prior"))
    relocated_parent = tmp_path / "relocated"
    sentinel = b"replacement tree"
    real_open_child = PrivateDirectory.open_child
    substituted = False

    def substitute_parent_before_child_open(
        directory: PrivateDirectory,
        name: str,
        *,
        create: bool = True,
    ) -> PrivateDirectory:
        nonlocal substituted
        if (
            not substituted
            and directory.path == active_parent
            and name == log.path.name
        ):
            substituted = True
            active_parent.rename(relocated_parent)
            active_parent.mkdir(mode=0o700)
            replacement_store = active_parent / log.path.name
            replacement_store.mkdir(mode=0o700)
            (replacement_store / "sentinel").write_bytes(sentinel)
        return real_open_child(directory, name, create=create)

    monkeypatch.setattr(
        PrivateDirectory,
        "open_child",
        substitute_parent_before_child_open,
    )

    if operation == "append":
        with pytest.raises(OSError):
            log.append(_record(unit="new", repeat_id=1))
    elif operation == "load":
        assert [record.unit for record in log.load()] == ["prior"]
    else:
        log.delete()

    replacement_store = active_parent / log.path.name
    assert (replacement_store / "sentinel").read_bytes() == sentinel
    assert _record_files(replacement_store) == []
    original_store = relocated_parent / log.path.name
    if operation == "delete":
        assert list(original_store.iterdir()) == []
    else:
        assert len(_record_files(original_store)) == 1


def test_delete_never_removes_replacement_at_former_rmdir_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = PartialLog(path=tmp_path / "calls.partial")
    log.append(_record())
    relocated = tmp_path / "relocated-partials"
    sentinel = b"replacement container"
    real_fsync = PrivateDirectory.fsync
    substituted = False

    def substitute_before_final_fsync(directory: PrivateDirectory) -> None:
        nonlocal substituted
        if not substituted and directory.path == log.path:
            substituted = True
            log.path.rename(relocated)
            log.path.mkdir(mode=0o700)
            (log.path / "sentinel").write_bytes(sentinel)
        real_fsync(directory)

    monkeypatch.setattr(
        PrivateDirectory,
        "fsync",
        substitute_before_final_fsync,
    )

    log.delete()

    assert (log.path / "sentinel").read_bytes() == sentinel
    assert relocated.is_dir()
    assert list(relocated.iterdir()) == []


def test_strict_orphan_temporary_is_ignored_until_verified_delete(
    tmp_path: Path,
) -> None:
    log = PartialLog(path=tmp_path / "calls.partial")
    log.append(_record(unit="first"))
    second = _record(unit="second", repeat_id=1)
    entry = log._entry_path(second)
    temporary = entry.with_name(f".{entry.stem}.{'1' * 32}.tmp")
    temporary.write_bytes(b'{"torn":')
    temporary.chmod(0o600)

    assert [record.unit for record in log.load()] == ["first"]
    log.append(second)
    assert {record.unit for record in log.load()} == {"first", "second"}
    assert temporary.read_bytes() == b'{"torn":'

    log.delete()
    assert log.path.is_dir()
    assert list(log.path.iterdir()) == []


def test_append_uses_fresh_exclusive_temp_and_does_not_predelete_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = PartialLog(path=tmp_path / "calls.partial")
    log.append(_record(unit="first"))
    second = _record(unit="second", repeat_id=1)
    entry = log._entry_path(second)
    collision = entry.with_name(f".{entry.stem}.{'1' * 32}.tmp")
    collision.write_bytes(b"existing orphan")
    collision.chmod(0o600)
    tokens = iter(["1" * 32, "2" * 32])
    sources: list[Path] = []
    real_replace = PrivateDirectory.replace

    monkeypatch.setattr(
        partials_module.secrets,
        "token_hex",
        lambda _: next(tokens),
    )

    def observe_replace(
        directory: PrivateDirectory,
        source: str,
        destination: str,
    ) -> None:
        sources.append(directory.path / source)
        real_replace(directory, source, destination)

    monkeypatch.setattr(PrivateDirectory, "replace", observe_replace)
    log.append(second)

    assert collision.read_bytes() == b"existing orphan"
    assert sources == [entry.with_name(f".{entry.stem}.{'2' * 32}.tmp")]


def test_legacy_predictable_temp_is_unknown_and_untouched(
    tmp_path: Path,
) -> None:
    log = PartialLog(path=tmp_path / "calls.partial")
    log.append(_record(unit="first"))
    second = _record(unit="second", repeat_id=1)
    legacy_temp = log._entry_path(second).with_name(
        f".{log._entry_path(second).stem}.tmp"
    )
    legacy_temp.write_bytes(b"unknown")

    log.append(second)
    with pytest.raises(ValueError, match="unexpected partial storage entry"):
        log.load()
    with pytest.raises(ValueError, match="unexpected partial storage entry"):
        log.delete()
    assert legacy_temp.read_bytes() == b"unknown"


@pytest.mark.process_integration
def test_killed_atomic_writer_leaves_ignored_bounded_orphan(
    tmp_path: Path,
) -> None:
    path = tmp_path / "calls.partial"
    log = PartialLog(path=path)
    log.append(_record(unit="first"))
    context = multiprocessing.get_context("spawn")
    started = context.Event()
    writer = context.Process(
        target=write_torn_partial_worker,
        args=(str(path), started),
    )
    processes = []
    try:
        writer.start()
        processes.append(writer)
        assert started.wait(timeout=10)
        writer.terminate()
        writer.join(timeout=10)
        assert writer.exitcode is not None
    finally:
        terminate_processes(processes, timeout=10)
    assert [record.unit for record in log.load()] == ["first"]
    assert len(list(path.glob(".*.tmp"))) == 1


def test_mutated_temporary_cannot_be_published(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = PartialLog(path=tmp_path / "calls.partial")
    real_replace = PrivateDirectory.replace

    def corrupt_then_replace(
        directory: PrivateDirectory,
        source: str,
        destination: str,
    ) -> None:
        source_path = directory.path / source
        body = source_path.read_bytes().replace(b"candidate-1", b"candidate-X")
        source_path.write_bytes(body)
        real_replace(directory, source, destination)

    monkeypatch.setattr(PrivateDirectory, "replace", corrupt_then_replace)
    with pytest.raises(ValueError, match="checksum mismatch"):
        log.append(_record())
    assert not log._entry_path(_record()).exists()


def test_visible_replacement_before_acknowledgement_is_detected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = PartialLog(path=tmp_path / "calls.partial")
    real_replace = PrivateDirectory.replace

    def replace_published_target(
        directory: PrivateDirectory,
        source: str,
        destination: str,
    ) -> None:
        real_replace(directory, source, destination)
        attacker = directory.path / "attacker"
        attacker.write_bytes(b'{"replaced":true}\n')
        real_replace(directory, attacker.name, destination)

    monkeypatch.setattr(PrivateDirectory, "replace", replace_published_target)
    with pytest.raises(ValueError, match="partial frame"):
        log.append(_record())
    assert not log._entry_path(_record()).exists()


def test_pre_replace_corruption_rolls_back_to_valid_prior_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = PartialLog(path=tmp_path / "calls.partial")
    record = _record(at="2026-07-31T12:00:00+00:00")
    log.append(record.model_copy(update={"score": 0.0}))
    entry = log._entry_path(record)
    prior_body = entry.read_bytes()
    real_replace = PrivateDirectory.replace
    injected = False

    def corrupt_first_source(
        directory: PrivateDirectory,
        source: str,
        destination: str,
    ) -> None:
        nonlocal injected
        if not injected:
            injected = True
            source_path = directory.path / source
            source_path.write_bytes(
                source_path.read_bytes().replace(
                    b'"score":1.0', b'"score":9.0'
                )
            )
        real_replace(directory, source, destination)

    monkeypatch.setattr(PrivateDirectory, "replace", corrupt_first_source)
    with pytest.raises(ValueError, match="checksum mismatch"):
        log.append(record.model_copy(update={"score": 1.0}))

    assert entry.read_bytes() == prior_body
    assert [loaded.score for loaded in log.load()] == [0.0]


def test_post_replace_corruption_rolls_back_to_valid_prior_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = PartialLog(path=tmp_path / "calls.partial")
    record = _record(at="2026-07-31T12:00:00+00:00")
    log.append(record.model_copy(update={"score": 0.0}))
    entry = log._entry_path(record)
    prior_body = entry.read_bytes()
    real_replace = PrivateDirectory.replace
    injected = False

    def replace_first_published_target(
        directory: PrivateDirectory,
        source: str,
        destination: str,
    ) -> None:
        nonlocal injected
        real_replace(directory, source, destination)
        if not injected:
            injected = True
            attacker = directory.path / "attacker"
            attacker.write_bytes(b'{"replaced":true}\n')
            real_replace(directory, attacker.name, destination)

    monkeypatch.setattr(
        PrivateDirectory,
        "replace",
        replace_first_published_target,
    )
    with pytest.raises(ValueError, match="partial frame"):
        log.append(record.model_copy(update={"score": 1.0}))

    assert entry.read_bytes() == prior_body
    assert [loaded.score for loaded in log.load()] == [0.0]


def test_append_work_is_linear_in_new_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = PartialLog(path=tmp_path / "calls.partial")
    reads = 0
    real_read = partials_module._read_entry

    def count_read(
        fd: int,
        *,
        directory: PrivateDirectory,
        name: str,
    ) -> tuple[PartialCallRecord, bytes, os.stat_result]:
        nonlocal reads
        reads += 1
        return real_read(fd, directory=directory, name=name)

    monkeypatch.setattr(partials_module, "_read_entry", count_read)
    record_count = 32
    for repeat_id in range(record_count):
        log.append(
            _record(
                unit=f"candidate-{repeat_id}",
                repeat_id=repeat_id,
                request_identity=f"{repeat_id:064x}",
            )
        )

    assert reads == record_count
    assert len(_record_files(log.path)) == record_count


def test_repeated_load_then_append_reads_only_new_target_on_append(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = PartialLog(path=tmp_path / "calls.partial")
    log.append(_record(unit="candidate-0"))
    reads = 0
    real_read = partials_module._read_entry

    def count_read(
        fd: int,
        *,
        directory: PrivateDirectory,
        name: str,
    ) -> tuple[PartialCallRecord, bytes, os.stat_result]:
        nonlocal reads
        reads += 1
        return real_read(fd, directory=directory, name=name)

    monkeypatch.setattr(partials_module, "_read_entry", count_read)
    for repeat_id in range(1, 5):
        assert len(log.load()) == repeat_id
        after_load = reads
        log.append(
            _record(
                unit=f"candidate-{repeat_id}",
                repeat_id=repeat_id,
                request_identity=f"{repeat_id:064x}",
            )
        )
        assert reads == after_load + 1


@pytest.mark.process_integration
def test_multiprocess_multi_megabyte_appends_all_validate(
    tmp_path: Path,
) -> None:
    path = tmp_path / "calls.partial"
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
    started = []
    try:
        for process in processes:
            process.start()
            started.append(process)
        join_processes(started, timeout=30)
    finally:
        barrier.abort()
        terminate_processes(started, timeout=30)

    loaded = PartialLog(path=path).load()
    assert len(loaded) == worker_count
    assert {len(record.output_text or "") for record in loaded} == {
        1024 * 1024
    }


@pytest.mark.process_integration
def test_append_survives_immediate_child_hard_exit(tmp_path: Path) -> None:
    path = tmp_path / "calls.partial"
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=append_partial_worker,
        args=(str(path), 1, 32),
        kwargs={"exit_immediately": True},
    )
    started = []
    try:
        process.start()
        started.append(process)
        join_processes(started, timeout=10)
    finally:
        terminate_processes(started, timeout=10)
    assert [record.unit for record in PartialLog(path=path).load()] == [
        "candidate-1"
    ]


@pytest.mark.parametrize("operation", ["append", "load", "delete"])
@pytest.mark.process_integration
def test_separate_instances_serialize_all_operations(
    tmp_path: Path,
    operation: str,
) -> None:
    path = tmp_path / "calls.partial"
    PartialLog(path=path).append(_record())
    context = multiprocessing.get_context("spawn")
    entered = context.Event()
    release = context.Event()
    holder = context.Process(
        target=hold_partial_lock,
        args=(str(path), entered, release),
    )
    output = context.Queue()
    attempted = context.Event()
    acquired = context.Event()
    operation_process = context.Process(
        target=run_partial_operation,
        args=(str(path), operation, output, attempted, acquired),
    )
    started = []
    try:
        holder.start()
        started.append(holder)
        assert entered.wait(timeout=10)
        operation_process.start()
        started.append(operation_process)
        assert attempted.wait(timeout=10)
        assert not acquired.is_set()
        release.set()
        assert acquired.wait(timeout=10)
        join_processes(started, timeout=10)
        assert output.get(timeout=5) in {"appended", "deleted", 1}
    finally:
        release.set()
        terminate_processes(started, timeout=10)


def test_short_writes_are_completed(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "calls.partial"
    original_write = os.write

    def short_write(fd: int, body) -> int:
        return original_write(fd, body[:7])

    monkeypatch.setattr(partials_module.os, "write", short_write)
    PartialLog(path=path).append(_record())
    assert PartialLog(path=path).load()[0].unit == "candidate-1"


@pytest.mark.parametrize("value", ["", "not-a-timestamp", "2026-07-31"])
def test_invalid_at_is_rejected(value: str) -> None:
    with pytest.raises(ValueError, match="partial row at"):
        _record(at=value)


@pytest.mark.parametrize(
    "value",
    [
        "2026-07-31T12:00:00+01:00",
        "2026-07-31T12:00:00Z",
    ],
)
def test_at_requires_canonical_utc_isoformat(value: str) -> None:
    with pytest.raises(ValueError, match="partial row at"):
        _record(at=value)


@pytest.mark.parametrize(
    ("update", "match"),
    [
        ({"request_identity": "A" * 64}, "request_identity"),
        ({"redrive_pending": 0}, "redrive_pending"),
        ({"observation_payload": {"nested": math.nan}}, "finite numbers"),
    ],
)
def test_append_revalidates_boundary_before_creating_directory(
    tmp_path: Path,
    update: dict[str, object],
    match: str,
) -> None:
    log = PartialLog(path=tmp_path / "calls.partial")
    invalid = _record().model_copy(update=update)

    with pytest.raises(ValueError, match=match):
        log.append(invalid)
    assert not log.path.exists()


@pytest.mark.parametrize("value", ["", "a" * 63, "A" * 64, "g" * 64, 1])
def test_request_identity_must_be_canonical_hex(value: object) -> None:
    with pytest.raises(ValueError, match="request_identity"):
        PartialCallRecord.model_validate(
            {
                "phase": "internal",
                "instance_id": "task-1",
                "unit": "candidate-1",
                "repeat_id": 0,
                "request_identity": value,
                "redrive_pending": False,
            }
        )


def test_redrive_pending_is_required_and_strict() -> None:
    base: dict[str, object] = {
        "phase": "internal",
        "instance_id": "task-1",
        "unit": "candidate-1",
        "repeat_id": 0,
        "request_identity": REQUEST_IDENTITY_A,
    }
    with pytest.raises(ValueError, match="redrive_pending"):
        PartialCallRecord.model_validate(base)
    with pytest.raises(ValueError, match="redrive_pending"):
        PartialCallRecord.model_validate({**base, "redrive_pending": 0})


def test_observation_payload_round_trips_separately_from_raw_response(
    tmp_path: Path,
) -> None:
    log = PartialLog(path=tmp_path / "calls.partial")
    observation_payload = {
        "reward": 1.0,
        "state": ["ready", True, None, {"ordinal": 1}],
    }
    record = PartialCallRecord(
        phase="internal",
        instance_id="task-1",
        unit="candidate-1",
        repeat_id=0,
        request_identity=REQUEST_IDENTITY_A,
        redrive_pending=True,
        raw_response='{"provider":"evidence"}',
        observation_payload=observation_payload,
    )
    log.append(record)

    loaded = log.load()[0]
    assert loaded.raw_response == '{"provider":"evidence"}'
    assert loaded.observation_payload == observation_payload
    assert loaded.redrive_pending is True


@pytest.mark.parametrize(
    "value",
    [
        {"nested": [math.nan]},
        {"nested": math.inf},
        {1: "non-string key"},
        ("tuple",),
    ],
)
def test_observation_payload_rejects_invalid_json(value: object) -> None:
    with pytest.raises(ValueError, match="observation_payload"):
        PartialCallRecord.model_validate(
            {
                "phase": "internal",
                "instance_id": "task-1",
                "unit": "candidate-1",
                "repeat_id": 0,
                "request_identity": REQUEST_IDENTITY_A,
                "redrive_pending": False,
                "observation_payload": value,
            }
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("score", math.nan),
        ("latency_s", math.inf),
        ("provider_error", {"outer": [{"inner": -math.inf}]}),
    ],
)
def test_non_finite_numbers_are_rejected(field: str, value: object) -> None:
    with pytest.raises(ValueError, match="finite numbers"):
        PartialCallRecord.model_validate(
            {
                "phase": "internal",
                "instance_id": "task-1",
                "unit": "candidate-1",
                "repeat_id": 0,
                "request_identity": REQUEST_IDENTITY_A,
                "redrive_pending": False,
                field: value,
            }
        )


@pytest.mark.parametrize("storage_kind", ["data", "lock"])
def test_storage_symlinks_do_not_touch_external_target(
    tmp_path: Path,
    storage_kind: str,
) -> None:
    path = tmp_path / "calls.partial"
    log = PartialLog(path=path)
    victim = tmp_path / f"{storage_kind}-victim"
    body = f"{storage_kind}-external".encode()
    victim.write_bytes(body)
    victim.chmod(0o644)
    target = path if storage_kind == "data" else log._lock_path
    target.symlink_to(victim)

    with pytest.raises((OSError, ValueError)):
        log.append(_record())
    assert victim.read_bytes() == body
    assert stat.S_IMODE(victim.stat().st_mode) == 0o644


def test_lock_hardlink_is_rejected_before_victim_repair(
    tmp_path: Path,
) -> None:
    log = PartialLog(path=tmp_path / "calls.partial")
    victim = tmp_path / "lock-victim"
    body = b"external lock contents"
    victim.write_bytes(body)
    victim.chmod(0o644)
    os.link(victim, log._lock_path)

    with pytest.raises(OSError, match="unexpected hard links"):
        log.append(_record())
    assert victim.read_bytes() == body
    assert stat.S_IMODE(victim.stat().st_mode) == 0o644


def test_record_hardlink_is_rejected_before_victim_repair_or_read(
    tmp_path: Path,
) -> None:
    log = PartialLog(path=tmp_path / "calls.partial")
    record = _record()
    log.append(record)
    entry = log._entry_path(record)
    entry.unlink()
    victim = tmp_path / "record-victim"
    body = b"external record contents"
    victim.write_bytes(body)
    victim.chmod(0o644)
    os.link(victim, entry)

    with pytest.raises(OSError, match="unexpected hard links"):
        log.append(record)
    assert victim.read_bytes() == body
    assert stat.S_IMODE(victim.stat().st_mode) == 0o644


def test_record_hardlink_aborts_delete_before_any_record_is_removed(
    tmp_path: Path,
) -> None:
    log = PartialLog(path=tmp_path / "calls.partial")
    log.append(_record(unit="first"))
    log.append(_record(unit="second", repeat_id=1))
    entries = _record_files(log.path)
    assert len(entries) == 2
    hardlinked_entry = entries[-1]
    safe_entry = entries[0]
    safe_body = safe_entry.read_bytes()
    hardlinked_entry.unlink()
    victim = tmp_path / "record-delete-victim"
    body = b"external delete victim"
    victim.write_bytes(body)
    victim.chmod(0o644)
    os.link(victim, hardlinked_entry)

    with pytest.raises(OSError, match="unsafe managed partial entry"):
        log.delete()
    assert safe_entry.read_bytes() == safe_body
    assert victim.read_bytes() == body
    assert stat.S_IMODE(victim.stat().st_mode) == 0o644


def test_temporary_hardlink_is_rejected_before_delete_or_victim_repair(
    tmp_path: Path,
) -> None:
    log = PartialLog(path=tmp_path / "calls.partial")
    record = _record()
    log.append(record)
    entry = log._entry_path(record)
    temporary = entry.with_name(f".{entry.stem}.{'1' * 32}.tmp")
    victim = tmp_path / "temporary-victim"
    body = b"external temporary contents"
    victim.write_bytes(body)
    victim.chmod(0o644)
    os.link(victim, temporary)

    with pytest.raises(OSError, match="unsafe managed partial entry"):
        log.delete()
    assert victim.read_bytes() == body
    assert stat.S_IMODE(victim.stat().st_mode) == 0o644
    assert temporary.exists()
    assert entry.exists()


def test_insecure_orphan_temporary_is_not_cleaned(tmp_path: Path) -> None:
    log = PartialLog(path=tmp_path / "calls.partial")
    record = _record()
    log.append(record)
    entry = log._entry_path(record)
    temporary = entry.with_name(f".{entry.stem}.{'1' * 32}.tmp")
    temporary.write_bytes(b"orphan")
    temporary.chmod(0o644)

    with pytest.raises(OSError, match="unsafe managed partial entry"):
        log.delete()
    assert temporary.read_bytes() == b"orphan"
    assert stat.S_IMODE(temporary.stat().st_mode) == 0o644
    assert entry.exists()


def test_cache_hit_requires_complete_provenance_and_null_latency() -> None:
    with pytest.raises(ValueError, match="complete provenance"):
        PartialCallRecord(
            phase="internal",
            instance_id="task-1",
            unit="candidate-1",
            repeat_id=0,
            request_identity=REQUEST_IDENTITY_A,
            redrive_pending=False,
            cache_hit=True,
            latency_s=0.0,
        )


def test_atomic_publication_and_delete_are_durable_and_private(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[Path] = []
    modes: list[int] = []
    real_replace = PrivateDirectory.replace
    real_fsync = PrivateDirectory.fsync

    def record_replace(
        directory: PrivateDirectory,
        source: str,
        destination: str,
    ) -> None:
        modes.append(stat.S_IMODE(directory.stat(source).st_mode))
        real_replace(directory, source, destination)

    def record_fsync(directory: PrivateDirectory) -> None:
        observed.append(directory.path)
        real_fsync(directory)

    monkeypatch.setattr(PrivateDirectory, "fsync", record_fsync)
    monkeypatch.setattr(PrivateDirectory, "replace", record_replace)
    log = PartialLog(path=tmp_path / "calls.partial")
    first = _record(unit="first")
    second = _record(unit="second", repeat_id=1)
    log.append(first)
    log.append(second)
    log.delete()

    assert observed == [
        log.path,
        log.path,
        log.path,
    ]
    assert modes == [0o600, 0o600]
    assert log.path.is_dir()
    assert list(log.path.iterdir()) == []


def test_all_masking_umask_preserves_private_retryable_storage(
    tmp_path: Path,
) -> None:
    log = PartialLog(path=tmp_path / "nested" / "calls.partial")
    previous_umask = os.umask(0o777)
    try:
        log.append(_record(unit="first"))
        log.append(_record(unit="second", repeat_id=1))
        observed_umask = os.umask(0o777)
        assert observed_umask == 0o777
    finally:
        os.umask(previous_umask)

    assert {record.unit for record in log.load()} == {"first", "second"}
    assert stat.S_IMODE(log.path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(log.path.stat().st_mode) == 0o700
    assert stat.S_IMODE(log._lock_path.stat().st_mode) == 0o600
    assert {
        stat.S_IMODE(entry.stat().st_mode) for entry in _record_files(log.path)
    } == {0o600}
    assert list(log.path.glob(".*.tmp")) == []


def test_unknown_storage_entry_fails_load_and_delete(tmp_path: Path) -> None:
    log = PartialLog(path=tmp_path / "calls.partial")
    log.append(_record())
    unknown = log.path / "unknown"
    unknown.write_text("unexpected")

    with pytest.raises(ValueError, match="unexpected partial storage entry"):
        log.load()
    with pytest.raises(ValueError, match="unexpected partial storage entry"):
        log.delete()
    assert unknown.exists()


def test_missing_storage_and_delete_are_idempotent(tmp_path: Path) -> None:
    log = PartialLog(path=tmp_path / "calls.partial")
    assert log.load() == []
    assert log.recorded_keys() == set()
    log.delete()
    log.append(_record())
    log.delete()
    assert log.path.is_dir()
    assert list(log.path.iterdir()) == []
    log.append(_record(unit="reused"))
    assert [record.unit for record in log.load()] == ["reused"]
