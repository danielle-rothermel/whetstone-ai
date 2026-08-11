from __future__ import annotations

import errno
import json
import os
import select
import selectors
import signal
import subprocess
import sys
import threading
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Protocol, cast

import pytest
from dr_serialize import StrictJsonDecodeError
from pydantic import JsonValue, ValidationError

import whetstone.execution.fanout as fanout_module
import whetstone.execution.process_worker as process_worker_module
from tests.execution.process_signals import ProcessSignals
from whetstone.execution.fanout import (
    CallSpec,
    FanoutStatus,
    ProcessCancellationError,
    ProcessJob,
    ProcessWorkerError,
    run_call_pool,
)

_WORKERS = "tests.execution.process_workers"


class _KqueueEvent(Protocol):
    fflags: int


class _Kqueue(Protocol):
    def control(
        self,
        changes: list[_KqueueEvent],
        max_events: int,
        timeout: float,
    ) -> list[_KqueueEvent]: ...

    def close(self) -> None: ...


class _KqueueApi(Protocol):
    KQ_FILTER_PROC: int
    KQ_EV_ADD: int
    KQ_EV_ONESHOT: int
    KQ_NOTE_EXIT: int

    def kqueue(self) -> _Kqueue: ...

    def kevent(
        self,
        ident: int,
        *,
        filter: int,
        flags: int,
        fflags: int,
    ) -> _KqueueEvent: ...


def _identity(value: JsonValue) -> JsonValue:
    return value


def _never_rate_limited(_value: JsonValue) -> bool:
    return False


def _job(function: str, payload: JsonValue) -> ProcessJob:
    return ProcessJob(
        entrypoint=f"{_WORKERS}:{function}",
        payload=payload,
    )


def _gated_spec(
    key: str,
    *,
    signals: ProcessSignals,
    value: JsonValue | None = None,
    deadline: float = 5.0,
    commit: Callable[[JsonValue], None] | None = None,
    fail: bool = False,
    decode: Callable[[JsonValue], JsonValue] = _identity,
    cancellation_barrier: Callable[[], None] | None = None,
) -> CallSpec[str, JsonValue]:
    payload: dict[str, JsonValue] = {
        "key": key,
        "signal_path": os.fspath(signals.path),
        "value": key if value is None else value,
        "fail": fail,
    }
    return CallSpec(
        key=key,
        job=_job("gated_event", payload),
        decode=decode,
        deadline_seconds=deadline,
        commit=commit,
        cancellation_barrier=cancellation_barrier,
    )


def _blocking_tree_spec(
    key: str,
    signals: ProcessSignals,
    *,
    deadline: float,
    commit: Callable[[JsonValue], None] | None = None,
    cancellation_barrier: Callable[[], None] | None = None,
) -> CallSpec[str, JsonValue]:
    return CallSpec(
        key=key,
        job=_job(
            "block_process_tree",
            {"signal_path": os.fspath(signals.path), "key": key},
        ),
        decode=_identity,
        deadline_seconds=deadline,
        commit=commit,
        cancellation_barrier=cancellation_barrier,
    )


def _assert_process_gone(pid: int) -> None:
    try:
        os.kill(pid, 0)
    except OSError as error:
        assert error.errno == errno.ESRCH
        return
    if hasattr(os, "pidfd_open"):
        try:
            descriptor = os.pidfd_open(pid)
        except ProcessLookupError:
            return
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(descriptor, selectors.EVENT_READ)
                assert selector.select(3.0), (
                    f"process {pid} survived scheduler return"
                )
        finally:
            os.close(descriptor)
        return
    kqueue_api = cast(_KqueueApi, cast(object, select))
    queue = kqueue_api.kqueue()
    try:
        event = kqueue_api.kevent(
            pid,
            filter=kqueue_api.KQ_FILTER_PROC,
            flags=kqueue_api.KQ_EV_ADD | kqueue_api.KQ_EV_ONESHOT,
            fflags=kqueue_api.KQ_NOTE_EXIT,
        )
        observed = queue.control([event], 1, 3.0)
    finally:
        queue.close()
    assert observed, f"process {pid} survived scheduler return"
    assert observed[0].fflags & kqueue_api.KQ_NOTE_EXIT


@pytest.mark.process_integration
def test_process_gone_accepts_exit_between_probe_and_pidfd_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.buffer.read(1)"],
        stdin=subprocess.PIPE,
    )
    real_kill = os.kill
    probed = False

    def probe(pid: int, sig: int) -> None:
        nonlocal probed
        real_kill(pid, sig)
        assert pid == process.pid
        assert sig == 0
        probed = True

    def exit_before_open(pid: int) -> int:
        assert pid == process.pid
        assert probed
        assert process.poll() is None
        assert process.stdin is not None
        process.stdin.close()
        process.wait(timeout=3.0)
        raise ProcessLookupError(errno.ESRCH, os.strerror(errno.ESRCH))

    monkeypatch.setattr(os, "kill", probe)
    monkeypatch.setattr(os, "pidfd_open", exit_before_open, raising=False)
    try:
        _assert_process_gone(process.pid)
    finally:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3.0)


def test_process_gone_propagates_unrelated_pidfd_open_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = PermissionError(errno.EPERM, os.strerror(errno.EPERM))

    def fail_open(_pid: int) -> int:
        raise error

    monkeypatch.setattr(os, "pidfd_open", fail_open, raising=False)

    with pytest.raises(PermissionError) as raised:
        _assert_process_gone(os.getpid())

    assert raised.value is error


def test_process_gone_rejects_live_pidfd_target_after_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    read_descriptor, write_descriptor = os.pipe()

    def open_live_target(_pid: int) -> int:
        return read_descriptor

    def report_live_target(
        _selector: selectors.BaseSelector,
        timeout: float | None = None,
    ) -> list[tuple[selectors.SelectorKey, int]]:
        assert timeout == 3.0
        return []

    monkeypatch.setattr(os, "pidfd_open", open_live_target, raising=False)
    monkeypatch.setattr(
        selectors.DefaultSelector, "select", report_live_target
    )
    try:
        with pytest.raises(
            AssertionError,
            match="survived scheduler return",
        ):
            _assert_process_gone(os.getpid())
    finally:
        os.close(write_descriptor)
    with pytest.raises(OSError) as closed:
        os.fstat(read_descriptor)
    assert closed.value.errno == errno.EBADF


def _assert_process_group_absent(process_group_id: int) -> None:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return
    except PermissionError:
        _assert_process_gone(process_group_id)
        return
    raise AssertionError(f"process group {process_group_id} survived return")


def _wait_for_eof(descriptor: int, *, timeout: float = 3.0) -> None:
    with selectors.DefaultSelector() as selector:
        selector.register(descriptor, selectors.EVENT_READ)
        assert selector.select(timeout), (
            "guardian pipe did not become readable"
        )
    assert os.read(descriptor, 1) == b""


class _ScriptedDeadline:
    def __init__(self, condition: threading.Event) -> None:
        self.condition = condition
        self.triggered = threading.Event()
        self.errors: list[str] = []

    def __call__(
        self,
        _deadline: float,
        stop: threading.Event,
        trigger: Callable[[], None],
    ) -> None:
        if not self.condition.wait(timeout=10):
            self.errors.append("deadline trigger condition was not reached")
        if not stop.is_set():
            trigger()
            self.triggered.set()

    def assert_satisfied(self) -> None:
        assert not self.errors
        assert self.triggered.is_set()


def _open_fd_count() -> int:
    descriptor_directory = (
        Path("/proc/self/fd")
        if Path("/proc/self/fd").exists()
        else Path("/dev/fd")
    )
    return len(tuple(descriptor_directory.iterdir()))


def test_process_job_pins_json_contract_and_rejects_nested_entrypoints() -> (
    None
):
    job = _job("return_payload", {"answer": 42})
    assert job.model_dump(mode="json") == {
        "schema_name": "whetstone.execution.process_job/v1",
        "entrypoint": "tests.execution.process_workers:return_payload",
        "payload": {"answer": 42},
    }
    with pytest.raises(ValidationError, match="top_level_callable"):
        ProcessJob(entrypoint="module:object.method", payload={})
    with pytest.raises(ValidationError):
        ProcessJob.model_validate(
            {"entrypoint": "module:function", "payload": object()}
        )


@pytest.mark.parametrize(
    "payload",
    [
        float("nan"),
        {"nested": [float("inf")]},
    ],
)
def test_process_job_rejects_non_finite_json_recursively(
    payload: JsonValue,
) -> None:
    with pytest.raises(ValidationError, match="non-finite JSON number"):
        _job("return_payload", payload)


def test_process_dispatch_marker_pins_json_contract_and_validation() -> None:
    marker = fanout_module._ProcessDispatchMarker(
        started_at_monotonic=1.25,
        guardian_pid=123,
    )
    assert marker.model_dump(mode="json") == {
        "schema_name": "whetstone.execution.process_dispatch/v1",
        "started_at_monotonic": 1.25,
        "guardian_pid": 123,
    }
    with pytest.raises(ValidationError, match="process dispatch schema"):
        fanout_module._ProcessDispatchMarker(
            schema_name="whetstone.execution.process_dispatch/v2",
            started_at_monotonic=1.25,
            guardian_pid=123,
        )
    for started_at in (
        float("nan"),
        float("inf"),
        -1.0,
        "1.25",
        True,
        False,
    ):
        with pytest.raises(ValidationError):
            fanout_module._ProcessDispatchMarker(
                started_at_monotonic=cast(float, started_at),
                guardian_pid=123,
            )
    with pytest.raises(ValidationError):
        fanout_module._ProcessDispatchMarker(
            started_at_monotonic=1.25,
            guardian_pid=0,
        )


@pytest.mark.parametrize(
    "payload",
    [b"{}", b'{"schema":"first","schema":"second"}', b'{"x":NaN}', b"\xff"],
)
def test_active_worker_rejects_any_visible_invalid_dispatch_marker(
    tmp_path: Path,
    payload: bytes,
) -> None:
    dispatch_path = tmp_path / "dispatch.json"
    dispatch_path.write_bytes(payload)
    process = fanout_module._ActiveProcess(
        index=0,
        spec=CallSpec(
            key="worker",
            job=_job("return_payload", None),
            decode=_identity,
            deadline_seconds=5.0,
        ),
        process=cast(subprocess.Popen[bytes], object()),
        directory=tmp_path,
        result_path=tmp_path / "result.json",
        stderr_path=tmp_path / "stderr.log",
        dispatch_path=dispatch_path,
        lifetime_writer=None,
        guardian_reader=None,
        start_writer=None,
    )

    with pytest.raises(ProcessWorkerError, match="invalid dispatch marker"):
        process.refresh_dispatch_marker(required=False)


@pytest.mark.parametrize(
    "payload",
    [b'{"schema":"first","schema":"second"}', b'{"x":NaN}', b"\xff"],
)
def test_worker_result_rejects_non_strict_json(
    tmp_path: Path,
    payload: bytes,
) -> None:
    result_path = tmp_path / "result.json"
    result_path.write_bytes(payload)
    process = fanout_module._ActiveProcess(
        index=0,
        spec=CallSpec(
            key="worker",
            job=_job("return_payload", None),
            decode=_identity,
            deadline_seconds=5.0,
        ),
        process=cast(subprocess.Popen[bytes], object()),
        directory=tmp_path,
        result_path=result_path,
        stderr_path=tmp_path / "stderr.log",
        dispatch_path=tmp_path / "dispatch.json",
        lifetime_writer=None,
        guardian_reader=None,
        start_writer=None,
    )

    with pytest.raises(ProcessWorkerError, match="invalid result envelope"):
        fanout_module._read_worker_result(process)


@pytest.mark.parametrize(
    "payload",
    [b'{"schema":"first","schema":"second"}', b'{"x":NaN}', b"\xff"],
)
def test_process_job_rejects_non_strict_json(
    tmp_path: Path,
    payload: bytes,
) -> None:
    path = tmp_path / "job.json"
    path.write_bytes(payload)

    with pytest.raises(StrictJsonDecodeError):
        process_worker_module._load_job(path)


@pytest.mark.process_integration
def test_worker_dispatch_marker_follows_start_gate_and_precedes_user_code(
    tmp_path: Path,
) -> None:
    job_path = tmp_path / "job.json"
    result_path = tmp_path / "result.json"
    dispatch_path = tmp_path / "dispatch.json"
    event_path = tmp_path / "event"
    job_path.write_text(
        _job(
            "require_path_then_return",
            {
                "required_path": os.fspath(dispatch_path),
                "event_path": os.fspath(event_path),
                "value": "complete",
            },
        ).model_dump_json(),
        encoding="utf-8",
    )
    parent_reader, parent_writer = os.pipe()
    guardian_reader, guardian_writer = os.pipe()
    start_reader, start_writer = os.pipe()
    ready_reader, ready_writer = os.pipe()
    worker = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "whetstone.execution.process_worker",
            os.fspath(job_path),
            os.fspath(result_path),
            os.fspath(dispatch_path),
            str(parent_reader),
            str(guardian_writer),
            str(start_reader),
            "none",
            str(ready_writer),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        pass_fds=(
            parent_reader,
            guardian_writer,
            start_reader,
            ready_writer,
        ),
        start_new_session=True,
    )
    os.close(parent_reader)
    os.close(guardian_writer)
    os.close(start_reader)
    os.close(ready_writer)
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(ready_reader, selectors.EVENT_READ)
            assert selector.select(3.0), "worker did not reach its start gate"
        assert os.read(ready_reader, 1) == b"\x01"
        assert worker.poll() is None
        assert not dispatch_path.exists()
        assert not event_path.exists()

        assert os.write(start_writer, b"\x01") == 1
        os.close(start_writer)
        start_writer = -1
        assert worker.wait(timeout=3.0) == 0
        assert dispatch_path.exists()
        assert event_path.read_text(encoding="utf-8") == "observed\n"
    finally:
        if start_writer >= 0:
            os.close(start_writer)
        if worker.poll() is None:
            os.killpg(worker.pid, signal.SIGKILL)
            worker.wait(timeout=3.0)
        os.close(parent_writer)
        _wait_for_eof(guardian_reader)
        os.close(guardian_reader)
        os.close(ready_reader)


@pytest.mark.process_integration
def test_worker_boundary_files_are_restrictive_and_validated(
    tmp_path: Path,
) -> None:
    job_path = tmp_path / "job.json"
    result_path = tmp_path / "result.json"
    dispatch_path = tmp_path / "dispatch.json"
    job_path.write_text(
        _job("return_payload", {"ok": True}).model_dump_json(),
        encoding="utf-8",
    )
    job_path.chmod(0o600)
    parent_reader, parent_writer = os.pipe()
    guardian_reader, guardian_writer = os.pipe()
    start_reader, start_writer = os.pipe()
    os.write(start_writer, b"\x01")
    os.close(start_writer)
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "whetstone.execution.process_worker",
                os.fspath(job_path),
                os.fspath(result_path),
                os.fspath(dispatch_path),
                str(parent_reader),
                str(guardian_writer),
                str(start_reader),
                "none",
            ],
            check=False,
            capture_output=True,
            pass_fds=(parent_reader, guardian_writer, start_reader),
            start_new_session=True,
            timeout=10.0,
        )
    finally:
        os.close(parent_reader)
        os.close(guardian_writer)
        os.close(start_reader)
    os.close(parent_writer)
    _wait_for_eof(guardian_reader)
    os.close(guardian_reader)
    assert completed.returncode == 0, completed.stderr.decode()
    assert completed.stderr == b""
    assert dispatch_path.stat().st_mode & 0o777 == 0o600
    assert result_path.stat().st_mode & 0o777 == 0o600
    marker = fanout_module._ProcessDispatchMarker.model_validate_json(
        dispatch_path.read_bytes()
    )
    assert marker.schema_name == "whetstone.execution.process_dispatch/v1"
    assert marker.started_at_monotonic >= 0
    assert marker.guardian_pid > 0
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert {
        "schema_name": result["schema_name"],
        "status": result["status"],
        "value": result["value"],
        "error": result["error"],
    } == {
        "schema_name": "whetstone.execution.process_result/v1",
        "status": "success",
        "value": {"ok": True},
        "error": None,
    }


@pytest.mark.process_integration
@pytest.mark.process_guardian
def test_guardians_start_under_high_worker_concurrency() -> None:
    worker_count = 32
    specs = [
        CallSpec(
            key=index,
            job=_job("return_payload", index),
            decode=_identity,
            deadline_seconds=30.0,
        )
        for index in range(worker_count)
    ]

    outcome = run_call_pool(
        specs,
        concurrency=worker_count,
        is_rate_limited=_never_rate_limited,
    )

    assert [result.status for result in outcome.results] == [
        FanoutStatus.COMPLETED
    ] * len(specs)
    assert [result.value for result in outcome.results] == list(
        range(worker_count)
    )


@pytest.mark.parametrize("parent_signal", [signal.SIGTERM, signal.SIGKILL])
@pytest.mark.process_integration
@pytest.mark.process_guardian
def test_parent_death_kills_fresh_worker_process_group(
    parent_signal: int,
) -> None:
    signals = ProcessSignals()
    scheduler_script = """
import os
import sys

from whetstone.execution.fanout import CallSpec, ProcessJob, run_call_pool

path = sys.argv[1]
job = ProcessJob(
    entrypoint="tests.execution.process_workers:block_process_tree",
    payload={"signal_path": path, "key": "worker"},
)
run_call_pool(
    [
        CallSpec(
            key="worker",
            job=job,
            decode=lambda value: value,
            deadline_seconds=30.0,
        )
    ],
    concurrency=1,
    is_rate_limited=lambda value: False,
)
"""
    scheduler = subprocess.Popen(
        [sys.executable, "-c", scheduler_script, os.fspath(signals.path)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        signals.wait_entered(["worker-worker", "worker-descendant"])
        os.kill(scheduler.pid, parent_signal)
        scheduler.wait(timeout=3.0)
        for pid in (
            signals.pid("worker-worker"),
            signals.pid("worker-descendant"),
        ):
            _assert_process_gone(pid)
    finally:
        if scheduler.poll() is None:
            scheduler.kill()
            scheduler.wait(timeout=3.0)
        signals.close()


@pytest.mark.process_integration
@pytest.mark.process_guardian
def test_stopped_guardian_on_completion_forces_local_containment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals = ProcessSignals()
    spawned: list[fanout_module._ActiveProcess[str, JsonValue]] = []
    spawned_event = threading.Event()
    real_spawn = fanout_module._spawn

    def record_spawn(
        index: int,
        spec: CallSpec[str, JsonValue],
        *,
        operation_deadline: float | None,
    ) -> fanout_module._ActiveProcess[str, JsonValue]:
        process = real_spawn(
            index,
            spec,
            operation_deadline=operation_deadline,
        )
        spawned.append(process)
        spawned_event.set()
        return process

    monkeypatch.setattr(fanout_module, "_spawn", record_spawn)
    failures: list[BaseException] = []

    def schedule() -> None:
        try:
            run_call_pool(
                [
                    CallSpec(
                        key="worker",
                        job=_job(
                            "spawn_descendant_and_return",
                            {
                                "signal_path": os.fspath(signals.path),
                                "release_key": "release",
                                "value": "complete",
                            },
                        ),
                        decode=_identity,
                        deadline_seconds=5.0,
                    )
                ],
                concurrency=1,
                is_rate_limited=_never_rate_limited,
            )
        except BaseException as error:
            failures.append(error)

    scheduler = threading.Thread(target=schedule)
    scheduler.start()
    assert spawned_event.wait(timeout=10)
    process = spawned[0]
    signals.wait_entered(["descendant", "release"])
    process.refresh_dispatch_marker(required=True)
    guardian_pid = process.guardian_pid
    assert guardian_pid is not None
    try:
        os.kill(guardian_pid, signal.SIGSTOP)
        signals.release("release")
        scheduler.join(timeout=3.0)
        assert not scheduler.is_alive()
        assert len(failures) == 1
        assert isinstance(failures[0], ProcessCancellationError)
        assert "did not exit" in str(failures[0])
        for pid in (
            process.process.pid,
            guardian_pid,
            signals.pid("descendant"),
        ):
            _assert_process_gone(pid)
        assert not process.directory.exists()
        assert fanout_module._parent_control_fds == set()
    finally:
        if "release" in signals.entered_keys:
            try:
                signals.release("release")
            except (AssertionError, BrokenPipeError, EOFError, OSError):
                pass
        if scheduler.is_alive():
            os.killpg(process.process_group_id, signal.SIGKILL)
            scheduler.join(timeout=3.0)
        signals.close()


@pytest.mark.process_integration
@pytest.mark.process_guardian
def test_harvest_retains_state_when_fallback_cannot_prove_containment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals = ProcessSignals()
    spawned: list[fanout_module._ActiveProcess[str, JsonValue]] = []
    spawned_event = threading.Event()
    real_spawn = fanout_module._spawn
    real_group_exists = fanout_module._process_group_exists

    def record_spawn(
        index: int,
        spec: CallSpec[str, JsonValue],
        *,
        operation_deadline: float | None,
    ) -> fanout_module._ActiveProcess[str, JsonValue]:
        process = real_spawn(
            index,
            spec,
            operation_deadline=operation_deadline,
        )
        spawned.append(process)
        spawned_event.set()
        return process

    monkeypatch.setattr(fanout_module, "_spawn", record_spawn)
    failures: list[BaseException] = []

    def schedule() -> None:
        try:
            run_call_pool(
                [
                    CallSpec(
                        key="worker",
                        job=_job(
                            "spawn_descendant_and_return",
                            {
                                "signal_path": os.fspath(signals.path),
                                "release_key": "release",
                                "value": "complete",
                            },
                        ),
                        decode=_identity,
                        deadline_seconds=5.0,
                    )
                ],
                is_rate_limited=_never_rate_limited,
            )
        except BaseException as error:
            failures.append(error)

    scheduler = threading.Thread(target=schedule)
    scheduler.start()
    assert spawned_event.wait(timeout=10)
    process = spawned[0]
    signals.wait_entered(["descendant", "release"])
    process.refresh_dispatch_marker(required=True)
    guardian_pid = process.guardian_pid
    assert guardian_pid is not None

    def deny_signal(
        candidate: fanout_module._ActiveProcess[str, JsonValue],
        _sig: int,
    ) -> None:
        if candidate is process:
            raise PermissionError("injected process-group signal denial")
        raise AssertionError("unexpected process")

    def retain_group(
        candidate: fanout_module._ActiveProcess[str, JsonValue],
    ) -> bool:
        if candidate is process:
            return True
        return real_group_exists(candidate)

    try:
        os.kill(guardian_pid, signal.SIGSTOP)
        monkeypatch.setattr(
            fanout_module,
            "_signal_process_group",
            deny_signal,
        )
        monkeypatch.setattr(
            fanout_module,
            "_process_group_exists",
            retain_group,
        )
        signals.release("release")
        scheduler.join(timeout=5.0)
        assert not scheduler.is_alive()
        assert len(failures) == 1
        failure = failures[0]
        assert isinstance(failure, ProcessCancellationError)
        assert "could not confirm terminal local process group" in str(failure)
        assert isinstance(failure.__cause__, ProcessCancellationError)
        assert "did not exit" in str(failure.__cause__)
        assert process.directory.exists()
        assert process.guardian_reader is not None
        assert process.guardian_reader in fanout_module._parent_control_fds
        assert not process.cleanup_allowed
        os.killpg(process.process_group_id, 0)
    finally:
        monkeypatch.undo()
        if "release" in signals.entered_keys:
            try:
                signals.release("release")
            except (AssertionError, BrokenPipeError, EOFError, OSError):
                pass
        try:
            os.kill(guardian_pid, signal.SIGCONT)
        except ProcessLookupError:
            pass
        try:
            os.killpg(process.process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.process.wait(timeout=3.0)
        assert not fanout_module._wait_for_process_group_absence([process])
        process.cleanup_allowed = True
        process.release_guardian_after_containment()
        process.cleanup()
        scheduler.join(timeout=3.0)
        signals.close()


@pytest.mark.parametrize(
    "failure_site",
    ["unit-expiration", "wall-watcher", "outer-exception"],
)
@pytest.mark.process_integration
def test_cancellation_failure_retains_uncontained_process_state(
    monkeypatch: pytest.MonkeyPatch,
    failure_site: str,
) -> None:
    signals = ProcessSignals()
    target: list[fanout_module._ActiveProcess[str, JsonValue]] = []
    target_spawned = threading.Event()
    failure_enabled = threading.Event()
    real_spawn = fanout_module._spawn
    real_signal = fanout_module._signal_process_group
    real_group_exists = fanout_module._process_group_exists

    def record_spawn(
        index: int,
        spec: CallSpec[str, JsonValue],
        *,
        operation_deadline: float | None,
    ) -> fanout_module._ActiveProcess[str, JsonValue]:
        process = real_spawn(
            index,
            spec,
            operation_deadline=operation_deadline,
        )
        if spec.key == "target":
            target.append(process)
            target_spawned.set()
        return process

    def controlled_signal(
        process: fanout_module._ActiveProcess[str, JsonValue],
        sig: int,
    ) -> None:
        if target and process is target[0] and failure_enabled.is_set():
            raise PermissionError("injected process-group signal denial")
        real_signal(process, sig)

    def controlled_group_exists(
        process: fanout_module._ActiveProcess[str, JsonValue],
    ) -> bool:
        if target and process is target[0] and failure_enabled.is_set():
            return True
        return real_group_exists(process)

    monkeypatch.setattr(fanout_module, "_spawn", record_spawn)
    monkeypatch.setattr(
        fanout_module,
        "_signal_process_group",
        controlled_signal,
    )
    monkeypatch.setattr(
        fanout_module,
        "_process_group_exists",
        controlled_group_exists,
    )
    specs = [
        _blocking_tree_spec(
            "target",
            signals,
            deadline=(0.2 if failure_site == "unit-expiration" else 5.0),
        )
    ]
    max_wall_seconds: float | None = None
    if failure_site == "wall-watcher":
        max_wall_seconds = sys.float_info.max
    elif failure_site == "outer-exception":
        specs.append(
            _gated_spec(
                "failed",
                signals=signals,
                fail=True,
            )
        )
    scripted_deadline: _ScriptedDeadline | None = None
    if failure_site == "wall-watcher":
        scripted_deadline = _ScriptedDeadline(failure_enabled)
        monkeypatch.setattr(
            fanout_module,
            "_wait_for_operation_deadline",
            scripted_deadline,
        )

    failures: list[BaseException] = []

    def schedule() -> None:
        try:
            run_call_pool(
                specs,
                concurrency=len(specs),
                max_wall_seconds=max_wall_seconds,
                is_rate_limited=_never_rate_limited,
            )
        except BaseException as error:
            failures.append(error)

    scheduler = threading.Thread(target=schedule)
    scheduler.start()
    assert target_spawned.wait(timeout=10)
    signals.wait_entered(["target-worker", "target-descendant"])
    process = target[0]
    failure_enabled.set()
    if failure_site == "outer-exception":
        signals.wait_entered(["failed"])
        signals.release("failed")
    try:
        scheduler.join(timeout=5.0)
        assert not scheduler.is_alive()
        assert len(failures) == 1
        failure = failures[0]
        assert isinstance(failure, ProcessCancellationError)
        assert "could not confirm terminal local process group" in str(failure)
        if failure_site == "outer-exception":
            assert isinstance(failure.__cause__, ProcessWorkerError)
            assert "requested failure" in str(failure.__cause__)
        assert process.directory.exists()
        assert process.lifetime_writer is not None
        assert process.guardian_reader is not None
        assert process.lifetime_writer in fanout_module._parent_control_fds
        assert process.guardian_reader in fanout_module._parent_control_fds
        assert not process.cleanup_allowed
        if scripted_deadline is not None:
            scripted_deadline.assert_satisfied()
    finally:
        failure_enabled.clear()
        monkeypatch.undo()
        try:
            os.killpg(process.process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.process.wait(timeout=3.0)
        assert not fanout_module._wait_for_process_group_absence([process])
        process.cleanup_allowed = True
        process.release_guardian_after_containment()
        process.cleanup()
        scheduler.join(timeout=3.0)
        signals.close()


@pytest.mark.process_integration
@pytest.mark.process_guardian
def test_worker_contains_group_when_guardian_and_scheduler_die() -> None:
    signals = ProcessSignals()
    scheduler_script = """
import os
import sys

import whetstone.execution.fanout as fanout
from whetstone.execution.fanout import CallSpec, ProcessJob, run_call_pool
from tests.execution.process_signals import publish_ready

signal_path = sys.argv[1]
real_refresh_dispatch_marker = fanout._ActiveProcess.refresh_dispatch_marker
published_guardian = False

def record_dispatch_marker(self, *, required):
    global published_guardian
    started_at = real_refresh_dispatch_marker(self, required=required)
    if (
        started_at is not None
        and self.guardian_pid is not None
        and not published_guardian
    ):
        published_guardian = True
        assert self.guardian_pid is not None
        publish_ready(signal_path, "guardian")
    return started_at

fanout._ActiveProcess.refresh_dispatch_marker = record_dispatch_marker
run_call_pool(
    [
        CallSpec(
            key="worker",
            job=ProcessJob(
                entrypoint="tests.execution.process_workers:block_process_tree",
                payload={"signal_path": signal_path, "key": "worker"},
            ),
            decode=lambda value: value,
            deadline_seconds=30.0,
        )
    ],
    concurrency=1,
    is_rate_limited=lambda value: False,
)
"""
    scheduler = subprocess.Popen(
        [
            sys.executable,
            "-c",
            scheduler_script,
            os.fspath(signals.path),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    observed_pids: list[int] = []
    try:
        signals.wait_entered(
            ["worker-worker", "worker-descendant", "guardian"]
        )
        worker_pid = signals.pid("worker-worker")
        guardian_pid = signals.pid("guardian")
        observed_pids = [
            worker_pid,
            guardian_pid,
            signals.pid("worker-descendant"),
        ]
        os.kill(guardian_pid, signal.SIGKILL)
        os.kill(scheduler.pid, signal.SIGKILL)
        scheduler.wait(timeout=3.0)
        for pid in observed_pids:
            _assert_process_gone(pid)
    finally:
        if scheduler.poll() is None:
            scheduler.kill()
            scheduler.wait(timeout=3.0)
        if observed_pids:
            try:
                os.killpg(observed_pids[0], signal.SIGKILL)
            except (PermissionError, ProcessLookupError):
                pass
        signals.close()
        for pid in observed_pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize("guardian_behavior", ["exit", "hang"])
@pytest.mark.process_integration
@pytest.mark.process_guardian
def test_guardian_pre_ready_failure_hard_contains_group(
    tmp_path: Path,
    guardian_behavior: str,
) -> None:
    ready_reader, ready_writer = os.pipe()
    fake_guardian_path = tmp_path / "fake_guardian.py"
    fake_guardian_path.write_text(
        """
import os
import signal
import sys

ready_writer, behavior = sys.argv[1:]
signal.signal(signal.SIGTERM, signal.SIG_IGN)
os.write(int(ready_writer), f"{os.getpid()}\\n".encode())
os.close(int(ready_writer))
if behavior == "exit":
    raise SystemExit(2)
signal.pause()
""",
        encoding="utf-8",
    )
    starter_script = """
import os
import subprocess
import sys

import whetstone.execution.process_worker as worker

fake_guardian_path, ready_writer, behavior = sys.argv[1:]
real_popen = subprocess.Popen
worker._GUARDIAN_READY_TIMEOUT_SECONDS = 0.1

def fake_popen(*args, **kwargs):
    return real_popen(
        [sys.executable, fake_guardian_path, ready_writer, behavior],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        pass_fds=(*kwargs["pass_fds"], int(ready_writer)),
    )

worker.subprocess.Popen = fake_popen
lifetime_reader, lifetime_writer = os.pipe()
done_reader, done_writer = os.pipe()
worker._start_guardian(lifetime_reader, done_writer)
"""
    starter = subprocess.Popen(
        [
            sys.executable,
            "-c",
            starter_script,
            os.fspath(fake_guardian_path),
            str(ready_writer),
            guardian_behavior,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        start_new_session=True,
        pass_fds=(ready_writer,),
    )
    os.close(ready_writer)
    ready_writer = -1
    child_pids: list[int] = []
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(ready_reader, selectors.EVENT_READ)
            assert selector.select(3.0), "fake guardian did not publish ready"
        child_pids = [int(os.read(ready_reader, 32).strip())]
        starter.wait(timeout=3.0)
        assert starter.returncode == -signal.SIGKILL
        for pid in child_pids:
            _assert_process_gone(pid)
    finally:
        if starter.poll() is None:
            os.killpg(starter.pid, signal.SIGKILL)
            starter.wait(timeout=3.0)
        for pid in child_pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if ready_writer >= 0:
            os.close(ready_writer)
        os.close(ready_reader)


@pytest.mark.process_integration
@pytest.mark.process_guardian
def test_forked_scheduler_sibling_cannot_keep_worker_group_alive() -> None:
    signals = ProcessSignals()
    scheduler_script = """
import os
import signal
import sys

import whetstone.execution.fanout as fanout
from whetstone.execution.fanout import CallSpec, ProcessJob, run_call_pool
from tests.execution.process_signals import publish_ready

signal_path = sys.argv[1]
real_spawn = fanout._spawn

def fork_after_spawn(*args, **kwargs):
    process = real_spawn(*args, **kwargs)
    sibling_pid = os.fork()
    if sibling_pid == 0:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        publish_ready(signal_path, "scheduler-sibling")
        signal.pause()
    return process

fanout._spawn = fork_after_spawn
run_call_pool(
    [
        CallSpec(
            key="worker",
            job=ProcessJob(
                entrypoint="tests.execution.process_workers:block_process_tree",
                payload={"signal_path": signal_path, "key": "worker"},
            ),
            decode=lambda value: value,
            deadline_seconds=30.0,
        )
    ],
    concurrency=1,
    is_rate_limited=lambda value: False,
)
"""
    scheduler = subprocess.Popen(
        [
            sys.executable,
            "-c",
            scheduler_script,
            os.fspath(signals.path),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    sibling_pid: int | None = None
    worker_pids: list[int] = []
    try:
        signals.wait_entered(
            ["scheduler-sibling", "worker-worker", "worker-descendant"]
        )
        sibling_pid = signals.pid("scheduler-sibling")
        worker_pids = [
            signals.pid("worker-worker"),
            signals.pid("worker-descendant"),
        ]
        os.kill(scheduler.pid, signal.SIGKILL)
        scheduler.wait(timeout=3.0)
        for pid in worker_pids:
            _assert_process_gone(pid)
        os.kill(sibling_pid, 0)
    finally:
        if scheduler.poll() is None:
            scheduler.kill()
            scheduler.wait(timeout=3.0)
        if sibling_pid is not None:
            try:
                os.kill(sibling_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        signals.close()
        for pid in worker_pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.process_integration
@pytest.mark.process_guardian
def test_scheduler_death_after_worker_return_kills_left_descendant() -> None:
    signals = ProcessSignals()
    scheduler_script = """
import os
import signal
import sys

import whetstone.execution.fanout as fanout
from whetstone.execution.fanout import CallSpec, ProcessJob, run_call_pool
from tests.execution.process_signals import publish_ready

signal_path = sys.argv[1]
real_read_worker_result = fanout._read_worker_result

def block_harvest(process):
    publish_ready(signal_path, "harvest")
    signal.pause()

fanout._read_worker_result = block_harvest
run_call_pool(
    [
        CallSpec(
            key="worker",
            job=ProcessJob(
                entrypoint=(
                    "tests.execution.process_workers:"
                    "spawn_descendant_and_return"
                ),
                payload={
                    "signal_path": signal_path,
                    "value": "complete",
                },
            ),
            decode=lambda value: value,
            deadline_seconds=30.0,
        )
    ],
    concurrency=1,
    is_rate_limited=lambda value: False,
)
"""
    scheduler = subprocess.Popen(
        [
            sys.executable,
            "-c",
            scheduler_script,
            os.fspath(signals.path),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    descendant_pids: list[int] = []
    try:
        signals.wait_entered(["harvest", "descendant"])
        descendant_pids = [signals.pid("descendant")]
        os.kill(scheduler.pid, signal.SIGKILL)
        scheduler.wait(timeout=3.0)
        for pid in descendant_pids:
            _assert_process_gone(pid)
    finally:
        if scheduler.poll() is None:
            scheduler.kill()
            scheduler.wait(timeout=3.0)
        for pid in descendant_pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        signals.close()


@pytest.mark.process_integration
def test_normal_completion_stops_left_descendant_before_acceptance() -> None:
    signals = ProcessSignals()
    try:
        outcome = run_call_pool(
            [
                CallSpec(
                    key="worker",
                    job=_job(
                        "spawn_descendant_and_return",
                        {
                            "signal_path": os.fspath(signals.path),
                            "value": "complete",
                        },
                    ),
                    decode=_identity,
                    deadline_seconds=5.0,
                )
            ],
            is_rate_limited=_never_rate_limited,
        )
    finally:
        signals.close()
    assert outcome.results[0].status is FanoutStatus.COMPLETED
    assert outcome.results[0].value == "complete"
    _assert_process_gone(signals.pid("descendant"))


@pytest.mark.process_integration
def test_control_descriptors_close_before_user_code_and_do_not_leak() -> None:
    baseline = _open_fd_count()
    for index in range(12):
        outcome = run_call_pool(
            [
                CallSpec(
                    key=str(index),
                    job=_job("open_file_descriptors", {}),
                    decode=_identity,
                    deadline_seconds=5.0,
                )
            ],
            is_rate_limited=_never_rate_limited,
        )
        assert outcome.results[0].value == []
    assert fanout_module._parent_control_fds == set()
    assert _open_fd_count() <= baseline


@pytest.mark.parametrize(
    "failure_stage",
    [
        "directory",
        "pipe-1",
        "pipe-2",
        "pipe-3",
        "job-write",
        "stderr-open",
        "popen",
    ],
)
def test_spawn_failure_releases_incremental_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    baseline_descriptors = _open_fd_count()
    baseline_registry = set(fanout_module._parent_control_fds)
    real_mkdtemp = fanout_module.tempfile.mkdtemp
    real_open_pipe = fanout_module._open_parent_control_pipe
    real_write_job = fanout_module._write_job
    real_open = fanout_module.os.open
    created_directories: list[Path] = []
    pipe_calls = 0

    def controlled_mkdtemp(*, prefix: str) -> str:
        if failure_stage == "directory":
            raise OSError("requested directory allocation failure")
        directory = Path(real_mkdtemp(prefix=prefix, dir=tmp_path))
        created_directories.append(directory)
        return os.fspath(directory)

    def controlled_open_pipe() -> tuple[int, int]:
        nonlocal pipe_calls
        pipe_calls += 1
        if failure_stage == f"pipe-{pipe_calls}":
            raise OSError("requested pipe allocation failure")
        return real_open_pipe()

    def controlled_write_job(path: Path, job: ProcessJob) -> None:
        if failure_stage == "job-write":
            raise OSError("requested job write failure")
        real_write_job(path, job)

    def controlled_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if (
            failure_stage == "stderr-open"
            and Path(os.fsdecode(path)).name == "stderr.log"
        ):
            raise OSError("requested stderr allocation failure")
        return real_open(path, flags, mode, dir_fd=dir_fd)

    def controlled_popen(
        *args: object,
        **kwargs: object,
    ) -> subprocess.Popen[bytes]:
        del args, kwargs
        raise OSError("requested Popen failure")

    monkeypatch.setattr(fanout_module.tempfile, "mkdtemp", controlled_mkdtemp)
    monkeypatch.setattr(
        fanout_module,
        "_open_parent_control_pipe",
        controlled_open_pipe,
    )
    monkeypatch.setattr(fanout_module, "_write_job", controlled_write_job)
    if failure_stage == "stderr-open":
        monkeypatch.setattr(fanout_module.os, "open", controlled_open)
    if failure_stage == "popen":
        monkeypatch.setattr(
            fanout_module.subprocess,
            "Popen",
            controlled_popen,
        )

    with pytest.raises(OSError, match="requested"):
        fanout_module._spawn(
            0,
            CallSpec(
                key="worker",
                job=_job("return_payload", {"ok": True}),
                decode=_identity,
                deadline_seconds=5.0,
            ),
            operation_deadline=None,
        )

    assert fanout_module._parent_control_fds == baseline_registry
    assert _open_fd_count() <= baseline_descriptors
    assert all(not directory.exists() for directory in created_directories)


@pytest.mark.parametrize("inherit_call", [1, 2])
def test_control_pipe_inherit_failure_closes_both_descriptors(
    monkeypatch: pytest.MonkeyPatch,
    inherit_call: int,
) -> None:
    baseline_descriptors = _open_fd_count()
    baseline_registry = set(fanout_module._parent_control_fds)
    real_set_inheritable = fanout_module.os.set_inheritable
    calls = 0

    def controlled_set_inheritable(
        descriptor: int,
        inheritable: bool,
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == inherit_call:
            raise OSError("requested inheritable failure")
        real_set_inheritable(descriptor, inheritable)

    monkeypatch.setattr(
        fanout_module.os,
        "set_inheritable",
        controlled_set_inheritable,
    )
    with pytest.raises(OSError, match="requested inheritable failure"):
        fanout_module._open_parent_control_pipe()
    assert fanout_module._parent_control_fds == baseline_registry
    assert _open_fd_count() <= baseline_descriptors


@pytest.mark.process_integration
def test_repeated_completed_process_trees_are_clean() -> None:
    all_pids: list[int] = []
    for index in range(5):
        signals = ProcessSignals()
        try:
            outcome = run_call_pool(
                [
                    CallSpec(
                        key=str(index),
                        job=_job(
                            "spawn_descendant_and_return",
                            {
                                "signal_path": os.fspath(signals.path),
                                "value": index,
                            },
                        ),
                        decode=_identity,
                        deadline_seconds=5.0,
                    )
                ],
                is_rate_limited=_never_rate_limited,
            )
            all_pids.append(signals.pid("descendant"))
        finally:
            signals.close()
        assert outcome.results[0].status is FanoutStatus.COMPLETED
    assert len(all_pids) == 5
    for pid in all_pids:
        _assert_process_gone(pid)


@pytest.mark.process_integration
def test_lazy_dispatch_never_starts_more_than_current_capacity() -> None:
    signals = ProcessSignals()
    specs = [
        CallSpec(
            key=str(index),
            job=_job(
                "gated_event",
                {
                    "key": str(index),
                    "signal_path": os.fspath(signals.path),
                    "value": str(index),
                },
            ),
            decode=_identity,
            deadline_seconds=5.0,
        )
        for index in range(6)
    ]
    outcome: list[object] = []
    failure: list[BaseException] = []

    def schedule() -> None:
        try:
            outcome.append(
                run_call_pool(
                    specs,
                    concurrency=2,
                    is_rate_limited=_never_rate_limited,
                )
            )
        except BaseException as error:
            failure.append(error)

    scheduler = threading.Thread(target=schedule)
    scheduler.start()
    try:
        signals.wait_entered(["0", "1"])
        assert signals.entered_keys == {"0", "1"}
        for index in range(6):
            key = str(index)
            signals.wait_entered([key])
            signals.release(key)
    finally:
        scheduler.join(timeout=5.0)
        signals.close()
    assert not scheduler.is_alive()
    assert not failure
    assert len(outcome) == 1


@pytest.mark.parametrize("failure_stage", ["decode", "predicate", "commit"])
@pytest.mark.process_integration
def test_accepted_worker_is_not_cancelled_when_parent_callback_fails(
    failure_stage: str,
) -> None:
    signals = ProcessSignals()
    barriers: list[str] = []

    def fail(stage: str) -> None:
        if failure_stage == stage:
            raise RuntimeError(f"{stage} failed")

    def decode(value: JsonValue) -> JsonValue:
        fail("decode")
        return value

    def predicate(_value: JsonValue) -> bool:
        fail("predicate")
        return False

    def commit(_value: JsonValue) -> None:
        fail("commit")

    failures: list[BaseException] = []

    def schedule() -> None:
        try:
            run_call_pool(
                [
                    _gated_spec(
                        "accepted",
                        signals=signals,
                        decode=decode,
                        commit=commit,
                        cancellation_barrier=lambda: barriers.append(
                            "accepted"
                        ),
                    ),
                    _blocking_tree_spec(
                        "sibling",
                        signals,
                        deadline=5.0,
                        cancellation_barrier=lambda: barriers.append(
                            "sibling"
                        ),
                    ),
                ],
                concurrency=2,
                is_rate_limited=predicate,
            )
        except BaseException as error:
            failures.append(error)

    scheduler = threading.Thread(target=schedule)
    scheduler.start()
    try:
        signals.wait_entered(
            ["accepted", "sibling-worker", "sibling-descendant"]
        )
        signals.release("accepted")
        scheduler.join(timeout=10)
    finally:
        signals.close()
    assert not scheduler.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], RuntimeError)
    assert str(failures[0]) == f"{failure_stage} failed"
    assert barriers == ["sibling"]
    for pid in (
        signals.pid("sibling-worker"),
        signals.pid("sibling-descendant"),
    ):
        _assert_process_gone(pid)


@pytest.mark.process_integration
def test_completion_order_drives_commits_but_results_preserve_input_order() -> (  # noqa: E501
    None
):
    signals = ProcessSignals()
    commits: list[JsonValue] = []
    committed = {key: threading.Event() for key in ("slow", "fast", "middle")}

    def record_commit(value: JsonValue) -> None:
        commits.append(value)
        assert isinstance(value, str)
        committed[value].set()

    specs = [
        _gated_spec(
            "slow",
            signals=signals,
            commit=record_commit,
        ),
        _gated_spec(
            "fast",
            signals=signals,
            commit=record_commit,
        ),
        _gated_spec(
            "middle",
            signals=signals,
            commit=record_commit,
        ),
    ]
    outcomes: list[object] = []

    def schedule() -> None:
        outcomes.append(
            run_call_pool(
                specs,
                concurrency=3,
                is_rate_limited=_never_rate_limited,
            )
        )

    scheduler = threading.Thread(target=schedule)
    scheduler.start()
    try:
        signals.wait_entered(["slow", "fast", "middle"])
        for key in ("fast", "middle", "slow"):
            signals.release(key)
            assert committed[key].wait(timeout=10)
        scheduler.join(timeout=10)
    finally:
        signals.close()
    assert not scheduler.is_alive()
    assert len(outcomes) == 1
    outcome = cast(fanout_module.PoolOutcome[str, JsonValue], outcomes[0])
    assert commits == ["fast", "middle", "slow"]
    assert [result.key for result in outcome.results] == [
        "slow",
        "fast",
        "middle",
    ]
    assert [result.value for result in outcome.results] == [
        "slow",
        "fast",
        "middle",
    ]


@pytest.mark.process_integration
def test_rate_feedback_reduces_capacity_before_filling_it() -> None:
    signals = ProcessSignals()
    committed = {
        key: threading.Event()
        for key in ("slow", "limited", "middle-1", "middle-2", "queued")
    }

    def record_commit(value: JsonValue) -> None:
        assert isinstance(value, str)
        committed[value].set()

    specs = [
        _gated_spec(
            "slow",
            signals=signals,
            commit=record_commit,
        ),
        _gated_spec(
            "limited",
            signals=signals,
            commit=record_commit,
        ),
        _gated_spec(
            "middle-1",
            signals=signals,
            commit=record_commit,
        ),
        _gated_spec(
            "middle-2",
            signals=signals,
            commit=record_commit,
        ),
        _gated_spec("queued", signals=signals, commit=record_commit),
    ]
    outcomes: list[fanout_module.PoolOutcome[str, JsonValue]] = []

    def schedule() -> None:
        outcomes.append(
            run_call_pool(
                specs,
                concurrency=4,
                is_rate_limited=lambda value: value == "limited",
            )
        )

    scheduler = threading.Thread(target=schedule)
    scheduler.start()
    try:
        initial = ["slow", "limited", "middle-1", "middle-2"]
        signals.wait_entered(initial)
        assert signals.entered_keys == set(initial)
        signals.release("limited")
        assert committed["limited"].wait(timeout=10)
        signals.release("middle-1")
        assert committed["middle-1"].wait(timeout=10)
        assert "queued" not in signals.entered_keys
        signals.release("middle-2")
        assert committed["middle-2"].wait(timeout=10)
        signals.wait_entered(["queued"])
        signals.release("queued")
        signals.release("slow")
        scheduler.join(timeout=10)
    finally:
        signals.close()
    assert not scheduler.is_alive()
    assert len(outcomes) == 1
    outcome = outcomes[0]
    assert outcome.concurrency_halved
    assert outcome.effective_concurrency == 2


@pytest.mark.process_integration
def test_unit_deadline_starts_when_each_child_starts() -> None:
    signals = ProcessSignals()
    outcomes: list[fanout_module.PoolOutcome[str, JsonValue]] = []

    def schedule() -> None:
        outcomes.append(
            run_call_pool(
                [
                    _blocking_tree_spec("first", signals, deadline=0.2),
                    _gated_spec("second", signals=signals, deadline=0.2),
                ],
                concurrency=1,
                is_rate_limited=_never_rate_limited,
            )
        )

    scheduler = threading.Thread(target=schedule)
    scheduler.start()
    try:
        signals.wait_entered(["first-worker", "first-descendant"])
        signals.wait_entered(["second"])
        signals.release("second")
        scheduler.join(timeout=10)
    finally:
        signals.close()
    assert not scheduler.is_alive()
    assert len(outcomes) == 1
    outcome = outcomes[0]
    assert [result.status for result in outcome.results] == [
        FanoutStatus.UNIT_TIMEOUT,
        FanoutStatus.COMPLETED,
    ]


@pytest.mark.process_integration
def test_unit_timeout_kills_process_and_prevents_late_commit() -> None:
    signals = ProcessSignals()
    commits: list[JsonValue] = []
    barriers: list[str] = []
    try:
        outcome = run_call_pool(
            [
                _blocking_tree_spec(
                    "hung",
                    signals,
                    deadline=0.3,
                    commit=commits.append,
                    cancellation_barrier=lambda: barriers.append("terminal"),
                )
            ],
            concurrency=1,
            is_rate_limited=_never_rate_limited,
        )
    finally:
        signals.close()
    assert outcome.results[0].status is FanoutStatus.UNIT_TIMEOUT
    assert outcome.guard_timeouts == 1
    assert commits == []
    assert barriers == ["terminal"]
    for pid in (
        signals.pid("hung-worker"),
        signals.pid("hung-descendant"),
    ):
        _assert_process_gone(pid)


@pytest.mark.process_integration
def test_operation_deadline_kills_active_and_never_dispatches_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals = ProcessSignals()
    commits: list[JsonValue] = []
    outcomes: list[fanout_module.PoolOutcome[str, JsonValue]] = []
    wall_ready = threading.Event()
    scripted_deadline = _ScriptedDeadline(wall_ready)
    monkeypatch.setattr(
        fanout_module,
        "_wait_for_operation_deadline",
        scripted_deadline,
    )

    def schedule() -> None:
        outcomes.append(
            run_call_pool(
                [
                    _blocking_tree_spec(
                        "active-1",
                        signals,
                        deadline=5.0,
                        commit=commits.append,
                    ),
                    _blocking_tree_spec(
                        "active-2",
                        signals,
                        deadline=5.0,
                        commit=commits.append,
                    ),
                    _gated_spec(
                        "queued-1",
                        signals=signals,
                        commit=commits.append,
                    ),
                    _gated_spec(
                        "queued-2",
                        signals=signals,
                        commit=commits.append,
                    ),
                ],
                concurrency=2,
                max_wall_seconds=sys.float_info.max,
                is_rate_limited=_never_rate_limited,
            )
        )

    scheduler = threading.Thread(target=schedule)
    scheduler.start()
    try:
        active_keys = [
            "active-1-worker",
            "active-1-descendant",
            "active-2-worker",
            "active-2-descendant",
        ]
        signals.wait_entered(active_keys)
        wall_ready.set()
        scheduler.join(timeout=10)
    finally:
        signals.close()
    assert not scheduler.is_alive()
    assert len(outcomes) == 1
    outcome = outcomes[0]
    assert [result.status for result in outcome.results] == [
        FanoutStatus.OPERATION_DEADLINE,
        FanoutStatus.OPERATION_DEADLINE,
        FanoutStatus.NOT_DISPATCHED,
        FanoutStatus.NOT_DISPATCHED,
    ]
    assert outcome.deadline_reached
    scripted_deadline.assert_satisfied()
    assert outcome.not_dispatched == ["queued-1", "queued-2"]
    assert commits == []
    assert not {"queued-1", "queued-2"} & signals.entered_keys
    for key in active_keys:
        pid = signals.pid(key)
        _assert_process_gone(pid)
    for key in ("active-1-worker", "active-2-worker"):
        _assert_process_group_absent(signals.pid(key))


@pytest.mark.process_integration
def test_wall_watcher_stops_sibling_while_decode_runs_past_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals = ProcessSignals()
    decode_entered = threading.Event()
    scripted_deadline = _ScriptedDeadline(decode_entered)
    monkeypatch.setattr(
        fanout_module,
        "_wait_for_operation_deadline",
        scripted_deadline,
    )
    sibling_stopped_during_decode: list[bool] = []
    sibling_terminal = threading.Event()

    def blocking_decode(value: JsonValue) -> JsonValue:
        decode_entered.set()
        assert scripted_deadline.triggered.wait(timeout=10)
        assert sibling_terminal.wait(timeout=10)
        for key in ("sibling-worker", "sibling-descendant"):
            _assert_process_gone(signals.pid(key))
        sibling_stopped_during_decode.append(True)
        return value

    outcomes: list[fanout_module.PoolOutcome[str, JsonValue]] = []

    def schedule() -> None:
        outcomes.append(
            run_call_pool(
                [
                    _gated_spec(
                        "completed",
                        signals=signals,
                        decode=blocking_decode,
                    ),
                    _blocking_tree_spec(
                        "sibling",
                        signals,
                        deadline=5.0,
                        cancellation_barrier=sibling_terminal.set,
                    ),
                ],
                concurrency=2,
                max_wall_seconds=60.0,
                is_rate_limited=_never_rate_limited,
            )
        )

    scheduler = threading.Thread(target=schedule)
    scheduler.start()
    try:
        signals.wait_entered(
            ["completed", "sibling-worker", "sibling-descendant"]
        )
        signals.release("completed")
        scheduler.join(timeout=10)
    finally:
        signals.close()
    assert not scheduler.is_alive()
    scripted_deadline.assert_satisfied()
    assert len(outcomes) == 1
    outcome = outcomes[0]
    assert sibling_stopped_during_decode == [True]
    assert [result.status for result in outcome.results] == [
        FanoutStatus.OPERATION_DEADLINE,
        FanoutStatus.OPERATION_DEADLINE,
    ]


@pytest.mark.process_integration
def test_slow_spawn_cannot_release_worker_after_wall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals = ProcessSignals()
    popen_entered = threading.Event()
    scripted_deadline = _ScriptedDeadline(popen_entered)
    monkeypatch.setattr(
        fanout_module,
        "_wait_for_operation_deadline",
        scripted_deadline,
    )
    real_popen = subprocess.Popen

    def slow_popen(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
        process = cast(
            "subprocess.Popen[bytes]",
            real_popen(*args, **kwargs),  # ty: ignore[no-matching-overload]
        )
        popen_entered.set()
        assert scripted_deadline.triggered.wait(timeout=10)
        return process

    monkeypatch.setattr(fanout_module.subprocess, "Popen", slow_popen)
    outcome = run_call_pool(
        [_gated_spec("queued", signals=signals)],
        concurrency=1,
        max_wall_seconds=60.0,
        is_rate_limited=_never_rate_limited,
    )
    assert outcome.results[0].status is FanoutStatus.NOT_DISPATCHED
    scripted_deadline.assert_satisfied()
    assert "queued" not in signals.entered_keys
    signals.close()


def test_slow_serialization_stops_before_spawn_after_wall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals = ProcessSignals()
    serialization_entered = threading.Event()
    scripted_deadline = _ScriptedDeadline(serialization_entered)
    monkeypatch.setattr(
        fanout_module,
        "_wait_for_operation_deadline",
        scripted_deadline,
    )
    real_write_job = fanout_module._write_job

    def slow_write_job(path: Path, job: ProcessJob) -> None:
        real_write_job(path, job)
        serialization_entered.set()
        assert scripted_deadline.triggered.wait(timeout=10)

    monkeypatch.setattr(fanout_module, "_write_job", slow_write_job)
    outcome = run_call_pool(
        [_gated_spec("queued", signals=signals)],
        concurrency=1,
        max_wall_seconds=60.0,
        is_rate_limited=_never_rate_limited,
    )
    assert outcome.results[0].status is FanoutStatus.NOT_DISPATCHED
    scripted_deadline.assert_satisfied()
    assert "queued" not in signals.entered_keys
    signals.close()


@pytest.mark.process_integration
def test_slow_commit_may_finish_but_wall_stops_later_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals = ProcessSignals()
    commit_entered = threading.Event()
    scripted_deadline = _ScriptedDeadline(commit_entered)
    monkeypatch.setattr(
        fanout_module,
        "_wait_for_operation_deadline",
        scripted_deadline,
    )

    def blocking_commit(_value: JsonValue) -> None:
        commit_entered.set()
        assert scripted_deadline.triggered.wait(timeout=10)

    outcomes: list[fanout_module.PoolOutcome[str, JsonValue]] = []

    def schedule() -> None:
        outcomes.append(
            run_call_pool(
                [
                    _gated_spec(
                        "committed",
                        signals=signals,
                        commit=blocking_commit,
                    ),
                    _gated_spec("queued", signals=signals),
                ],
                concurrency=1,
                max_wall_seconds=60.0,
                is_rate_limited=_never_rate_limited,
            )
        )

    scheduler = threading.Thread(target=schedule)
    scheduler.start()
    try:
        signals.wait_entered(["committed"])
        signals.release("committed")
        scheduler.join(timeout=10)
    finally:
        signals.close()
    assert not scheduler.is_alive()
    scripted_deadline.assert_satisfied()
    assert len(outcomes) == 1
    outcome = outcomes[0]
    assert [result.status for result in outcome.results] == [
        FanoutStatus.COMPLETED,
        FanoutStatus.NOT_DISPATCHED,
    ]
    assert outcome.deadline_reached
    assert "queued" not in signals.entered_keys


@pytest.mark.process_integration
def test_wall_crossing_during_cancellation_never_dispatches_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals = ProcessSignals()
    barrier_entered = threading.Event()
    scripted_deadline = _ScriptedDeadline(barrier_entered)
    monkeypatch.setattr(
        fanout_module,
        "_wait_for_operation_deadline",
        scripted_deadline,
    )

    def blocking_barrier() -> None:
        barrier_entered.set()
        assert scripted_deadline.triggered.wait(timeout=10)

    outcome = run_call_pool(
        [
            _blocking_tree_spec(
                "timeout",
                signals,
                deadline=0.2,
                cancellation_barrier=blocking_barrier,
            ),
            _gated_spec("queued", signals=signals),
        ],
        concurrency=1,
        max_wall_seconds=60.0,
        is_rate_limited=_never_rate_limited,
    )
    assert [result.status for result in outcome.results] == [
        FanoutStatus.UNIT_TIMEOUT,
        FanoutStatus.NOT_DISPATCHED,
    ]
    assert outcome.deadline_reached
    scripted_deadline.assert_satisfied()
    assert "queued" not in signals.entered_keys
    signals.close()


@pytest.mark.process_integration
def test_unexpected_child_failure_cancels_siblings_before_raise() -> None:
    signals = ProcessSignals()
    failures: list[BaseException] = []

    def schedule() -> None:
        try:
            run_call_pool(
                [
                    _blocking_tree_spec("sibling", signals, deadline=5.0),
                    _gated_spec(
                        "failed",
                        signals=signals,
                        fail=True,
                    ),
                ],
                concurrency=2,
                is_rate_limited=_never_rate_limited,
            )
        except BaseException as error:
            failures.append(error)

    scheduler = threading.Thread(target=schedule)
    scheduler.start()
    try:
        signals.wait_entered(
            ["sibling-worker", "sibling-descendant", "failed"]
        )
        signals.release("failed")
        scheduler.join(timeout=10)
    finally:
        signals.close()
    assert not scheduler.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], ProcessWorkerError)
    assert "requested failure for failed" in str(failures[0])
    for pid in (
        signals.pid("sibling-worker"),
        signals.pid("sibling-descendant"),
    ):
        _assert_process_gone(pid)


@pytest.mark.parametrize(
    "concurrency",
    [0, -1, True, 1.0, "1"],
)
def test_invalid_concurrency_fails_before_dispatch(
    concurrency: object,
) -> None:
    with pytest.raises(ValueError, match="positive non-bool integer"):
        run_call_pool(
            [],
            concurrency=cast(int, concurrency),
            is_rate_limited=lambda _value: False,
        )


@pytest.mark.parametrize(
    "duration",
    [-1, True, float("nan"), float("inf"), 10**400, Decimal("1")],
)
def test_invalid_wall_fails_before_dispatch(duration: object) -> None:
    with pytest.raises(ValueError, match="finite nonnegative real"):
        run_call_pool(
            [],
            max_wall_seconds=cast(float, duration),
            is_rate_limited=lambda _value: False,
        )


def test_platform_sized_finite_wall_is_supported() -> None:
    outcome = run_call_pool(
        [],
        max_wall_seconds=sys.float_info.max,
        is_rate_limited=_never_rate_limited,
    )
    assert outcome.results == ()
    assert not outcome.deadline_reached


@pytest.mark.parametrize(
    "duration",
    [-1, True, float("nan"), float("inf"), 10**400, Decimal("1")],
)
def test_invalid_unit_deadline_fails_before_dispatch(
    duration: object,
) -> None:
    signals = ProcessSignals()
    spec = _gated_spec(
        "invalid",
        signals=signals,
        deadline=cast(float, duration),
    )
    with pytest.raises(ValueError, match="finite nonnegative real"):
        run_call_pool(
            [spec],
            is_rate_limited=_never_rate_limited,
        )
    assert "invalid" not in signals.entered_keys
    signals.close()


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.process_integration
def test_worker_rejects_non_finite_result_recursively(
    tmp_path: Path,
    nested: bool,
) -> None:
    event_path = tmp_path / "events"
    with pytest.raises(ProcessWorkerError, match="non-finite JSON number"):
        run_call_pool(
            [
                CallSpec(
                    key="invalid-output",
                    job=_job(
                        "non_finite_result",
                        {
                            "event_path": os.fspath(event_path),
                            "nested": nested,
                        },
                    ),
                    decode=_identity,
                    deadline_seconds=5.0,
                )
            ],
            is_rate_limited=_never_rate_limited,
        )
