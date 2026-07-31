"""Spawned-process fanout scheduling and cancellation contracts."""

from __future__ import annotations

import errno
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import cast

import pytest
from pydantic import JsonValue, ValidationError

import whetstone.execution.fanout as fanout_module
from whetstone.execution.fanout import (
    CallSpec,
    FanoutStatus,
    ProcessCancellationError,
    ProcessJob,
    ProcessWorkerError,
    run_call_pool,
)

_WORKERS = "tests.execution.process_workers"


def _identity(value: JsonValue) -> JsonValue:
    return value


def _never_rate_limited(_value: JsonValue) -> bool:
    return False


def _job(function: str, payload: JsonValue) -> ProcessJob:
    return ProcessJob(
        entrypoint=f"{_WORKERS}:{function}",
        payload=payload,
    )


def _delayed_spec(
    key: str,
    *,
    event_path: Path,
    delay: float,
    value: JsonValue | None = None,
    deadline: float = 5.0,
    commit: Callable[[JsonValue], None] | None = None,
    fail: bool = False,
    wait_for_started: int | None = None,
    decode: Callable[[JsonValue], JsonValue] = _identity,
    cancellation_barrier: Callable[[], None] | None = None,
) -> CallSpec[str, JsonValue]:
    payload: dict[str, JsonValue] = {
        "key": key,
        "event_path": os.fspath(event_path),
        "delay": delay,
        "value": key if value is None else value,
        "fail": fail,
    }
    if wait_for_started is not None:
        payload["wait_for_started"] = wait_for_started
    return CallSpec(
        key=key,
        job=_job("delayed_event", payload),
        decode=decode,
        deadline_seconds=deadline,
        commit=commit,
        cancellation_barrier=cancellation_barrier,
    )


def _heartbeat_spec(
    key: str,
    path: Path,
    *,
    deadline: float,
    commit: Callable[[JsonValue], None] | None = None,
    cancellation_barrier: Callable[[], None] | None = None,
) -> CallSpec[str, JsonValue]:
    return CallSpec(
        key=key,
        job=_job(
            "heartbeat_forever",
            {"heartbeat_path": os.fspath(path)},
        ),
        decode=_identity,
        deadline_seconds=deadline,
        commit=commit,
        cancellation_barrier=cancellation_barrier,
    )


def _wait_until(
    predicate: Callable[[], bool],
    *,
    timeout: float = 3.0,
) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition was not met before timeout")
        time.sleep(0.005)


def _lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []


def _pid_lines(path: Path) -> list[int]:
    return [
        int(line.split("|", maxsplit=1)[1])
        for line in _lines(path)
        if line.startswith("pid|")
    ]


def _assert_process_gone(pid: int) -> None:
    deadline = time.monotonic() + 1.0
    while True:
        try:
            os.kill(pid, 0)
        except OSError as error:
            assert error.errno == errno.ESRCH
            return
        if time.monotonic() >= deadline:
            raise AssertionError(f"process {pid} survived scheduler return")
        time.sleep(0.01)


def _open_fd_count() -> int:
    descriptor_directory = (
        Path("/proc/self/fd")
        if Path("/proc/self/fd").exists()
        else Path("/dev/fd")
    )
    return len(tuple(descriptor_directory.iterdir()))


def _guardian_pid(worker_pid: int) -> int | None:
    completed = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,command="],
        check=True,
        capture_output=True,
        text=True,
    )
    for raw_line in completed.stdout.splitlines():
        columns = raw_line.strip().split(maxsplit=2)
        if len(columns) != 3:
            continue
        pid, parent_pid, command = columns
        if (
            int(parent_pid) == worker_pid
            and "whetstone.execution.process_guardian" in command
        ):
            return int(pid)
    return None


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


def test_worker_result_file_is_restrictive(tmp_path: Path) -> None:
    job_path = tmp_path / "job.json"
    result_path = tmp_path / "result.json"
    started_path = tmp_path / "started.bin"
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
                os.fspath(started_path),
                str(parent_reader),
                str(guardian_writer),
                str(start_reader),
                "none",
            ],
            check=False,
            capture_output=True,
            pass_fds=(parent_reader, guardian_writer, start_reader),
            start_new_session=True,
        )
    finally:
        os.close(parent_reader)
        os.close(guardian_writer)
        os.close(start_reader)
    os.close(parent_writer)
    assert os.read(guardian_reader, 1) == b""
    os.close(guardian_reader)
    assert completed.returncode == 0, completed.stderr.decode()
    assert completed.stderr == b""
    assert result_path.stat().st_mode & 0o777 == 0o600
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


@pytest.mark.parametrize("parent_signal", [signal.SIGTERM, signal.SIGKILL])
def test_parent_death_kills_fresh_worker_process_group(
    tmp_path: Path,
    parent_signal: int,
) -> None:
    heartbeat_path = tmp_path / "heartbeat"
    scheduler_script = """
import os
import sys

from whetstone.execution.fanout import CallSpec, ProcessJob, run_call_pool

path = sys.argv[1]
job = ProcessJob(
    entrypoint="tests.execution.process_workers:heartbeat_forever",
    payload={"heartbeat_path": path},
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
        [sys.executable, "-c", scheduler_script, os.fspath(heartbeat_path)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        _wait_until(lambda: len(_pid_lines(heartbeat_path)) == 2)
        os.kill(scheduler.pid, parent_signal)
        scheduler.wait(timeout=3.0)
        pids = _pid_lines(heartbeat_path)
        for pid in pids:
            _assert_process_gone(pid)
        returned_content = heartbeat_path.read_bytes()
        time.sleep(0.1)
        assert heartbeat_path.read_bytes() == returned_content
    finally:
        if scheduler.poll() is None:
            scheduler.kill()
            scheduler.wait()


def test_stopped_guardian_on_completion_forces_local_containment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    heartbeat_path = tmp_path / "heartbeat"
    release_path = tmp_path / "release"
    spawned: list[fanout_module._ActiveProcess[str, JsonValue]] = []
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
                                "heartbeat_path": os.fspath(heartbeat_path),
                                "release_path": os.fspath(release_path),
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
    _wait_until(lambda: len(spawned) == 1)
    process = spawned[0]
    _wait_until(lambda: len(_pid_lines(heartbeat_path)) == 1)
    guardian_pids: list[int] = []

    def capture_guardian() -> bool:
        guardian_pid = _guardian_pid(process.process.pid)
        if guardian_pid is None:
            return False
        guardian_pids.append(guardian_pid)
        return True

    _wait_until(capture_guardian)
    guardian_pid = guardian_pids[0]
    try:
        os.kill(guardian_pid, signal.SIGSTOP)
        release_path.touch()
        scheduler.join(timeout=3.0)
        assert not scheduler.is_alive()
        assert len(failures) == 1
        assert isinstance(failures[0], ProcessCancellationError)
        assert "did not exit" in str(failures[0])
        for pid in (
            process.process.pid,
            guardian_pid,
            *_pid_lines(heartbeat_path),
        ):
            _assert_process_gone(pid)
        returned_content = heartbeat_path.read_bytes()
        time.sleep(0.1)
        assert heartbeat_path.read_bytes() == returned_content
        assert not process.directory.exists()
        assert fanout_module._parent_control_fds == set()
    finally:
        release_path.touch(exist_ok=True)
        if scheduler.is_alive():
            os.killpg(process.process_group_id, signal.SIGKILL)
            scheduler.join(timeout=3.0)


def test_harvest_retains_state_when_fallback_cannot_prove_containment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    heartbeat_path = tmp_path / "heartbeat"
    release_path = tmp_path / "release"
    spawned: list[fanout_module._ActiveProcess[str, JsonValue]] = []
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
                                "heartbeat_path": os.fspath(heartbeat_path),
                                "release_path": os.fspath(release_path),
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
    _wait_until(lambda: len(spawned) == 1)
    process = spawned[0]
    _wait_until(lambda: len(_pid_lines(heartbeat_path)) == 1)
    guardian_pids: list[int] = []

    def capture_guardian() -> bool:
        guardian_pid = _guardian_pid(process.process.pid)
        if guardian_pid is None:
            return False
        guardian_pids.append(guardian_pid)
        return True

    _wait_until(capture_guardian)
    guardian_pid = guardian_pids[0]

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
        release_path.touch()
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
        release_path.touch(exist_ok=True)
        try:
            os.kill(guardian_pid, signal.SIGCONT)
        except ProcessLookupError:
            pass
        try:
            os.killpg(process.process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.process.wait(timeout=3.0)
        _wait_until(lambda: not real_group_exists(process))
        process.cleanup_allowed = True
        process.release_guardian_after_containment()
        process.cleanup()
        scheduler.join(timeout=3.0)


@pytest.mark.parametrize(
    "failure_site",
    ["unit-expiration", "wall-watcher", "outer-exception"],
)
def test_cancellation_failure_retains_uncontained_process_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_site: str,
) -> None:
    heartbeat_path = tmp_path / "heartbeat"
    event_path = tmp_path / "events"
    target: list[fanout_module._ActiveProcess[str, JsonValue]] = []
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
        _heartbeat_spec(
            "target",
            heartbeat_path,
            deadline=(0.2 if failure_site == "unit-expiration" else 5.0),
        )
    ]
    max_wall_seconds: float | None = None
    if failure_site == "wall-watcher":
        max_wall_seconds = 0.2
    elif failure_site == "outer-exception":
        specs.append(
            _delayed_spec(
                "failed",
                event_path=event_path,
                delay=0.3,
                fail=True,
            )
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
    _wait_until(lambda: len(target) == 1 and bool(_pid_lines(heartbeat_path)))
    process = target[0]
    failure_enabled.set()
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
    finally:
        failure_enabled.clear()
        monkeypatch.undo()
        try:
            os.killpg(process.process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.process.wait(timeout=3.0)
        _wait_until(lambda: not real_group_exists(process))
        process.cleanup_allowed = True
        process.release_guardian_after_containment()
        process.cleanup()
        scheduler.join(timeout=3.0)


def test_worker_contains_group_when_guardian_and_scheduler_die(
    tmp_path: Path,
) -> None:
    heartbeat_path = tmp_path / "heartbeat"
    worker_pid_path = tmp_path / "worker-pid"
    scheduler_script = """
import os
import sys

import whetstone.execution.fanout as fanout
from whetstone.execution.fanout import CallSpec, ProcessJob, run_call_pool

heartbeat_path, worker_pid_path = sys.argv[1:]
real_spawn = fanout._spawn

def record_spawn(*args, **kwargs):
    process = real_spawn(*args, **kwargs)
    with open(worker_pid_path, "w", encoding="utf-8") as output:
        output.write(str(process.process.pid))
        output.flush()
        os.fsync(output.fileno())
    return process

fanout._spawn = record_spawn
run_call_pool(
    [
        CallSpec(
            key="worker",
            job=ProcessJob(
                entrypoint="tests.execution.process_workers:heartbeat_forever",
                payload={"heartbeat_path": heartbeat_path},
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
            os.fspath(heartbeat_path),
            os.fspath(worker_pid_path),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    worker_pid: int | None = None
    observed_pids: list[int] = []
    try:
        _wait_until(
            lambda: (
                worker_pid_path.exists()
                and len(_pid_lines(heartbeat_path)) == 2
            )
        )
        worker_pid = int(worker_pid_path.read_text(encoding="utf-8"))
        guardian_pids: list[int] = []

        def capture_guardian() -> bool:
            assert worker_pid is not None
            guardian_pid = _guardian_pid(worker_pid)
            if guardian_pid is None:
                return False
            guardian_pids.append(guardian_pid)
            return True

        _wait_until(capture_guardian)
        guardian_pid = guardian_pids[0]
        observed_pids = [
            worker_pid,
            guardian_pid,
            *_pid_lines(heartbeat_path),
        ]
        os.kill(guardian_pid, signal.SIGKILL)
        os.kill(scheduler.pid, signal.SIGKILL)
        scheduler.wait(timeout=3.0)
        for pid in observed_pids:
            _assert_process_gone(pid)
        returned_content = heartbeat_path.read_bytes()
        time.sleep(0.1)
        assert heartbeat_path.read_bytes() == returned_content
    finally:
        if scheduler.poll() is None:
            scheduler.kill()
            scheduler.wait()
        if worker_pid is not None:
            try:
                os.killpg(worker_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for pid in observed_pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize("guardian_behavior", ["exit", "hang"])
def test_guardian_pre_ready_failure_hard_contains_group(
    tmp_path: Path,
    guardian_behavior: str,
) -> None:
    heartbeat_path = tmp_path / "heartbeat"
    fake_guardian_path = tmp_path / "fake_guardian.py"
    fake_guardian_path.write_text(
        """
import os
import signal
import sys
import time

path, behavior = sys.argv[1:]
signal.signal(signal.SIGTERM, signal.SIG_IGN)
descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
os.write(descriptor, f"pid|{os.getpid()}\\n".encode())
if behavior == "exit":
    raise SystemExit(2)
while True:
    os.write(descriptor, f"tick|{time.monotonic()}\\n".encode())
    time.sleep(0.01)
""",
        encoding="utf-8",
    )
    starter_script = """
import os
import subprocess
import sys

import whetstone.execution.process_worker as worker

fake_guardian_path, heartbeat_path, behavior = sys.argv[1:]
real_popen = subprocess.Popen

def fake_popen(*args, **kwargs):
    return real_popen(
        [sys.executable, fake_guardian_path, heartbeat_path, behavior],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        pass_fds=kwargs["pass_fds"],
    )

worker.subprocess.Popen = fake_popen
lifetime_reader, lifetime_writer = os.pipe()
done_reader, done_writer = os.pipe()
worker._start_guardian(lifetime_reader, done_writer)
"""
    started_at = time.monotonic()
    starter = subprocess.Popen(
        [
            sys.executable,
            "-c",
            starter_script,
            os.fspath(fake_guardian_path),
            os.fspath(heartbeat_path),
            guardian_behavior,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    child_pids: list[int] = []
    try:
        _wait_until(lambda: len(_pid_lines(heartbeat_path)) == 1)
        child_pids = _pid_lines(heartbeat_path)
        starter.wait(timeout=3.0)
        assert time.monotonic() - started_at < 2.0
        assert starter.returncode == -signal.SIGKILL
        for pid in child_pids:
            _assert_process_gone(pid)
        returned_content = heartbeat_path.read_bytes()
        time.sleep(0.1)
        assert heartbeat_path.read_bytes() == returned_content
    finally:
        if starter.poll() is None:
            os.killpg(starter.pid, signal.SIGKILL)
            starter.wait()
        for pid in child_pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_forked_scheduler_sibling_cannot_keep_worker_group_alive(
    tmp_path: Path,
) -> None:
    heartbeat_path = tmp_path / "heartbeat"
    sibling_pid_path = tmp_path / "sibling-pid"
    scheduler_script = """
import os
import signal
import sys
import time

import whetstone.execution.fanout as fanout
from whetstone.execution.fanout import CallSpec, ProcessJob, run_call_pool

heartbeat_path, sibling_pid_path = sys.argv[1:]
real_spawn = fanout._spawn

def fork_after_spawn(*args, **kwargs):
    process = real_spawn(*args, **kwargs)
    sibling_pid = os.fork()
    if sibling_pid == 0:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        with open(sibling_pid_path, "w", encoding="utf-8") as output:
            output.write(str(os.getpid()))
            output.flush()
            os.fsync(output.fileno())
        while True:
            time.sleep(1.0)
    return process

fanout._spawn = fork_after_spawn
run_call_pool(
    [
        CallSpec(
            key="worker",
            job=ProcessJob(
                entrypoint="tests.execution.process_workers:heartbeat_forever",
                payload={"heartbeat_path": heartbeat_path},
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
            os.fspath(heartbeat_path),
            os.fspath(sibling_pid_path),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    sibling_pid: int | None = None
    worker_pids: list[int] = []
    try:
        _wait_until(
            lambda: (
                sibling_pid_path.exists()
                and len(_pid_lines(heartbeat_path)) == 2
            )
        )
        sibling_pid = int(sibling_pid_path.read_text(encoding="utf-8"))
        worker_pids = _pid_lines(heartbeat_path)
        os.kill(scheduler.pid, signal.SIGKILL)
        scheduler.wait(timeout=3.0)
        for pid in worker_pids:
            _assert_process_gone(pid)
        os.kill(sibling_pid, 0)
    finally:
        if scheduler.poll() is None:
            scheduler.kill()
            scheduler.wait()
        if sibling_pid is not None:
            try:
                os.kill(sibling_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for pid in worker_pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_scheduler_death_after_worker_return_kills_left_descendant(
    tmp_path: Path,
) -> None:
    heartbeat_path = tmp_path / "heartbeat"
    harvest_path = tmp_path / "harvest-entered"
    scheduler_script = """
import os
import sys
import time

import whetstone.execution.fanout as fanout
from whetstone.execution.fanout import CallSpec, ProcessJob, run_call_pool

heartbeat_path, harvest_path = sys.argv[1:]
real_read_worker_result = fanout._read_worker_result

def block_harvest(process):
    with open(harvest_path, "wb") as output:
        output.write(b"entered")
        output.flush()
        os.fsync(output.fileno())
    while True:
        time.sleep(1.0)

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
                    "heartbeat_path": heartbeat_path,
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
            os.fspath(heartbeat_path),
            os.fspath(harvest_path),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    descendant_pids: list[int] = []
    try:
        _wait_until(
            lambda: (
                harvest_path.exists() and len(_pid_lines(heartbeat_path)) == 1
            )
        )
        descendant_pids = _pid_lines(heartbeat_path)
        os.kill(scheduler.pid, signal.SIGKILL)
        scheduler.wait(timeout=3.0)
        for pid in descendant_pids:
            _assert_process_gone(pid)
    finally:
        if scheduler.poll() is None:
            scheduler.kill()
            scheduler.wait()
        for pid in descendant_pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_normal_completion_stops_left_descendant_before_acceptance(
    tmp_path: Path,
) -> None:
    heartbeat_path = tmp_path / "heartbeat"
    outcome = run_call_pool(
        [
            CallSpec(
                key="worker",
                job=_job(
                    "spawn_descendant_and_return",
                    {
                        "heartbeat_path": os.fspath(heartbeat_path),
                        "value": "complete",
                    },
                ),
                decode=_identity,
                deadline_seconds=5.0,
            )
        ],
        is_rate_limited=_never_rate_limited,
    )
    assert outcome.results[0].status is FanoutStatus.COMPLETED
    assert outcome.results[0].value == "complete"
    pids = _pid_lines(heartbeat_path)
    assert len(pids) == 1
    for pid in pids:
        _assert_process_gone(pid)
    returned_content = heartbeat_path.read_bytes()
    time.sleep(0.1)
    assert heartbeat_path.read_bytes() == returned_content


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


def test_repeated_completed_process_trees_are_clean(tmp_path: Path) -> None:
    all_pids: list[int] = []
    for index in range(5):
        heartbeat_path = tmp_path / f"heartbeat-{index}"
        outcome = run_call_pool(
            [
                CallSpec(
                    key=str(index),
                    job=_job(
                        "spawn_descendant_and_return",
                        {
                            "heartbeat_path": os.fspath(heartbeat_path),
                            "value": index,
                        },
                    ),
                    decode=_identity,
                    deadline_seconds=5.0,
                )
            ],
            is_rate_limited=_never_rate_limited,
        )
        assert outcome.results[0].status is FanoutStatus.COMPLETED
        all_pids.extend(_pid_lines(heartbeat_path))
    assert len(all_pids) == 5
    for pid in all_pids:
        _assert_process_gone(pid)


def test_lazy_dispatch_never_starts_more_than_current_capacity(
    tmp_path: Path,
) -> None:
    event_path = tmp_path / "events"
    release_path = tmp_path / "release"
    specs = [
        CallSpec(
            key=str(index),
            job=_job(
                "wait_for_release",
                {
                    "key": str(index),
                    "event_path": os.fspath(event_path),
                    "release_path": os.fspath(release_path),
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
        _wait_until(
            lambda: (
                sum(line.startswith("start|") for line in _lines(event_path))
                == 2
            )
        )
        time.sleep(0.05)
        assert (
            sum(line.startswith("start|") for line in _lines(event_path)) == 2
        )
    finally:
        release_path.touch()
        scheduler.join(timeout=5.0)
    assert not scheduler.is_alive()
    assert not failure
    assert len(outcome) == 1


@pytest.mark.parametrize("failure_stage", ["decode", "predicate", "commit"])
def test_accepted_worker_is_not_cancelled_when_parent_callback_fails(
    tmp_path: Path,
    failure_stage: str,
) -> None:
    event_path = tmp_path / "events"
    heartbeat_path = tmp_path / "heartbeat"
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

    with pytest.raises(RuntimeError, match=f"{failure_stage} failed"):
        run_call_pool(
            [
                _delayed_spec(
                    "accepted",
                    event_path=event_path,
                    delay=0.01,
                    decode=decode,
                    commit=commit,
                    cancellation_barrier=lambda: barriers.append("accepted"),
                ),
                _heartbeat_spec(
                    "sibling",
                    heartbeat_path,
                    deadline=5.0,
                    cancellation_barrier=lambda: barriers.append("sibling"),
                ),
            ],
            concurrency=2,
            is_rate_limited=predicate,
        )
    assert barriers == ["sibling"]
    for pid in _pid_lines(heartbeat_path):
        _assert_process_gone(pid)


def test_completion_order_drives_commits_but_results_preserve_input_order(
    tmp_path: Path,
) -> None:
    event_path = tmp_path / "events"
    commits: list[JsonValue] = []
    specs = [
        _delayed_spec(
            "slow",
            event_path=event_path,
            delay=0.15,
            commit=commits.append,
        ),
        _delayed_spec(
            "fast",
            event_path=event_path,
            delay=0.01,
            commit=commits.append,
        ),
        _delayed_spec(
            "middle",
            event_path=event_path,
            delay=0.07,
            commit=commits.append,
        ),
    ]
    outcome = run_call_pool(
        specs,
        concurrency=3,
        is_rate_limited=_never_rate_limited,
    )
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


def test_rate_feedback_reduces_capacity_before_filling_it(
    tmp_path: Path,
) -> None:
    event_path = tmp_path / "events"
    specs = [
        _delayed_spec(
            "slow",
            event_path=event_path,
            delay=0.25,
            wait_for_started=4,
        ),
        _delayed_spec(
            "limited",
            event_path=event_path,
            delay=0.01,
            wait_for_started=4,
        ),
        _delayed_spec(
            "middle-1",
            event_path=event_path,
            delay=0.08,
            wait_for_started=4,
        ),
        _delayed_spec(
            "middle-2",
            event_path=event_path,
            delay=0.15,
            wait_for_started=4,
        ),
        _delayed_spec("queued", event_path=event_path, delay=0.01),
    ]
    outcome = run_call_pool(
        specs,
        concurrency=4,
        is_rate_limited=lambda value: value == "limited",
    )
    events = _lines(event_path)
    queued_start = next(
        index
        for index, event in enumerate(events)
        if event.startswith("start|queued|")
    )
    second_middle_finish = next(
        index
        for index, event in enumerate(events)
        if event.startswith("finish|middle-2|")
    )
    assert queued_start > second_middle_finish
    assert outcome.concurrency_halved
    assert outcome.effective_concurrency == 2


def test_unit_deadline_starts_when_each_child_starts(
    tmp_path: Path,
) -> None:
    event_path = tmp_path / "events"
    outcome = run_call_pool(
        [
            _delayed_spec(
                "first",
                event_path=event_path,
                delay=0.4,
                deadline=1.5,
            ),
            _delayed_spec(
                "second",
                event_path=event_path,
                delay=0.03,
                deadline=0.35,
            ),
        ],
        concurrency=1,
        is_rate_limited=_never_rate_limited,
    )
    assert [result.status for result in outcome.results] == [
        FanoutStatus.COMPLETED,
        FanoutStatus.COMPLETED,
    ]


def test_unit_timeout_kills_process_and_prevents_late_commit(
    tmp_path: Path,
) -> None:
    heartbeat_path = tmp_path / "heartbeat"
    commits: list[JsonValue] = []
    barriers: list[str] = []
    started_at = time.monotonic()
    outcome = run_call_pool(
        [
            _heartbeat_spec(
                "hung",
                heartbeat_path,
                deadline=0.3,
                commit=commits.append,
                cancellation_barrier=lambda: barriers.append("terminal"),
            )
        ],
        concurrency=1,
        is_rate_limited=_never_rate_limited,
    )
    elapsed = time.monotonic() - started_at
    assert elapsed < 1.0
    assert outcome.results[0].status is FanoutStatus.UNIT_TIMEOUT
    assert outcome.guard_timeouts == 1
    assert commits == []
    assert barriers == ["terminal"]
    pids = _pid_lines(heartbeat_path)
    assert len(pids) == 2
    for pid in pids:
        _assert_process_gone(pid)
    returned_content = heartbeat_path.read_bytes()
    time.sleep(0.1)
    assert heartbeat_path.read_bytes() == returned_content


def test_operation_deadline_kills_active_and_never_dispatches_queue(
    tmp_path: Path,
) -> None:
    heartbeat_path = tmp_path / "heartbeat"
    queued_events = tmp_path / "queued-events"
    outcome = run_call_pool(
        [
            _heartbeat_spec("active-1", heartbeat_path, deadline=5.0),
            _heartbeat_spec("active-2", heartbeat_path, deadline=5.0),
            _delayed_spec(
                "queued-1",
                event_path=queued_events,
                delay=0.0,
            ),
            _delayed_spec(
                "queued-2",
                event_path=queued_events,
                delay=0.0,
            ),
        ],
        concurrency=2,
        max_wall_seconds=0.75,
        is_rate_limited=_never_rate_limited,
    )
    assert [result.status for result in outcome.results] == [
        FanoutStatus.OPERATION_DEADLINE,
        FanoutStatus.OPERATION_DEADLINE,
        FanoutStatus.NOT_DISPATCHED,
        FanoutStatus.NOT_DISPATCHED,
    ]
    assert outcome.deadline_reached
    assert outcome.not_dispatched == ["queued-1", "queued-2"]
    assert not queued_events.exists()
    returned_content = heartbeat_path.read_bytes()
    for pid in _pid_lines(heartbeat_path):
        _assert_process_gone(pid)
    time.sleep(0.1)
    assert heartbeat_path.read_bytes() == returned_content


def test_wall_watcher_stops_sibling_while_decode_runs_past_deadline(
    tmp_path: Path,
) -> None:
    event_path = tmp_path / "events"
    heartbeat_path = tmp_path / "heartbeat"
    sibling_stopped_during_decode: list[bool] = []

    def slow_decode(value: JsonValue) -> JsonValue:
        time.sleep(0.65)
        before = heartbeat_path.read_bytes()
        time.sleep(0.1)
        sibling_stopped_during_decode.append(
            heartbeat_path.read_bytes() == before
        )
        return value

    outcome = run_call_pool(
        [
            _delayed_spec(
                "completed",
                event_path=event_path,
                delay=0.0,
                decode=slow_decode,
            ),
            _heartbeat_spec("sibling", heartbeat_path, deadline=5.0),
        ],
        concurrency=2,
        max_wall_seconds=0.5,
        is_rate_limited=_never_rate_limited,
    )
    assert sibling_stopped_during_decode == [True]
    assert [result.status for result in outcome.results] == [
        FanoutStatus.OPERATION_DEADLINE,
        FanoutStatus.OPERATION_DEADLINE,
    ]
    for pid in _pid_lines(heartbeat_path):
        _assert_process_gone(pid)


def test_slow_spawn_cannot_release_worker_after_wall(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event_path = tmp_path / "events"
    real_popen = subprocess.Popen

    def slow_popen(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
        process = cast(
            "subprocess.Popen[bytes]",
            real_popen(*args, **kwargs),  # ty: ignore[no-matching-overload]
        )
        time.sleep(0.1)
        return process

    monkeypatch.setattr(fanout_module.subprocess, "Popen", slow_popen)
    outcome = run_call_pool(
        [_delayed_spec("queued", event_path=event_path, delay=0.0)],
        concurrency=1,
        max_wall_seconds=0.03,
        is_rate_limited=_never_rate_limited,
    )
    assert outcome.results[0].status is FanoutStatus.NOT_DISPATCHED
    assert not event_path.exists()


def test_slow_serialization_stops_before_spawn_after_wall(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event_path = tmp_path / "events"
    real_write_job = fanout_module._write_job

    def slow_write_job(path: Path, job: ProcessJob) -> None:
        real_write_job(path, job)
        time.sleep(0.06)

    monkeypatch.setattr(fanout_module, "_write_job", slow_write_job)
    outcome = run_call_pool(
        [_delayed_spec("queued", event_path=event_path, delay=0.0)],
        concurrency=1,
        max_wall_seconds=0.02,
        is_rate_limited=_never_rate_limited,
    )
    assert outcome.results[0].status is FanoutStatus.NOT_DISPATCHED
    assert not event_path.exists()


def test_slow_commit_may_finish_but_wall_stops_later_dispatch(
    tmp_path: Path,
) -> None:
    event_path = tmp_path / "events"
    queued_path = tmp_path / "queued"

    def slow_commit(_value: JsonValue) -> None:
        time.sleep(0.55)

    outcome = run_call_pool(
        [
            _delayed_spec(
                "committed",
                event_path=event_path,
                delay=0.0,
                commit=slow_commit,
            ),
            _delayed_spec("queued", event_path=queued_path, delay=0.0),
        ],
        concurrency=1,
        max_wall_seconds=0.4,
        is_rate_limited=_never_rate_limited,
    )
    assert [result.status for result in outcome.results] == [
        FanoutStatus.COMPLETED,
        FanoutStatus.NOT_DISPATCHED,
    ]
    assert outcome.deadline_reached
    assert not queued_path.exists()


def test_wall_crossing_during_cancellation_never_dispatches_queue(
    tmp_path: Path,
) -> None:
    heartbeat_path = tmp_path / "heartbeat"
    queued_path = tmp_path / "queued"

    def slow_barrier() -> None:
        time.sleep(0.35)

    outcome = run_call_pool(
        [
            _heartbeat_spec(
                "timeout",
                heartbeat_path,
                deadline=0.2,
                cancellation_barrier=slow_barrier,
            ),
            _delayed_spec("queued", event_path=queued_path, delay=0.0),
        ],
        concurrency=1,
        max_wall_seconds=0.45,
        is_rate_limited=_never_rate_limited,
    )
    assert [result.status for result in outcome.results] == [
        FanoutStatus.UNIT_TIMEOUT,
        FanoutStatus.NOT_DISPATCHED,
    ]
    assert outcome.deadline_reached
    assert not queued_path.exists()


def test_unexpected_child_failure_cancels_siblings_before_raise(
    tmp_path: Path,
) -> None:
    heartbeat_path = tmp_path / "heartbeat"
    event_path = tmp_path / "events"
    with pytest.raises(
        ProcessWorkerError,
        match="requested failure for failed",
    ):
        run_call_pool(
            [
                _heartbeat_spec("sibling", heartbeat_path, deadline=5.0),
                _delayed_spec(
                    "failed",
                    event_path=event_path,
                    delay=0.05,
                    fail=True,
                ),
            ],
            concurrency=2,
            is_rate_limited=_never_rate_limited,
        )
    pids = _pid_lines(heartbeat_path)
    assert len(pids) == 2
    for pid in pids:
        _assert_process_gone(pid)
    returned_content = heartbeat_path.read_bytes()
    time.sleep(0.1)
    assert heartbeat_path.read_bytes() == returned_content


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
    tmp_path: Path,
    duration: object,
) -> None:
    event_path = tmp_path / "events"
    spec = _delayed_spec(
        "invalid",
        event_path=event_path,
        delay=0.0,
        deadline=cast(float, duration),
    )
    with pytest.raises(ValueError, match="finite nonnegative real"):
        run_call_pool(
            [spec],
            is_rate_limited=_never_rate_limited,
        )
    assert not event_path.exists()


@pytest.mark.parametrize("nested", [False, True])
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
