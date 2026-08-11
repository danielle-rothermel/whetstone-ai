"""Shared helpers for fanout guardian process integration tests."""

from __future__ import annotations

import errno
import os
import select
import selectors
import signal
import subprocess
import sys
import threading
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast

from tests.execution.fanout_guardian_schedulers import _FAKE_GUARDIAN
from tests.execution.process_signals import ProcessSignals

_WAIT_TIMEOUT_SECONDS = 3.0
_FAKE_GUARDIAN_PATH = os.fspath(_FAKE_GUARDIAN)


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


def assert_process_gone(pid: int) -> None:
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
                assert selector.select(_WAIT_TIMEOUT_SECONDS), (
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
        observed = queue.control([event], 1, _WAIT_TIMEOUT_SECONDS)
    finally:
        queue.close()
    assert observed, f"process {pid} survived scheduler return"
    assert observed[0].fflags & kqueue_api.KQ_NOTE_EXIT


def launch_scheduler(
    scenario: str,
    signals: ProcessSignals,
    *,
    capture_stderr: bool = False,
) -> subprocess.Popen[bytes]:
    command = [
        sys.executable,
        "-m",
        "tests.execution.fanout_guardian_schedulers",
        scenario,
        os.fspath(signals.path),
    ]
    return subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE if capture_stderr else subprocess.DEVNULL,
    )


def kill_processes(
    pids: Collection[int],
    *,
    use_process_group_for_first: bool = False,
) -> None:
    pid_list = list(pids)
    if use_process_group_for_first and pid_list:
        try:
            os.killpg(pid_list[0], signal.SIGKILL)
        except (PermissionError, ProcessLookupError):
            pass
    for pid in pid_list:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def terminate_scheduler(scheduler: subprocess.Popen[bytes] | None) -> None:
    if scheduler is None:
        return
    if scheduler.poll() is None:
        scheduler.kill()
        scheduler.wait(timeout=_WAIT_TIMEOUT_SECONDS)


def release_signal_if_entered(signals: ProcessSignals, key: str) -> None:
    if key not in signals.entered_keys:
        return
    try:
        signals.release(key)
    except (AssertionError, BrokenPipeError, EOFError, OSError):
        pass


class SchedulerProcess:
    def __init__(
        self,
        scenario: str,
        *,
        capture_stderr: bool = False,
    ) -> None:
        self.scenario = scenario
        self.capture_stderr = capture_stderr
        self.signals = ProcessSignals()
        self.scheduler: subprocess.Popen[bytes] | None = None
        self._extra_pids: list[int] = []

    def track_pids(self, pids: Collection[int]) -> None:
        self._extra_pids.extend(pids)

    def __enter__(self) -> SchedulerProcess:
        self.scheduler = launch_scheduler(
            self.scenario,
            self.signals,
            capture_stderr=self.capture_stderr,
        )
        return self

    def __exit__(self, *_exc: object) -> None:
        terminate_scheduler(self.scheduler)
        kill_processes(
            self._extra_pids,
            use_process_group_for_first=bool(self._extra_pids),
        )
        self.signals.close()


class GuardianThreadRun:
    def __init__(self) -> None:
        self.failures: list[BaseException] = []
        self._thread: threading.Thread | None = None

    def start(self, schedule: Callable[[], None]) -> None:
        def wrapped() -> None:
            try:
                schedule()
            except BaseException as error:
                self.failures.append(error)

        self._thread = threading.Thread(target=wrapped)
        self._thread.start()

    def join(self, timeout: float) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def kill_process_group(self, process_group_id: int) -> None:
        if not self.is_alive:
            return
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.join(timeout=_WAIT_TIMEOUT_SECONDS)


@dataclass
class PreReadyGuardianRun:
    starter: subprocess.Popen[bytes]
    ready_reader: int
    ready_writer: int
    child_pids: list[int] = field(default_factory=list)

    def close_ready_writer(self) -> None:
        if self.ready_writer >= 0:
            os.close(self.ready_writer)
            self.ready_writer = -1

    def cleanup(self) -> None:
        terminate_scheduler(self.starter)
        if self.starter.poll() is None:
            os.killpg(self.starter.pid, signal.SIGKILL)
            self.starter.wait(timeout=_WAIT_TIMEOUT_SECONDS)
        kill_processes(self.child_pids)
        self.close_ready_writer()
        os.close(self.ready_reader)


def run_pre_ready_guardian_starter(
    behavior: str,
    *,
    fake_guardian_path: Path | None = None,
) -> PreReadyGuardianRun:
    ready_reader, ready_writer = os.pipe()
    guardian_path = (
        os.fspath(fake_guardian_path)
        if fake_guardian_path is not None
        else _FAKE_GUARDIAN_PATH
    )
    starter = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tests.execution.fanout_guardian_schedulers",
            "pre_ready_starter",
            guardian_path,
            str(ready_writer),
            behavior,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        start_new_session=True,
        pass_fds=(ready_writer,),
    )
    os.close(ready_writer)
    return PreReadyGuardianRun(
        starter=starter,
        ready_reader=ready_reader,
        ready_writer=-1,
    )
