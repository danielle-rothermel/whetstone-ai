"""Fanout guardian and containment pathway tests.

Linux CI (``depot-ubuntu-24.04``) is the authoritative environment for these
tests. Fork and signal behavior may differ on macOS; fork-specific cases are
skipped when ``multiprocessing`` fork is unavailable locally.
"""

from __future__ import annotations

import multiprocessing
import os
import selectors
import signal
import sys
import threading
from collections.abc import Callable

import pytest
from pydantic import JsonValue

import whetstone.execution.fanout as fanout_module
from tests.execution.fanout_guardian_support import (
    GuardianThreadRun,
    SchedulerProcess,
    release_signal_if_entered,
    run_pre_ready_guardian_starter,
)
from tests.execution.fanout_guardian_support import (
    assert_process_gone as _assert_process_gone,
)
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
    with SchedulerProcess("parent_death_pool") as run:
        run.signals.wait_entered(["worker-worker", "worker-descendant"])
        assert run.scheduler is not None
        os.kill(run.scheduler.pid, parent_signal)
        run.scheduler.wait(timeout=3.0)
        for pid in (
            run.signals.pid("worker-worker"),
            run.signals.pid("worker-descendant"),
        ):
            _assert_process_gone(pid)


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
    thread_run = GuardianThreadRun()

    def schedule() -> None:
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

    thread_run.start(schedule)
    assert spawned_event.wait(timeout=10)
    process = spawned[0]
    signals.wait_entered(["descendant", "release"])
    process.refresh_dispatch_marker(required=True)
    guardian_pid = process.guardian_pid
    assert guardian_pid is not None
    try:
        os.kill(guardian_pid, signal.SIGSTOP)
        signals.release("release")
        thread_run.join(timeout=3.0)
        assert not thread_run.is_alive
        assert len(thread_run.failures) == 1
        assert isinstance(thread_run.failures[0], ProcessCancellationError)
        assert "did not exit" in str(thread_run.failures[0])
        for pid in (
            process.process.pid,
            guardian_pid,
            signals.pid("descendant"),
        ):
            _assert_process_gone(pid)
        assert not process.directory.exists()
        assert fanout_module._parent_control_fds == set()
    finally:
        release_signal_if_entered(signals, "release")
        thread_run.kill_process_group(process.process_group_id)
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
    thread_run = GuardianThreadRun()

    def schedule() -> None:
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

    thread_run.start(schedule)
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
        thread_run.join(timeout=5.0)
        assert not thread_run.is_alive
        assert len(thread_run.failures) == 1
        failure = thread_run.failures[0]
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
        release_signal_if_entered(signals, "release")
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
        thread_run.join(timeout=3.0)
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
    with SchedulerProcess(
        "guardian_scheduler_die",
        capture_stderr=True,
    ) as run:
        run.signals.wait_entered(
            ["worker-worker", "worker-descendant", "guardian"]
        )
        worker_pid = run.signals.pid("worker-worker")
        guardian_pid = run.signals.pid("guardian")
        observed_pids = [
            worker_pid,
            guardian_pid,
            run.signals.pid("worker-descendant"),
        ]
        run.track_pids(observed_pids)
        assert run.scheduler is not None
        os.kill(guardian_pid, signal.SIGKILL)
        os.kill(run.scheduler.pid, signal.SIGKILL)
        run.scheduler.wait(timeout=3.0)
        for pid in observed_pids:
            _assert_process_gone(pid)


@pytest.mark.parametrize("guardian_behavior", ["exit", "hang"])
@pytest.mark.process_integration
@pytest.mark.process_guardian
def test_guardian_pre_ready_failure_hard_contains_group(
    guardian_behavior: str,
) -> None:
    run = run_pre_ready_guardian_starter(guardian_behavior)
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(run.ready_reader, selectors.EVENT_READ)
            assert selector.select(3.0), "fake guardian did not publish ready"
        run.child_pids = [int(os.read(run.ready_reader, 32).strip())]
        run.starter.wait(timeout=3.0)
        assert run.starter.returncode == -signal.SIGKILL
        for pid in run.child_pids:
            _assert_process_gone(pid)
    finally:
        run.cleanup()


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="fork is unavailable on this platform",
)
@pytest.mark.process_integration
@pytest.mark.process_guardian
def test_forked_scheduler_sibling_cannot_keep_worker_group_alive() -> None:
    sibling_pid: int | None = None
    worker_pids: list[int] = []
    with SchedulerProcess("forked_scheduler_sibling") as run:
        run.signals.wait_entered(
            ["scheduler-sibling", "worker-worker", "worker-descendant"]
        )
        sibling_pid = run.signals.pid("scheduler-sibling")
        worker_pids = [
            run.signals.pid("worker-worker"),
            run.signals.pid("worker-descendant"),
        ]
        run.track_pids([sibling_pid, *worker_pids])
        assert run.scheduler is not None
        os.kill(run.scheduler.pid, signal.SIGKILL)
        run.scheduler.wait(timeout=3.0)
        for pid in worker_pids:
            _assert_process_gone(pid)
        os.kill(sibling_pid, 0)


@pytest.mark.process_integration
@pytest.mark.process_guardian
def test_scheduler_death_after_worker_return_kills_left_descendant() -> None:
    with SchedulerProcess("block_harvest") as run:
        run.signals.wait_entered(["harvest", "descendant"])
        descendant_pids = [run.signals.pid("descendant")]
        run.track_pids(descendant_pids)
        assert run.scheduler is not None
        os.kill(run.scheduler.pid, signal.SIGKILL)
        run.scheduler.wait(timeout=3.0)
        for pid in descendant_pids:
            _assert_process_gone(pid)


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
