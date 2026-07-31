"""Lazy process-isolated fanout with parent-owned result commits."""

from __future__ import annotations

import math
import os
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from enum import UNIQUE, StrEnum, verify
from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)

__all__ = [
    "DEFAULT_CONCURRENCY",
    "CallSpec",
    "FanoutResult",
    "FanoutStatus",
    "PoolOutcome",
    "ProcessCancellationError",
    "ProcessJob",
    "ProcessWorkerError",
    "run_call_pool",
]

DEFAULT_CONCURRENCY = 5
_CANCELLATION_GRACE_SECONDS = 0.1
_DEADLINE_WAIT_CHUNK_SECONDS = 86_400.0
_GUARDIAN_EXIT_TIMEOUT_SECONDS = 1.0
_POLL_INTERVAL_SECONDS = 0.005
_PROCESS_DISPATCH_SCHEMA = "whetstone.execution.process_dispatch/v1"
_PROCESS_JOB_SCHEMA = "whetstone.execution.process_job/v1"
_PROCESS_RESULT_SCHEMA = "whetstone.execution.process_result/v1"
_START_TOKEN = b"\x01"
_parent_control_fds: set[int] = set()
_fork_child_control_fds: tuple[int, ...] = ()
_parent_control_lock = threading.Lock()


def _before_fork() -> None:
    global _fork_child_control_fds
    _parent_control_lock.acquire()
    _fork_child_control_fds = tuple(_parent_control_fds)


def _after_fork_in_parent() -> None:
    global _fork_child_control_fds
    _fork_child_control_fds = ()
    _parent_control_lock.release()


def _after_fork_in_child() -> None:
    """Drop scheduler authority without acquiring an inherited lock."""
    global _fork_child_control_fds, _parent_control_fds, _parent_control_lock
    for descriptor in _fork_child_control_fds:
        try:
            os.close(descriptor)
        except OSError:
            pass
    _fork_child_control_fds = ()
    _parent_control_fds = set()
    _parent_control_lock = threading.Lock()


os.register_at_fork(
    before=_before_fork,
    after_in_parent=_after_fork_in_parent,
    after_in_child=_after_fork_in_child,
)


def _open_parent_control_pipe() -> tuple[int, int]:
    with _parent_control_lock:
        reader, writer = os.pipe()
        try:
            os.set_inheritable(reader, False)
            os.set_inheritable(writer, False)
        except BaseException:
            os.close(reader)
            os.close(writer)
            raise
        else:
            _parent_control_fds.update((reader, writer))
    return reader, writer


def _close_parent_control_fd(descriptor: int) -> None:
    with _parent_control_lock:
        if descriptor not in _parent_control_fds:
            return
        _parent_control_fds.remove(descriptor)
        os.close(descriptor)


def _wait_for_guardian_exit(
    descriptor: int,
) -> ProcessCancellationError | None:
    deadline = time.monotonic() + _GUARDIAN_EXIT_TIMEOUT_SECONDS
    with selectors.DefaultSelector() as selector:
        selector.register(descriptor, selectors.EVENT_READ)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not selector.select(remaining):
                return ProcessCancellationError(
                    "process guardian did not exit after its group was stopped"
                )
            try:
                payload = os.read(descriptor, 1)
            except InterruptedError:
                continue
            if payload:
                return ProcessCancellationError(
                    "process guardian could not stop its process group"
                )
            return None


def _premature_guardian_error(
    descriptor: int,
) -> ProcessCancellationError | None:
    with selectors.DefaultSelector() as selector:
        selector.register(descriptor, selectors.EVENT_READ)
        if not selector.select(0):
            return None
    try:
        payload = os.read(descriptor, 1)
    except InterruptedError:
        return None
    if payload:
        return ProcessCancellationError(
            "process guardian failed while scheduler authority remained open"
        )
    return ProcessCancellationError(
        "process guardian exited while scheduler authority remained open"
    )


def _require_finite_json(value: object, *, path: str) -> object:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite JSON number")
        return value
    if isinstance(value, dict):
        for key, item in value.items():
            _require_finite_json(item, path=f"{path}.{key}")
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _require_finite_json(item, path=f"{path}[{index}]")
    return value


class ProcessJob(BaseModel):
    """Validated subprocess input naming one importable top-level callable."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_name: StrictStr = _PROCESS_JOB_SCHEMA
    entrypoint: StrictStr
    payload: JsonValue

    @field_validator("schema_name")
    @classmethod
    def _schema_is_current(cls, value: str) -> str:
        if value != _PROCESS_JOB_SCHEMA:
            raise ValueError("unsupported process job schema")
        return value

    @field_validator("entrypoint")
    @classmethod
    def _entrypoint_is_top_level(cls, value: str) -> str:
        module_name, separator, function_name = value.partition(":")
        if (
            not separator
            or not module_name
            or not function_name
            or ":" in function_name
            or "." in function_name
            or any(not part.isidentifier() for part in module_name.split("."))
            or not function_name.isidentifier()
        ):
            raise ValueError(
                "entrypoint must be 'importable.module:top_level_callable'"
            )
        return value

    @field_validator("payload", mode="before")
    @classmethod
    def _payload_is_finite_json(cls, value: object) -> object:
        return _require_finite_json(value, path="payload")


class _ProcessDispatchMarker(BaseModel):
    """Validated worker dispatch evidence crossing the subprocess boundary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_name: StrictStr = _PROCESS_DISPATCH_SCHEMA
    started_at_monotonic: float = Field(
        ge=0,
        allow_inf_nan=False,
        strict=True,
    )
    guardian_pid: int = Field(gt=0, strict=True)

    @field_validator("schema_name")
    @classmethod
    def _schema_is_current(cls, value: str) -> str:
        if value != _PROCESS_DISPATCH_SCHEMA:
            raise ValueError("unsupported process dispatch schema")
        return value


@verify(UNIQUE)
class _ProcessResultStatus(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    NOT_STARTED = "not-started"


class _ProcessWorkerFailure(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    error_type: StrictStr
    message: StrictStr


class _ProcessWorkerResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_name: StrictStr = _PROCESS_RESULT_SCHEMA
    status: _ProcessResultStatus
    completed_at_monotonic: float = Field(ge=0, allow_inf_nan=False)
    value: JsonValue = None
    error: _ProcessWorkerFailure | None = None

    @field_validator("value", mode="before")
    @classmethod
    def _value_is_finite_json(cls, value: object) -> object:
        return _require_finite_json(value, path="value")

    @model_validator(mode="after")
    def _valid_terminal_shape(self) -> _ProcessWorkerResult:
        if self.schema_name != _PROCESS_RESULT_SCHEMA:
            raise ValueError("unsupported process result schema")
        if self.status is _ProcessResultStatus.SUCCESS:
            if self.error is not None:
                raise ValueError(
                    "successful process result cannot carry an error"
                )
        elif self.status is _ProcessResultStatus.FAILURE:
            if self.error is None:
                raise ValueError("failed process result must carry an error")
            if self.value is not None:
                raise ValueError("failed process result cannot carry a value")
        elif self.error is not None or self.value is not None:
            raise ValueError(
                "not-started process result cannot carry a value or error"
            )
        return self


@verify(UNIQUE)
class FanoutStatus(StrEnum):
    """Terminal state of one fanout unit."""

    COMPLETED = "completed"
    UNIT_TIMEOUT = "unit-timeout"
    OPERATION_DEADLINE = "operation-deadline"
    NOT_DISPATCHED = "not-dispatched"


class ProcessWorkerError(RuntimeError):
    """A child failed outside the fanout unit's expected result contract."""


class ProcessCancellationError(RuntimeError):
    """A cancellation barrier could not confirm terminal external work."""


@dataclass(frozen=True, slots=True)
class CallSpec[K: Hashable, R]:
    """One process job and its parent-side acceptance callbacks."""

    key: K
    job: ProcessJob
    decode: Callable[[JsonValue], R]
    deadline_seconds: float
    commit: Callable[[R], None] | None = None
    cancellation_barrier: Callable[[], None] | None = None


@dataclass(frozen=True, slots=True)
class FanoutResult[K: Hashable, R]:
    """The accepted value or terminal non-completion state for one unit."""

    key: K
    status: FanoutStatus
    value: R | None = None

    @property
    def completed(self) -> bool:
        return self.status is FanoutStatus.COMPLETED


@dataclass(frozen=True, slots=True)
class PoolOutcome[K: Hashable, R]:
    """Terminal pool report, preserving caller input order."""

    results: tuple[FanoutResult[K, R], ...]
    effective_concurrency: int
    concurrency_halved: bool
    deadline_reached: bool
    guard_timeouts: int

    @property
    def not_dispatched(self) -> list[K]:
        return [
            result.key
            for result in self.results
            if result.status is FanoutStatus.NOT_DISPATCHED
        ]


@dataclass(slots=True)
class _ActiveProcess[K: Hashable, R]:
    index: int
    spec: CallSpec[K, R]
    process: subprocess.Popen[bytes]
    directory: Path
    result_path: Path
    stderr_path: Path
    dispatch_path: Path
    lifetime_writer: int | None
    guardian_reader: int | None
    start_writer: int | None
    started_at: float | None = None
    guardian_pid: int | None = None
    guardian_error: ProcessCancellationError | None = None
    cleanup_allowed: bool = False

    @property
    def process_group_id(self) -> int:
        return self.process.pid

    def release(self) -> None:
        descriptor = self.start_writer
        if descriptor is None:
            raise AssertionError("worker start gate was already released")
        self.start_writer = None
        try:
            written = os.write(descriptor, _START_TOKEN)
            if written != len(_START_TOKEN):
                raise OSError("short write while releasing worker start gate")
        finally:
            _close_parent_control_fd(descriptor)

    def refresh_dispatch_marker(self, *, required: bool) -> float | None:
        if self.started_at is not None:
            return self.started_at
        try:
            payload = self.dispatch_path.read_bytes()
        except FileNotFoundError:
            if required:
                raise ProcessWorkerError(
                    f"worker for key {self.spec.key!r} exited without a "
                    "dispatch marker"
                ) from None
            return None
        except OSError as error:
            raise ProcessWorkerError(
                f"could not read dispatch marker for key {self.spec.key!r}"
            ) from error
        try:
            marker = _ProcessDispatchMarker.model_validate_json(payload)
        except ValidationError as error:
            raise ProcessWorkerError(
                f"worker for key {self.spec.key!r} wrote an invalid "
                "dispatch marker"
            ) from error
        self.started_at = marker.started_at_monotonic
        self.guardian_pid = marker.guardian_pid
        return marker.started_at_monotonic

    def finish_guardian(self) -> None:
        writer = self.lifetime_writer
        if writer is not None:
            self.lifetime_writer = None
            _close_parent_control_fd(writer)
        if self.guardian_error is not None:
            raise self.guardian_error
        reader = self.guardian_reader
        if reader is None:
            return
        error = _wait_for_guardian_exit(reader)
        if error is not None:
            self.guardian_error = error
            raise error
        if _wait_for_process_group_absence([self]):
            error = ProcessCancellationError(
                "process group remained alive after its guardian exited"
            )
            self.guardian_error = error
            raise error
        self.cleanup_allowed = True
        self.guardian_reader = None
        _close_parent_control_fd(reader)

    def require_guardian_alive(self) -> None:
        if self.guardian_error is None and self.guardian_reader is not None:
            self.guardian_error = _premature_guardian_error(
                self.guardian_reader
            )
        if self.guardian_error is not None:
            raise self.guardian_error

    def release_guardian_after_containment(self) -> None:
        if self.lifetime_writer is not None:
            _close_parent_control_fd(self.lifetime_writer)
            self.lifetime_writer = None
        if self.guardian_reader is not None:
            _close_parent_control_fd(self.guardian_reader)
            self.guardian_reader = None

    def cleanup(self) -> None:
        if not self.cleanup_allowed:
            raise AssertionError(
                "process state cannot be discarded before terminal proof"
            )
        if self.start_writer is not None:
            _close_parent_control_fd(self.start_writer)
            self.start_writer = None
        if (
            self.lifetime_writer is not None
            or self.guardian_reader is not None
        ):
            raise AssertionError(
                "guardian state cannot be discarded before terminal proof"
            )
        shutil.rmtree(self.directory, ignore_errors=True)


@dataclass(frozen=True, slots=True)
class _CompletedProcess[K: Hashable, R]:
    index: int
    spec: CallSpec[K, R]
    started_at: float | None
    envelope: _ProcessWorkerResult


class _DispatchDeadlineReached(Exception):
    pass


def _write_job(path: Path, job: ProcessJob) -> None:
    payload = job.model_dump_json().encode("utf-8")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written == 0:
                raise OSError("zero-byte write while serializing process job")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _spawn[K: Hashable, R](
    index: int,
    spec: CallSpec[K, R],
    *,
    operation_deadline: float | None,
) -> _ActiveProcess[K, R]:
    directory: Path | None = None
    lifetime_reader: int | None = None
    lifetime_writer: int | None = None
    guardian_reader: int | None = None
    guardian_writer: int | None = None
    start_reader: int | None = None
    start_writer: int | None = None
    try:
        directory = Path(tempfile.mkdtemp(prefix="whetstone-process-job-"))
        job_path = directory / "job.json"
        result_path = directory / "result.json"
        stderr_path = directory / "stderr.log"
        dispatch_path = directory / "dispatch.json"
        lifetime_reader, lifetime_writer = _open_parent_control_pipe()
        guardian_reader, guardian_writer = _open_parent_control_pipe()
        start_reader, start_writer = _open_parent_control_pipe()
        _write_job(job_path, spec.job)
        if (
            operation_deadline is not None
            and time.monotonic() >= operation_deadline
        ):
            raise _DispatchDeadlineReached
        stderr_descriptor = os.open(
            stderr_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "whetstone.execution.process_worker",
                    os.fspath(job_path),
                    os.fspath(result_path),
                    os.fspath(dispatch_path),
                    str(lifetime_reader),
                    str(guardian_writer),
                    str(start_reader),
                    (
                        "none"
                        if operation_deadline is None
                        else repr(operation_deadline)
                    ),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=stderr_descriptor,
                start_new_session=True,
                pass_fds=(lifetime_reader, guardian_writer, start_reader),
            )
        finally:
            os.close(stderr_descriptor)
    except BaseException:
        for descriptor in (
            lifetime_reader,
            lifetime_writer,
            guardian_reader,
            guardian_writer,
            start_reader,
            start_writer,
        ):
            if descriptor is not None:
                _close_parent_control_fd(descriptor)
        if directory is not None:
            shutil.rmtree(directory, ignore_errors=True)
        raise
    assert directory is not None
    assert lifetime_reader is not None
    assert lifetime_writer is not None
    assert guardian_reader is not None
    assert guardian_writer is not None
    assert start_reader is not None
    assert start_writer is not None
    _close_parent_control_fd(lifetime_reader)
    _close_parent_control_fd(guardian_writer)
    _close_parent_control_fd(start_reader)
    return _ActiveProcess(
        index=index,
        spec=spec,
        process=process,
        directory=directory,
        result_path=result_path,
        stderr_path=stderr_path,
        dispatch_path=dispatch_path,
        lifetime_writer=lifetime_writer,
        guardian_reader=guardian_reader,
        start_writer=start_writer,
    )


def _signal_process_group[K: Hashable, R](
    process: _ActiveProcess[K, R], sig: int
) -> None:
    try:
        os.killpg(process.process_group_id, sig)
    except ProcessLookupError:
        pass
    except PermissionError:
        # macOS reports EPERM for a group whose only member is a terminal,
        # not-yet-reaped leader. A live leader remains an error.
        if process.process.poll() is None:
            raise


def _process_group_exists[K: Hashable, R](
    process: _ActiveProcess[K, R],
) -> bool:
    try:
        os.killpg(process.process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return process.process.poll() is None
    return True


def _wait_for_process_group_absence[K: Hashable, R](
    processes: Sequence[_ActiveProcess[K, R]],
) -> list[_ActiveProcess[K, R]]:
    deadline = time.monotonic() + _GUARDIAN_EXIT_TIMEOUT_SECONDS
    unterminated = [
        process for process in processes if _process_group_exists(process)
    ]
    while unterminated and time.monotonic() < deadline:
        time.sleep(_POLL_INTERVAL_SECONDS)
        unterminated = [
            process
            for process in unterminated
            if _process_group_exists(process)
        ]
    return unterminated


def _terminate_processes[K: Hashable, R](
    processes: Sequence[_ActiveProcess[K, R]],
) -> None:
    if not processes:
        return
    signal_errors: dict[int, OSError | subprocess.TimeoutExpired] = {}
    for process in processes:
        try:
            _signal_process_group(process, signal.SIGTERM)
        except OSError as error:
            signal_errors.setdefault(process.index, error)

    grace_deadline = time.monotonic() + _CANCELLATION_GRACE_SECONDS
    while time.monotonic() < grace_deadline and any(
        _process_group_exists(process) for process in processes
    ):
        time.sleep(_POLL_INTERVAL_SECONDS)

    # The leader may honor TERM while a descendant does not. Signal the group
    # independently of the leader's return code.
    for process in processes:
        try:
            _signal_process_group(process, signal.SIGKILL)
        except OSError as error:
            signal_errors.setdefault(process.index, error)
    for process in processes:
        try:
            process.process.wait(timeout=_GUARDIAN_EXIT_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as error:
            signal_errors.setdefault(process.index, error)

    unterminated = _wait_for_process_group_absence(processes)
    unterminated_ids = {id(process) for process in unterminated}
    containment_error: ProcessCancellationError | None = None
    guardian_error: BaseException | None = None
    for process in processes:
        if id(process) in unterminated_ids:
            if containment_error is None:
                cause = signal_errors.get(process.index)
                if cause is None:
                    cause = OSError(
                        "local process group remained alive after SIGKILL"
                    )
                containment_error = ProcessCancellationError(
                    "could not confirm terminal local process group "
                    f"for key {process.spec.key!r}"
                )
                containment_error.__cause__ = cause
            continue
        process.cleanup_allowed = True
        try:
            process.finish_guardian()
        except BaseException as error:
            guardian_error = guardian_error or error
            process.release_guardian_after_containment()
    if containment_error is not None:
        raise containment_error
    if guardian_error is not None:
        raise guardian_error


def _started_processes[K: Hashable, R](
    processes: Sequence[_ActiveProcess[K, R]],
) -> list[_ActiveProcess[K, R]]:
    return [
        process
        for process in processes
        if process.refresh_dispatch_marker(required=False) is not None
    ]


def _run_cancellation_barriers[K: Hashable, R](
    processes: Sequence[_ActiveProcess[K, R]],
) -> None:
    barrier_error: ProcessCancellationError | None = None
    for process in processes:
        barrier = process.spec.cancellation_barrier
        if barrier is None:
            continue
        try:
            barrier()
        except Exception as error:
            process.cleanup_allowed = False
            barrier_error = barrier_error or ProcessCancellationError(
                "external cancellation did not reach a confirmed terminal "
                f"state for key {process.spec.key!r}"
            )
            barrier_error.__cause__ = error
    if barrier_error is not None:
        raise barrier_error


def _cancel_processes[K: Hashable, R](
    processes: Sequence[_ActiveProcess[K, R]],
) -> list[_ActiveProcess[K, R]]:
    termination_error: BaseException | None = None
    try:
        _terminate_processes(processes)
    except BaseException as error:
        termination_error = error
    contained = [process for process in processes if process.cleanup_allowed]
    try:
        started = _started_processes(contained)
    except BaseException:
        for process in contained:
            process.cleanup_allowed = False
        raise
    barrier_error: BaseException | None = None
    try:
        _run_cancellation_barriers(started)
    except BaseException as error:
        barrier_error = error
    if termination_error is not None:
        raise termination_error
    if barrier_error is not None:
        raise barrier_error
    return started


def _read_worker_result[K: Hashable, R](
    process: _ActiveProcess[K, R],
) -> _ProcessWorkerResult:
    try:
        raw = process.result_path.read_text(encoding="utf-8")
    except OSError as error:
        stderr = _read_stderr(process.stderr_path)
        raise ProcessWorkerError(
            f"worker for key {process.spec.key!r} exited with code "
            f"{process.process.returncode} without a result envelope"
            f"{stderr}"
        ) from error
    try:
        result = _ProcessWorkerResult.model_validate_json(raw)
    except ValidationError as error:
        raise ProcessWorkerError(
            f"worker for key {process.spec.key!r} wrote an invalid result "
            "envelope"
        ) from error
    expected_success_exit = result.status is not _ProcessResultStatus.FAILURE
    if expected_success_exit != (process.process.returncode == 0):
        raise ProcessWorkerError(
            f"worker for key {process.spec.key!r} published a result "
            "inconsistent with its exit code"
        )
    return result


def _read_stderr(path: Path) -> str:
    try:
        stderr = path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    return f": {stderr}" if stderr else ""


def _worker_failure[K: Hashable, R](
    process: _CompletedProcess[K, R],
) -> ProcessWorkerError:
    detail = process.envelope.error
    if detail is None:
        return ProcessWorkerError(
            f"worker for key {process.spec.key!r} failed without error detail"
        )
    return ProcessWorkerError(
        f"worker for key {process.spec.key!r} failed with "
        f"{detail.error_type}: {detail.message}"
    )


def _validate_duration(name: str, value: object) -> None:
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or type(value) not in (int, float)
    ):
        raise ValueError(f"{name} must be a finite nonnegative real number")
    try:
        seconds = float(value)
    except OverflowError:
        raise ValueError(
            f"{name} must be a finite nonnegative real number "
            "representable as seconds"
        ) from None
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"{name} must be a finite nonnegative real number")


def _validate_bounds[K: Hashable, R](
    specs: Sequence[CallSpec[K, R]],
    *,
    concurrency: int,
    max_wall_seconds: float | None,
) -> None:
    if type(concurrency) is not int or concurrency < 1:
        raise ValueError("concurrency must be a positive non-bool integer")
    if max_wall_seconds is not None:
        _validate_duration("max_wall_seconds", max_wall_seconds)
    for spec in specs:
        _validate_duration("deadline_seconds", spec.deadline_seconds)


def run_call_pool[K: Hashable, R](
    specs: Sequence[CallSpec[K, R]],
    *,
    concurrency: int = DEFAULT_CONCURRENCY,
    is_rate_limited: Callable[[R], bool],
    max_wall_seconds: float | None = None,
) -> PoolOutcome[K, R]:
    """Run JSON jobs lazily in fresh process groups.

    The absolute wall gates every child dispatch. A deadline watcher cancels
    active siblings even while trusted parent-side decode, predicate, commit,
    or cancellation callbacks run. Python cannot preempt those callbacks: one
    begun before the wall may delay return. A decode or predicate that crosses
    the wall prevents later acceptance steps; a commit that crosses it remains
    completed because its side effect has already occurred.
    """
    _validate_bounds(
        specs,
        concurrency=concurrency,
        max_wall_seconds=max_wall_seconds,
    )
    initial_concurrency = concurrency
    effective_concurrency = concurrency
    concurrency_halved = False
    deadline_reached = False
    guard_timeouts = 0
    operation_started_at = time.monotonic()
    operation_deadline = (
        operation_started_at + float(max_wall_seconds)
        if max_wall_seconds is not None
        else None
    )
    active: dict[int, _ActiveProcess[K, R]] = {}
    active_lock = threading.Lock()
    wall_triggered = threading.Event()
    wall_stop = threading.Event()
    wall_cancelled: list[tuple[_ActiveProcess[K, R], bool]] = []
    wall_errors: list[BaseException] = []
    results: list[FanoutResult[K, R] | None] = [None] * len(specs)
    next_index = 0

    def cleanup(processes: Sequence[_ActiveProcess[K, R]]) -> None:
        for process in processes:
            if process.cleanup_allowed:
                process.cleanup()

    def cancel(
        processes: Sequence[_ActiveProcess[K, R]],
    ) -> list[_ActiveProcess[K, R]]:
        return _cancel_processes(processes)

    def deadline_has_arrived() -> bool:
        return wall_triggered.is_set() or (
            operation_deadline is not None
            and time.monotonic() >= operation_deadline
        )

    def take_all_active() -> list[_ActiveProcess[K, R]]:
        with active_lock:
            processes = list(active.values())
            active.clear()
        return processes

    def deadline_watch() -> None:
        if operation_deadline is None:
            return
        while True:
            delay = operation_deadline - time.monotonic()
            if delay <= 0:
                break
            if wall_stop.wait(min(delay, _DEADLINE_WAIT_CHUNK_SECONDS)):
                return
        wall_triggered.set()
        claimed: list[_ActiveProcess[K, R]] = []
        with active_lock:
            for index, process in list(active.items()):
                if process.process.poll() is None or _process_group_exists(
                    process
                ):
                    claimed.append(process)
                    active.pop(index)
        started_indexes: set[int] = set()
        try:
            started_indexes = {process.index for process in cancel(claimed)}
        except BaseException as error:
            wall_errors.append(error)
            try:
                started_indexes = {
                    process.index for process in _started_processes(claimed)
                }
            except BaseException as marker_error:
                wall_errors.append(marker_error)
        finally:
            cleanup(claimed)
            wall_cancelled.extend(
                (
                    process,
                    process.index in started_indexes,
                )
                for process in claimed
            )

    deadline_thread = (
        threading.Thread(
            target=deadline_watch,
            name="whetstone-operation-deadline",
            daemon=False,
        )
        if operation_deadline is not None
        else None
    )
    if deadline_thread is not None:
        deadline_thread.start()

    def join_deadline_thread() -> None:
        if deadline_thread is not None:
            deadline_thread.join()

    def mark_remaining_not_dispatched() -> None:
        nonlocal next_index
        while next_index < len(specs):
            spec = specs[next_index]
            results[next_index] = FanoutResult(
                key=spec.key,
                status=FanoutStatus.NOT_DISPATCHED,
            )
            next_index += 1

    def collect_wall_cancellations() -> None:
        nonlocal deadline_reached
        if not wall_triggered.is_set():
            return
        deadline_reached = True
        join_deadline_thread()
        for process, started in wall_cancelled:
            results[process.index] = FanoutResult(
                key=process.spec.key,
                status=(
                    FanoutStatus.OPERATION_DEADLINE
                    if started
                    else FanoutStatus.NOT_DISPATCHED
                ),
            )
        mark_remaining_not_dispatched()
        if wall_errors:
            raise wall_errors[0]

    def dispatch_available() -> None:
        nonlocal deadline_reached, next_index
        while next_index < len(specs):
            with active_lock:
                if len(active) >= effective_concurrency:
                    return
            if deadline_has_arrived():
                deadline_reached = True
                collect_wall_cancellations()
                mark_remaining_not_dispatched()
                return
            index = next_index
            try:
                process = _spawn(
                    index,
                    specs[index],
                    operation_deadline=operation_deadline,
                )
            except _DispatchDeadlineReached:
                deadline_reached = True
                wall_triggered.set()
                next_index += 1
                results[index] = FanoutResult(
                    key=specs[index].key,
                    status=FanoutStatus.NOT_DISPATCHED,
                )
                mark_remaining_not_dispatched()
                return

            release_error: BaseException | None = None
            registered = False
            with active_lock:
                if not deadline_has_arrived():
                    active[index] = process
                    registered = True
                    try:
                        process.release()
                    except BaseException as error:
                        active.pop(index)
                        release_error = error
            next_index += 1
            if not registered:
                deadline_reached = True
                try:
                    _terminate_processes([process])
                except BaseException:
                    cleanup([process])
                    raise
                cleanup([process])
                results[index] = FanoutResult(
                    key=process.spec.key,
                    status=FanoutStatus.NOT_DISPATCHED,
                )
                mark_remaining_not_dispatched()
                return
            if release_error is not None:
                try:
                    _terminate_processes([process])
                except BaseException as containment_error:
                    cleanup([process])
                    raise containment_error from release_error
                cleanup([process])
                raise ProcessWorkerError(
                    f"could not release worker for key {process.spec.key!r}"
                ) from release_error

    def harvest_completed() -> list[_CompletedProcess[K, R]]:
        with active_lock:
            completed = [
                process
                for process in active.values()
                if process.process.poll() is not None
            ]
            for process in completed:
                active.pop(process.index)
        harvested: list[_CompletedProcess[K, R]] = []
        first_error: BaseException | None = None
        containment_failure: tuple[BaseException, BaseException] | None = None
        for process in completed:
            try:
                process.require_guardian_alive()
                envelope = _read_worker_result(process)
                process.finish_guardian()
                started_at = (
                    None
                    if envelope.status is _ProcessResultStatus.NOT_STARTED
                    else process.refresh_dispatch_marker(required=True)
                )
                harvested.append(
                    _CompletedProcess(
                        index=process.index,
                        spec=process.spec,
                        started_at=started_at,
                        envelope=envelope,
                    )
                )
            except BaseException as error:
                first_error = first_error or error
                try:
                    _terminate_processes([process])
                except BaseException as containment_error:
                    if not process.cleanup_allowed:
                        containment_failure = containment_failure or (
                            containment_error,
                            error,
                        )
            finally:
                cleanup([process])
        if containment_failure is not None:
            containment_error, original_error = containment_failure
            raise containment_error from original_error
        if first_error is not None:
            raise first_error
        harvested.sort(key=lambda item: item.envelope.completed_at_monotonic)
        return harvested

    try:
        dispatch_available()
        while True:
            collect_wall_cancellations()
            with active_lock:
                guardian_snapshot = list(active.values())
            for process in guardian_snapshot:
                process.require_guardian_alive()
            completed = harvest_completed()
            for process in completed:
                envelope = process.envelope
                if envelope.status is _ProcessResultStatus.NOT_STARTED:
                    deadline_reached = True
                    results[process.index] = FanoutResult(
                        key=process.spec.key,
                        status=FanoutStatus.NOT_DISPATCHED,
                    )
                    continue
                started_at = process.started_at
                if started_at is None:
                    raise AssertionError("started worker lacks a start time")
                unit_deadline = started_at + float(
                    process.spec.deadline_seconds
                )
                if (
                    operation_deadline is not None
                    and envelope.completed_at_monotonic > operation_deadline
                ):
                    deadline_reached = True
                    results[process.index] = FanoutResult(
                        key=process.spec.key,
                        status=FanoutStatus.OPERATION_DEADLINE,
                    )
                elif envelope.completed_at_monotonic > unit_deadline:
                    guard_timeouts += 1
                    results[process.index] = FanoutResult(
                        key=process.spec.key,
                        status=FanoutStatus.UNIT_TIMEOUT,
                    )
                elif envelope.status is _ProcessResultStatus.FAILURE:
                    raise _worker_failure(process)
                elif deadline_has_arrived():
                    deadline_reached = True
                    results[process.index] = FanoutResult(
                        key=process.spec.key,
                        status=FanoutStatus.OPERATION_DEADLINE,
                    )
                else:
                    value = process.spec.decode(envelope.value)
                    if deadline_has_arrived():
                        deadline_reached = True
                        results[process.index] = FanoutResult(
                            key=process.spec.key,
                            status=FanoutStatus.OPERATION_DEADLINE,
                        )
                        continue
                    limited = is_rate_limited(value)
                    if limited and not concurrency_halved:
                        concurrency_halved = True
                        effective_concurrency = max(
                            1, initial_concurrency // 2
                        )
                    if deadline_has_arrived():
                        deadline_reached = True
                        results[process.index] = FanoutResult(
                            key=process.spec.key,
                            status=FanoutStatus.OPERATION_DEADLINE,
                        )
                        continue
                    if process.spec.commit is not None:
                        process.spec.commit(value)
                    results[process.index] = FanoutResult(
                        key=process.spec.key,
                        status=FanoutStatus.COMPLETED,
                        value=value,
                    )

            collect_wall_cancellations()
            now = time.monotonic()
            expired: list[_ActiveProcess[K, R]] = []
            with active_lock:
                for index, process in list(active.items()):
                    started_at = process.refresh_dispatch_marker(
                        required=False
                    )
                    if started_at is not None and now >= started_at + float(
                        process.spec.deadline_seconds
                    ):
                        expired.append(process)
                        active.pop(index)
            if expired:
                try:
                    cancel(expired)
                finally:
                    cleanup(expired)
                guard_timeouts += len(expired)
                for process in expired:
                    results[process.index] = FanoutResult(
                        key=process.spec.key,
                        status=FanoutStatus.UNIT_TIMEOUT,
                    )

            collect_wall_cancellations()
            dispatch_available()
            collect_wall_cancellations()
            with active_lock:
                active_snapshot = list(active.values())
            if not active_snapshot and next_index >= len(specs):
                break

            next_event_at = now + _POLL_INTERVAL_SECONDS
            if operation_deadline is not None:
                next_event_at = min(next_event_at, operation_deadline)
            started_deadlines = [
                started_at + float(process.spec.deadline_seconds)
                for process in active_snapshot
                if (
                    started_at := process.refresh_dispatch_marker(
                        required=False
                    )
                )
                is not None
            ]
            if started_deadlines:
                next_event_at = min(next_event_at, min(started_deadlines))
            delay = next_event_at - time.monotonic()
            if delay > 0:
                time.sleep(delay)
    except BaseException as original_error:
        wall_stop.set()
        join_deadline_thread()
        remaining = take_all_active()
        containment_error: BaseException | None = None
        try:
            cancel(remaining)
        except BaseException as error:
            containment_error = error
        finally:
            cleanup(remaining)
        if containment_error is not None:
            raise containment_error from original_error
        raise
    finally:
        wall_stop.set()
        join_deadline_thread()

    if any(result is None for result in results):
        raise AssertionError("fanout scheduler left a unit without a result")
    return PoolOutcome(
        results=tuple(result for result in results if result is not None),
        effective_concurrency=effective_concurrency,
        concurrency_halved=concurrency_halved,
        deadline_reached=deadline_reached,
        guard_timeouts=guard_timeouts,
    )
