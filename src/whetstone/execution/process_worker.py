from __future__ import annotations

import importlib
import math
import os
import selectors
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType

from dr_serialize import decode_strict_json_bytes
from pydantic import JsonValue

from whetstone.execution.fanout import (
    ProcessJob,
    _ProcessDispatchMarker,
    _ProcessResultStatus,
    _ProcessWorkerFailure,
    _ProcessWorkerResult,
)

__all__: list[str] = []

_START_TOKEN = b"\x01"
_GUARDIAN_READY_TOKEN = b"\x01"


_GUARDIAN_READY_TIMEOUT_SECONDS = 10.0
_PRE_GATE_READY_TOKEN = b"\x01"


def _resolve_entrypoint(entrypoint: str) -> Callable[[JsonValue], JsonValue]:
    module_name, function_name = entrypoint.split(":", maxsplit=1)
    module: ModuleType = importlib.import_module(module_name)
    candidate = getattr(module, function_name)
    if not callable(candidate):
        raise TypeError(f"{entrypoint!r} does not resolve to a callable")
    if (
        getattr(candidate, "__module__", None) != module_name
        or getattr(candidate, "__name__", None) != function_name
    ):
        raise TypeError(
            f"{entrypoint!r} does not name a top-level callable defined in "
            "its module"
        )
    return candidate


def _write_exclusive(path: Path, payload: bytes, *, label: str) -> None:
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
                raise OSError(f"zero-byte write while serializing {label}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_result(path: Path, result: _ProcessWorkerResult) -> None:
    _write_exclusive(
        path,
        result.model_dump_json().encode("utf-8"),
        label="process result",
    )


def _write_dispatch_marker(
    path: Path,
    marker: _ProcessDispatchMarker,
) -> None:
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        _write_exclusive(
            temporary_path,
            marker.model_dump_json().encode("utf-8"),
            label="process dispatch marker",
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _load_job(path: Path) -> ProcessJob:
    raw = path.read_bytes()
    decode_strict_json_bytes(
        raw,
        max_bytes=len(raw),
        max_depth=len(raw),
    )
    return ProcessJob.model_validate_json(raw)


def _failure(error: Exception) -> _ProcessWorkerResult:
    return _ProcessWorkerResult(
        status=_ProcessResultStatus.FAILURE,
        completed_at_monotonic=time.monotonic(),
        error=_ProcessWorkerFailure(
            error_type=type(error).__qualname__,
            message=str(error),
        ),
    )


def _start_guardian(
    parent_reader: int,
    done_writer: int,
) -> subprocess.Popen[bytes]:
    ready_reader, ready_writer = os.pipe()
    try:
        guardian = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "whetstone.execution.process_guardian",
                str(parent_reader),
                str(done_writer),
                str(ready_writer),
                str(os.getpgrp()),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            pass_fds=(parent_reader, done_writer, ready_writer),
        )
    finally:
        os.close(parent_reader)
        os.close(done_writer)
        os.close(ready_writer)
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(ready_reader, selectors.EVENT_READ)
            ready_available = selector.select(_GUARDIAN_READY_TIMEOUT_SECONDS)
        ready = os.read(ready_reader, 1) if ready_available else b""
    finally:
        os.close(ready_reader)
    if ready != _GUARDIAN_READY_TOKEN:
        _kill_own_process_group()
    monitor = threading.Thread(
        target=_kill_group_if_guardian_exits,
        args=(guardian,),
        name="whetstone-process-guardian-monitor",
        daemon=True,
    )
    monitor.start()
    return guardian


def _kill_own_process_group() -> None:
    os.killpg(os.getpgrp(), signal.SIGKILL)
    raise AssertionError("SIGKILL unexpectedly returned")


def _kill_group_if_guardian_exits(
    guardian: subprocess.Popen[bytes],
) -> None:
    guardian.wait()
    _kill_own_process_group()


def _operation_deadline(value: str) -> float | None:
    if value == "none":
        return None
    deadline = float(value)
    if not math.isfinite(deadline) or deadline < 0:
        raise ValueError("operation deadline must be finite and nonnegative")
    return deadline


def _wait_for_start(start_reader: int) -> bool:
    try:
        return os.read(start_reader, 1) == _START_TOKEN
    finally:
        os.close(start_reader)


def _publish_pre_gate_ready(writer: int | None) -> None:
    if writer is None:
        return
    try:
        if os.write(writer, _PRE_GATE_READY_TOKEN) != len(
            _PRE_GATE_READY_TOKEN
        ):
            raise OSError("short write while publishing pre-gate readiness")
    finally:
        os.close(writer)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) not in {7, 8}:
        sys.stderr.write(
            "usage: python -m whetstone.execution.process_worker "
            "JOB_PATH RESULT_PATH DISPATCH_PATH PARENT_FD GUARDIAN_DONE_FD "
            "START_FD OPERATION_DEADLINE [PRE_GATE_READY_FD]\n"
        )
        return 2
    job_path = Path(arguments[0])
    result_path = Path(arguments[1])
    dispatch_path = Path(arguments[2])
    parent_reader = int(arguments[3])
    guardian_done_writer = int(arguments[4])
    start_reader = int(arguments[5])
    deadline = _operation_deadline(arguments[6])
    pre_gate_ready_writer = int(arguments[7]) if len(arguments) == 8 else None
    guardian = _start_guardian(parent_reader, guardian_done_writer)
    _publish_pre_gate_ready(pre_gate_ready_writer)

    if not _wait_for_start(start_reader) or (
        deadline is not None and time.monotonic() >= deadline
    ):
        _write_result(
            result_path,
            _ProcessWorkerResult(
                status=_ProcessResultStatus.NOT_STARTED,
                completed_at_monotonic=time.monotonic(),
            ),
        )
        return 0

    _write_dispatch_marker(
        dispatch_path,
        _ProcessDispatchMarker(
            started_at_monotonic=time.monotonic(),
            guardian_pid=guardian.pid,
        ),
    )
    try:
        job = _load_job(job_path)
        entrypoint = _resolve_entrypoint(job.entrypoint)
        value = entrypoint(job.payload)
        result = _ProcessWorkerResult(
            status=_ProcessResultStatus.SUCCESS,
            completed_at_monotonic=time.monotonic(),
            value=value,
        )
    except Exception as error:
        result = _failure(error)
    _write_result(result_path, result)
    del guardian
    return 0 if result.status is _ProcessResultStatus.SUCCESS else 1


if __name__ == "__main__":
    raise SystemExit(main())
