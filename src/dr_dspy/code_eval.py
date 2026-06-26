"""Reusable helpers for extracting and evaluating generated Python code."""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import platform
import threading
from multiprocessing.connection import Connection
from typing import Any

from pydantic import BaseModel, ConfigDict, StrictFloat

DEFAULT_MEM_LIMIT_BYTES = 1 << 30
DEFAULT_CPU_LIMIT_SECONDS = 17
DEFAULT_CAPTURE_LIMIT_BYTES = 4096

__all__ = [
    "DEFAULT_CAPTURE_LIMIT_BYTES",
    "DEFAULT_CPU_LIMIT_SECONDS",
    "DEFAULT_MEM_LIMIT_BYTES",
    "CodeExecutionResult",
    "extract_dspy_code",
    "run_python_check",
]


class CodeExecutionResult(BaseModel):
    """Result of running generated code against a check function."""

    model_config = ConfigDict(extra="forbid")

    score: StrictFloat
    error: str | None
    stdout: str = ""
    stderr: str = ""
    stdout_truncated: bool = False
    stderr_truncated: bool = False


class StreamCapture(BaseModel):
    """Bounded bytes captured from a redirected child process stream."""

    model_config = ConfigDict(extra="forbid")

    text: str
    truncated: bool


def _capture_fd(read_fd: int, *, limit_bytes: int) -> StreamCapture:
    chunks: list[bytes] = []
    total_size = 0
    captured_size = 0
    with os.fdopen(read_fd, "rb", closefd=True) as stream:
        while True:
            chunk = stream.read(8192)
            if not chunk:
                break
            total_size += len(chunk)
            remaining = limit_bytes - captured_size
            if remaining > 0:
                captured = chunk[:remaining]
                chunks.append(captured)
                captured_size += len(captured)
    return StreamCapture(
        text=b"".join(chunks).decode("utf-8", errors="replace"),
        truncated=total_size > limit_bytes,
    )


def _redirect_child_streams(stdout_fd: int, stderr_fd: int) -> None:
    os.dup2(stdout_fd, 1)
    os.dup2(stderr_fd, 2)
    import sys

    sys.stdout = os.fdopen(1, "w", closefd=False)
    sys.stderr = os.fdopen(2, "w", closefd=False)
    with contextlib.suppress(OSError):
        os.close(stdout_fd)
    with contextlib.suppress(OSError):
        os.close(stderr_fd)


def _worker(
    code: str,
    test: str,
    entry_point: str,
    mem_limit_bytes: int,
    cpu_limit_seconds: int,
    stdout_fd: int,
    stderr_fd: int,
    conn: Connection,
) -> None:
    """Child process body: apply rlimits, exec code+test, send status."""
    try:
        _redirect_child_streams(stdout_fd, stderr_fd)

        import resource
        import signal

        with contextlib.suppress(ValueError, OSError):
            resource.setrlimit(
                resource.RLIMIT_AS, (mem_limit_bytes, mem_limit_bytes)
            )
        with contextlib.suppress(ValueError, OSError):
            resource.setrlimit(
                resource.RLIMIT_CPU, (cpu_limit_seconds, cpu_limit_seconds)
            )

        def alarm_handler(_sig: int, _frame: Any) -> None:
            raise TimeoutError("CPU alarm fired")

        signal.signal(signal.SIGALRM, alarm_handler)
        signal.alarm(int(cpu_limit_seconds))

        namespace: dict[str, Any] = {}
        exec(code, namespace)
        exec(test, namespace)
        check_fn = namespace.get("check")
        candidate = namespace.get(entry_point)
        if check_fn is None or candidate is None:
            conn.send(("err", f"missing check or entry_point={entry_point!r}"))
            return
        check_fn(candidate)
        conn.send(("ok", None))
    except BaseException as e:
        conn.send(("err", f"{type(e).__name__}: {e}"))
    finally:
        with contextlib.suppress(Exception):
            conn.close()


def run_python_check(
    *,
    code: str,
    test: str,
    entry_point: str,
    timeout: float,
    mem_limit_bytes: int = DEFAULT_MEM_LIMIT_BYTES,
    cpu_limit_seconds: int = DEFAULT_CPU_LIMIT_SECONDS,
    capture_limit_bytes: int = DEFAULT_CAPTURE_LIMIT_BYTES,
) -> CodeExecutionResult:
    """Run code plus a HumanEval-style test in a sandboxed subprocess."""
    method = "fork" if platform.system() != "Windows" else "spawn"
    ctx: Any = mp.get_context(method)
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    stdout_read_fd, stdout_write_fd = os.pipe()
    stderr_read_fd, stderr_write_fd = os.pipe()
    os.set_inheritable(stdout_write_fd, True)
    os.set_inheritable(stderr_write_fd, True)
    stdout_capture: StreamCapture | None = None
    stderr_capture: StreamCapture | None = None

    def read_stdout() -> None:
        nonlocal stdout_capture
        stdout_capture = _capture_fd(
            stdout_read_fd, limit_bytes=capture_limit_bytes
        )

    def read_stderr() -> None:
        nonlocal stderr_capture
        stderr_capture = _capture_fd(
            stderr_read_fd, limit_bytes=capture_limit_bytes
        )

    proc = ctx.Process(
        target=_worker,
        args=(
            code,
            test,
            entry_point,
            mem_limit_bytes,
            cpu_limit_seconds,
            stdout_write_fd,
            stderr_write_fd,
            child_conn,
        ),
    )
    proc.start()
    child_conn.close()
    os.close(stdout_write_fd)
    os.close(stderr_write_fd)
    stdout_thread = threading.Thread(target=read_stdout, daemon=True)
    stderr_thread = threading.Thread(target=read_stderr, daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    score = 0.0
    error: str | None = None
    timed_out = False
    try:
        if parent_conn.poll(timeout):
            try:
                status, msg = parent_conn.recv()
            except EOFError:
                status, msg = "err", "EOF from worker"
            if status == "ok":
                score = 1.0
            else:
                error = str(msg)
        else:
            error = f"timeout after {timeout}s"
            timed_out = True
    finally:
        if timed_out and proc.is_alive():
            proc.terminate()
            proc.join(timeout=1.0)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=1.0)
        else:
            proc.join(timeout=1.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1.0)
        with contextlib.suppress(Exception):
            parent_conn.close()
        stdout_thread.join(timeout=1.0)
        stderr_thread.join(timeout=1.0)
    stdout_capture = stdout_capture or StreamCapture(text="", truncated=False)
    stderr_capture = stderr_capture or StreamCapture(text="", truncated=False)
    return CodeExecutionResult(
        score=score,
        error=error,
        stdout=stdout_capture.text,
        stderr=stderr_capture.text,
        stdout_truncated=stdout_capture.truncated,
        stderr_truncated=stderr_capture.truncated,
    )


def extract_dspy_code(pred: Any, *, field_name: str = "code") -> str:
    """Pull Python source out of a DSPy prediction field."""
    code_field = getattr(pred, field_name, None)
    if code_field is None:
        return ""
    inner = getattr(code_field, "code", None)
    if isinstance(inner, str):
        return inner
    if isinstance(code_field, str):
        return code_field
    try:
        as_str = str(code_field)
    except Exception:
        return ""
    if as_str.startswith("code="):
        try:
            return as_str.split("=", 1)[1].strip().strip("'\"")
        except Exception:
            return as_str
    return as_str
