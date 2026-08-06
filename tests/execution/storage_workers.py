from __future__ import annotations

import fcntl
import os
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from dr_providers import (
    GenerationControls,
    MessageRole,
    PromptMessage,
    ProviderCallRequest,
    Transcript,
    openrouter_chat_config,
)

import whetstone.execution._file_lock as file_lock_module
from tests.provider import support as s
from whetstone.execution._file_lock import FileLock, open_private_directory
from whetstone.execution.partials import (
    PartialCallRecord,
    PartialLog,
    _create_temporary,
    _encode_frame,
)
from whetstone.execution.prompt_cache import PromptResultCache, execute_call


class _CrashAfterPublicationCache(PromptResultCache):
    def _finalize_accounting_locked(
        self,
        key: Any,
        journal: Any,
    ) -> NoReturn:
        os._exit(86)


class _CrashAfterPendingCache(PromptResultCache):
    def _store_entry(self, **kwargs: Any) -> NoReturn:
        os._exit(87)


class _CrashAfterStatsWriteCache(PromptResultCache):
    def _write_stats(self, stats: Any) -> None:
        super()._write_stats(stats)
        if stats.inflight_publication_ids:
            os._exit(88)


class _CrashAfterAppliedRenameCache(PromptResultCache):
    def _cleanup_applied_accounting_locked(
        self,
        journal: Any,
    ) -> NoReturn:
        os._exit(89)


def cache_request() -> ProviderCallRequest:
    return ProviderCallRequest(
        config=openrouter_chat_config(
            model="x/y",
            controls=GenerationControls(temperature=0.0),
        ),
        transcript=Transcript(
            messages=(PromptMessage(role=MessageRole.USER, content="hello"),)
        ),
    )


@dataclass(slots=True)
class _FileTransport:
    invocation_path: Path
    text: str
    started: Any | None = None
    block: bool = False

    def __call__(self, request: ProviderCallRequest):
        body = f"{os.getpid()}:{self.text}\n".encode()
        fd = os.open(
            self.invocation_path,
            os.O_WRONLY | os.O_APPEND | os.O_CREAT,
            0o600,
        )
        try:
            os.write(fd, body)
            os.fsync(fd)
        finally:
            os.close(fd)
        if self.started is not None:
            self.started.set()
        if self.block:
            signal.pause()
            raise AssertionError("signal.pause returned without termination")
        return s.build_evidence(
            request=request,
            policy=s.build_transport_policy(),
            outcome=s.response_outcome(text=self.text),
        )


def execute_cache_worker(
    root: str,
    invocation_path: str,
    worker_id: int,
    barrier: Any | None,
    output: Any,
    *,
    block: bool = False,
    started: Any | None = None,
    lock_attempted: Any | None = None,
    lock_acquired: Any | None = None,
    crash_after_publication: bool = False,
    crash_after_pending: bool = False,
    crash_after_stats_write: bool = False,
    crash_after_applied_rename: bool = False,
    umask_value: int | None = None,
) -> None:
    if umask_value is not None:
        os.umask(umask_value)
    request = cache_request()
    policy = s.build_execution_policy(max_attempts=1)
    if barrier is not None:
        barrier.wait(timeout=30)
    if lock_attempted is not None and lock_acquired is not None:
        _observe_lock_boundary(lock_attempted, lock_acquired)
    if crash_after_publication:
        cache = _CrashAfterPublicationCache(root=Path(root))
    elif crash_after_pending:
        cache = _CrashAfterPendingCache(root=Path(root))
    elif crash_after_stats_write:
        cache = _CrashAfterStatsWriteCache(root=Path(root))
    elif crash_after_applied_rename:
        cache = _CrashAfterAppliedRenameCache(root=Path(root))
    else:
        cache = PromptResultCache(root=Path(root))
    execution = execute_call(
        request=request,
        policy=policy,
        transport=_FileTransport(
            invocation_path=Path(invocation_path),
            text=f"result-{worker_id}",
            started=started,
            block=block,
        ),
        logical_call_id=f"call-{worker_id}",
        repeat_index=0,
        drive_ordinal=0,
        cache=cache,
        phase="worker",
        unit=f"unit-{worker_id}",
        clock=s.FakeClock(),
        sleep=s.SleepRecorder(),
    )
    output.put(
        {
            "result": execution.result.model_dump(mode="json"),
            "cache_hit": execution.cache_hit,
            "provenance": (
                execution.provenance.model_dump(mode="json")
                if execution.provenance is not None
                else None
            ),
        }
    )


def recover_cache_worker(root: str, key: str, output: Any) -> None:
    cache = PromptResultCache(root=Path(root))
    entry = cache.get_result(key)
    counters = cache.counters()
    stats = cache._read_stats()
    output.put(
        {
            "entry_readable": entry is not None,
            "counters": counters,
            "inflight_publication_ids": list(stats.inflight_publication_ids),
            "pending_exists": cache._pending_accounting_path_for(key).exists(),
            "applied_exists": cache._applied_accounting_path_for(key).exists(),
        }
    )


def append_partial_worker(
    path: str,
    worker_id: int,
    payload_size: int,
    barrier: Any | None = None,
    *,
    exit_immediately: bool = False,
) -> None:
    if barrier is not None:
        barrier.wait(timeout=30)
    PartialLog(path=Path(path)).append(
        PartialCallRecord(
            phase="worker",
            instance_id=f"task-{worker_id}",
            unit=f"candidate-{worker_id}",
            repeat_id=worker_id,
            request_identity=f"{worker_id:064x}",
            redrive_pending=False,
            output_text=str(worker_id) * payload_size,
        )
    )
    if exit_immediately:
        os._exit(0)


def write_torn_partial_worker(
    path: str,
    started: Any,
) -> None:
    log = PartialLog(path=Path(path))
    record = PartialCallRecord(
        phase="worker",
        instance_id="torn-task",
        unit="torn-candidate",
        repeat_id=99,
        request_identity="c" * 64,
        redrive_pending=False,
        at="2026-07-31T12:00:00+00:00",
    )
    body = _encode_frame(record)
    with FileLock(log._lock_path):
        with open_private_directory(log.path) as directory:
            _temporary, fd = _create_temporary(
                directory,
                log._entry_path(record).name,
            )
            try:
                os.write(fd, body[: len(body) // 2])
                os.fsync(fd)
            finally:
                os.close(fd)
        started.set()
        signal.pause()
        raise AssertionError("signal.pause returned without termination")


def hold_partial_lock(
    path: str,
    entered: Any,
    release: Any,
) -> None:
    log = PartialLog(path=Path(path))
    with FileLock(log._lock_path):
        entered.set()
        if not release.wait(timeout=30):
            raise TimeoutError("partial lock holder was not released")


def run_partial_operation(
    path: str,
    operation: str,
    output: Any,
    attempted: Any,
    acquired: Any,
) -> None:
    _observe_lock_boundary(attempted, acquired)
    log = PartialLog(path=Path(path))
    if operation == "append":
        log.append(
            PartialCallRecord(
                phase="worker",
                instance_id="task-operation",
                unit="candidate-operation",
                repeat_id=20,
                request_identity="d" * 64,
                redrive_pending=False,
            )
        )
        output.put("appended")
    elif operation == "load":
        output.put(len(log.load()))
    elif operation == "delete":
        log.delete()
        output.put("deleted")
    else:  # pragma: no cover - test fixture misuse
        raise ValueError(f"unsupported partial operation: {operation}")


def _observe_lock_boundary(attempted: Any, acquired: Any) -> None:
    real_flock = file_lock_module.fcntl.flock
    observed = False

    def observed_flock(fd: int, operation: int) -> None:
        nonlocal observed
        acquiring = operation & (fcntl.LOCK_EX | fcntl.LOCK_SH)
        if acquiring and not observed:
            observed = True
            attempted.set()
            real_flock(fd, operation)
            acquired.set()
            return
        real_flock(fd, operation)

    setattr(file_lock_module.fcntl, "flock", observed_flock)  # noqa: B010
