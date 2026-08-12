"""Importable fanout guardian scheduler scenarios."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import whetstone.execution.fanout as fanout
import whetstone.execution.process_worker as worker
from tests.execution.process_signals import publish_ready
from whetstone.execution.fanout import CallSpec, ProcessJob, run_call_pool

_WORKERS = "tests.execution.process_workers"
_FAKE_GUARDIAN = (
    Path(__file__).resolve().parent / "fixtures" / "fake_guardian.py"
)


def _identity(value: object) -> object:
    return value


def _never_rate_limited(_value: object) -> bool:
    return False


def _block_process_tree_job(
    signal_path: str, *, key: str = "worker"
) -> ProcessJob:
    return ProcessJob(
        entrypoint=f"{_WORKERS}:block_process_tree",
        payload={"signal_path": signal_path, "key": key},
    )


def _run_block_process_tree_pool(
    signal_path: str,
    *,
    key: str = "worker",
    spawn_hook: Callable[..., Any] | None = None,
    pool_setup: Callable[[], None] | None = None,
) -> None:
    if spawn_hook is not None:
        real_spawn = fanout._spawn

        def wrapped_spawn(*args: object, **kwargs: object) -> object:
            return spawn_hook(real_spawn, *args, **kwargs)

        fanout._spawn = wrapped_spawn  # ty: ignore[invalid-assignment]
    if pool_setup is not None:
        pool_setup()
    run_call_pool(
        [
            CallSpec(
                key=key,
                job=_block_process_tree_job(signal_path, key=key),
                decode=_identity,
                deadline_seconds=30.0,
            )
        ],
        concurrency=1,
        is_rate_limited=_never_rate_limited,
    )


def parent_death_pool(signal_path: str) -> None:
    _run_block_process_tree_pool(signal_path)


def guardian_scheduler_die(signal_path: str) -> None:
    real_refresh_dispatch_marker = (
        fanout._ActiveProcess.refresh_dispatch_marker
    )
    published_guardian = False

    def record_dispatch_marker(
        self: fanout._ActiveProcess[Any, Any],
        *,
        required: bool,
    ) -> float | None:
        nonlocal published_guardian
        started_at = real_refresh_dispatch_marker(self, required=required)
        if (
            started_at is not None
            and self.guardian_pid is not None
            and not published_guardian
        ):
            published_guardian = True
            publish_ready(signal_path, "guardian")
        return started_at

    def pool_setup() -> None:
        fanout._ActiveProcess.refresh_dispatch_marker = record_dispatch_marker

    _run_block_process_tree_pool(signal_path, pool_setup=pool_setup)


def forked_scheduler_sibling(signal_path: str) -> None:
    def fork_after_spawn(
        real_spawn: Callable[..., fanout._ActiveProcess[Any, Any]],
        index: int,
        spec: CallSpec[Any, Any],
        *,
        operation_deadline: float | None,
    ) -> fanout._ActiveProcess[Any, Any]:
        process = real_spawn(
            index,
            spec,
            operation_deadline=operation_deadline,
        )
        sibling_pid = os.fork()
        if sibling_pid == 0:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            publish_ready(signal_path, "scheduler-sibling")
            signal.pause()
        return process

    _run_block_process_tree_pool(signal_path, spawn_hook=fork_after_spawn)


def block_harvest(signal_path: str) -> None:
    real_read_worker_result = fanout._read_worker_result

    def block_harvest_hook(
        process: fanout._ActiveProcess[Any, Any],
    ) -> object:
        publish_ready(signal_path, "harvest")
        signal.pause()
        return real_read_worker_result(process)

    fanout._read_worker_result = block_harvest_hook  # ty: ignore[invalid-assignment]
    run_call_pool(
        [
            CallSpec(
                key="worker",
                job=ProcessJob(
                    entrypoint=f"{_WORKERS}:spawn_descendant_and_return",
                    payload={
                        "signal_path": signal_path,
                        "value": "complete",
                    },
                ),
                decode=_identity,
                deadline_seconds=30.0,
            )
        ],
        concurrency=1,
        is_rate_limited=_never_rate_limited,
    )


def pre_ready_starter(
    fake_guardian_path: str,
    ready_writer: str,
    behavior: str,
) -> None:
    real_popen = subprocess.Popen
    worker._GUARDIAN_READY_TIMEOUT_SECONDS = 0.1

    def fake_popen(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
        pass_fds_raw = kwargs.get("pass_fds", ())
        if not isinstance(pass_fds_raw, tuple):
            raise TypeError("pass_fds must be a tuple")
        inherited = cast(tuple[int, ...], pass_fds_raw)
        pass_fds = (*inherited, int(ready_writer))
        return real_popen(
            [
                sys.executable,
                fake_guardian_path,
                ready_writer,
                behavior,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            pass_fds=pass_fds,
        )

    worker.subprocess.Popen = fake_popen  # ty: ignore[invalid-assignment]
    lifetime_reader, _lifetime_writer = os.pipe()
    _done_reader, done_writer = os.pipe()
    worker._start_guardian(lifetime_reader, done_writer)


_SCENARIOS: dict[str, Callable[..., None]] = {
    "parent_death_pool": parent_death_pool,
    "guardian_scheduler_die": guardian_scheduler_die,
    "forked_scheduler_sibling": forked_scheduler_sibling,
    "block_harvest": block_harvest,
    "pre_ready_starter": pre_ready_starter,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run a fanout guardian scheduler scenario.",
    )
    parser.add_argument(
        "scenario",
        choices=sorted(_SCENARIOS),
    )
    parser.add_argument("args", nargs="*")
    parsed = parser.parse_args(argv)
    if parsed.scenario == "pre_ready_starter":
        if len(parsed.args) != 3:
            parser.error(
                "pre_ready_starter requires "
                "fake_guardian_path ready_writer behavior"
            )
        _SCENARIOS[parsed.scenario](*parsed.args)
        return
    if len(parsed.args) != 1:
        parser.error(f"{parsed.scenario} requires signal_path")
    _SCENARIOS[parsed.scenario](parsed.args[0])


if __name__ == "__main__":
    main()
