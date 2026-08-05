"""Bounded process joins and escalation for optimization integration tests."""

from __future__ import annotations

import multiprocessing
from collections.abc import Iterable
from typing import Any


def in_process_start_methods() -> tuple[str, ...]:
    """Return supported methods that never clone the pytest process."""
    return tuple(
        method
        for method in multiprocessing.get_all_start_methods()
        if method != "fork"
    )


def join_processes(
    processes: Iterable[Any],
    *,
    timeout: float,
) -> None:
    """Join successful workers and require clean zero exit codes."""
    workers = tuple(processes)
    for process in workers:
        process.join(timeout=timeout)
    alive = [process.name for process in workers if process.is_alive()]
    if alive:
        raise AssertionError(f"workers did not exit: {alive}")
    failed = [
        (process.name, process.exitcode)
        for process in workers
        if process.exitcode != 0
    ]
    if failed:
        raise AssertionError(f"workers exited unsuccessfully: {failed}")


def terminate_processes(
    processes: Iterable[Any],
    *,
    timeout: float,
) -> None:
    """Terminate, then kill, and require every started worker to be reaped."""
    workers = tuple(processes)
    for process in workers:
        if process.is_alive():
            process.terminate()
    for process in workers:
        process.join(timeout=timeout)
    for process in workers:
        if process.is_alive():
            process.kill()
    for process in workers:
        process.join(timeout=timeout)
    alive = [process.name for process in workers if process.is_alive()]
    if alive:
        raise AssertionError(f"workers survived termination and kill: {alive}")
