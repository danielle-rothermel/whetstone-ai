"""Task-screen operations over environment-owned transformations."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from whetstone.envs.input_transform import (
    direct_body,
    direct_prompt,
    rename_identifier,
    renamed_task,
    split_prompt,
)
from whetstone.runner.cell import CellConfig, CellOutcome, run_cell
from whetstone.runner.events import (
    EventStream,
    EventUnit,
    screen_key_locked_event,
)


class ScreenKeyLocked(RuntimeError):
    """Another process owns the exact screen output key."""


def sidecar_lock_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.lock")


@contextmanager
def screen_key_lock(
    path: Path,
    *,
    event_stream: EventStream | None = None,
    unit: EventUnit | None = None,
) -> Iterator[None]:
    """Hold a non-blocking one-writer lock for a screen artifact."""
    lock_path = sidecar_lock_path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            if event_stream is not None and unit is not None:
                event_stream.emit(
                    screen_key_locked_event(
                        unit=unit,
                        screen_key=str(path),
                        lock_path=str(lock_path),
                    )
                )
            raise ScreenKeyLocked(str(path)) from exc
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def write_screen_summary_atomic(
    path: Path, rows: tuple[dict[str, object], ...]
) -> None:
    """Replace one complete screen summary without exposing a torn file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(list(rows), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


@dataclass(frozen=True, slots=True)
class TaskScreenReport:
    outcomes: tuple[CellOutcome, ...]
    summary_path: Path


def run_task_screen(
    configs: tuple[CellConfig, ...],
    *,
    summary_path: Path,
    event_stream: EventStream | None = None,
    unit: EventUnit | None = None,
) -> TaskScreenReport:
    """Run screen cells and atomically project their validated records."""
    with screen_key_lock(
        summary_path,
        event_stream=event_stream,
        unit=unit,
    ):
        outcomes = tuple(run_cell(config) for config in configs)
        write_screen_summary_atomic(
            summary_path,
            tuple(
                outcome.record.model_dump(mode="json", by_alias=True)
                for outcome in outcomes
            ),
        )
    return TaskScreenReport(outcomes, summary_path)


__all__ = [
    "ScreenKeyLocked",
    "TaskScreenReport",
    "direct_body",
    "direct_prompt",
    "rename_identifier",
    "renamed_task",
    "run_task_screen",
    "screen_key_lock",
    "sidecar_lock_path",
    "split_prompt",
    "write_screen_summary_atomic",
]
