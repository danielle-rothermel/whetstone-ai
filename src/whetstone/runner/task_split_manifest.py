"""Filesystem adapter for environment-owned task-selection contracts."""

from __future__ import annotations

from pathlib import Path

from whetstone.envs.task_selection import (
    ResolvedSplit,
    TaskSplitManifest,
    TaskSplitManifestError,
    TaskSplitRoles,
    parse_task_split_manifest,
    resolve_manifest_split,
)


def load_task_split_manifest(path: Path) -> TaskSplitManifest:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise TaskSplitManifestError(
            f"cannot read task-selection manifest {path}: {exc}"
        ) from exc
    return parse_task_split_manifest(payload)


__all__ = [
    "ResolvedSplit",
    "TaskSplitManifest",
    "TaskSplitManifestError",
    "TaskSplitRoles",
    "load_task_split_manifest",
    "resolve_manifest_split",
]
