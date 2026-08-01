"""Secure and durable filesystem foundations for execution storage."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

import whetstone.execution._file_lock as file_lock_module
from whetstone.execution._file_lock import (
    FileLock,
    PrivateDirectory,
    ensure_private_directory,
)


def _identity(fd: int) -> tuple[int, int]:
    status = os.fstat(fd)
    return status.st_dev, status.st_ino


def test_nested_creation_fsyncs_each_receiving_parent_in_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "new" / "nested" / "run.partial"
    events: list[tuple[str, tuple[int, int], str | None]] = []
    real_mkdir = os.mkdir
    real_fsync = os.fsync

    def observe_mkdir(
        path: str,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> None:
        assert dir_fd is not None
        events.append(("mkdir", _identity(dir_fd), path))
        real_mkdir(path, mode, dir_fd=dir_fd)

    def observe_fsync(fd: int) -> None:
        events.append(("fsync", _identity(fd), None))
        real_fsync(fd)

    monkeypatch.setattr(file_lock_module.os, "mkdir", observe_mkdir)
    monkeypatch.setattr(file_lock_module.os, "fsync", observe_fsync)

    ensure_private_directory(target)

    expected = [
        ("mkdir", (tmp_path.stat().st_dev, tmp_path.stat().st_ino), "new"),
        (
            "fsync",
            (
                (tmp_path / "new").stat().st_dev,
                (tmp_path / "new").stat().st_ino,
            ),
            None,
        ),
        ("fsync", (tmp_path.stat().st_dev, tmp_path.stat().st_ino), None),
        (
            "mkdir",
            (
                (tmp_path / "new").stat().st_dev,
                (tmp_path / "new").stat().st_ino,
            ),
            "nested",
        ),
        (
            "fsync",
            (
                (tmp_path / "new" / "nested").stat().st_dev,
                (tmp_path / "new" / "nested").stat().st_ino,
            ),
            None,
        ),
        (
            "fsync",
            (
                (tmp_path / "new").stat().st_dev,
                (tmp_path / "new").stat().st_ino,
            ),
            None,
        ),
        (
            "mkdir",
            (
                (tmp_path / "new" / "nested").stat().st_dev,
                (tmp_path / "new" / "nested").stat().st_ino,
            ),
            "run.partial",
        ),
        (
            "fsync",
            (target.stat().st_dev, target.stat().st_ino),
            None,
        ),
        (
            "fsync",
            (
                (tmp_path / "new" / "nested").stat().st_dev,
                (tmp_path / "new" / "nested").stat().st_ino,
            ),
            None,
        ),
    ]
    assert events == expected
    for directory in [
        tmp_path / "new",
        tmp_path / "new" / "nested",
        target,
    ]:
        status = directory.stat()
        assert stat.S_IMODE(status.st_mode) == 0o700
        assert status.st_uid == os.geteuid()


def test_existing_ancestor_is_not_repaired_or_recreated(
    tmp_path: Path,
) -> None:
    ancestor = tmp_path / "existing"
    ancestor.mkdir(mode=0o755)
    ancestor.chmod(0o755)

    ensure_private_directory(ancestor / "run.partial")

    assert stat.S_IMODE(ancestor.stat().st_mode) == 0o755
    assert stat.S_IMODE((ancestor / "run.partial").stat().st_mode) == 0o700


def test_intermediate_symlink_is_not_followed(tmp_path: Path) -> None:
    external = tmp_path / "external"
    external.mkdir(mode=0o755)
    external.chmod(0o755)
    link = tmp_path / "linked"
    link.symlink_to(external, target_is_directory=True)

    with pytest.raises(OSError):
        ensure_private_directory(link / "run.partial")

    assert not (external / "run.partial").exists()
    assert stat.S_IMODE(external.stat().st_mode) == 0o755


def test_filesystem_root_is_never_a_managed_private_directory() -> None:
    with pytest.raises(ValueError, match="filesystem root"):
        ensure_private_directory(Path("/"))


def test_all_masking_umask_succeeds_and_retry_is_clean(tmp_path: Path) -> None:
    target = tmp_path / "masked" / "nested" / "run.partial"
    previous_umask = os.umask(0o777)
    try:
        ensure_private_directory(target)
        ensure_private_directory(target)
        observed_umask = os.umask(0o777)
        assert observed_umask == 0o777
    finally:
        os.umask(previous_umask)

    for directory in [target.parent.parent, target.parent, target]:
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
        assert directory.stat().st_uid == os.geteuid()


def test_lock_open_remains_bound_to_verified_parent_after_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    managed = tmp_path / "managed"
    managed.mkdir(mode=0o700)
    relocated = tmp_path / "relocated"
    external = tmp_path / "external"
    external.mkdir(mode=0o700)
    lock_path = managed / "storage.lock"
    real_open = PrivateDirectory.open_regular
    substituted = False

    def substitute_before_open(
        directory: PrivateDirectory,
        name: str,
        flags: int,
        mode: int = 0o600,
    ) -> int:
        nonlocal substituted
        if not substituted and directory.path == managed:
            substituted = True
            managed.rename(relocated)
            managed.symlink_to(external, target_is_directory=True)
        return real_open(directory, name, flags, mode)

    monkeypatch.setattr(
        PrivateDirectory,
        "open_regular",
        substitute_before_open,
    )
    with FileLock(lock_path):
        pass

    assert (relocated / "storage.lock").is_file()
    assert not (external / "storage.lock").exists()
