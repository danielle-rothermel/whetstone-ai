"""POSIX file locking and durability helpers for execution storage."""

from __future__ import annotations

import fcntl
import os
import stat
from pathlib import Path
from types import TracebackType
from typing import Self

__all__ = [
    "FileLock",
    "ensure_private_directory",
    "fsync_file",
    "fsync_parent_directory",
    "open_private_regular_file",
]


def _descriptor_flags(flags: int) -> int:
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _validate_owned_descriptor(
    fd: int,
    *,
    path: Path,
    expected_type: int,
) -> None:
    status = os.fstat(fd)
    if stat.S_IFMT(status.st_mode) != expected_type:
        expected = (
            "directory" if expected_type == stat.S_IFDIR else "regular file"
        )
        raise OSError(f"execution storage path is not a {expected}: {path}")
    if status.st_uid != os.geteuid():
        raise PermissionError(
            f"execution storage path is not owned by the current user: {path}"
        )


def ensure_private_directory(path: Path) -> None:
    """Create or tighten one owned, non-symlink storage directory."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        pass
    flags = _descriptor_flags(os.O_RDONLY)
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    fd = os.open(path, flags)
    try:
        _validate_owned_descriptor(
            fd,
            path=path,
            expected_type=stat.S_IFDIR,
        )
        os.fchmod(fd, 0o700)
    finally:
        os.close(fd)


def open_private_regular_file(
    path: Path,
    flags: int,
    mode: int = 0o600,
) -> int:
    """Open one owned regular file without following its final component."""
    # O_NONBLOCK prevents an unexpected FIFO from blocking before fstat can
    # reject it. It has no effect on regular-file I/O.
    fd = os.open(
        path,
        _descriptor_flags(flags | os.O_NONBLOCK),
        mode,
    )
    try:
        _validate_owned_descriptor(
            fd,
            path=path,
            expected_type=stat.S_IFREG,
        )
        os.fchmod(fd, 0o600)
    except BaseException:
        os.close(fd)
        raise
    return fd


class FileLock:
    """An advisory sidecar lock shared by threads and peer processes."""

    def __init__(self, path: Path, *, shared: bool = False) -> None:
        self.path = path
        self.shared = shared
        self._fd: int | None = None

    def __enter__(self) -> Self:
        ensure_private_directory(self.path.parent)
        fd = open_private_regular_file(
            self.path,
            os.O_RDWR | os.O_CREAT,
        )
        try:
            operation = fcntl.LOCK_SH if self.shared else fcntl.LOCK_EX
            fcntl.flock(fd, operation)
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        fd = self._fd
        if fd is None:
            return
        self._fd = None
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def fsync_file(fd: int) -> None:
    """Persist file contents and metadata acknowledged through ``fd``."""
    os.fsync(fd)


def fsync_parent_directory(path: Path) -> None:
    """Persist a path's directory entry after creation, replace, or delete."""
    flags = _descriptor_flags(os.O_RDONLY)
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    fd = os.open(path.parent, flags)
    try:
        _validate_owned_descriptor(
            fd,
            path=path.parent,
            expected_type=stat.S_IFDIR,
        )
        os.fsync(fd)
    finally:
        os.close(fd)
