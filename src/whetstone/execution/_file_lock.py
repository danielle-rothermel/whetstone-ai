"""POSIX file locking and durability helpers for execution storage."""

from __future__ import annotations

import fcntl
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

__all__ = [
    "FileLock",
    "PrivateDirectory",
    "ensure_private_directory",
    "fsync_file",
    "fsync_parent_directory",
    "open_private_directory",
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
) -> os.stat_result:
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
    if expected_type == stat.S_IFREG and status.st_nlink != 1:
        raise OSError(
            f"execution storage path has unexpected hard links: {path}"
        )
    return status


def _directory_flags() -> int:
    flags = _descriptor_flags(os.O_RDONLY)
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    return flags


def _validate_component_name(name: str) -> None:
    if name in {"", ".", ".."} or Path(name).name != name:
        raise ValueError(f"managed path component must be one name: {name!r}")


def _open_private_regular_at(
    directory_fd: int,
    name: str,
    *,
    display_path: Path,
    flags: int,
    mode: int = 0o600,
) -> int:
    _validate_component_name(name)
    attempts = 4 if flags & os.O_CREAT else 1
    for attempt in range(attempts):
        try:
            fd = os.open(
                name,
                _descriptor_flags(flags | os.O_NONBLOCK),
                mode,
                dir_fd=directory_fd,
            )
        except FileNotFoundError:
            if attempt + 1 == attempts:
                raise
            continue
        break
    try:
        status = _validate_owned_descriptor(
            fd,
            path=display_path,
            expected_type=stat.S_IFREG,
        )
        if stat.S_IMODE(status.st_mode) != 0o600:
            os.fchmod(fd, 0o600)
    except BaseException:
        os.close(fd)
        raise
    return fd


@dataclass(slots=True)
class PrivateDirectory:
    """An owned private directory retained as a descriptor capability."""

    path: Path
    fd: int
    _closed: bool = False

    def __enter__(self) -> Self:
        if self._closed:
            raise RuntimeError("private directory is already closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            os.close(self.fd)

    def _active_fd(self) -> int:
        if self._closed:
            raise RuntimeError("private directory is already closed")
        return self.fd

    def open_regular(
        self,
        name: str,
        flags: int,
        mode: int = 0o600,
    ) -> int:
        return _open_private_regular_at(
            self._active_fd(),
            name,
            display_path=self.path / name,
            flags=flags,
            mode=mode,
        )

    def open_child(
        self,
        name: str,
        *,
        create: bool = True,
    ) -> Self:
        _validate_component_name(name)
        parent_fd = self._active_fd()
        child_path = self.path / name
        created = False
        try:
            child_fd = os.open(
                name,
                _directory_flags(),
                dir_fd=parent_fd,
            )
        except FileNotFoundError:
            if not create:
                raise
            try:
                os.mkdir(name, mode=0o700, dir_fd=parent_fd)
                created = True
            except FileExistsError:
                pass
            if created:
                os.chmod(
                    name,
                    0o700,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
            child_fd = os.open(
                name,
                _directory_flags(),
                dir_fd=parent_fd,
            )
        try:
            status = _validate_owned_descriptor(
                child_fd,
                path=child_path,
                expected_type=stat.S_IFDIR,
            )
            if stat.S_IMODE(status.st_mode) != 0o700:
                os.fchmod(child_fd, 0o700)
                os.fsync(child_fd)
            if created:
                os.fsync(child_fd)
                os.fsync(parent_fd)
        except BaseException:
            os.close(child_fd)
            raise
        return type(self)(path=child_path, fd=child_fd)

    def stat(self, name: str) -> os.stat_result:
        _validate_component_name(name)
        return os.stat(
            name,
            dir_fd=self._active_fd(),
            follow_symlinks=False,
        )

    def list_names(self) -> list[str]:
        return os.listdir(self._active_fd())

    def replace(self, source: str, destination: str) -> None:
        _validate_component_name(source)
        _validate_component_name(destination)
        fd = self._active_fd()
        os.replace(
            source,
            destination,
            src_dir_fd=fd,
            dst_dir_fd=fd,
        )

    def unlink(self, name: str) -> None:
        _validate_component_name(name)
        os.unlink(name, dir_fd=self._active_fd())

    def fsync(self) -> None:
        os.fsync(self._active_fd())


def open_private_directory(
    path: Path,
    *,
    create: bool = True,
) -> PrivateDirectory:
    """Open an owned private directory without releasing its verified FD."""
    absolute = Path(os.path.abspath(path))
    if len(absolute.parts) == 1:
        raise ValueError(
            "private directory path must not be a filesystem root"
        )
    current_path = Path(absolute.anchor)
    current_fd = os.open(current_path, _directory_flags())
    try:
        private_chain = False
        for index, component in enumerate(absolute.parts[1:]):
            next_path = current_path / component
            created = False
            try:
                next_fd = os.open(
                    component,
                    _directory_flags(),
                    dir_fd=current_fd,
                )
            except FileNotFoundError:
                if not create:
                    raise
                private_chain = True
                try:
                    os.mkdir(component, mode=0o700, dir_fd=current_fd)
                    created = True
                except FileExistsError:
                    pass
                if created:
                    os.chmod(
                        component,
                        0o700,
                        dir_fd=current_fd,
                        follow_symlinks=False,
                    )
                next_fd = os.open(
                    component,
                    _directory_flags(),
                    dir_fd=current_fd,
                )
            try:
                is_leaf = index == len(absolute.parts) - 2
                if private_chain or is_leaf:
                    status = _validate_owned_descriptor(
                        next_fd,
                        path=next_path,
                        expected_type=stat.S_IFDIR,
                    )
                    if stat.S_IMODE(status.st_mode) != 0o700:
                        os.fchmod(next_fd, 0o700)
                        os.fsync(next_fd)
                if created:
                    os.fsync(next_fd)
                    os.fsync(current_fd)
            except BaseException:
                os.close(next_fd)
                raise
            os.close(current_fd)
            current_fd = next_fd
            current_path = next_path
    except BaseException:
        os.close(current_fd)
        raise
    return PrivateDirectory(path=absolute, fd=current_fd)


def ensure_private_directory(path: Path) -> None:
    """Durably create and tighten an owned, non-symlink directory path."""
    with open_private_directory(path):
        pass


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
        status = _validate_owned_descriptor(
            fd,
            path=path,
            expected_type=stat.S_IFREG,
        )
        if stat.S_IMODE(status.st_mode) != 0o600:
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
        self._directory: PrivateDirectory | None = None

    @property
    def directory(self) -> PrivateDirectory:
        directory = self._directory
        if directory is None:
            raise RuntimeError("file lock is not held")
        return directory

    def __enter__(self) -> Self:
        directory = open_private_directory(self.path.parent)
        try:
            fd = directory.open_regular(
                self.path.name,
                os.O_RDWR | os.O_CREAT,
            )
            operation = fcntl.LOCK_SH if self.shared else fcntl.LOCK_EX
            fcntl.flock(fd, operation)
        except BaseException:
            if "fd" in locals():
                os.close(fd)
            directory.close()
            raise
        self._fd = fd
        self._directory = directory
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
        directory = self._directory
        self._directory = None
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            try:
                os.close(fd)
            finally:
                if directory is not None:
                    directory.close()


def fsync_file(fd: int) -> None:
    """Persist file contents and metadata acknowledged through ``fd``."""
    os.fsync(fd)


def fsync_parent_directory(path: Path) -> None:
    """Persist a path's directory entry after creation, replace, or delete."""
    fd = os.open(path.parent, _directory_flags())
    try:
        _validate_owned_descriptor(
            fd,
            path=path.parent,
            expected_type=stat.S_IFDIR,
        )
        os.fsync(fd)
    finally:
        os.close(fd)
