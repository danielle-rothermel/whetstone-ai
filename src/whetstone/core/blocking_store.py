from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from dr_store import BindStatus, ObjectStore, PutStatus, SqliteBackend
from dr_store.content_addressing import ObjectReference

__all__ = [
    "BlockingObjectStore",
    "open_blocking_sqlite_store",
    "persistent_blocking_sqlite_store",
]


class BlockingObjectStore:
    """Sync facade over dr-store 0.2.x async ``ObjectStore`` APIs."""

    def __init__(
        self,
        store: ObjectStore,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._store = store
        self._loop = loop

    def _run(self, coro: Any) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def put(
        self, schema: str, record: Any
    ) -> tuple[ObjectReference, PutStatus]:
        return self._run(self._store.put(schema, record))

    def get(self, reference: ObjectReference) -> Any:
        return self._run(self._store.get(reference))

    def bind(self, key: str, reference: ObjectReference) -> BindStatus:
        return self._run(self._store.bind(key, reference))

    def resolve(self, key: str) -> ObjectReference | None:
        return self._run(self._store.resolve(key))


class _StoreSession:
    def __init__(self, path: str) -> None:
        self._path = path
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever,
            name="dr-store-sqlite-loop",
            daemon=True,
        )
        self._backend: SqliteBackend | None = None
        self.store: BlockingObjectStore | None = None

    def open(self) -> BlockingObjectStore:
        if self.store is not None:
            return self.store
        self._thread.start()
        backend = asyncio.run_coroutine_threadsafe(
            SqliteBackend.open(self._path),
            self._loop,
        ).result()
        self._backend = backend
        self.store = BlockingObjectStore(ObjectStore(backend), self._loop)
        return self.store

    def close(self) -> None:
        if self._backend is not None:
            asyncio.run_coroutine_threadsafe(
                self._backend.aclose(),
                self._loop,
            ).result()
            self._backend = None
        if self._thread.is_alive():
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)
        self.store = None


@contextmanager
def open_blocking_sqlite_store(path: str) -> Iterator[BlockingObjectStore]:
    session = _StoreSession(path)
    try:
        yield session.open()
    finally:
        session.close()


_sessions: dict[str, _StoreSession] = {}


def persistent_blocking_sqlite_store(path: str) -> BlockingObjectStore:
    """Open or reuse a process-lifetime blocking SQLite store."""
    session = _sessions.get(path)
    if session is None:
        session = _StoreSession(path)
        session.open()
        _sessions[path] = session
    assert session.store is not None
    return session.store
