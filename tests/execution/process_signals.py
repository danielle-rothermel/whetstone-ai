"""Explicit parent/child synchronization for execution integration tests."""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
from collections.abc import Callable, Collection
from multiprocessing.connection import Client, Connection, Listener
from pathlib import Path

from pydantic import JsonValue

_AUTHKEY = b"whetstone-execution-tests"
_WATCHDOG_SECONDS = 10.0


def wait_for_release(socket_path: str, key: str) -> None:
    """Publish entry, block for release, then publish completion."""
    connection = Client(socket_path, family="AF_UNIX", authkey=_AUTHKEY)
    try:
        connection.send(("entered", key, os.getpid()))
        if connection.recv() != "release":
            raise RuntimeError("test gate returned an invalid release token")
        connection.send(("completed", key))
    finally:
        connection.close()


def publish_ready(socket_path: str, key: str) -> None:
    """Publish one readiness record and await parent acknowledgement."""
    connection = Client(socket_path, family="AF_UNIX", authkey=_AUTHKEY)
    try:
        connection.send(("ready", key, os.getpid()))
        if connection.recv() != "acknowledged":
            raise RuntimeError(
                "test signal returned an invalid acknowledgement"
            )
    finally:
        connection.close()


class ProcessSignals:
    """Collect process events and release blocked workers by stable key."""

    def __init__(self) -> None:
        self._directory = Path(
            tempfile.mkdtemp(prefix="whetstone-test-signals-")
        )
        self.path = self._directory / "socket"
        self._listener = Listener(
            os.fspath(self.path),
            family="AF_UNIX",
            backlog=32,
            authkey=_AUTHKEY,
        )
        self._condition = threading.Condition()
        self._connections: dict[str, Connection] = {}
        self._pids: dict[str, int] = {}
        self._completed: set[str] = set()
        self._errors: list[BaseException] = []
        self._closed = False
        self._handlers: list[threading.Thread] = []
        self._server = threading.Thread(
            target=self._serve,
            name="whetstone-test-process-signals",
            daemon=True,
        )
        self._server.start()

    def _serve(self) -> None:
        while True:
            try:
                connection = self._listener.accept()
            except (OSError, EOFError):
                return
            handler = threading.Thread(
                target=self._handle,
                args=(connection,),
                name="whetstone-test-process-signal",
                daemon=True,
            )
            with self._condition:
                self._handlers.append(handler)
            handler.start()

    def _handle(self, connection: Connection) -> None:
        try:
            message = connection.recv()
            if (
                not isinstance(message, tuple)
                or len(message) != 3
                or message[0] not in {"entered", "ready"}
                or not isinstance(message[1], str)
                or not isinstance(message[2], int)
            ):
                raise AssertionError(f"invalid process event: {message!r}")
            event, key, pid = message
            with self._condition:
                if key in self._pids:
                    raise AssertionError(f"duplicate process event key: {key}")
                self._pids[key] = pid
                if event == "entered":
                    self._connections[key] = connection
                self._condition.notify_all()
            if event == "ready":
                connection.send("acknowledged")
                return
            completed = connection.recv()
            if completed != ("completed", key):
                raise AssertionError(
                    f"invalid process completion event: {completed!r}"
                )
            with self._condition:
                self._completed.add(key)
                self._connections.pop(key, None)
                self._condition.notify_all()
        except (EOFError, OSError):
            if not self._closed:
                with self._condition:
                    self._errors.append(
                        AssertionError(
                            "process signal connection closed early"
                        )
                    )
                    self._condition.notify_all()
        except BaseException as error:
            with self._condition:
                self._errors.append(error)
                self._condition.notify_all()
        finally:
            connection.close()

    def _wait_for(
        self,
        predicate: Callable[[], bool],
        description: str,
    ) -> None:
        with self._condition:
            reached = self._condition.wait_for(
                lambda: bool(predicate()) or bool(self._errors),
                timeout=_WATCHDOG_SECONDS,
            )
            if self._errors:
                raise self._errors[0]
            if not reached:
                raise AssertionError(
                    f"timed out waiting for process signal: {description}"
                )

    def wait_entered(self, keys: Collection[str]) -> None:
        expected = set(keys)
        self._wait_for(
            lambda: expected <= self._pids.keys(),
            f"entered keys {sorted(expected)!r}",
        )

    def wait_completed(self, keys: Collection[str]) -> None:
        expected = set(keys)
        self._wait_for(
            lambda: expected <= self._completed,
            f"completed keys {sorted(expected)!r}",
        )

    def release(self, key: str) -> None:
        self.wait_entered([key])
        with self._condition:
            connection = self._connections.get(key)
            if connection is None:
                raise AssertionError(f"process gate {key!r} is not blocked")
            connection.send("release")

    @property
    def entered_keys(self) -> set[str]:
        with self._condition:
            return set(self._pids)

    def pid(self, key: str) -> int:
        self.wait_entered([key])
        with self._condition:
            return self._pids[key]

    def close(self) -> None:
        self._closed = True
        self._listener.close()
        with self._condition:
            connections = list(self._connections.values())
        for connection in connections:
            connection.close()
        shutil.rmtree(self._directory, ignore_errors=True)

    def __enter__(self) -> ProcessSignals:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def json_string(payload: dict[str, JsonValue], key: str) -> str:
    """Return one required string from a validated JSON test payload."""
    value = payload[key]
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string")
    return value
