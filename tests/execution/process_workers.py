"""Importable process jobs used by fanout integration tests."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from pydantic import JsonValue


def _mapping(payload: JsonValue) -> dict[str, JsonValue]:
    if not isinstance(payload, dict):
        raise TypeError("test worker payload must be an object")
    return payload


def _string(payload: dict[str, JsonValue], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string")
    return value


def _number(payload: dict[str, JsonValue], key: str) -> float:
    value = payload[key]
    if not isinstance(value, int | float):
        raise TypeError(f"{key} must be a number")
    return float(value)


def _append(path: Path, line: str) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o600,
    )
    try:
        payload = f"{line}\n".encode()
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise OSError("short test event write")
    finally:
        os.close(descriptor)


def _started_count(path: Path) -> int:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return 0
    return sum(line.startswith("start|") for line in lines)


def delayed_event(payload: JsonValue) -> JsonValue:
    """Record start/finish around a delay, optionally failing."""
    body = _mapping(payload)
    key = _string(body, "key")
    event_path = Path(_string(body, "event_path"))
    _append(event_path, f"start|{key}|{time.monotonic()}")
    wait_for_started = body.get("wait_for_started")
    if wait_for_started is not None:
        if not isinstance(wait_for_started, int):
            raise TypeError("wait_for_started must be an integer")
        deadline = time.monotonic() + 5.0
        while _started_count(event_path) < wait_for_started:
            if time.monotonic() >= deadline:
                raise TimeoutError("initial test workers did not all start")
            time.sleep(0.005)
    delay = _number(body, "delay")
    if delay:
        time.sleep(delay)
    _append(event_path, f"finish|{key}|{time.monotonic()}")
    if body.get("fail") is True:
        raise RuntimeError(f"requested failure for {key}")
    return body.get("value")


def wait_for_release(payload: JsonValue) -> JsonValue:
    """Record start and block until a test-owned release file appears."""
    body = _mapping(payload)
    key = _string(body, "key")
    event_path = Path(_string(body, "event_path"))
    release_path = Path(_string(body, "release_path"))
    _append(event_path, f"start|{key}|{time.monotonic()}")
    deadline = time.monotonic() + 5.0
    while not release_path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("test release file was not created")
        time.sleep(0.005)
    _append(event_path, f"finish|{key}|{time.monotonic()}")
    return key


def require_path_then_return(payload: JsonValue) -> JsonValue:
    """Prove a required boundary file exists before user code begins."""
    body = _mapping(payload)
    required_path = Path(_string(body, "required_path"))
    event_path = Path(_string(body, "event_path"))
    if not required_path.exists():
        raise AssertionError("required path was absent when user code began")
    _append(event_path, "observed")
    return body.get("value")


def heartbeat_forever(payload: JsonValue) -> JsonValue:
    """Write from a process tree until the scheduler escalates to KILL."""
    body = _mapping(payload)
    heartbeat_path = Path(_string(body, "heartbeat_path"))
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    descendant_script = """
import os
import signal
import sys
import time

path = sys.argv[1]
signal.signal(signal.SIGTERM, signal.SIG_IGN)
descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
os.write(descriptor, f"pid|{os.getpid()}\\n".encode())
while True:
    os.write(
        descriptor,
        f"tick|{os.getpid()}|{time.monotonic()}\\n".encode(),
    )
    time.sleep(0.01)
"""
    subprocess.Popen(
        [sys.executable, "-c", descendant_script, os.fspath(heartbeat_path)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _append(heartbeat_path, f"pid|{os.getpid()}")
    while True:
        _append(heartbeat_path, f"tick|{os.getpid()}|{time.monotonic()}")
        time.sleep(0.01)


def spawn_descendant_and_return(payload: JsonValue) -> JsonValue:
    """Leave a same-group descendant running after the worker returns."""
    body = _mapping(payload)
    heartbeat_path = Path(_string(body, "heartbeat_path"))
    descendant_script = """
import os
import signal
import sys
import time

path = sys.argv[1]
signal.signal(signal.SIGTERM, signal.SIG_IGN)
descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
os.write(descriptor, f"pid|{os.getpid()}\\n".encode())
while True:
    os.write(
        descriptor,
        f"tick|{os.getpid()}|{time.monotonic()}\\n".encode(),
    )
    time.sleep(0.01)
"""
    subprocess.Popen(
        [sys.executable, "-c", descendant_script, os.fspath(heartbeat_path)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 3.0
    while not _pid_lines_for_worker(heartbeat_path):
        if time.monotonic() >= deadline:
            raise TimeoutError("descendant did not publish its pid")
        time.sleep(0.005)
    release_path = body.get("release_path")
    if release_path is not None:
        if not isinstance(release_path, str):
            raise TypeError("release_path must be a string")
        release = Path(release_path)
        while not release.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("test release file was not created")
            time.sleep(0.005)
    return body.get("value")


def open_file_descriptors(payload: JsonValue) -> JsonValue:
    """Return unexpected descriptors visible when user code begins."""
    del payload
    descriptors: list[int] = []
    for descriptor in range(3, 4096):
        try:
            os.fstat(descriptor)
        except OSError:
            continue
        descriptors.append(descriptor)
    return descriptors


def _pid_lines_for_worker(path: Path) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    return [line for line in lines if line.startswith("pid|")]


def return_payload(payload: JsonValue) -> JsonValue:
    """Return the validated JSON payload unchanged."""
    return payload


def non_finite_result(payload: JsonValue) -> JsonValue:
    """Return an invalid scalar or nested JSON number."""
    body = _mapping(payload)
    event_path = Path(_string(body, "event_path"))
    _append(event_path, "started")
    if body.get("nested") is True:
        return {"nested": [float("inf")]}
    return float("nan")
