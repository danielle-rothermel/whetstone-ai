from __future__ import annotations

import os
import selectors
import signal
import subprocess
import sys
from pathlib import Path

from pydantic import JsonValue

from tests.execution.process_signals import publish_ready, wait_for_release


def _mapping(payload: JsonValue) -> dict[str, JsonValue]:
    if not isinstance(payload, dict):
        raise TypeError("test worker payload must be an object")
    return payload


def _string(payload: dict[str, JsonValue], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string")
    return value


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


def gated_event(payload: JsonValue) -> JsonValue:
    body = _mapping(payload)
    key = _string(body, "key")
    wait_for_release(_string(body, "signal_path"), key)
    if body.get("fail") is True:
        raise RuntimeError(f"requested failure for {key}")
    return body.get("value")


def require_path_then_return(payload: JsonValue) -> JsonValue:
    body = _mapping(payload)
    required_path = Path(_string(body, "required_path"))
    event_path = Path(_string(body, "event_path"))
    if not required_path.exists():
        raise AssertionError("required path was absent when user code began")
    _append(event_path, "observed")
    return body.get("value")


def descendant_ready(signal_path: str, key: str) -> None:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    publish_ready(signal_path, f"{key}-descendant")
    signal.pause()


def block_process_tree(payload: JsonValue) -> JsonValue:
    body = _mapping(payload)
    signal_path = _string(body, "signal_path")
    key = _string(body, "key")
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tests.execution.process_workers",
            "descendant_ready",
            signal_path,
            key,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    publish_ready(signal_path, f"{key}-worker")
    signal.pause()
    raise AssertionError(
        "signal.pause returned without terminating the worker"
    )


def spawn_descendant_and_return(payload: JsonValue) -> JsonValue:
    body = _mapping(payload)
    signal_path = _string(body, "signal_path")
    ready_reader, ready_writer = os.pipe()
    descendant_script = """
import os
import signal
import sys

from tests.execution.process_signals import publish_ready

path, ready_writer = sys.argv[1:]
signal.signal(signal.SIGTERM, signal.SIG_IGN)
publish_ready(path, "descendant")
os.write(int(ready_writer), b"1")
os.close(int(ready_writer))
signal.pause()
"""
    try:
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                descendant_script,
                signal_path,
                str(ready_writer),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            pass_fds=(ready_writer,),
        )
    finally:
        os.close(ready_writer)
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(ready_reader, selectors.EVENT_READ)
            if not selector.select(timeout=5.0):
                raise TimeoutError("descendant did not publish readiness")
            if os.read(ready_reader, 1) != b"1":
                raise RuntimeError("descendant readiness pipe closed early")
    finally:
        os.close(ready_reader)
    release_key = body.get("release_key")
    if release_key is not None:
        if not isinstance(release_key, str):
            raise TypeError("release_key must be a string")
        wait_for_release(signal_path, release_key)
    return body.get("value")


def open_file_descriptors(payload: JsonValue) -> JsonValue:
    del payload
    descriptors: list[int] = []
    for descriptor in range(3, 4096):
        try:
            os.fstat(descriptor)
        except OSError:
            continue
        descriptors.append(descriptor)
    return descriptors


def return_payload(payload: JsonValue) -> JsonValue:
    return payload


def non_finite_result(payload: JsonValue) -> JsonValue:
    body = _mapping(payload)
    event_path = Path(_string(body, "event_path"))
    _append(event_path, "started")
    if body.get("nested") is True:
        return {"nested": [float("inf")]}
    return float("nan")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: process_workers <command> [args...]")
    command = sys.argv[1]
    if command == "descendant_ready":
        if len(sys.argv) != 4:
            raise SystemExit(
                "usage: process_workers descendant_ready <signal_path> <key>"
            )
        descendant_ready(sys.argv[2], sys.argv[3])
        return
    raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    main()
