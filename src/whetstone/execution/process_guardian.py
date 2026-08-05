"""Independent process-group guardian for fanout workers."""

from __future__ import annotations

import os
import signal
import sys
from collections.abc import Sequence

__all__: list[str] = []

_READY_TOKEN = b"\x01"
_FAILURE_TOKEN = b"\x01"


def _close(descriptor: int) -> None:
    try:
        os.close(descriptor)
    except OSError:
        pass


def _report_failure(descriptor: int) -> None:
    try:
        os.write(descriptor, _FAILURE_TOKEN)
    except OSError:
        pass


def main(argv: Sequence[str] | None = None) -> int:
    """Kill the guarded process group when scheduler authority reaches EOF."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 4:
        sys.stderr.write(
            "usage: python -m whetstone.execution.process_guardian "
            "LIFETIME_FD DONE_FD READY_FD PROCESS_GROUP_ID\n"
        )
        return 2
    lifetime_reader = int(arguments[0])
    done_writer = int(arguments[1])
    ready_writer = int(arguments[2])
    process_group_id = int(arguments[3])
    for descriptor in (lifetime_reader, done_writer, ready_writer):
        os.set_inheritable(descriptor, False)

    if os.getpgrp() != process_group_id:
        _report_failure(done_writer)
        return 2
    try:
        if os.write(ready_writer, _READY_TOKEN) != len(_READY_TOKEN):
            _report_failure(done_writer)
            return 2
    finally:
        _close(ready_writer)

    try:
        while True:
            try:
                payload = os.read(lifetime_reader, 1)
            except InterruptedError:
                continue
            except OSError:
                break
            if not payload:
                break
    finally:
        _close(lifetime_reader)

    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except OSError:
        _report_failure(done_writer)
        return 1
    raise AssertionError("SIGKILL unexpectedly returned")


if __name__ == "__main__":
    raise SystemExit(main())
