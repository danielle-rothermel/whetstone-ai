"""Fake guardian process for pre-ready failure containment tests."""

from __future__ import annotations

import os
import signal
import sys


def main() -> None:
    ready_writer, behavior = sys.argv[1:]
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    os.write(int(ready_writer), f"{os.getpid()}\n".encode())
    os.close(int(ready_writer))
    if behavior == "exit":
        raise SystemExit(2)
    signal.pause()


if __name__ == "__main__":
    main()
