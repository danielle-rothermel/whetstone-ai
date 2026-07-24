"""Execution-mode detection for the validation runner.

The runner records an ``execution_mode`` per cell. Three modes, detected in
degrade-gracefully order:

1. ``postgres`` -- a usable Postgres for the DBOS orchestration path, detected
   with the SAME mechanism dr-platform's own tests use (a
   ``DR_PLATFORM_TEST_DATABASE_URL`` connect probe, default
   ``postgresql+psycopg:///dr_platform_test``).
2. ``docker-postgres`` -- if no ambient Postgres but the Docker daemon is
   running, a vanilla (pgvector-less) Postgres container can be brought up. The
   detector only reports *availability*; the runner does not silently spin up a
   container inside tests.
3. ``in-process`` -- always available. This is the stage-03 attempt driver +
   graph run + Result Store persistence path that already exists (wired, not
   duplicated, by :mod:`whetstone.runner.eval_run`).

Both the DBOS path and the in-process path produce **identical Result Store
artifacts** -- the same terminal Rollout Result / aggregate content -- because
the durable executor drives the exact same stage-03 pure driver
(:func:`whetstone.provider.driver.run_provider_call`) the in-process path uses.
Detection makes no live paid call; the probes touch only a database socket and
the Docker daemon.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "DEFAULT_TEST_DATABASE_URL",
    "ExecutionMode",
    "ExecutionModeDecision",
    "detect_execution_mode",
    "docker_daemon_running",
    "postgres_available",
]

#: The dr-platform test database URL (identical default to ``tests/conftest``).
DEFAULT_TEST_DATABASE_URL = "postgresql+psycopg:///dr_platform_test"


class ExecutionMode(StrEnum):
    """The recorded execution mode for a cell."""

    POSTGRES = "postgres"
    DOCKER_POSTGRES = "docker-postgres"
    IN_PROCESS = "in-process"


@dataclass(frozen=True, slots=True)
class ExecutionModeDecision:
    """The detected mode plus a human-readable reason for the report."""

    mode: ExecutionMode
    reason: str
    database_url: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "execution_mode": self.mode.value,
            "reason": self.reason,
            "database_url_present": self.database_url is not None,
        }


def postgres_available(database_url: str | None = None) -> bool:
    """Probe a usable Postgres exactly as dr-platform's tests do.

    Uses ``DR_PLATFORM_TEST_DATABASE_URL`` (or the passed URL, or the default),
    attempts a single SQLAlchemy connect, and returns whether it succeeded. Any
    connect failure is a clean ``False`` (never raised) -- the runner degrades.
    """
    url = database_url or os.environ.get(
        "DR_PLATFORM_TEST_DATABASE_URL", DEFAULT_TEST_DATABASE_URL
    )
    try:
        from sqlalchemy import create_engine

        engine = create_engine(url)
        try:
            with engine.connect():
                pass
        finally:
            engine.dispose()
    except Exception:
        return False
    return True


def docker_daemon_running() -> bool:
    """Whether the Docker daemon is up (``docker info`` succeeds).

    Returns ``False`` when the ``docker`` binary is absent or the daemon is not
    reachable, so a container-backed Postgres is only a *candidate* when Docker
    is genuinely running.
    """
    binary = shutil.which("docker")
    if binary is None:
        return False
    try:
        completed = subprocess.run(
            [binary, "info"],
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def detect_execution_mode(
    *,
    database_url: str | None = None,
    allow_docker: bool = True,
    force: ExecutionMode | None = None,
) -> ExecutionModeDecision:
    """Detect the execution mode in degrade-gracefully order.

    ``force`` pins a mode (used by tests and the ``--execution-mode`` flag) and
    skips probing. Otherwise: usable Postgres -> ``postgres``; else Docker
    daemon running (and ``allow_docker``) -> ``docker-postgres``; else
    ``in-process``.
    """
    if force is not None:
        return ExecutionModeDecision(
            mode=force, reason="forced by caller", database_url=None
        )
    url = database_url or os.environ.get(
        "DR_PLATFORM_TEST_DATABASE_URL", DEFAULT_TEST_DATABASE_URL
    )
    if postgres_available(url):
        return ExecutionModeDecision(
            mode=ExecutionMode.POSTGRES,
            reason=f"ambient Postgres reachable at {url}",
            database_url=url,
        )
    if allow_docker and docker_daemon_running():
        return ExecutionModeDecision(
            mode=ExecutionMode.DOCKER_POSTGRES,
            reason=(
                "no ambient Postgres; Docker daemon running -> a vanilla "
                "pgvector-less Postgres container can back the DBOS path"
            ),
            database_url=None,
        )
    return ExecutionModeDecision(
        mode=ExecutionMode.IN_PROCESS,
        reason=(
            "no ambient Postgres and no Docker daemon; using the in-process "
            "stage-03 driver + graph run + Result Store persistence path"
        ),
        database_url=None,
    )
