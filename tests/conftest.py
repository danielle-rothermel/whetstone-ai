from __future__ import annotations

from pathlib import Path

import pytest

_INTEGRATION_MARKERS = frozenset(
    {
        "process_integration",
        "process_guardian",
        "postgres_integration",
        "sqlite_time_integration",
        "sqlite_contention",
    }
)

_SLOW_PATH_PARTS = (
    "/tests/runner/",
    "/tests/evaluation/preview/",
)


def _is_slow_path(path: Path) -> bool:
    posix = path.as_posix()
    return any(part in posix for part in _SLOW_PATH_PARTS)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Apply ``slow`` to integration, runner, and preview tests."""
    for item in items:
        if item.get_closest_marker("slow") is not None:
            continue
        if _is_slow_path(item.path):
            item.add_marker(pytest.mark.slow)
            continue
        if any(item.get_closest_marker(name) for name in _INTEGRATION_MARKERS):
            item.add_marker(pytest.mark.slow)
