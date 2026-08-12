from __future__ import annotations

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


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Apply ``slow`` to integration tests."""
    for item in items:
        if item.get_closest_marker("slow") is not None:
            continue
        if any(item.get_closest_marker(name) for name in _INTEGRATION_MARKERS):
            item.add_marker(pytest.mark.slow)
