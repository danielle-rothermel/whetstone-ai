"""Shared hooks for pathway integration lanes."""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark every collected pathway test for lane filtering."""
    for item in items:
        if item.get_closest_marker("pathway") is None:
            item.add_marker(pytest.mark.pathway)
