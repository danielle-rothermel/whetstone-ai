"""Shared hooks for pathway integration lanes."""

from __future__ import annotations

from pathlib import Path

import pytest

_PATHWAYS_ROOT = Path(__file__).resolve().parent


def _is_pathway_test(item: pytest.Item) -> bool:
    try:
        item.path.relative_to(_PATHWAYS_ROOT)
    except ValueError:
        return False
    return True


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark every collected pathway test for lane filtering."""
    for item in items:
        if not _is_pathway_test(item):
            continue
        if item.get_closest_marker("pathway") is None:
            item.add_marker(pytest.mark.pathway)
        if item.get_closest_marker("slow") is None:
            item.add_marker(pytest.mark.slow)
