"""Tier 2 platform integration tests (Postgres + DBOS + dr-platform).

Run locally:
    uv sync --extra platform
    uv run pytest -m integration tests/integration/
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.environ.get("WHETSTONE_PLATFORM_INTEGRATION") != "1",
    reason="set WHETSTONE_PLATFORM_INTEGRATION=1 with Postgres+DBOS configured",
)
def test_inline_platform_copro_submit_to_result() -> None:
    """Full submit → admission → stages → run completion → OptimResult."""
    pytest.fail(
        "integration harness not wired in CI yet; implement against local "
        "Postgres using submit_optim_run and register_optim_pipeline"
    )
