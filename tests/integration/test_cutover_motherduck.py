from __future__ import annotations

import os

import pytest
from sqlalchemy import text

from whetstone.platform.cutover_tooling import _sqlalchemy_engine


@pytest.mark.integration
def test_opt_in_motherduck_connection_is_read_only() -> None:
    """Prove dialect initialization and SELECT only; never create or mutate."""
    if os.environ.get("WHETSTONE_MOTHERDUCK_READ_ONLY_PROBE") != "1":
        pytest.skip("MotherDuck read-only probe is not explicitly enabled")
    url = os.environ.get("MOTHERDUCK_DATABASE_URL")
    if not url:
        pytest.skip("MOTHERDUCK_DATABASE_URL is unavailable")

    engine = _sqlalchemy_engine(
        url, environment="MOTHERDUCK_DATABASE_URL"
    )
    try:
        with engine.connect() as connection:
            assert connection.execute(text("SELECT 1")).scalar_one() == 1
    finally:
        engine.dispose()
