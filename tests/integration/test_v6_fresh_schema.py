"""Fresh-schema integration contracts run by ``scripts/ci/integration.sh``."""

from __future__ import annotations

import os

import pytest
from sqlalchemy import create_engine, inspect

from whetstone.db import schema


@pytest.mark.integration
def test_fresh_database_has_the_complete_v6_domain_schema() -> None:
    """The disposable database is migrated from baseline, never stamped."""
    engine = create_engine(os.environ["DATABASE_URL"])
    try:
        tables = set(inspect(engine).get_table_names())
    finally:
        engine.dispose()

    assert set(schema.V6_TABLE_NAMES).issubset(tables)
    assert schema.GENERATION_RUNS_TABLE in tables
    assert schema.SCORE_ATTEMPTS_TABLE in tables
    assert schema.SCORE_HARNESS_FAILURES_TABLE in tables
