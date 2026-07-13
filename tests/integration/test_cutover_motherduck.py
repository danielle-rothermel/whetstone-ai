from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from whetstone.platform.cutover_tooling import (
    _create_schema,
    _drop_schema,
    _require_schema_owner,
    _sqlalchemy_engine,
)


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


@pytest.mark.integration
def test_opt_in_motherduck_marker_constraints_reject_replacement() -> None:
    """Create one temporary schema only after an explicit DDL opt-in."""
    if os.environ.get("WHETSTONE_MOTHERDUCK_MARKER_DDL_PROBE") != "1":
        pytest.skip("MotherDuck marker DDL probe is not explicitly enabled")
    url = os.environ.get("MOTHERDUCK_DATABASE_URL")
    if not url:
        pytest.skip("MOTHERDUCK_DATABASE_URL is unavailable")
    suffix = uuid.uuid4().hex[:12]
    run_id = f"probe_{suffix}"
    schema = f"whetstone_marker_probe_{suffix}"
    digest = "a" * 64
    _create_schema(
        url,
        schema,
        environment="MOTHERDUCK_DATABASE_URL",
        run_id=run_id,
        descriptor_sha256=digest,
    )
    try:
        engine = _sqlalchemy_engine(
            url, environment="MOTHERDUCK_DATABASE_URL"
        )
        try:
            with pytest.raises(DBAPIError), engine.begin() as connection:
                connection.execute(
                    text(
                        f'UPDATE "{schema}"."whetstone_cutover_ownership" '
                        "SET descriptor_sha256=:replacement"
                    ),
                    {"replacement": "b" * 64},
                )
        finally:
            engine.dispose()
        _require_schema_owner(
            url,
            schema,
            environment="MOTHERDUCK_DATABASE_URL",
            run_id=run_id,
            descriptor_sha256=digest,
        )
    finally:
        _drop_schema(
            url, schema, environment="MOTHERDUCK_DATABASE_URL"
        )
