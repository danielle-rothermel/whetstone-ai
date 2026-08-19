"""Shared fixtures for Postgres + DBOS platform integration tests."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from sqlalchemy import Engine, create_engine, make_url, text

from dr_platform._core.ledger.schema import LedgerSchema
from dr_platform.runtime.database.migrate import upgrade_platform_schema

from tests.integration.platform_helpers import NOW

pytest.importorskip("dr_platform")

TEST_DATABASE_URL = os.environ.get(
    "WHETSTONE_TEST_DATABASE_URL",
    os.environ.get(
        "DR_PLATFORM_TEST_DATABASE_URL",
        "postgresql+psycopg:///whetstone_platform_test",
    ),
)


def _validate_test_database_url(database_url: str) -> None:
    url = make_url(database_url)
    if url.get_backend_name() != "postgresql":
        raise ValueError("integration tests require PostgreSQL")
    database_name = url.database
    if database_name is None or not database_name.endswith("_test"):
        raise ValueError("integration database name must end with '_test'")


def _verify_postgres_available(database_url: str) -> None:
    _validate_test_database_url(database_url)
    engine = create_engine(database_url)
    try:
        with engine.connect():
            pass
    except Exception:
        ci_value = os.environ.get("CI", "").lower()
        if ci_value and ci_value not in {"false", "0"}:
            raise
        pytest.skip(
            "postgres unavailable for platform integration tests "
            f"({database_url}); set WHETSTONE_TEST_DATABASE_URL or create "
            "whetstone_platform_test"
        )
    finally:
        engine.dispose()


def _reset_test_database(database_url: str) -> None:
    engine = create_engine(database_url)
    try:
        with engine.begin() as connection:
            connection.execute(text("DROP EXTENSION IF EXISTS pgcrypto"))
            connection.execute(text("DROP SCHEMA IF EXISTS dr_store CASCADE"))
            connection.execute(text("DROP SCHEMA public CASCADE"))
            connection.execute(text("CREATE SCHEMA public"))
            connection.execute(text("CREATE EXTENSION pgcrypto"))
    finally:
        engine.dispose()


def migrate_platform_schema(engine: Engine) -> LedgerSchema:
    upgrade_platform_schema(engine.url.render_as_string(hide_password=False))
    return LedgerSchema()


@pytest.fixture(scope="session")
def pg_url() -> str:
    _verify_postgres_available(TEST_DATABASE_URL)
    return TEST_DATABASE_URL


@pytest.fixture
def clean_pg(pg_url: str) -> str:
    _reset_test_database(pg_url)
    return pg_url


@pytest.fixture
def pg_engine(clean_pg: str) -> Iterator[Engine]:
    engine = create_engine(clean_pg)
    yield engine
    engine.dispose()
