from __future__ import annotations

import importlib
import os
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from dbos import DBOS
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Engine

import dr_dspy.platform.graph_workflow  # noqa: F401
import tests.integration.dbos_test_workflows  # noqa: F401
from dr_dspy.db.migrations.url import normalize_postgresql_driver_url
from dr_dspy.harness import dbos as shared_dbos
from dr_dspy.platform.worker import DBOS_APP_NAME


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: Postgres and DBOS integration tests (opt-in)",
    )


def _normalize_database_url(database_url: str) -> str:
    if database_url.startswith("postgresql://"):
        return normalize_postgresql_driver_url(database_url)
    return database_url


def database_url_with_search_path(
    database_url: str,
    schema_name: str,
) -> str:
    option = quote(f"-csearch_path={schema_name},public", safe="")
    separator = "&" if "?" in database_url else "?"
    return f"{database_url}{separator}options={option}"


def _postgres_available(database_url: str) -> str | None:
    try:
        engine = create_engine(database_url)
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        engine.dispose()
    except Exception as exc:
        return str(exc)
    return None


def _apply_v1_migrations(
    connection: Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    domain_migration = importlib.import_module(
        "dr_dspy.db.migrations.versions.20260629_0001_v1_domain_schema"
    )
    append_only_migration = importlib.import_module(
        "dr_dspy.db.migrations.versions."
        "20260630_0001_append_only_outcome_triggers"
    )
    context = MigrationContext.configure(cast(Any, connection))
    monkeypatch.setattr(domain_migration, "op", Operations(context))
    domain_migration.upgrade()
    context = MigrationContext.configure(cast(Any, connection))
    monkeypatch.setattr(append_only_migration, "op", Operations(context))
    append_only_migration.upgrade()


@dataclass(frozen=True)
class AppPostgresSchema:
    engine: Engine
    schema_name: str
    database_url: str


@pytest.fixture()
def postgres_base_url() -> str:
    database_url = _normalize_database_url(
        os.environ.get(
            "DATABASE_URL",
            "postgresql+psycopg:///dr_dspy",
        )
    )
    skip_reason = _postgres_available(database_url)
    if skip_reason is not None:
        message = f"PostgreSQL unavailable: {skip_reason}"
        pytest.skip(message)  # ty: ignore[too-many-positional-arguments]
    return database_url


@pytest.fixture()
def app_postgres_schema(
    postgres_base_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[AppPostgresSchema]:
    schema_name = f"dr_dspy_it_{uuid.uuid4().hex}"
    engine = create_engine(postgres_base_url)
    try:
        with engine.begin() as connection:
            connection.execute(text(f"CREATE SCHEMA {schema_name}"))
            connection.execute(
                text(f"SET search_path TO {schema_name}, public")
            )
            _apply_v1_migrations(connection, monkeypatch)
        yield AppPostgresSchema(
            engine=engine,
            schema_name=schema_name,
            database_url=database_url_with_search_path(
                postgres_base_url,
                schema_name,
            ),
        )
    finally:
        with engine.begin() as connection:
            connection.execute(
                text(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE")
            )
        engine.dispose()


@pytest.fixture()
def reset_dbos(
    app_postgres_schema: AppPostgresSchema,
    tmp_path: Path,
) -> Iterator[shared_dbos.EvalDbosConfig]:
    shared_dbos.destroy_dbos_runtime()
    system_db_path = tmp_path / "dbos_system.sqlite"
    system_database_url = f"sqlite:///{system_db_path}"
    config = shared_dbos.build_eval_dbos_config(
        database_url=app_postgres_schema.database_url,
        dbos_system_database_url=system_database_url,
        generation_concurrency=1,
        scoring_concurrency=1,
        database_url_error_suffix="for integration tests",
    )
    DBOS(
        config=shared_dbos.build_dbos_config(
            config,
            app_name=DBOS_APP_NAME,
        )
    )
    DBOS.reset_system_database()
    DBOS.listen_queues([])
    DBOS.launch()
    try:
        yield config
    finally:
        shared_dbos.destroy_dbos_runtime()
