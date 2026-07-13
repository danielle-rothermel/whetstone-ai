"""Small, typed SQLAlchemy connection policy for Whetstone boundaries."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Literal

from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import URL, make_url
from sqlalchemy.pool import NullPool

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class DatabaseBoundary(StrEnum):
    APPLICATION = "application"
    DBOS_SYSTEM = "dbos_system"
    SOURCE_ADMIN = "source_admin"
    SOURCE_SCHEMA = "source_schema"
    MOTHERDUCK_POSTGRES = "motherduck_postgres"
    NEON_POSTGRES = "neon_postgres"


PoolMode = Literal["default", "ephemeral"]


def normalize_url(value: str | URL) -> URL:
    """Parse a URL and select psycopg 3 for a bare PostgreSQL scheme."""
    url = value if isinstance(value, URL) else make_url(value)
    if url.drivername == "postgresql":
        return url.set(drivername="postgresql+psycopg")
    return url


def render_connection_url(url: str | URL) -> str:
    """Render credentials for consumers that cannot accept a URL object."""
    return normalize_url(url).render_as_string(hide_password=False)


def bind_schema(url: str | URL, schema: str) -> URL:
    """Bind PostgreSQL search_path while retaining all existing URL options."""
    if _IDENTIFIER.fullmatch(schema) is None:
        raise ValueError("schema must be a SQL identifier")
    normalized = normalize_url(url)
    existing_options = normalized.query.get("options")
    options = f"-c search_path={schema},public"
    if isinstance(existing_options, str) and existing_options:
        options = f"{existing_options} {options}"
    return normalized.update_query_dict(
        {"options": options}
    )


def create_whetstone_engine(
    url: str | URL,
    *,
    boundary: DatabaseBoundary,
    pool_mode: PoolMode = "default",
) -> Engine:
    """Create a PostgreSQL engine with boundary-specific policy applied."""
    normalized = normalize_url(url)
    backend = normalized.get_backend_name()
    if backend != "postgresql" and not (
        boundary is DatabaseBoundary.MOTHERDUCK_POSTGRES
        and backend == "duckdb"
    ):
        raise ValueError(f"{boundary.value} requires a PostgreSQL URL")
    options: dict[str, object] = {}
    if (
        boundary is DatabaseBoundary.MOTHERDUCK_POSTGRES
        and backend == "postgresql"
    ):
        options["use_native_hstore"] = False
    if pool_mode == "ephemeral":
        options["poolclass"] = NullPool
    return create_engine(normalized, **options)
