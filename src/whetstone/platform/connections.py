"""Small, typed SQLAlchemy connection policy for Whetstone boundaries."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Literal

from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import URL, make_url
from sqlalchemy.pool import NullPool

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_POOLED_NEON_HOST = re.compile(
    r"^[a-z0-9.-]*-pooler\.[a-z0-9.-]+\.neon\.tech$", re.IGNORECASE
)


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


def require_direct_endpoint(url: str | URL) -> URL:
    """Reject pooled Neon hosts, which drop startup search_path options."""
    normalized = normalize_url(url)
    host = normalized.host or ""
    if _POOLED_NEON_HOST.fullmatch(host) is not None:
        raise ValueError(
            "pooled Neon endpoint (*-pooler.*.neon.tech) rejects startup "
            "search_path options and is unsafe for Whetstone boundaries; "
            "supply the explicitly direct (unpooled) Neon URL. Hostnames "
            "are never derived or rewritten."
        )
    return normalized


def _bind_search_path(url: str | URL, schema: str, search_path: str) -> URL:
    if _IDENTIFIER.fullmatch(schema) is None:
        raise ValueError("schema must be a SQL identifier")
    normalized = require_direct_endpoint(url)
    existing_options = normalized.query.get("options")
    options = f"-c search_path={search_path}"
    if isinstance(existing_options, str) and existing_options:
        options = f"{existing_options} {options}"
    return normalized.update_query_dict(
        {"options": options}
    )


def bind_schema(url: str | URL, schema: str) -> URL:
    """Bind a runtime search_path keeping the public extension fallback."""
    return _bind_search_path(url, schema, f"{schema},public")


def bind_schema_strict(url: str | URL, schema: str) -> URL:
    """Bind a migration/admin search_path to exactly the run schema."""
    return _bind_search_path(url, schema, schema)


def create_whetstone_engine(
    url: str | URL,
    *,
    boundary: DatabaseBoundary,
    pool_mode: PoolMode = "default",
) -> Engine:
    """Create a PostgreSQL engine with boundary-specific policy applied."""
    normalized = normalize_url(url)
    backend = normalized.get_backend_name()
    if backend == "postgresql":
        require_direct_endpoint(normalized)
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
