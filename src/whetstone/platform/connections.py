"""Small, typed SQLAlchemy connection policy for Whetstone boundaries."""

from __future__ import annotations

import re

from sqlalchemy.engine import URL, make_url

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def normalize_url(value: str | URL) -> URL:
    """Parse a URL and select psycopg 3 for a bare PostgreSQL scheme."""
    url = value if isinstance(value, URL) else make_url(value)
    if url.drivername == "postgresql":
        return url.set(drivername="postgresql+psycopg")
    return url


def render_connection_url(url: str | URL) -> str:
    """Render credentials for consumers that cannot accept a URL object."""
    return normalize_url(url).render_as_string(hide_password=False)


def _bind_search_path(url: str | URL, schema: str, search_path: str) -> URL:
    if _IDENTIFIER.fullmatch(schema) is None:
        raise ValueError("schema must be a SQL identifier")
    normalized = normalize_url(url)
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
