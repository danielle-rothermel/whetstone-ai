from __future__ import annotations

import pytest

from whetstone.platform.connections import (
    bind_schema,
    bind_schema_strict,
    normalize_url,
    render_connection_url,
)


@pytest.mark.parametrize("scheme", ["postgresql", "postgresql+psycopg"])
def test_postgres_connection_policy_preserves_encoded_url_parts(
    scheme: str,
) -> None:
    url = normalize_url(
        f"{scheme}://operator:p%2Fss%40word@db.example/test"
        "?sslmode=require&application_name=fixture"
    )

    assert url.drivername == "postgresql+psycopg"
    assert url.username == "operator"
    assert url.password == "p/ss@word"
    assert url.query == {
        "sslmode": "require",
        "application_name": "fixture",
    }
    assert "***" in str(url)
    assert "p%2Fss%40word" in render_connection_url(url)


def test_runtime_schema_binding_keeps_public_extension_fallback() -> None:
    """Runtime paths need unqualified extension/function fallback to public."""
    bound = bind_schema(
        "postgresql://operator:p%2Fss@db.example/test?sslmode=require",
        "run_owned",
    )

    assert bound.drivername == "postgresql+psycopg"
    assert bound.query == {
        "sslmode": "require",
        "options": "-c search_path=run_owned,public",
    }
    rendered = render_connection_url(bound)
    assert "operator:p%2Fss@" in rendered
    assert "sslmode=require" in rendered
    assert "options=-c+search_path%3Drun_owned%2Cpublic" in rendered


def test_strict_schema_binding_excludes_public() -> None:
    """Migration/admin binding must never see same-named public relations."""
    bound = bind_schema_strict(
        "postgresql://operator:p%2Fss@db.example/test?sslmode=require",
        "run_owned",
    )

    assert bound.drivername == "postgresql+psycopg"
    assert bound.query == {
        "sslmode": "require",
        "options": "-c search_path=run_owned",
    }
    rendered = render_connection_url(bound)
    assert "operator:p%2Fss@" in rendered
    assert "public" not in rendered
    assert "options=-c+search_path%3Drun_owned" in rendered


def test_strict_schema_binding_rejects_non_identifier_schema() -> None:
    with pytest.raises(ValueError, match="SQL identifier"):
        bind_schema_strict("postgresql://db.example/test", "bad-schema")


def test_schema_binding_retains_existing_options() -> None:
    bound = bind_schema(
        "postgresql://operator:password@db.example/test"
        "?options=-c%20statement_timeout%3D1000&sslmode=require",
        "run_owned",
    )

    assert bound.query["options"] == (
        "-c statement_timeout=1000 -c search_path=run_owned,public"
    )
    assert bound.query["sslmode"] == "require"
