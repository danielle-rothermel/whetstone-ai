from __future__ import annotations

from collections.abc import Callable

import pytest
from sqlalchemy.engine import URL
from sqlalchemy.pool import NullPool

from whetstone.platform import connections, cutover_tooling
from whetstone.platform.connections import (
    DatabaseBoundary,
    bind_schema,
    bind_schema_strict,
    create_whetstone_engine,
    normalize_url,
    render_connection_url,
    require_direct_endpoint,
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


@pytest.mark.parametrize("binder", [bind_schema, bind_schema_strict])
def test_schema_binding_rejects_pooled_neon_endpoint(
    binder: Callable[[str, str], URL],
) -> None:
    with pytest.raises(ValueError, match=r"direct \(unpooled\) Neon URL"):
        binder(
            "postgresql://operator:secret"
            "@ep-cool-name-123456-pooler.us-east-2.aws.neon.tech/neondb"
            "?sslmode=require",
            "run_owned",
        )


def test_direct_neon_and_non_neon_pooler_hosts_are_accepted() -> None:
    direct = bind_schema_strict(
        "postgresql://operator:secret"
        "@ep-cool-name-123456.us-east-2.aws.neon.tech/neondb",
        "run_owned",
    )
    unrelated = bind_schema(
        "postgresql://operator:secret@db-pooler.example.com/test",
        "run_owned",
    )

    assert direct.host == "ep-cool-name-123456.us-east-2.aws.neon.tech"
    assert unrelated.host == "db-pooler.example.com"


def test_engine_policy_rejects_pooled_neon_endpoint() -> None:
    with pytest.raises(ValueError, match=r"direct \(unpooled\) Neon URL"):
        create_whetstone_engine(
            "postgresql://operator:secret"
            "@ep-cool-name-123456-pooler.us-east-2.aws.neon.tech/neondb",
            boundary=DatabaseBoundary.NEON_POSTGRES,
        )


def test_engine_policy_accepts_direct_neon_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[URL] = []
    marker = object()
    monkeypatch.setattr(
        connections,
        "create_engine",
        lambda url, **options: captured.append(url) or marker,
    )

    assert create_whetstone_engine(
        "postgresql://operator:secret"
        "@ep-cool-name-123456.us-east-2.aws.neon.tech/neondb",
        boundary=DatabaseBoundary.NEON_POSTGRES,
    ) is marker
    assert captured[0].host == "ep-cool-name-123456.us-east-2.aws.neon.tech"


def test_require_direct_endpoint_returns_normalized_url() -> None:
    url = require_direct_endpoint("postgresql://db.example/test")

    assert url.drivername == "postgresql+psycopg"


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


def test_engine_policy_applies_boundary_and_pool_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[URL, dict[str, object]]] = []
    marker = object()
    monkeypatch.setattr(
        connections,
        "create_engine",
        lambda url, **options: captured.append((url, options)) or marker,
    )

    assert create_whetstone_engine(
        "postgresql://operator:p%2Fss@db.example/test?sslmode=require",
        boundary=DatabaseBoundary.MOTHERDUCK_POSTGRES,
        pool_mode="ephemeral",
    ) is marker
    assert captured == [
        (
            normalize_url(
                "postgresql://operator:p%2Fss@db.example/test?sslmode=require"
            ),
            {"use_native_hstore": False, "poolclass": NullPool},
        )
    ]


def test_neon_engine_policy_does_not_disable_hstore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[URL, dict[str, object]]] = []
    monkeypatch.setattr(
        connections,
        "create_engine",
        lambda url, **options: captured.append((url, options)) or object(),
    )

    create_whetstone_engine(
        "postgresql://operator:password@db.example/test",
        boundary=DatabaseBoundary.NEON_POSTGRES,
    )

    assert captured[0][0].drivername == "postgresql+psycopg"
    assert captured[0][1] == {}


def test_engine_policy_rejects_non_postgres_boundary() -> None:
    with pytest.raises(ValueError, match="requires a PostgreSQL URL"):
        create_whetstone_engine(
            "duckdb:///md:warehouse",
            boundary=DatabaseBoundary.NEON_POSTGRES,
        )


def test_cutover_delegates_url_and_engine_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound_calls: list[tuple[str, str]] = []
    engine_calls: list[tuple[str, DatabaseBoundary]] = []
    marker = object()
    monkeypatch.setattr(
        cutover_tooling,
        "bind_schema",
        lambda url, schema: bound_calls.append((url, schema))
        or normalize_url(url),
    )
    monkeypatch.setattr(
        cutover_tooling,
        "render_connection_url",
        lambda url: "rendered-url",
    )
    monkeypatch.setattr(
        cutover_tooling,
        "create_whetstone_engine",
        lambda url, *, boundary: engine_calls.append((url, boundary))
        or marker,
    )

    assert (
        cutover_tooling._bound_url("postgresql://db/test", "run_schema")
        == "rendered-url"
    )
    assert cutover_tooling._sqlalchemy_engine(
        "postgresql://db/test", environment="MOTHERDUCK_DATABASE_URL"
    ) is marker

    assert bound_calls == [("postgresql://db/test", "run_schema")]
    assert engine_calls == [
        ("postgresql://db/test", DatabaseBoundary.MOTHERDUCK_POSTGRES)
    ]
