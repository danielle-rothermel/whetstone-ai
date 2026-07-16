"""Run-store schema isolation regressions (``scripts/ci/integration.sh``).

The integration harness migrates the disposable database's ``public``
schema to head before this suite runs, so ``public`` already contains the
same-named Whetstone baseline — tables and append-only triggers — that
produced the ``v6accept_0713d`` duplicate-trigger failure. Every test here
proves the production migration path writes only to the fresh run schema.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, make_url

from whetstone.db import schema
from whetstone.platform.connections import (
    bind_schema,
    bind_schema_strict,
    render_connection_url,
)
from whetstone.platform.platform_db import (
    PLATFORM_VERSION_TABLE,
    assert_run_schema_owns_objects,
    ensure_whetstone_application_schema,
)

_PUBLIC_TABLE_OIDS = text(
    "SELECT c.relname, c.oid FROM pg_class c "
    "JOIN pg_namespace n ON n.oid = c.relnamespace "
    "WHERE n.nspname = 'public' AND c.relkind = 'r' "
    "AND c.relname LIKE 'whetstone%' ORDER BY c.relname"
)
_PUBLIC_TRIGGER_OIDS = text(
    "SELECT t.tgname, t.oid, t.tgrelid FROM pg_trigger t "
    "JOIN pg_class c ON c.oid = t.tgrelid "
    "JOIN pg_namespace n ON n.oid = c.relnamespace "
    "WHERE n.nspname = 'public' AND NOT t.tgisinternal "
    "ORDER BY t.tgname"
)


def _public_whetstone_oids(
    engine: Engine,
) -> tuple[tuple[tuple[str, int], ...], tuple[tuple[str, int, int], ...]]:
    with engine.connect() as connection:
        tables = tuple(
            (str(name), int(oid))
            for name, oid in connection.execute(_PUBLIC_TABLE_OIDS)
        )
        triggers = tuple(
            (str(name), int(oid), int(relid))
            for name, oid, relid in connection.execute(_PUBLIC_TRIGGER_OIDS)
        )
    return tables, triggers


def _assert_public_baseline_installed(engine: Engine) -> None:
    tables, triggers = _public_whetstone_oids(engine)
    assert schema.GENERATION_RUNS_TABLE in {name for name, _ in tables}
    assert f"tr_{schema.GENERATION_RUNS_TABLE}_append_only" in {
        name for name, _oid, _relid in triggers
    }


@pytest.fixture()
def admin_engine(postgres_base_url: str) -> Iterator[Engine]:
    engine = create_engine(postgres_base_url)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture()
def fresh_schema(admin_engine: Engine) -> Iterator[str]:
    name = f"whetstone_run_it{uuid.uuid4().hex[:10]}"
    with admin_engine.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA "{name}"'))
    try:
        yield name
    finally:
        with admin_engine.begin() as connection:
            connection.execute(
                text(f'DROP SCHEMA IF EXISTS "{name}" CASCADE')
            )


@pytest.mark.integration
def test_ensure_migrates_fresh_schema_despite_public_collision(
    postgres_base_url: str, admin_engine: Engine, fresh_schema: str
) -> None:
    """The 0713d regression: same-named public baseline must be untouched."""
    _assert_public_baseline_installed(admin_engine)
    before = _public_whetstone_oids(admin_engine)

    ensure_whetstone_application_schema(
        postgres_base_url, expected_schema=fresh_schema
    )

    assert _public_whetstone_oids(admin_engine) == before
    strict = create_engine(
        bind_schema_strict(postgres_base_url, fresh_schema)
    )
    runtime = create_engine(bind_schema(postgres_base_url, fresh_schema))
    try:
        with strict.connect() as connection:
            assert_run_schema_owns_objects(connection, fresh_schema)
        # Under the runtime fallback binding, run-schema objects must
        # shadow the same-named public baseline for unqualified names.
        with runtime.connect() as connection:
            assert_run_schema_owns_objects(connection, fresh_schema)
    finally:
        strict.dispose()
        runtime.dispose()


@pytest.mark.integration
def test_absent_expected_schema_fails_before_ddl(
    postgres_base_url: str, admin_engine: Engine
) -> None:
    before = _public_whetstone_oids(admin_engine)
    absent = f"whetstone_run_absent_{uuid.uuid4().hex[:8]}"

    with pytest.raises(ValueError, match="does not exist"):
        ensure_whetstone_application_schema(
            postgres_base_url, expected_schema=absent
        )

    with admin_engine.connect() as connection:
        still_absent = not connection.execute(
            text(
                "SELECT EXISTS (SELECT 1 FROM pg_namespace "
                "WHERE nspname = :name)"
            ),
            {"name": absent},
        ).scalar_one()
    assert still_absent
    assert _public_whetstone_oids(admin_engine) == before


@pytest.mark.integration
def test_pre_existing_application_table_in_fresh_schema_is_rejected(
    postgres_base_url: str, admin_engine: Engine, fresh_schema: str
) -> None:
    before = _public_whetstone_oids(admin_engine)
    with admin_engine.begin() as connection:
        connection.execute(
            text(
                f'CREATE TABLE "{fresh_schema}".'
                f"{schema.GENERATION_RUNS_TABLE} (run_id TEXT)"
            )
        )

    with pytest.raises(ValueError, match="not fresh"):
        ensure_whetstone_application_schema(
            postgres_base_url, expected_schema=fresh_schema
        )

    assert _public_whetstone_oids(admin_engine) == before


@pytest.mark.integration
def test_pre_existing_kernel_version_table_is_rejected(
    postgres_base_url: str, admin_engine: Engine, fresh_schema: str
) -> None:
    """A pre-stamped kernel version table must not silently no-op alembic."""
    before = _public_whetstone_oids(admin_engine)
    with admin_engine.begin() as connection:
        connection.execute(
            text(
                f'CREATE TABLE "{fresh_schema}".'
                f"{PLATFORM_VERSION_TABLE} (version_num TEXT)"
            )
        )

    with pytest.raises(ValueError, match="not fresh"):
        ensure_whetstone_application_schema(
            postgres_base_url, expected_schema=fresh_schema
        )

    assert _public_whetstone_oids(admin_engine) == before


@pytest.mark.integration
def test_url_object_rendered_string_psycopg_schema_and_auth(
    postgres_base_url: str, fresh_schema: str
) -> None:
    """URL object → rendered string → psycopg selects exactly the schema."""
    base = make_url(postgres_base_url)
    strict = bind_schema_strict(base, fresh_schema)
    rendered = render_connection_url(strict)

    assert "***" not in rendered
    if base.password is not None:
        assert make_url(rendered).password == base.password

    for engine in (create_engine(strict), create_engine(rendered)):
        try:
            with engine.connect() as connection:
                current = connection.execute(
                    text("SELECT current_schema()")
                ).scalar_one()
                effective = connection.execute(
                    text("SELECT current_schemas(false)")
                ).scalar_one()
            assert current == fresh_schema
            assert list(effective) == [fresh_schema]
        finally:
            engine.dispose()

    runtime = create_engine(bind_schema(base, fresh_schema))
    try:
        with runtime.connect() as connection:
            effective = connection.execute(
                text("SELECT current_schemas(false)")
            ).scalar_one()
        assert list(effective) == [fresh_schema, "public"]
    finally:
        runtime.dispose()
